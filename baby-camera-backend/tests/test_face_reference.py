from dataclasses import replace

import numpy as np

from backend.detector.face_reference import FaceReferenceMonitor
from backend.schemas import DetectionResult
from tests.test_event_alert_policy import make_manager


def parts(box=(20,20,80,80)):
    return DetectionResult("test", "face_parts_visibility", False, 0., "", {
        "detections": [{"label":"face_estimate", "box":box}]})


def test_reference_needs_consecutive_stable_frames_and_never_learns_occlusion():
    guard = FaceReferenceMonitor()
    frame = np.random.default_rng(12).integers(0,256,(100,100,3),dtype=np.uint8)
    assert not guard.observe(frame,parts(),True,5).extra["reference_ready"]
    guard.observe(frame,None,True,5)
    assert not guard.observe(frame,parts(),True,5).extra["reference_ready"]
    guard.observe(frame,parts(),True,5)
    assert guard.observe(frame,parts(),True,5).extra["capture_reference"]
    reference = guard.reference.copy()
    assert not guard.observe(frame,parts(),True,5).is_abnormal
    for _ in range(4):
        assert guard.observe(frame,None,True,5).is_abnormal
    assert np.array_equal(reference,guard.reference)
    assert not guard.observe(frame,parts(),True,5).is_abnormal
    assert guard.observe(frame,parts(),False,5).is_abnormal


def test_appearance_change_and_gap(monkeypatch):
    now=[10.]
    monkeypatch.setattr("backend.detector.face_reference.time.monotonic",lambda:now[0])
    guard=FaceReferenceMonitor()
    frame=np.zeros((100,100,3),dtype=np.uint8)
    frame[:,::2]=255
    guard.observe(frame,parts(),True,5)
    guard.observe(frame,parts(),True,5)
    now[0]+=10
    assert not guard.observe(frame,parts(),True,5).extra["reference_ready"]
    guard.observe(frame,parts(),True,5)
    guard.observe(frame,parts(),True,5)
    result=guard.observe(255-frame,parts(),True,5)
    assert result.is_abnormal and not result.extra["obstruction_confirmed"]


def test_snapshot_on_presence_and_reference_with_local_alarm(monkeypatch):
    now=[1000.]
    monkeypatch.setattr("backend.event_manager.time.monotonic",lambda:now[0])
    manager,outbound=make_manager()
    captures=[]
    manager.snapshot_service.save_snapshot=lambda *a,**kw: captures.append(a[1].event_type) or "local.jpg"
    frame=np.zeros((10,10,3),dtype=np.uint8)
    presence=DetectionResult("test","bed_occupancy",False,0.,"",{"state":"present"})
    manager.handle_result(frame,presence)
    manager.handle_result(frame,presence)
    assert captures==["bed_occupancy"]
    result=DetectionResult("test","face_reference_warning",False,0.,"",{"capture_reference":True})
    manager.handle_result(frame,result)
    assert manager.reference_snapshot=="local.jpg"
    abnormal=replace(result,is_abnormal=True,extra={"max_observation_gap_sec":5})
    manager.handle_result(frame,abnormal)
    now[0]+=4
    manager.handle_result(frame,abnormal)
    assert manager.local_airway_alert() is None
    now[0]+=1
    assert manager.handle_result(frame,abnormal) is not None
    assert manager.local_airway_alert() is not None
    assert not outbound.sent


def test_crop_landmarks_restore_full_image_coordinates():
    from backend.detector.baby_face_parts import BabyFacePartsDetector
    from tests.test_baby_face_parts import mesh
    observed=[]
    detector=BabyFacePartsDetector(predict_fn=lambda image: observed.append(image.shape) or [mesh()])
    result=detector.detect(np.zeros((200,400,3),dtype=np.uint8),region=(.4,.4,.6,.6))
    assert observed[0][0] < 200 and observed[0][1] < 400
    box=result.extra["detections"][0]["box"]
    assert 100 < box[0] < box[2] < 300
    assert not result.extra["visibility_confirmed"]


def test_pose_rotation_maps_boxes_back_to_original_frame():
    from types import SimpleNamespace
    from backend.detector.safety_pipeline import SceneModels, SafetyOptions
    class Values:
        def __init__(self, data): self.data=data
        def cpu(self): return self
        def tolist(self): return self.data
    class Boxes:
        xyxy=Values([[20,30,80,150]])
        def __len__(self): return 1
    result=SimpleNamespace(boxes=Boxes(),keypoints=SimpleNamespace(
        xy=Values([[[40,60],[60,60],[50,80],[50,90],[50,100]]]),
        conf=Values([[1,1,1,1,1]])))
    model=SceneModels("","","",SafetyOptions(pose_rotation=90))
    seen=[]
    model.pose=SimpleNamespace(predict=lambda image,**kw: seen.append(image.shape) or [result])
    scene=model(np.zeros((100,200,3),dtype=np.uint8))
    assert seen==[(200,100,3)]
    assert np.allclose(scene["people"][0]["body"],(.25,.2,.85,.8))
