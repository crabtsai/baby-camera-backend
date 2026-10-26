import numpy as np
import cv2

from backend.high_frequency_motion import HighFrequencyMotion, rhythmic_motion
from backend.schemas import DetectionResult
from tests.test_event_alert_policy import make_manager


def signal(hz, fps=20, amplitude=1):
    t=np.arange(0,4.1,1/fps)
    return t,(amplitude*np.sin(2*np.pi*hz*t))[:,None]


def test_rhythmic_fast_motion_but_not_slow_static_or_single_jerk():
    assert rhythmic_motion(*signal(3))["abnormal"]
    assert not rhythmic_motion(*signal(.7))["abnormal"]
    assert not rhythmic_motion(*signal(3,amplitude=.02))["abnormal"]
    t,x=signal(3);x[:]=0;x[35:38]=2
    assert not rhythmic_motion(t,x)["abnormal"]
    assert not rhythmic_motion(t,np.zeros_like(x))["abnormal"]


def test_insufficient_sampling_and_discontinuity_are_unknown():
    assert not rhythmic_motion(*signal(3,fps=8))["evaluable"]
    t,x=signal(3);t[40:]+=.5
    assert not rhythmic_motion(t,x)["evaluable"]


def test_rhythm_fit_uses_actual_irregular_frame_times():
    t=np.cumsum(np.tile([.04,.09,.06,.07],20))
    x=np.sin(2*np.pi*3*t)[:,None]
    result=rhythmic_motion(t,x)
    assert result["evaluable"] and result["abnormal"]
    assert result["peak_hz"]==3.
    t,x=signal(3);t[20]=t[19]
    assert not rhythmic_motion(t,x)["evaluable"]


def texture_frame(offset=0,global_offset=0):
    rng=np.random.default_rng(9)
    image=rng.integers(30,220,(180,320),dtype=np.uint8)
    patch=rng.integers(30,220,(80,80),dtype=np.uint8)
    # Fixed ROI with texture moving inside it; background remains static.
    patch=cv2.warpAffine(patch,np.float32([[1,0,offset],[0,1,0]]),(80,80),borderMode=cv2.BORDER_REFLECT)
    image[60:140,120:200]=patch
    image=cv2.warpAffine(image,np.float32([[1,0,global_offset],[0,1,0]]),(320,180),borderMode=cv2.BORDER_REFLECT)
    return cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)


def test_real_optical_flow_rhythmic_roi_and_global_camera_shake():
    monitor=HighFrequencyMotion(lambda *a:None)
    region=(120/320,60/180,200/320,140/180)
    out=None
    for t in np.arange(0,4.4,.05):
        out=monitor.process(texture_frame(offset=2*np.sin(2*np.pi*3*t)),float(t),(region,float(t)))
    assert out["evaluable"] and out["abnormal"]
    monitor._reset()
    for t in np.arange(0,4.4,.05):
        out=monitor.process(texture_frame(global_offset=3*np.sin(2*np.pi*3*t)),float(t),(region,float(t)))
    assert not out.get("abnormal",False)
    assert not monitor.process(texture_frame(),5,None)["evaluable"]
    assert not monitor.process(texture_frame(),5,(region,1))["evaluable"]


def test_alert_needs_sustained_results_and_stays_local(monkeypatch):
    now=[1000.]
    monkeypatch.setattr("backend.event_manager.time.monotonic",lambda:now[0])
    manager,outbound=make_manager()
    frame=np.zeros((20,20,3),dtype=np.uint8)
    result=DetectionResult("test","high_frequency_motion_warning",True,0.,"test",{
        "evaluable":True,"max_observation_gap_sec":1.})
    for _ in range(4):
        assert manager.handle_result(frame,result) is None
        now[0]+=.5
    assert manager.handle_result(frame,result) is not None
    assert manager.local_airway_alert() is not None
    assert not outbound.sent
