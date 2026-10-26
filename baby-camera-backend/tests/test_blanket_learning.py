import json

import numpy as np
import pytest
from fastapi import HTTPException

from backend.api import blanket_api
from backend.api.blanket_api import Annotation, validate_annotation
from backend.detector.safety_pipeline import object_head_overlap
from backend.schemas import DetectionResult
from backend.services.blanket_collection import BlanketCollection
from scripts.blanket_model import export_reviewed, validate_label


def observation(state):
    return DetectionResult("test","bed_occupancy",False,0.,"",{"state":state})


def test_collection_presence_interval_limit_and_restart(tmp_path,monkeypatch):
    now=[100.]
    monkeypatch.setattr("backend.services.blanket_collection.time.monotonic",lambda:now[0])
    collector=BlanketCollection("camera",True,60,3,tmp_path)
    frame=np.zeros((30,40,3),dtype=np.uint8)
    collector.observe(frame,observation("unknown"));assert collector.count==0
    collector.observe(frame,observation("present"));assert collector.count==1
    collector.observe(frame,observation("present"));assert collector.count==1
    now[0]+=60
    collector.observe(frame,observation("present"));assert collector.count==2
    session=collector.session
    collector.observe(frame,observation("absent"))
    collector.observe(frame,observation("present"));assert collector.count==3
    assert collector.session!=session
    now[0]+=60
    collector.observe(frame,observation("present"));assert collector.count==3
    assert collector.error
    assert BlanketCollection("camera",True,60,3,tmp_path).count==3
    assert all(json.loads(p.read_text())["reviewed"] is False for p in tmp_path.glob("*/*.json"))


def test_write_failure_does_not_break_monitor(tmp_path,monkeypatch):
    collector=BlanketCollection("camera",True,root=tmp_path)
    monkeypatch.setattr("backend.services.blanket_collection.cv2.imwrite",lambda *a:False)
    collector.observe(np.zeros((10,10,3),dtype=np.uint8),observation("present"))
    assert collector.error and collector.count==0


@pytest.mark.parametrize("kwargs",[
    {"polygons":[],"reviewed":False,"negative":True},
    {"polygons":[],"reviewed":True},
    {"polygons":[[(0,0),(1,1),(2,0)]],"reviewed":True},
    {"polygons":[[(0,0),(1,1),(.5,.5)]],"reviewed":True},
])
def test_bad_annotations_rejected(kwargs):
    with pytest.raises(HTTPException):validate_annotation(Annotation(**kwargs))


def test_mask_does_not_warn_from_empty_box_corner():
    obj={"box":(0,0,1,1),"polygon":[[0,0],[1,0],[0,1]]}
    assert object_head_overlap(obj,(.8,.8,1,1))==0
    assert object_head_overlap(obj,(0,0,.2,.2))>.95


def test_review_and_export_require_explicit_human_labels(tmp_path,monkeypatch):
    pending=tmp_path/"pending"
    monkeypatch.setattr(blanket_api,"ROOT",pending)
    collector=BlanketCollection("camera",True,root=pending)
    collector.observe(np.zeros((30,40,3),dtype=np.uint8),observation("present"))
    image=next(pending.glob("*/*.jpg"))
    image.with_suffix(".proposal.json").write_text(json.dumps({"polygons":[[[0,0],[1,0],[0,1]]]}))
    with pytest.raises(ValueError,match="human-reviewed"):
        export_reviewed(tmp_path/"base",pending,tmp_path/"output")
    with pytest.raises(HTTPException):blanket_api.sample_path("..",image.name)
    annotation=Annotation(polygons=[[(0,0),(1,0),(0,1)]],reviewed=True)
    blanket_api.save_label("camera",image.name,annotation)
    base=tmp_path/"base"
    for s in ("train","val","test"):
        for k in ("images","labels"):(base/k/s).mkdir(parents=True)
    result=export_reviewed(base,pending,tmp_path/"output")
    assert result=={"train":1,"val":0,"test":0}
    label=next((tmp_path/"output"/"labels"/"train").glob("*.txt"))
    validate_label(label.read_text())


def test_shadow_model_never_triggers_confirmed_obstruction():
    from tests.test_safety_pipeline import pipeline,scene,results
    data=scene()
    data["objects"]=[{"label":"blanket","box":(.3,.1,.8,.8),"polygon":[[.3,.1],[.8,.1],[.8,.8],[.3,.8]]}]
    result=results(pipeline(data,objects_shadow=True))["head_obstruction_warning"]
    assert result.is_abnormal and not result.extra["evaluable"]
    assert result.extra["experimental"]


def test_automatic_proposals_match_captured_frame(tmp_path):
    frame=np.zeros((30,40,3),dtype=np.uint8)
    collector=BlanketCollection("camera",True,root=tmp_path)
    collector.observe(frame,observation("present"))
    polygon=[[0,0],[1,0],[0,1]]
    result=DetectionResult("test","sleep_area_object",True,0.,"",{
        "objects":[{"label":"blanket","polygon":polygon,"confidence":.6}]})
    collector.observe(frame,result)
    proposal=next(tmp_path.glob("*/*.proposal.json"))
    assert json.loads(proposal.read_text())["polygons"]==[polygon]
    assert not json.loads(proposal.read_text())["reviewed"]


def test_annotation_http_and_session_split_guard(tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setattr(blanket_api,"ROOT",tmp_path)
    collector=BlanketCollection("camera",True,interval=1,root=tmp_path)
    frame=np.zeros((20,20,3),dtype=np.uint8)
    collector.observe(frame,observation("present"))
    collector.last_at=None
    collector.observe(frame,observation("present"))
    app=FastAPI();app.include_router(blanket_api.router)
    with TestClient(app) as client:
        assert client.get('/blanket-learning').status_code==200
        samples=client.get('/blanket-learning/samples').json()
        assert len(samples)==2
        url='/blanket-learning/label/camera/'
        body={"polygons":[],"reviewed":True,"negative":True,"split":"train"}
        assert client.post(url+samples[0]['name'],json=body).status_code==200
        body['split']='test'
        assert client.post(url+samples[1]['name'],json=body).status_code==409
