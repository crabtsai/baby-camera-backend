"""Local YOLO blanket segmentation: prepare public labels, train, propose annotations.

Predictions are never silently promoted to ground truth or deployed as alarms.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import yaml


def validate_label(text):
    for line in text.splitlines():
        values = [float(v) for v in line.split()]
        if not values:
            continue
        if len(values) < 7 or len(values) % 2 != 1 or values[0] != 0:
            raise ValueError("Expected class 0 blanket polygon with at least 3 points")
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values[1:]):
            raise ValueError("Polygon coordinates must be finite and normalized")


def prepare(source, output):
    if output.exists():
        raise ValueError("Output already exists; choose a new dataset directory")
    samples=[]
    for src, split in [("train","train"),("valid","val"),("test","test")]:
        for image in sorted((source/src/"images").glob("*")):
            if image.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
            label=source/src/"labels"/(image.stem+".txt")
            text=label.read_text(encoding="utf-8")
            validate_label(text)
            if cv2.imread(str(image)) is None: raise ValueError("Unreadable image")
            samples.append((image,label,split))
    if any(not any(r[2]==s for r in samples) for s in ("train","val","test")):
        raise ValueError("All source splits must contain annotated images")
    for image,label,split in samples:
        for kind,src in [("images",image),("labels",label)]:
            target=output/kind/split/src.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,target)
    (output/"dataset.yaml").write_text(yaml.safe_dump({"path":str(output.resolve()),
        "train":"images/train","val":"images/val","test":"images/test","names":{0:"blanket"}}),encoding="utf-8")
    return {s:sum(r[2]==s for r in samples) for s in ("train","val","test")}


def export_reviewed(base, pending, output):
    """Only explicit reviews enter training; preserve whole capture sessions."""
    from backend.api.blanket_api import Annotation, validate_annotation
    if output.exists():raise ValueError("Choose a new output directory")
    rows=[]
    sessions={}
    for review in sorted(pending.glob("*/*.review.json")):
        data=json.loads(review.read_text(encoding="utf-8"))
        validate_annotation(Annotation(**data))
        image=review.with_name(review.name.removesuffix(".review.json")+".jpg")
        metadata=json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))
        if not data.get("session") or data["session"]!=metadata.get("session"):
            raise ValueError("Review does not match capture session")
        session=(review.parent.name,data["session"])
        if session in sessions and sessions[session]!=data["split"]:raise ValueError("Session leakage")
        sessions[session]=data["split"]
        if cv2.imread(str(image)) is None:raise ValueError("Unreadable reviewed image")
        rows.append((image,data))
    if not rows:raise ValueError("No human-reviewed samples; proposals cannot be training labels")
    # Read a prepared base dataset; never modify its original hold-out partitions.
    for split in ("train","val","test"):
        for kind in ("images","labels"):
            source=base/kind/split
            if not source.is_dir():raise ValueError("Base must be a prepared dataset")
    shutil.copytree(base,output,ignore=shutil.ignore_patterns("*.cache"))
    counts={s:0 for s in ("train","val","test")}
    for image,data in rows:
        split=data["split"]
        name="local_"+image.parent.name+"_"+image.stem
        shutil.copy2(image,output/"images"/split/(name+".jpg"))
        lines=["0 "+" ".join(str(v) for point in poly for v in point) for poly in data["polygons"]]
        (output/"labels"/split/(name+".txt")).write_text("\n".join(lines),encoding="utf-8")
        counts[split]+=1
    (output/"dataset.yaml").write_text(yaml.safe_dump({"path":str(output.resolve()),
        "train":"images/train","val":"images/val","test":"images/test","names":{0:"blanket"}}),encoding="utf-8")
    (output/"local-review-report.json").write_text(json.dumps({"local_samples":counts,
        "target_camera_validated":False,"note":"Training does not automatically deploy weights"}),encoding="utf-8")
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare")
    prep.add_argument("--source",type=Path,required=True);prep.add_argument("--output",type=Path,required=True)
    train=sub.add_parser("train")
    train.add_argument("--data",type=Path,required=True)
    train.add_argument("--weights",type=Path,required=True)
    train.add_argument("--epochs",type=int,default=20)
    train.add_argument("--name",default="blanket-pilot")
    train.add_argument("--imgsz",type=int,default=320)
    propose=sub.add_parser("propose")
    propose.add_argument("--weights",type=Path,required=True)
    propose.add_argument("--pending",type=Path,default=Path("data/airway/blanket/pending"))
    export=sub.add_parser("export-reviewed")
    export.add_argument("--base",type=Path,required=True)
    export.add_argument("--pending",type=Path,default=Path("data/airway/blanket/pending"))
    export.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.command=="prepare":
        print(json.dumps(prepare(args.source,args.output)));return
    if args.command=="export-reviewed":
        print(json.dumps(export_reviewed(args.base,args.pending,args.output)));return
    if not args.weights.is_file(): raise ValueError("Download weights explicitly before running")
    import torch
    from ultralytics import YOLO, settings
    torch.set_num_threads(2)
    settings.update({"sync":False,"wandb":False,"comet":False,"mlflow":False})
    model=YOLO(str(args.weights))
    if model.task!="segment": raise ValueError("A YOLO segmentation model is required")
    if args.command=="train":
        model.train(data=str(args.data.resolve()),epochs=args.epochs,imgsz=args.imgsz,batch=8,
                    device="cpu",workers=0,project=str(Path("runs/airway").resolve()),name=args.name,
                    exist_ok=False,plots=False,seed=17,amp=False,freeze=10)
        best=Path(model.trainer.best)
        evaluation=YOLO(str(best)).val(data=str(args.data.resolve()),split="test",device="cpu",
                                       workers=0,imgsz=args.imgsz,plots=False,project=str(Path("runs/airway").resolve()),name=args.name+"-test")
        report={"weights":str(best),"test_metrics":evaluation.results_dict,
                "deployment":"candidate_only_not_validated_on_live_camera"}
        (best.parent.parent/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report));return
    if set(model.names.values())!={"blanket"}: raise ValueError("Expected blanket-only model")
    count=0
    for image in sorted(args.pending.glob("*/*.jpg")):
        target=image.with_suffix(".proposal.json")
        if target.exists():continue
        result=model.predict(str(image),verbose=False,conf=.25,device="cpu")[0]
        polygons=[] if result.masks is None else [m.tolist() for m in result.masks.xyn]
        target.write_text(json.dumps({"reviewed":False,"weights":str(args.weights),
            "polygons":polygons,"confidence":result.boxes.conf.tolist(),
            "note":"Unreviewed predictions; do not use as training labels"}),encoding="utf-8")
        count+=1
    print(json.dumps({"proposals":count}))


if __name__=="__main__": main()
