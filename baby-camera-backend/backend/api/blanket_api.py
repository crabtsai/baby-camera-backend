"""Local polygon annotation UI. Saving a label never deploys a model."""
import json
import math
import re
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

router=APIRouter()
ROOT=Path("data/airway/blanket/pending")
LOCK=threading.Lock()


def sample_path(camera, name):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",camera) or not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg",name):
        raise HTTPException(400,"Invalid sample name")
    path=(ROOT/camera/name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():raise HTTPException(404,"Sample not found")
    return path


class Annotation(BaseModel):
    polygons: list[list[tuple[float,float]]] = Field(max_length=30)
    reviewed: bool
    negative: bool=False
    split: str="train"


def validate_annotation(annotation):
    if not annotation.reviewed or annotation.split not in {"train","val","test"}:
        raise HTTPException(400,"Confirm review and choose a dataset split")
    if not annotation.polygons and not annotation.negative:
        raise HTTPException(400,"Draw blanket polygons or explicitly confirm no blanket")
    if annotation.polygons and annotation.negative:raise HTTPException(400,"Negative image cannot have polygons")
    for polygon in annotation.polygons:
        if not 3<=len(polygon)<=2000:raise HTTPException(400,"Polygon needs 3 to 2000 vertices")
        if any(not math.isfinite(v) or not 0<=v<=1 for point in polygon for v in point):
            raise HTTPException(400,"Invalid polygon coordinates")
        area=abs(sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1]-polygon[(i+1)%len(polygon)][0]*polygon[i][1]
                     for i in range(len(polygon))))/2
        if area<.00001:raise HTTPException(400,"Polygon area is too small")


@router.get("/blanket-learning/samples")
def samples():
    return [{"camera":p.parent.name,"name":p.name,"reviewed":p.with_suffix(".review.json").exists()}
            for p in sorted(ROOT.glob("*/*.jpg")) if p.with_suffix(".json").is_file()]


@router.get("/blanket-learning/image/{camera}/{name}")
def sample_image(camera:str,name:str):
    return FileResponse(sample_path(camera,name),media_type="image/jpeg")


@router.get("/blanket-learning/label/{camera}/{name}")
def label(camera:str,name:str):
    path=sample_path(camera,name)
    for suffix in [".review.json",".proposal.json"]:
        if path.with_suffix(suffix).exists():return json.loads(path.with_suffix(suffix).read_text(encoding="utf-8"))
    return {"polygons":[],"reviewed":False}


@router.post("/blanket-learning/label/{camera}/{name}")
def save_label(camera:str,name:str,annotation:Annotation):
    validate_annotation(annotation)
    path=sample_path(camera,name)
    with LOCK:
        metadata=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        # One capture session must never span training and held-out splits.
        for reviewed in path.parent.glob("*.review.json"):
            if reviewed==path.with_suffix(".review.json"):continue
            other=json.loads(reviewed.read_text(encoding="utf-8"))
            if other.get("session")==metadata["session"] and other["split"]!=annotation.split:
                raise HTTPException(409,"同一拍攝時段必須使用相同資料集；另一次拍攝可留作驗證／測試")
        data={**annotation.model_dump(),"session":metadata["session"],"camera":camera}
        target=path.with_suffix(".review.json")
        temporary=path.with_suffix(".review.tmp")
        temporary.write_text(json.dumps(data),encoding="utf-8")
        temporary.replace(target)
    return {"saved":True,"training_started":False}


@router.get("/blanket-learning",response_class=HTMLResponse)
def page():
    return '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>棉被學習資料</title>
<style>body{font:18px system-ui;max-width:980px;margin:24px auto;padding:16px;background:#f5f6fa}canvas{width:100%;background:#ddd;cursor:crosshair}button,select{font:inherit;margin:5px;padding:8px}#status{white-space:pre-wrap}</style>
<h1>棉被學習資料</h1><a href="/camera">返回即時監看</a>
<p>照片只存本機。沿棉被邊緣依序點選，再按「完成輪廓」。綠線可能是模型建議，請確認或重畫後儲存。拍照不等於模型已學會。</p>
<select id="samples"></select><button id="reload">重新載入</button><canvas id="canvas"></canvas>
<button id="finish">完成輪廓</button><button id="undo">撤銷一點</button><button id="clear">清除輪廓</button>
<p><label><input type="checkbox" id="negative">這張照片沒有棉被</label></p>
<p><label><input type="checkbox" id="reviewed">我已確認所有棉被輪廓，沒有把寶寶或床墊誤標成棉被</label></p>
<label>用途 <select id="split"><option value="train">訓練</option><option value="val">驗證</option><option value="test">保留測試</option></select></label>
<button id="save">儲存確認標註</button><p>同一拍攝時段必須使用相同用途，另一次拍攝留作驗證或測試。儲存不會立即重訓或更換警報模型。</p><p id="status" role="status"></p>
<script>
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d'),img=new Image();
let items=[],current=null,polygons=[],points=[],generation=0,ready=false;
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0);for(const poly of [...polygons,points]){if(!poly.length)continue;ctx.beginPath();poly.forEach((p,i)=>i?ctx.lineTo(p[0]*canvas.width,p[1]*canvas.height):ctx.moveTo(p[0]*canvas.width,p[1]*canvas.height));if(poly!==points)ctx.closePath();ctx.strokeStyle='#00ff77';ctx.lineWidth=2;ctx.stroke();}}
async function select(){const token=++generation;ready=false;ctx.clearRect(0,0,canvas.width,canvas.height);current=items[$('samples').value];polygons=[];points=[];$('reviewed').checked=false;$('negative').checked=false;$('status').textContent='載入中';if(!current)return;const key=encodeURIComponent(current.camera)+'/'+encodeURIComponent(current.name);const response=await fetch('/blanket-learning/label/'+key);if(!response.ok)throw Error('標註讀取失敗');const data=await response.json();if(token!==generation)return;polygons=data.polygons||[];$('split').value=data.split||'train';img.onload=()=>{if(token!==generation)return;canvas.width=img.width;canvas.height=img.height;ready=true;draw();$('status').textContent=data.reviewed?'已標註，可修正':'待確認；模型建議不會直接當訓練答案';};img.src='/blanket-learning/image/'+key;}
async function reload(){const response=await fetch('/blanket-learning/samples');if(!response.ok)throw Error('讀取失敗');items=await response.json();$('samples').replaceChildren(...items.map((s,i)=>new Option(s.camera+' '+s.name+(s.reviewed?' 已標註':''),i)));if(!items.length){current=null;$('status').textContent='尚無照片；確認人物在床後會自動收集';return;}await select();}
function run(fn){return async()=>{try{await fn();}catch(e){$('status').textContent=e.message;}};}
canvas.onclick=e=>{if(!current || !ready)return;const r=canvas.getBoundingClientRect();points.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);$('reviewed').checked=false;draw();};
$('finish').onclick=()=>{if(points.length<3)return;polygons.push(points);points=[];$('reviewed').checked=false;draw();};
$('undo').onclick=()=>{points.pop();$('reviewed').checked=false;draw();};$('clear').onclick=()=>{polygons=[];points=[];$('reviewed').checked=false;draw();};
$('reload').onclick=run(reload);$('samples').onchange=run(select);
$('save').onclick=run(async()=>{if(!current || !ready)return;if(points.length)throw Error('請先完成輪廓');const response=await fetch('/blanket-learning/label/'+encodeURIComponent(current.camera)+'/'+encodeURIComponent(current.name),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({polygons,reviewed:$('reviewed').checked,negative:$('negative').checked,split:$('split').value})});const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'儲存失敗，請檢查輪廓與確認勾選');$('status').textContent='已儲存人工確認的標註，供下一次訓練使用';});run(reload)();
</script></html>'''
