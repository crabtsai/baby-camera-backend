"""Experimental rhythmic image-motion monitor; not seizure detection or diagnosis."""
import queue
import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

from backend.schemas import DetectionResult

logger=logging.getLogger(__name__)


def rhythmic_motion(times, vectors, low=2., high=4.):
    """Signed optical-flow samples, seconds and pixels at 320px image width."""
    t=np.asarray(times,dtype=float)
    values=np.asarray(vectors,dtype=float)
    if len(t)<40 or t[-1]-t[0]<3.8:
        return {"evaluable":False,"reason":"累積約四秒影像中"}
    dt=np.diff(t)
    fps=(len(t)-1)/(t[-1]-t[0])
    if np.any(dt<=0) or dt.max()>.18 or fps<max(12,high*3):
        return {"evaluable":False,"reason":"影像幀率不足或時間間隔不穩，無法判讀高頻抖動","fps":round(fps,1)}
    # Fit sinusoids at actual arrival times rather than assuming uniform RTSP
    # delivery. R-squared rejects isolated jerks and broadband movement.
    x=values-values.mean(axis=0)
    total=np.sum(x*x,axis=0)+1e-12
    frequencies=np.arange(low,high+.001,.25)
    powers=[]
    for hz in frequencies:
        phase=2*np.pi*hz*(t-t[0])
        design=np.stack([np.sin(phase),np.cos(phase),np.ones_like(t)],axis=1)
        fitted=design@np.linalg.lstsq(design,x,rcond=None)[0]
        powers.append(np.maximum(0,1-np.sum((x-fitted)**2,axis=0)/total))
    power=np.asarray(powers)
    peak_indices=np.argmax(power,axis=0)
    hz=frequencies[peak_indices]
    concentration=power[peak_indices,np.arange(x.shape[1])]
    amplitude=np.std(x,axis=0)
    active=np.mean(np.abs(x)>.12,axis=0)
    qualifies=(concentration>=.6)&(amplitude>=.2)&(active>=.5)
    index=int(np.argmax(np.where(qualifies,concentration,0)))
    return {"evaluable":True,"abnormal":bool(qualifies.any()),"fps":round(fps,1),
            "peak_hz":round(float(hz[index]),2) if qualifies.any() else None,
            "rhythm_fit_r_squared":float(concentration[index]),"flow_std_px":float(amplitude[index])}


class HighFrequencyMotion:
    def __init__(self, callback):
        self.callback=callback
        self.jobs=queue.Queue(maxsize=2)
        self.stop_event=threading.Event()
        self.lock=threading.Lock()
        self.subject=None
        self.last_submit=0.
        self.last_report=0.
        self.previous=None
        self.previous_at=None
        self.previous_roi=None
        self.samples=deque()
        self.dropped=0
        self.errors=0
        self.last_result={"evaluable":False,"message":"等待人物定位及高幀率影像"}
        self.thread=threading.Thread(target=self._loop,name="high-frequency-motion",daemon=True)

    def start(self):self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)

    def update_subject(self,result,captured_at):
        if result.event_type!="bed_occupancy":return
        with self.lock:
            region=result.extra.get("subject_region")
            self.subject=(region,captured_at) if result.extra.get("state")=="present" and region else None

    def submit(self,frame,captured_at):
        if captured_at-self.last_submit<.035:return
        self.last_submit=captured_at
        h,w=frame.shape[:2]
        small=cv2.resize(frame,(320,max(2,round(h*320/w))))
        with self.lock:subject=self.subject
        job=(small,captured_at,subject)
        try:self.jobs.put_nowait(job)
        except queue.Full:
            try:self.jobs.get_nowait()
            except queue.Empty:pass
            self.dropped+=1
            try:self.jobs.put_nowait(job)
            except queue.Full:pass

    def _reset(self):
        self.previous=None;self.previous_at=None;self.previous_roi=None;self.samples.clear()

    def process(self,frame,at,subject):
        unknown=lambda reason:{"evaluable":False,"reason":reason}
        if subject is None or not 0<=at-subject[1]<=2.5:
            self._reset();return unknown("人物定位未確認或已過期，暫停抖動判讀")
        region=np.asarray(subject[0],dtype=float)
        if region.shape!=(4,) or not np.isfinite(region).all() or np.any(region<0) or np.any(region>1):
            self._reset();return unknown("人物區域無效")
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape
        x1,y1,x2,y2=(region*[w,h,w,h]).astype(int)
        if x2-x1<24 or y2-y1<24 or gray.mean()<15 or cv2.Laplacian(gray,cv2.CV_64F).var()<12:
            self._reset();return unknown("人物太小、影像過暗或模糊")
        roi=(x1,y1,x2,y2)
        if self.previous is None or self.previous.shape!=gray.shape or not 0<at-self.previous_at<=.18:
            self.samples.clear();self.previous=gray;self.previous_at=at;self.previous_roi=roi
            return unknown("建立連續影像參考")
        if max(abs(a-b) for a,b in zip(roi,self.previous_roi))>20 or abs(float(gray.mean())-float(self.previous.mean()))>12:
            self._reset();return unknown("人物位置或亮度大幅改變，重新建立參考")
        flow=cv2.calcOpticalFlowFarneback(self.previous,gray,None,.5,3,15,3,5,1.2,0)
        elapsed=at-self.previous_at
        sample_time=(at+self.previous_at)/2
        self.previous=gray;self.previous_at=at;self.previous_roi=roi
        background=np.ones((h,w),dtype=bool)
        background[max(0,y1-8):min(h,y2+8),max(0,x1-8):min(w,x2+8)]=False
        background[:min(20,h)]=False  # exclude camera timestamp text
        texture=cv2.cornerMinEigenVal(gray,3)>1e-4
        background &= texture
        if background.sum()<100:
            self.samples.clear();return unknown("缺少穩定背景，無法排除鏡頭晃動")
        bg=flow[background]
        global_motion=np.median(bg,axis=0)
        if np.linalg.norm(global_motion)>.4 or np.mean(np.linalg.norm(bg,axis=1)>.8)>.25:
            self.samples.clear();return unknown("鏡頭或背景晃動，暫停高頻判讀")
        flow-=global_motion
        flow *= (1/15)/elapsed  # comparable motion amplitude at differing frame intervals
        vectors=[]
        for ys in np.array_split(np.arange(y1,y2),3):
            for xs in np.array_split(np.arange(x1,x2),3):
                tile=flow[ys[0]:ys[-1]+1,xs[0]:xs[-1]+1]
                vectors.extend(np.median(tile.reshape(-1,2),axis=0).tolist())
        self.samples.append((sample_time,vectors))
        while self.samples and sample_time-self.samples[0][0]>4.2:self.samples.popleft()
        return rhythmic_motion([s[0] for s in self.samples],[s[1] for s in self.samples])

    def _loop(self):
        while not self.stop_event.is_set():
            try:frame,at,subject=self.jobs.get(timeout=.2)
            except queue.Empty:continue
            try:
                if time.monotonic()-at>.3:
                    self._reset();data={"evaluable":False,"reason":"處理延遲，影像已過期"}
                else:data=self.process(frame,at,subject)
            except Exception:
                self.errors+=1
                logger.exception("High frequency motion analysis failed")
                self._reset();data={"evaluable":False,"reason":"抖動分析失敗"}
            # Unknown frames reset event continuity immediately; valid decisions at 2Hz.
            if data["evaluable"] and at-self.last_report<.5:continue
            self.last_report=at
            abnormal=data.get("abnormal",False)
            message=("偵測到持續高頻抖動，請立即查看寶寶；這不是抽搐診斷" if abnormal else
                     "未達高頻抖動提醒門檻；不代表已排除抽搐" if data["evaluable"] else data["reason"])
            extra={**data,"assessment":"abnormal" if abnormal else "unknown",
                   "observed_at":time.time(),"max_observation_gap_sec":1.,"experimental":True,
                   "diagnosis":False,"band_hz":[2,4],"dropped_frames":self.dropped}
            self.last_result={**extra,"message":message}
            try:self.callback(frame,DetectionResult("high_frequency_motion","high_frequency_motion_warning",abnormal,0.,message,extra))
            except Exception:
                self.errors+=1
                logger.exception("High frequency motion event recording failed")
                self.last_result={**self.last_result,"evaluable":False,"message":"高頻動作事件儲存失敗，請查看現場"}

    def status(self):
        return {"enabled":True,"alive":self.thread.is_alive(),"errors":self.errors,**self.last_result}
