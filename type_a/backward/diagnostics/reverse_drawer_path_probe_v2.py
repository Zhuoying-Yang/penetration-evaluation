#!/usr/bin/env python3
"""Relative gripper/drawer path diagnostic, not a 3D penetration classifier.

Tracks manually seeded 2D landmarks on the gripper and *structural* drawer
front-panel rim/edge. Quantifies a screen-space front-edge side change and
whether the gripper's chosen low tip stays below the projected top rim.

IMPORTANT limitations:
- A 2D sign change is not proof of passing through a 3D solid surface.
- A gripper tip above a projected rim is not proof the *entire gripper* cleared it.
- Transparent drawer points can drift to background features. Inspect overlays.
- Results from manually selected frames/seed locations are DEVELOPMENT diagnostics.

CPU only: cv2, numpy, Python stdlib. No API, SAM or CUDA.

Example on ERIS:
  python reverse_drawer_path_probe_v2.py --case 0056 \\
      --video /path/to/seed103/0056.mp4 --start 148 --end 156 \\
      --output /path/to/reverse_drawer_path_v2/0056 --preview-only
  python reverse_drawer_path_probe_v2.py --case 0056 \\
      --video /path/to/seed103/0056.mp4 --start 148 --end 156 \\
      --output /path/to/reverse_drawer_path_v2/0056

Seeds are tentative pixels in original 1280x720 source frame f148; inspect
seed_preview.jpg and adjust with --tip / --wall-top / --wall-bottom / --rim-far.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

# DEVELOPMENT landmark seeds only: do not regard these as validated annotations.
DEFAULT_SEEDS = {
    '0056': {
        'start':148,'end':156,
        'gripper_box':(556,269,613,387),
        'tip':(582,382),              # black lower gripper tip: inspect image
        'wall_top':(602,374),         # top of transparent front panel's LEFT vertical edge
        'wall_bottom':(598,452),      # same vertical edge at bottom
        'rim_far':(696,329),          # right end of *same* front panel's top rim
    },
}
COLOR = {'tip':(70,70,255), 'wall_top':(20,215,20),
         'wall_bottom':(0,155,0), 'rim_far':(255,170,0)}


def parse_xy(x, default=None):
    if x is None:
        if default is None: raise ValueError('Missing landmark coordinate')
        return np.array(default,dtype=np.float32)
    a=x.split(',')
    if len(a)!=2: raise ValueError(f'Point must be x,y: {x!r}')
    return np.array([float(a[0]),float(a[1])],dtype=np.float32)


def extract_start(cap, first):
    cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    ok,frame=cap.read()
    if not ok:raise RuntimeError(f'Could not decode frame {first}')
    return frame


def get_patch(gray, xy, radius=13):
    x=int(round(float(xy[0])));y=int(round(float(xy[1])))
    if x-radius<0 or y-radius<0 or x+radius>=gray.shape[1] or y+radius>=gray.shape[0]:return None
    return gray[y-radius:y+radius+1,x-radius:x+radius+1].copy()


def track_point(prev,cur, xy, patch_radius=13, search=20):
    """Local template match checked against bidirectional LK; no blind reseeding."""
    patch=get_patch(prev,xy,patch_radius)
    if patch is None:return None,0.,float('inf'),'patch_out_of_frame'
    x,y=[float(v) for v in xy]
    R=patch_radius;S=search
    x0=max(0,int(round(x))-R-S);x1=min(cur.shape[1],int(round(x))+R+S+1)
    y0=max(0,int(round(y))-R-S);y1=min(cur.shape[0],int(round(y))+R+S+1)
    roi=cur[y0:y1,x0:x1]
    if roi.shape[0]<patch.shape[0] or roi.shape[1]<patch.shape[1]:return None,0.,float('inf'),'search_out_of_frame'
    values=cv2.matchTemplate(roi,patch,cv2.TM_CCOEFF_NORMED)
    _,ncc,_,loc=cv2.minMaxLoc(values)
    proposal=np.array([x0+loc[0]+R,y0+loc[1]+R],np.float32)
    old=np.asarray(xy,np.float32).reshape(1,1,2)
    lk,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,old,None,winSize=(25,25),maxLevel=3,
                                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,35,.01))
    fb=float('inf'); lkpos=None
    if lk is not None and st is not None and st[0,0]:
        back,bst,_=cv2.calcOpticalFlowPyrLK(cur,prev,lk,None,winSize=(25,25),maxLevel=3,
                                            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,35,.01))
        if back is not None and bst is not None and bst[0,0]:
            fb=float(np.linalg.norm(back[0,0]-old[0,0]));lkpos=lk[0,0]
    step=float(np.linalg.norm(proposal-xy))
    if step>search+3:return None,float(ncc),fb,'jump'
    if not np.isfinite(ncc) or ncc<0.57:return None,float(ncc),fb,'weak_template'
    if lkpos is not None and fb<=2.5:
        agreement=float(np.linalg.norm(lkpos-proposal))
        if agreement>9:
            return None,float(ncc),fb,'LK_template_disagree'
    elif float(ncc)<0.84:
        return None,float(ncc),fb,'unverified_template'
    return proposal,float(ncc),fb,'ok'


def segment_projection(p,q,target_axis,axis=0,max_extrap=28):
    """Interpolate coordinate at given axis; reject unsupported extrapolation."""
    v=float(q[axis]-p[axis]);arr=np.array([p,q])[:,axis]
    if abs(v)<3:return float('nan')
    a=(float(target_axis)-float(p[axis]))/v
    if target_axis < min(arr)-max_extrap or target_axis > max(arr)+max_extrap:
        return float('nan')
    return float(p[1-axis]+a*(q[1-axis]-p[1-axis]))


def add_marks(im,points,framenum,validity,lines=True):
    im=im.copy()
    if lines and all(validity.get(x,False) for x in ('wall_top','wall_bottom')):
        a=tuple(np.round(points['wall_top']).astype(int));b=tuple(np.round(points['wall_bottom']).astype(int))
        cv2.line(im,a,b,(40,225,40),3,cv2.LINE_AA)
    if lines and all(validity.get(x,False) for x in ('wall_top','rim_far')):
        a=tuple(np.round(points['wall_top']).astype(int));b=tuple(np.round(points['rim_far']).astype(int))
        cv2.line(im,a,b,(10,235,230),3,cv2.LINE_AA)
    for name,v in points.items():
        if not np.isfinite(v).all(): continue
        xy=tuple(np.round(v).astype(int));c=COLOR[name] if validity.get(name,False) else (90,90,90)
        cv2.circle(im,xy,7,c,2,cv2.LINE_AA)
        cv2.drawMarker(im,xy,c,markerType=cv2.MARKER_CROSS,markerSize=13,thickness=1)
        cv2.putText(im,name,(xy[0]+7,xy[1]-7),cv2.FONT_HERSHEY_SIMPLEX,.5,c,2,cv2.LINE_AA)
    cv2.rectangle(im,(0,0),(900,50),(15,15,15),-1)
    cv2.putText(im,f'f{framenum}  gripper-tip / moving panel EDGE / rim   [2D DIAGNOSTIC]',
                (10,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),2,cv2.LINE_AA)
    return im


def geometry(points,valid):
    if not all(valid.values()):return float('nan'),float('nan'),'INVALID_TRACK'
    tip,top,bot,far=[points[k] for k in ('tip','wall_top','wall_bottom','rim_far')]
    # Compare tip x to tracked *vertical solid front-panel edge* at the tip height.
    wx=segment_projection(top,bot,tip[1],axis=1,max_extrap=30)
    # Calculate signed *image-plane* vertical tip clearance above front-panel rim.
    ry=segment_projection(top,far,tip[0],axis=0,max_extrap=35)
    if not np.isfinite(wx) or not np.isfinite(ry):return float('nan'),float('nan'),'OUTSIDE_PROJECTION'
    side_dx=float(tip[0]-wx)
    clearance=float(ry-tip[1]) # + pixel value means tip higher than rim in the IMAGE
    return side_dx,clearance,'OK'


def plot(rows,path):
    good=[r for r in rows if r['geometry_quality']=='OK']
    if not good:return
    W,H=1150,650;im=np.full((H,W,3),255,np.uint8)
    L,R,T,B=95,W-40,90,H-120
    lo=min(min(float(r['signed_side_dx_px']),float(r['projected_tip_clearance_px'])) for r in good)
    hi=max(max(float(r['signed_side_dx_px']),float(r['projected_tip_clearance_px'])) for r in good)
    p=max(8,(hi-lo)*.2);lo-=p;hi+=p
    xx=lambda f:round(L+(R-L)*(f-rows[0]['frame'])/max(1,rows[-1]['frame']-rows[0]['frame']))
    yy=lambda v:round(B-(v-lo)/(hi-lo)*(B-T))
    cv2.rectangle(im,(L,T),(R,B),(170,170,170),1)
    if lo<=0<=hi:cv2.line(im,(L,yy(0)),(R,yy(0)),(120,120,120),1)
    for k,color in [('signed_side_dx_px',(175,65,175)),('projected_tip_clearance_px',(25,150,40))]:
        prev=None
        for row in rows:
            v=float(row[k]);valid=np.isfinite(v) and row['geometry_quality']=='OK'
            if valid:
                now=(xx(row['frame']),yy(v))
                if prev is not None:cv2.line(im,prev,now,color,3,cv2.LINE_AA)
                cv2.circle(im,now,4,color,-1)
                prev=now
            else:prev=None
    cv2.putText(im,'Projected wall-side and gripper-tip rim clearance (IMAGE PLANE ONLY)',
                (L,43),cv2.FONT_HERSHEY_SIMPLEX,.68,(20,20,20),2)
    cv2.putText(im,'Purple: tip x - front-panel edge x at same y',
                (L,H-79),cv2.FONT_HERSHEY_SIMPLEX,.55,(175,65,175),2)
    cv2.putText(im,'Green: rim y - tip y; > 0 is higher in the IMAGE, NOT verified 3D clearance',
                (L,H-44),cv2.FONT_HERSHEY_SIMPLEX,.47,(25,130,30),2)
    for f in [rows[0]['frame'],rows[-1]['frame']]:
        cv2.putText(im,f'f{f}',(xx(f)-15,B+29),cv2.FONT_HERSHEY_SIMPLEX,.55,(50,50,50),1)
    cv2.imwrite(str(path),im)


def summarize(rows, margin=4, clearance_margin=5):
    reliable=[r for r in rows if r['geometry_quality']=='OK']
    # Require a pair of stable observations of opposite signed projected sides.
    states=[]
    for r in rows:
        d=float(r['signed_side_dx_px'])
        s=0 if r['geometry_quality']!='OK' or abs(d)<=margin else (1 if d>0 else -1)
        states.append(s)
    intervals=[]
    for i in range(1,len(rows)):
        if states[i]==0:continue
        for j in range(i):
            if states[j] and states[j]!=states[i]:
                # Both endpoints confident and no tracking gaps; don't claim a
                # zero-crossing across an unmeasured interval.
                subset=rows[j:i+1]
                if not all(r['geometry_quality']=='OK' for r in subset):continue
                checks=[float(r['projected_tip_clearance_px']) for r in subset]
                clearance_observed=any(c>clearance_margin for c in checks)
                intervals.append({'from_frame':rows[j]['frame'],'to_frame':rows[i]['frame'],
                                  'second_corner_verified_through_crossing':all(
                                     r.get('two_corner_agreement')=='AGREE' for r in subset),
                                  'from_side':states[j], 'to_side':states[i],
                                  'projected_tip_above_rim_observed':clearance_observed,
                                  'candidate_type':('POSSIBLE_OPENING_ROUTE' if clearance_observed
                                      else 'BELOW_RIM_2D_SIDE_SWITCH_UNVERIFIED')})
                break
        if intervals:break
    result={'frames_total':len(rows),'frames_usable_geometry':len(reliable),
            'usable_frames_with_independent_second_corner':sum(
                r['geometry_quality']=='OK' and r.get('two_corner_agreement')=='AGREE' for r in rows),
            'unverified_2d_side_switch_candidates':intervals,
            'candidate_count':len(intervals),
            'interpretation':'No valid 3D penetration / legal-exit classification is possible from these 2D tracks alone.',
            'warning':'Rim clearance is projected pixel-space only; full gripper volume and true wall depth unknown.'}
    if reliable:
        a,b=reliable[0],reliable[-1]
        result.update(first_usable_frame=a['frame'],last_usable_frame=b['frame'],
                      projected_side_dx_start=a['signed_side_dx_px'],
                      projected_side_dx_end=b['signed_side_dx_px'],
                      projected_clearance_start=a['projected_tip_clearance_px'],
                      projected_clearance_end=b['projected_tip_clearance_px'])
    return result



def detect_features(gray,box,maxcorners=80):
    x1,y1,x2,y2=box
    mask=np.zeros_like(gray,np.uint8)
    mask[y1:y2,x1:x2]=255
    v=cv2.goodFeaturesToTrack(gray,maxCorners=maxcorners,qualityLevel=.012,
                              minDistance=4,blockSize=5,mask=mask)
    if v is None:return np.empty((0,1,2),dtype=np.float32)
    return np.asarray(v,np.float32)


def track_features(prev,cur,old):
    if len(old)==0:return old,np.empty((0,),bool)
    new,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,old,None,winSize=(23,23),maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,35,.015))
    if new is None:return old,np.zeros(len(old),bool)
    back,bst,_=cv2.calcOpticalFlowPyrLK(cur,prev,new,None,winSize=(23,23),maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,35,.015))
    if back is None:return new,np.zeros(len(old),bool)
    fb=np.linalg.norm((back-old).reshape(-1,2),axis=1)
    step=np.linalg.norm((new-old).reshape(-1,2),axis=1)
    valid=(st.reshape(-1)>0)&(bst.reshape(-1)>0)&(fb<1.7)&(step<22)
    if int(valid.sum())>=4:
        d=(new-old).reshape(-1,2)
        med=np.median(d[valid],axis=0)
        res=np.linalg.norm(d-med,axis=1)
        mad=np.median(res[valid])
        valid&=res<max(2.5,3*mad)
    return new,valid


def parse_box(box,default,w,h):
    if box is None:
        if default is None:raise ValueError('Missing --gripper-box')
        values=list(default)
    else:values=[int(v) for v in box.split(',')]
    if len(values)!=4 or not (0<=values[0]<values[2]<=w and 0<=values[1]<values[3]<=h):
        raise ValueError('Invalid --gripper-box x1,y1,x2,y2')
    return tuple(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--start',type=int)
    p.add_argument('--end',type=int)
    p.add_argument('--tip',help='initial gripper lower-tip x,y in source-video pixels')
    p.add_argument('--gripper-box',help='initial gripper ROI x1,y1,x2,y2 (avoid drawer and table grid)')
    p.add_argument('--wall-top',help='front panel vertical edge top x,y')
    p.add_argument('--wall-bottom',help='same panel edge bottom x,y')
    p.add_argument('--rim-far',help='clearly visible *moving* top rim corner x,y')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview-only',action='store_true')
    p.add_argument('--patch-radius',type=int,default=13)
    p.add_argument('--search',type=int,default=20)
    a=p.parse_args()
    defaults=DEFAULT_SEEDS.get(a.case,{})
    start=a.start if a.start is not None else defaults.get('start')
    end=a.end if a.end is not None else defaults.get('end')
    if start is None or end is None or not (0<=start<end):p.error('Provide --start and --end, end > start >= 0')
    if a.patch_radius<6 or a.search<5:p.error('patch-radius>=6, search>=5')
    points0={k:parse_xy(getattr(a,k),defaults.get(k)) for k in ('tip','wall_top','wall_bottom','rim_far')}
    a.output.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(a.video))
    if not cap.isOpened():raise RuntimeError('Cannot open video '+str(a.video))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end>=total:raise RuntimeError(f'End {end} exceeds video n={total}')
    fps=float(cap.get(cv2.CAP_PROP_FPS)) or 24.
    frame=extract_start(cap,start);h,w=frame.shape[:2]
    box=parse_box(a.gripper_box,defaults.get('gripper_box'),w,h)
    for k,v in points0.items():
        if not (a.patch_radius+1<=v[0]<w-a.patch_radius-1 and a.patch_radius+1<=v[1]<h-a.patch_radius-1):
            raise ValueError(f'{k} seed {v} outside image {w}x{h}')
    gray0=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    grips0=detect_features(gray0,box)
    if len(grips0)<8:raise RuntimeError('Too few gripper KLT points; change --gripper-box')
    gripper_pts=grips0.copy();gripper_valid=np.ones(len(grips0),bool)
    rim=points0['rim_far'].copy()
    secondary=points0['wall_bottom'].copy()
    secondary_good=True
    quality={k:True for k in points0}
    preview=add_marks(frame,points0,start,quality)
    cv2.rectangle(preview,box[:2],box[2:],(70,70,255),2)
    for v in grips0.reshape(-1,2):cv2.circle(preview,tuple(np.rint(v).astype(int)),2,(50,70,255),-1)
    cv2.imwrite(str(a.output/'seed_preview.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,94])
    print('VIDEO:',a.video,'SIZE:',w,h,'FRAMES:',start,end,flush=True)
    for k in points0:print('SEED',k,points0[k].tolist(),flush=True)
    print('GRIPPER FEATURES:',len(grips0),'BOX:',box,flush=True)
    print('SEED PREVIEW:',a.output/'seed_preview.jpg',flush=True)
    print('METHOD: KLT median gripper translation; one rigid drawer rim corner; translated front-panel edge.',flush=True)
    if a.preview_only:
        print('PREVIEW ONLY: check landmarks are on real solid drawer edges and actual gripper.',flush=True)
        cap.release();return
    writer=cv2.VideoWriter(str(a.output/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
                           max(1,min(fps,20)),(w,h))
    if not writer.isOpened():raise RuntimeError('Cannot write overlay.mp4')
    rows=[];lastgray=gray0
    for t in range(start,end+1):
        if t>start:
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Frame decode failed at f'+str(t))
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            active=np.flatnonzero(gripper_valid)
            if len(active)>=3:
                after,accept=track_features(lastgray,gray,gripper_pts[active])
                gripper_valid[active[~accept]]=False
                gripper_pts[active[accept]]=after[accept]
            if rim is not None:
                new,ncc,fb,rimstatus=track_point(lastgray,gray,rim,a.patch_radius,a.search)
                rim=new
            else:rimstatus='LOST'
            if secondary_good:
                new,ncc2,fb2,secondstatus=track_point(lastgray,gray,secondary,a.patch_radius,a.search)
                if new is None:secondary_good=False
                else:secondary=new
            else:secondstatus='LOST'
            lastgray=gray
        else:
            rimstatus='seed';secondstatus='seed'
        gn=int(gripper_valid.sum());offset=np.array([np.nan,np.nan],np.float32)
        if gn>=8:
            differences=(gripper_pts[gripper_valid]-grips0[gripper_valid]).reshape(-1,2)
            offset=np.median(differences,axis=0)
            spread=float(np.median(np.linalg.norm(differences-offset,axis=1)))
            if spread>8:offset[:]=np.nan
        else:spread=float('nan')
        if rim is not None:
            woffset=rim-points0['rim_far']
        else:woffset=np.array([np.nan,np.nan],np.float32)
        # Every drawer polygon point follows the same explicitly marked rim-corner translation.
        estimated={
            'tip':points0['tip']+offset,
            'wall_top':points0['wall_top']+woffset,
            'wall_bottom':points0['wall_bottom']+woffset,
            'rim_far':points0['rim_far']+woffset
        }
        valid={k:bool(np.isfinite(v).all()) for k,v in estimated.items()}
        side,clearance,geoquality=geometry(estimated,valid)
        corner_validation='UNAVAILABLE'
        if secondary_good and rim is not None:
            diff=secondary-points0['wall_bottom']
            corner_err=float(np.linalg.norm(diff-woffset))
            corner_validation='AGREE' if corner_err<9 else 'DISAGREE'
        else:corner_err=float('nan')
        if geoquality=='OK':
            if corner_validation=='DISAGREE':geoquality='LANDMARK_DISAGREE'
            elif gn<10:geoquality='GRIPPER_TRACK_WEAK'
        row={'frame':t,'signed_side_dx_px':side if geoquality=='OK' else float('nan'),
             'projected_tip_clearance_px':clearance if geoquality=='OK' else float('nan'),
             'geometry_quality':geoquality,'gripper_dx':float(offset[0]),'gripper_dy':float(offset[1]),
             'drawer_dx':float(woffset[0]),'drawer_dy':float(woffset[1]),
             'gripper_features':gn,'gripper_spread_px':spread,
             'rim_corner_status':rimstatus,'secondary_corner_status':secondstatus,
             'two_corner_agreement':corner_validation,'two_corner_error_px':corner_err}
        for k,v in estimated.items():
            row[f'{k}_x']=float(v[0]) if valid[k] else float('nan')
            row[f'{k}_y']=float(v[1]) if valid[k] else float('nan')
        rows.append(row)
        vis=add_marks(frame,estimated,t,valid)
        for xy in gripper_pts[gripper_valid].reshape(-1,2):
            cv2.circle(vis,tuple(np.rint(xy).astype(int)),2,(65,65,255),-1)
        # Overlay diagnostic values below title without pretending measured 3D clearance.
        cv2.rectangle(vis,(0,51),(960,96),(15,15,15),-1)
        cv2.putText(vis,f'2D side_dx={side:+.1f}  rim projection={clearance:+.1f}  '
                    f'Gpts={gn}  corner={corner_validation}  status={geoquality}',
                    (10,79),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
        writer.write(vis)
        if t==start or (t-start)%2==0 or t==end:
            print(f'f{t}: relative_side={side:+.2f}px rim_projection={clearance:+.2f}px '
                  f'Gpts={gn} corner={corner_validation} quality={geoquality} '
                  f'rim_track={rimstatus}',flush=True)
    cap.release();writer.release()
    with (a.output/'motion.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    plot(rows,a.output/'relative_geometry.png')
    result=summarize(rows)
    result.update(case=a.case,video=str(a.video),start=start,end=end,
                  seed_points={k:points0[k].tolist() for k in points0},
                  gripper_seed_box=list(box),output=str(a.output),
                  method='KLT median gripper displacement and one tracked panel rim corner; approximate rigid translation; optional second-corner cross-check',
                  warning='This is a manually seeded projected 2D path hypothesis; not a signed 3D solid-surface intersection test. Inspect overlay and corner tracks.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('OUTPUT:',a.output,flush=True)
    print('SUMMARY:',json.dumps(result,indent=2,allow_nan=False),flush=True)

if __name__=='__main__':main()
