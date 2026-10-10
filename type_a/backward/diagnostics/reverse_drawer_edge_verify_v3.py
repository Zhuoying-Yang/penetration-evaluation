#!/usr/bin/env python3
"""Experimental transparent-drawer front-edge verification, CPU only.

Use v2's gripper KLT and rim corner tracker. Crucially, detect the front wall
VERTICAL EDGE independently in each image with edge/Hough evidence above and
below the gripper. Do not silently extrapolate a missing wall from one rim point.

Outputs per-frame diagnostics, an overlay video, a CSV, a graph and conservative
2D candidate metadata. NO 3D penetration verdict is issued.

Initial seeds are manual for 0056/f148. They are not GT annotations and must
not be assumed to generalize to other videos. A transparent panel may be
indistinguishable from gripper/background lines. Inspect overlays carefully.

Run on ERIS (alongside reverse_drawer_path_probe_v2.py):
  python reverse_drawer_edge_verify_v3.py --case 0056 \\
    --video /path/to/0056.mp4 --start 148 --end 156 \\
    --output /path/to/edge_v3_0056
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np
import reverse_drawer_path_probe_v2 as prior

DEFAULT = prior.DEFAULT_SEEDS


def interpolate_x(p, q, y):
    dy = float(q[1]-p[1])
    return float('nan') if abs(dy)<2 else float(p[0]+(y-p[1])*(q[0]-p[0])/dy)


def line_candidates(img, pred_top, pred_bottom, search=25):
    """Find *observed* near-vertical line pairs on separate panel-edge bands.

    pred_* comes from rim motion only to locate a search neighborhood;
    the reported observed edge is fitted from image gradients/Hough pixels.
    """
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    edge=cv2.Canny(gray,28,95)
    H,W=edge.shape
    xmid=(float(pred_top[0])+float(pred_bottom[0]))/2
    xlo=max(0,int(round(min(pred_top[0],pred_bottom[0])-search)))
    xhi=min(W,int(round(max(pred_top[0],pred_bottom[0])+search+1)))
    ylo=max(0,int(round(pred_top[1]-32)))
    yhi=min(H,int(round(pred_bottom[1]+22)))
    crop=edge[ylo:yhi,xlo:xhi]
    if crop.shape[0]<40 or crop.shape[1]<25:
        return [],'ROI_TOO_SMALL',(xlo,ylo,xhi,yhi)
    detected=cv2.HoughLinesP(crop,1,np.pi/180,threshold=15,
                             minLineLength=18,maxLineGap=11)
    segments=[]
    if detected is not None:
        for raw in detected.reshape(-1,4):
            x0,y0,x1,y1=(raw.astype(float)+np.array([xlo,ylo,xlo,ylo])).tolist()
            dy=y1-y0;dx=x1-x0
            if abs(dy)<18 or abs(dx)>.32*abs(dy)+2:continue
            ymid=(y0+y1)/2
            if abs(ymid-(pred_top[1]+pred_bottom[1])/2)>75:continue
            slope=dx/dy
            xc=x0+slope*((pred_top[1]+pred_bottom[1])/2-y0)
            if abs(xc-xmid)>search+2:continue
            segments.append({'x0':x0,'y0':y0,'x1':x1,'y1':y1,'slope':slope,
                             'center_x':xc,'center_y':ymid,'length':abs(dy),
                             'ylo':min(y0,y1),'yhi':max(y0,y1)})
    mid_y=(float(pred_top[1])+float(pred_bottom[1]))/2
    upper=[s for s in segments if s['center_y']<mid_y-12]
    lower=[s for s in segments if s['center_y']>mid_y+12]
    solutions=[]
    for a in upper:
        for b in lower:
            if abs(a['center_x']-b['center_x'])>9:continue
            span=max(a['yhi'],b['yhi'])-min(a['ylo'],b['ylo'])
            if span<47:continue
            y=np.array([a['y0'],a['y1'],b['y0'],b['y1']],dtype=float)
            x=np.array([a['x0'],a['x1'],b['x0'],b['x1']],dtype=float)
            A=np.column_stack((y,np.ones_like(y)))
            slope,intercept=np.linalg.lstsq(A,x,rcond=None)[0]
            residual=float(np.sqrt(np.mean((A@np.array([slope,intercept])-x)**2)))
            if abs(slope)>.35 or residual>6:continue
            x_center=float(slope*mid_y+intercept)
            if abs(x_center-xmid)>search+1:continue
            # Modest proximity term only; Hough edge evidence must dominate.
            score=(0.50*span+0.16*(a['length']+b['length'])
                  +0.10*abs(a['center_y']-b['center_y'])
                  -0.9*abs(x_center-xmid)-1.5*residual)
            solutions.append({'score':float(score),'slope':float(slope),
                              'intercept':float(intercept),'residual':residual,
                              'center_x':x_center,'span':float(span),
                              'segments':[(a['x0'],a['y0'],a['x1'],a['y1']),
                                          (b['x0'],b['y0'],b['x1'],b['y1'])]})
    solutions.sort(key=lambda s:s['score'],reverse=True)
    if not solutions:
        return [],'NO_TWO_BAND_EDGE',(xlo,ylo,xhi,yhi)
    best=solutions[0]
    # Competition between materially different vertical lines means identity
    # is ambiguous, even if the top score is slightly larger.
    rivals=[s for s in solutions[1:] if abs(s['center_x']-best['center_x'])>=6
            and s['score']>=best['score']-8]
    status='AMBIGUOUS_PARALLEL_EDGES' if rivals else 'TWO_BAND_EDGE'
    return solutions,status,(xlo,ylo,xhi,yhi)


def write_curve(rows,path):
    H,W=610,1150
    im=np.full((H,W,3),255,np.uint8)
    xmin,xmax=88,W-45;ytop,ybot=75,H-135
    valid=[r for r in rows if r['edge_quality']=='VALID' and np.isfinite(r['side_dx_px'])]
    vals=[v for r in valid for v in [r['side_dx_px'],r['projected_tip_clearance_px']]]
    lo,hi=(min(vals)-8,max(vals)+8) if vals else (-30,30)
    if hi-lo<20:lo-=10;hi+=10
    X=lambda f:round(xmin+(f-rows[0]['frame'])*(xmax-xmin)/max(1,rows[-1]['frame']-rows[0]['frame']))
    Y=lambda v:round(ybot-(v-lo)*(ybot-ytop)/(hi-lo))
    cv2.rectangle(im,(xmin,ytop),(xmax,ybot),(180,180,180),1)
    if lo<=0<=hi:cv2.line(im,(xmin,Y(0)),(xmax,Y(0)),(130,130,130),1)
    for col,color in [('side_dx_px',(180,65,175)),('projected_tip_clearance_px',(30,145,40))]:
        p=None
        for r in rows:
            ok=r['edge_quality']=='VALID' and np.isfinite(r[col])
            if ok:
                q=(X(r['frame']),Y(r[col]));cv2.circle(im,q,4,color,-1)
                if p is not None:cv2.line(im,p,q,color,3,cv2.LINE_AA)
                p=q
            else:p=None
    cv2.putText(im,'Independent image-edge confirmation of moving front panel (2D only)',
                (xmin,40),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(im,'Purple: gripper tip x - INDEPENDENTLY DETECTED panel edge x',
                (xmin,H-90),cv2.FONT_HERSHEY_SIMPLEX,.53,(180,65,175),2,cv2.LINE_AA)
    cv2.putText(im,'Green: projected tip height above top rim (image coordinates, NOT 3D clearance)',
                (xmin,H-57),cv2.FONT_HERSHEY_SIMPLEX,.47,(30,145,40),2,cv2.LINE_AA)
    cv2.putText(im,'Missing points mean unreliable edge or gripper tracking; gaps are not interpolated.',
                (xmin,H-26),cv2.FONT_HERSHEY_SIMPLEX,.48,(50,50,50),1,cv2.LINE_AA)
    for t in [rows[0]['frame'],rows[-1]['frame']]:
        cv2.putText(im,f'f{t}',(X(t)-20,ybot+24),cv2.FONT_HERSHEY_SIMPLEX,.52,(50,50,50),1)
    cv2.imwrite(str(path),im)


def analyze(args):
    defaults=DEFAULT.get(args.case,{})
    start=args.start if args.start is not None else defaults.get('start')
    end=args.end if args.end is not None else defaults.get('end')
    if start is None or end is None or not 0<=start<end:raise ValueError('Require a valid --start and --end')
    pts0={k:prior.parse_xy(getattr(args,k),defaults.get(k))
          for k in ('tip','wall_top','wall_bottom','rim_far')}
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise RuntimeError(f'Cannot open video {args.video}')
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end>=n:raise RuntimeError(f'Video has {n} frames, requested f{end}')
    fps=float(cap.get(cv2.CAP_PROP_FPS)) or 24.
    frame=prior.extract_start(cap,start)
    h,w=frame.shape[:2]
    gripper_box=prior.parse_box(args.gripper_box,defaults.get('gripper_box'),w,h)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gripper_init=prior.detect_features(gray,gripper_box)
    if len(gripper_init)<8:raise RuntimeError('Not enough gripper KLT points; adjust --gripper-box')
    gripper=gripper_init.copy()
    keep=np.ones(len(gripper_init),dtype=bool)
    rim=pts0['rim_far'].copy()
    prevgray=gray
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    writer=None
    if not args.preview_only:
        writer=cv2.VideoWriter(str(args.output/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
                               min(fps,20),(w,h))
        if not writer.isOpened():raise RuntimeError('Cannot write overlay.mp4')
    try:
        for f in range(start,end+1):
            if f>start:
                ok,frame=cap.read()
                if not ok:raise RuntimeError(f'Video decode failed at f{f}')
                currgray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                ix=np.flatnonzero(keep)
                if len(ix)>=3:
                    new,accept=prior.track_features(prevgray,currgray,gripper[ix])
                    keep[ix[~accept]]=False
                    gripper[ix[accept]]=new[accept]
                if rim is not None:
                    new,ncc,fb,status=prior.track_point(prevgray,currgray,rim,
                                                        args.patch_radius,args.search)
                    rim=new
                else:status='LOST'
                prevgray=currgray
            else:status='seed'
            n_g=int(keep.sum())
            offset=np.array([np.nan,np.nan],dtype=float)
            spread=float('nan')
            if n_g>=8:
                d=(gripper[keep]-gripper_init[keep]).reshape(-1,2)
                offset=np.median(d,axis=0)
                spread=float(np.median(np.linalg.norm(d-offset,axis=1)))
                if spread>8:offset[:]=np.nan
            tip=pts0['tip']+offset
            rshift=(rim-pts0['rim_far']) if rim is not None else np.array([np.nan,np.nan])
            pred_top=pts0['wall_top']+rshift
            pred_bot=pts0['wall_bottom']+rshift
            candidate,status_edge,search_box=([], 'RIM_LOST',(0,0,0,0))
            best=None
            if np.isfinite(rshift).all():
                candidate,status_edge,search_box=line_candidates(frame,pred_top,pred_bot,args.edge_search)
                if candidate:best=candidate[0]
            tip_good=bool(np.isfinite(tip).all() and n_g>=10)
            if best is None:wall_good=False
            else:
                wall_good=status_edge=='TWO_BAND_EDGE'
                # Conservative agreement with rim only used as a consistency gate;
                # measured wall coordinates come solely from detected edges.
                if abs(best['center_x']-(pred_top[0]+pred_bot[0])/2)>args.rim_tolerance:
                    wall_good=False
                    status_edge='EDGE_RIM_MOTION_DISAGREE'
            valid=bool(tip_good and wall_good)
            side=float('nan');clear=float('nan');observed_x=float('nan')
            if best is not None and np.isfinite(tip).all():
                observed_x=best['slope']*tip[1]+best['intercept']
                side=float(tip[0]-observed_x)
                # Rim via tracked drawer corner is still PROJECTED rim only.
                rtop=pred_top;rfar=rim
                ry=prior.segment_projection(rtop,rfar,tip[0],axis=0,max_extrap=38)
                if np.isfinite(ry):clear=float(ry-tip[1])
            if valid and not np.isfinite([side,clear]).all():
                valid=False;status_edge='PROJECTION_INVALID'
            quality='VALID' if valid else status_edge if tip_good else 'GRIPPER_UNCERTAIN'
            row=dict(frame=f,edge_quality=quality,edge_candidate_count=len(candidate),
                     edge_rank_margin=(best['score']-candidate[1]['score'] if len(candidate)>1 else float('nan')),
                     edge_fit_residual_px=best['residual'] if best else float('nan'),
                     edge_center_x_px=best['center_x'] if best else float('nan'),
                     rim_pred_center_x_px=(float((pred_top[0]+pred_bot[0])/2) if np.isfinite(rshift).all() else float('nan')),
                     edge_minus_rim_pred_x_px=(best['center_x']-float((pred_top[0]+pred_bot[0])/2)
                                                if best else float('nan')),
                     tip_x_px=float(tip[0]),tip_y_px=float(tip[1]),
                     side_dx_px=side if valid else float('nan'),
                     projected_tip_clearance_px=clear if valid else float('nan'),
                     gripper_features=n_g,gripper_spread_px=spread,
                     rim_status=status,edge_detection_status=status_edge)
            rows.append(row)
            overlay=frame.copy()
            if best is not None:
                ytop=int(pred_top[1]-15);ybot=int(pred_bot[1]+15)
                p=(int(round(best['slope']*ytop+best['intercept'])),ytop)
                q=(int(round(best['slope']*ybot+best['intercept'])),ybot)
                cv2.line(overlay,p,q,(50,220,60) if wall_good else (0,200,250),3)
                for seg in best['segments']:
                    x0,y0,x1,y1=[int(round(v)) for v in seg]
                    cv2.line(overlay,(x0,y0),(x1,y1),(0,230,0),2,cv2.LINE_AA)
            if np.isfinite(rshift).all():
                p=tuple(np.round(pred_top).astype(int));q=tuple(np.round(pred_bot).astype(int))
                cv2.line(overlay,p,q,(0,155,255),1,cv2.LINE_AA)
                cv2.line(overlay,p,tuple(np.round(rim).astype(int)),(230,230,15),2,cv2.LINE_AA)
                cv2.circle(overlay,tuple(np.round(rim).astype(int)),6,(230,230,15),2)
            if tip_good:
                cv2.circle(overlay,tuple(np.rint(tip).astype(int)),8,(60,60,255),2)
            for q in gripper[keep].reshape(-1,2):
                cv2.circle(overlay,tuple(np.round(q).astype(int)),1,(60,60,255),-1)
            cv2.rectangle(overlay,(0,0),(w,68),(14,14,14),-1)
            cv2.putText(overlay,f'f{f}  observed panel edge=GREEN  rim-only prediction=ORANGE  gripper=RED',
                        (10,25),cv2.FONT_HERSHEY_SIMPLEX,.58,(245,245,245),2,cv2.LINE_AA)
            cv2.putText(overlay,f'edge: {status_edge}   gripper_pts={n_g}   2D_side={side:+.1f} '
                        f'  clearance={clear:+.1f}  quality={quality}',
                        (10,53),cv2.FONT_HERSHEY_SIMPLEX,.51,(245,245,245),1,cv2.LINE_AA)
            if f==start:
                cv2.imwrite(str(args.output/'seed_preview.jpg'),overlay,[cv2.IMWRITE_JPEG_QUALITY,93])
            if writer is not None:writer.write(overlay)
            if f==end:
                cv2.imwrite(str(args.output/'last_overlay.jpg'),overlay,[cv2.IMWRITE_JPEG_QUALITY,93])
            print(f'f{f}: edge={status_edge} observed_side={side:+.1f}px '
                  f'clearance={clear:+.1f}px rim={status} quality={quality}',flush=True)
            if args.preview_only:break
    finally:
        cap.release()
        if writer:writer.release()
    if args.preview_only:
        print('PREVIEW ONLY: no scoring and no API calls');return
    with (args.output/'motion.csv').open('w',newline='') as file:
        writer_csv=csv.DictWriter(file,fieldnames=list(rows[0].keys()))
        writer_csv.writeheader();writer_csv.writerows(rows)
    write_curve(rows,args.output/'relative_geometry.png')
    # Requiring BOTH independent edge support and complete reliable observation
    # through a sign switch, plus excluding an observed projected clearance.
    margin=4
    confident=[]
    states=[]
    for r in rows:
        d=float(r['side_dx_px'])
        states.append(0 if r['edge_quality']!='VALID' or not math.isfinite(d) or abs(d)<=margin
                      else (1 if d>0 else -1))
    for i,si in enumerate(states):
        if si==0:continue
        for j in range(i):
            if states[j]==-si:
                subsection=rows[j:i+1]
                if all(r['edge_quality']=='VALID' for r in subsection):
                    # Even a valid 2D side switch is only a candidate, not 3D proof.
                    clear_seen=any(r['projected_tip_clearance_px']>5 for r in subsection)
                    confident.append({'from_frame':rows[j]['frame'],'to_frame':rows[i]['frame'],
                                      'projected_clearance_seen':clear_seen,
                                      'label':'2D_SIDE_SWITCH_NOT_3D_PROOF'})
                break
        if confident:break
    report={'case':args.case,'frames':[start,end],
            'verified_edge_frames':sum(r['edge_quality']=='VALID' for r in rows),
            'total_frames':len(rows),
            'independently_observed_2d_switches':confident,
            '3d_penetration_decision':None,
            'unverified_intervals':[
                {'frame':r['frame'],'reason':r['edge_quality']} for r in rows if r['edge_quality']!='VALID'],
            'warning':'Observed Hough line pairs in transparent panels may still be distractor edges. 2D clearance does not verify any legal 3D exit. Review overlay before using signals.'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('OUTPUT:',args.output)
    print('SUMMARY:',json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--start',type=int)
    p.add_argument('--end',type=int)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gripper-box')
    p.add_argument('--tip')
    p.add_argument('--wall-top')
    p.add_argument('--wall-bottom')
    p.add_argument('--rim-far')
    p.add_argument('--patch-radius',type=int,default=13)
    p.add_argument('--search',type=int,default=20)
    p.add_argument('--edge-search',type=int,default=25)
    p.add_argument('--rim-tolerance',type=float,default=17.)
    p.add_argument('--preview-only',action='store_true')
    args=p.parse_args()
    if not 12<=args.edge_search<=60: p.error('--edge-search must be 12..60')
    if not 5<=args.rim_tolerance<=45:p.error('--rim-tolerance must be 5..45')
    analyze(args)


if __name__=='__main__':main()
