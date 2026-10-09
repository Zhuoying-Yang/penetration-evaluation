#!/usr/bin/env python3
"""Three subtype-specific Qwen visual agents plus one evidence adjudicator.

A: ALL adjacent original-frame pairs (type-A solid-boundary crossing).
B: gripper+object interface local crop for each adjacent pair; scout local box
   with Wilson Qwen unless a saved B ROI, saved boxes or SAM mask is provided.
C: same local crop at normal reference and candidate frame, each candidate frame;
   persistence computed from repeated local visual findings.
D: once A/B/C complete for this case, review their evidence + original images.

Run inside an existing H200 srun allocation with Wilson Qwen environment.
Existing Wilson files are not modified. Outputs are checkpointed per query.
Research pilot, NOT calibrated likelihoods. Do not pass ground truth to agents.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from functools import lru_cache
import traceback
from typing import Any

import cv2
from PIL import Image, ImageDraw

BASE = Path('/PHShome/zy992/Wilson/deformationdetection')
WS = BASE / 'workspace-cosmos3-0019-masking-20261007'
ROOT = Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
CATALOG = ROOT / 'wilson_batch_gt10_v1/catalog.json'
DEFAULT_MODEL = Path('/PHShome/zy992/Wilson/dependency/models/qwen_model')
SYSTEM_SHARED = '''You are a visual observation agent for physically impossible object interpenetration in generated robot-manipulation videos.
The displayed images are original frames with source frame numbers. Treat each image as visual evidence, not ground truth.
Describe concrete changes in material boundaries, not overall task quality. Normal occlusion, legal openings, deformity alone, object disappearance alone, and camera projection alone do not prove penetration.
Respond with ONLY a valid JSON object and no markdown. Be observational: never assume an event occurred because the task is supposed to do it.
All categorical fields must be one of YES, NO, UNCLEAR. No explanations of expected outcomes or target labels will be provided.'''
A_PROMPT = '''ROLE A: CONSECUTIVE-FRAME CHANGE DETECTOR, NOT A SINGLE-FRAME CONTAINER-ENTRY CLASSIFIER.
Image 1 is source frame {prev} BEFORE; Image 2 is source frame {curr} AFTER. They are consecutive frames.

Identify the MOVING ACTOR (the object held by the gripper if present) and the OTHER object's SOLID BARRIER (e.g. the large bin side wall). Do not confuse a carried cup with the large bin, or the gripper with the cup.

IMPORTANT: A low cup bottom relative to a bin rim, or an object merely being near/inside a bin, does NOT constitute a change. Only report an A event for a SPECIFIC OBSERVED TRANSITION between these exact two frames. If it is outside in BOTH frames, or inside in BOTH frames, or only gets slightly closer, you MUST answer changed_during_pair=NO and crosses_solid_side_in_pair=NO.

Follow the order:
1. Describe the moving actor's location in Image 1 and Image 2 separately. States:
   OUTSIDE_SIDE = laterally outside the bin's solid side wall;
   INSIDE_CAVITY = visibly inside the bin interior behind its side wall;
   ABOVE_OPENING = outside the wall but at/above the legal top opening;
   STRADDLING_RIM = overlaps the rim/side boundary;
   OCCLUDED = cannot locate the actor due to occlusion;
   UNCLEAR = depth/location uncertain.
2. Compare the SAME actor between images. State exactly WHAT MOVED or CHANGED, not what is generally happening in the task. If both states are the same without a boundary-crossing change, answer NO.
3. A suspicious Type-A candidate is an OUTSIDE_SIDE -> INSIDE_CAVITY change across a SOLID SIDE region, while actor's bottom appears below the rim, with no visible above-rim route. The presence of a legal opening elsewhere does not excuse a side-wall crossing. However, do NOT invent unobserved 3D heights; a 2D overlap may be normal occlusion, so use UNCLEAR when warranted.
4. Mark crossing YES only if the actor changes from one side of the SPECIFIC SOLID boundary to the other BETWEEN these two frames. If the scene merely contains a below-rim actor in both frames, crossing is NO.

Return EXACTLY one JSON object; category fields must be EXACT spellings below:
{{"actor":"short exact moving object name","barrier":"other object's solid wall","position_before":"OUTSIDE_SIDE|INSIDE_CAVITY|ABOVE_OPENING|STRADDLING_RIM|OCCLUDED|UNCLEAR","position_after":"OUTSIDE_SIDE|INSIDE_CAVITY|ABOVE_OPENING|STRADDLING_RIM|OCCLUDED|UNCLEAR","changed_during_pair":"YES|NO|UNCLEAR","motion_direction":"LATERAL_INTO_WALL|UP_OVER_RIM|DOWN_THROUGH_OPENING|OTHER|NONE|UNCLEAR","crosses_solid_side_in_pair":"YES|NO|UNCLEAR","bottom_below_rim_before":"YES|NO|UNCLEAR","bottom_below_rim_after":"YES|NO|UNCLEAR","clearance_above_rim_observed":"YES|NO|UNCLEAR","legal_opening_path_supported":"YES|NO|UNCLEAR","ordinary_occlusion_possible":"YES|NO|UNCLEAR","observation":"One sentence: exact before vs after physical change, or explicitly NO crossing this pair"}}'''
B_PROMPT = '''ROLE B: INSPECT LOCAL GRIPPER–TARGET INTERFACE IN ADJACENT FRAMES.
Image 1 is a local crop around gripper and target in frame {prev}; Image 2 is same-region crop in frame {curr}.
Look carefully at the physical gripper FINGERS, their tips, and the target object's intact material/silhouette. Watch for finger(s) initially outside the object's solid body, then appearing physically embedded deep inside the solid body, fused into it, or remaining embedded (not just grasping its surface).
A transparent 2D image overlap, the finger going behind a target, or normal grasp is NOT proof. If the gripper seems inside the solid body in the new frame without a physical opening, identify this as observable suspicious embedding even if a single pair cannot prove 3D contact.
Do not assume the crop is a perfect gripper mask.
Return exact JSON:
{{"gripper_visible":"YES|NO|UNCLEAR","target_visible":"YES|NO|UNCLEAR","gripper_outside_before":"YES|NO|UNCLEAR","gripper_inside_solid_after":"YES|NO|UNCLEAR","embedded_or_fused_now":"YES|NO|UNCLEAR","ordinary_grasp_or_occlusion_possible":"YES|NO|UNCLEAR","observation":"one short before/after comparison"}}'''
C_PROMPT = '''ROLE C: CONTRASTIVE LOCAL SOLID-BOUNDARY TRUNCATION DETECTION.
Image 1 is the earlier REFERENCE local crop from frame {ref}. Image 2 is the later CANDIDATE local crop from frame {curr}.
Do NOT decide penetration. Do NOT judge whether overall manipulation is physically valid.
Compare ONLY the same visible manipulated object (for example, a banana) and the drawer/bin/other container material.

QUESTION: Compared with Image 1, does Image 2 show STRONGER evidence that the visible manipulated object is being geometrically TRUNCATED OR OCCLUDED BY SOLID CONTAINER MATERIAL?
Strong evidence includes:
- a substantial object segment remains visible OUTSIDE;
- the visible object appears CUT OFF at a solid drawer edge/panel, rather than simply moving into the OPEN EMPTY CAVITY;
- the object's direction at the cutoff points toward SOLID material;
- the same physical portion of the object cannot simply pass through an ordinary opening.
Normal occlusion is possible; report exactly the local visual cues, not a final penetration verdict.
Return ONLY this JSON:
{{"candidate_more_suspicious":"YES|NO|UNCLEAR","object_more_truncated":"YES|NO|UNCLEAR","solid_boundary_evidence_stronger":"YES|NO|UNCLEAR","outside_segment_visible":"YES|NO|UNCLEAR","partial_containment":"YES|NO|UNCLEAR","occlusion_location":"SOLID_MATERIAL|OPEN_CAVITY|GRIPPER|OTHER|UNCLEAR","direction_toward_solid":"YES|NO|UNCLEAR","normal_opening_or_occlusion_possible":"YES|NO|UNCLEAR","observation":"short contrastive comparison"}}'''
SCOUT_PROMPT = '''Find the gripper FINGERS and the manipulated target object where they interact in this image.
Return a single rectangle tightly enclosing the gripper fingertips PLUS the nearby target object contact region. Coordinates are pixel values in the displayed image, origin upper left. If the region cannot be located, use null.
Return ONLY JSON: {"box_xyxy":[x1,y1,x2,y2] or null,"observation":"short"}.'''
D_PROMPT = '''ROLE D: INDEPENDENT, EVIDENCE-GROUNDED ADJUDICATOR.
Your visual input consists of EXPLICITLY LABELLED before/after SOURCE FRAME PAIR contact sheets, optionally with immediate context pairs. The frame numbers are written ON each image. Review them, not only agent conclusions. The accompanying EVIDENCE INDEX specifies exactly which source frames appear in each displayed image.
A = solid wall crossing / physically impossible entry path, B = gripper inserted into solid object, C = persistent object straddling a solid boundary.

REQUIRED REASONING FOR TYPE A:
* Identify the MOVING ITEM (which may be a cup carried by a gripper) and the OTHER object's solid barrier (e.g. bin side wall). Never accidentally assess "gripper through the carried cup" if the concern is "carried cup through the bin side wall".
* Compare the item's LOWEST POINT against the opening RIM before and after its lateral outside->inside transition. Legal top-entry requires an above-rim clearance path before descent. A nearby open top is NOT evidence that that path occurred. Analyze the transition, not just the final item location.
* A 2D image may be ambiguous due to depth projection/occlusion. Describe what is visually observable. If an A agent finds a specific outside->inside/below-rim event and the images cannot refute it, answer UNCERTAIN rather than automatic NO_PENETRATION. Do NOT assume a top opening validates every entry.
For Type B distinguish gripper material inside the target from grasping / ordinary occlusion. For Type C compare fixed reference and candidate; favor persistence.
Do not claim a particular frame is absent unless it is absent from the EVIDENCE INDEX. No GT labels. Do not uncritically accept self-reported confidence.
Return ONLY a JSON object:
{"category":"PENETRATION|NO_PENETRATION|UNCERTAIN","subtype":"A|B|C|NONE|UNCERTAIN","event_frame":null,"evidence_score":0.0,"supporting_evidence":"short","contradicting_evidence":"short","reason":"clear frame-number-specific causal explanation"}
evidence_score is UNCALIBRATED visual strength, not probability; event_frame may be an integer if visible.'''

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)

def read_json(path: Path) -> Any:
    return json.loads(path.read_text())

def normalize_yes(x: Any) -> str:
    x=str(x).strip().upper()
    return x if x in ('YES','NO','UNCLEAR') else 'UNCLEAR'

def get_obj(raw: str) -> dict:
    decoder=json.JSONDecoder()
    for match in re.finditer(r'\{',raw):
        try: obj,_=decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError: continue
        if isinstance(obj,dict): return obj
    raise ValueError('No valid JSON object in response: '+raw[:300])

def truth_score(row: dict, kind: str) -> float:
    """Visual indicator proxy, not calibrated probability or ground-truth class."""
    yes=lambda k: normalize_yes(row.get(k))=='YES'
    no=lambda k: normalize_yes(row.get(k))=='NO'
    if kind=='A':
        # A is a TRANSITION detector. Static below-rim positions are not evidence.
        # All yes/no fields come from model; no unobserved fact is inferred.
        before=str(row.get('position_before','UNCLEAR')).upper()
        after=str(row.get('position_after','UNCLEAR')).upper()
        if not (before=='OUTSIDE_SIDE' and after=='INSIDE_CAVITY'):
            return 0.0
        if not yes('changed_during_pair'):
            return 0.0
        if not yes('crosses_solid_side_in_pair'):
            return 0.0
        if str(row.get('motion_direction','UNCLEAR')).upper()!='LATERAL_INTO_WALL':
            return 0.0
        if yes('legal_opening_path_supported'):
            return 0.0
        if yes('ordinary_occlusion_possible'):
            return .30  # geometric ambiguity, not a clear crossing
        below=yes('bottom_below_rim_before') and yes('bottom_below_rim_after')
        if below and no('clearance_above_rim_observed') and no('ordinary_occlusion_possible'):
            return 1.0
        if below and not yes('clearance_above_rim_observed'):
            return .70
        return .50
    if kind=='B':
        if yes('gripper_inside_solid_after') and yes('embedded_or_fused_now'):
            return .9 if no('ordinary_grasp_or_occlusion_possible') else .65
        if yes('embedded_or_fused_now') and not yes('ordinary_grasp_or_occlusion_possible'): return .55
        return 0.0
    if kind=='C':
        if yes('candidate_more_suspicious') and yes('object_more_truncated') and \
           yes('solid_boundary_evidence_stronger') and yes('outside_segment_visible') and \
           str(row.get('occlusion_location','')).upper()=='SOLID_MATERIAL':
            return .95 if no('normal_opening_or_occlusion_possible') else .75
        if yes('candidate_more_suspicious') and yes('solid_boundary_evidence_stronger'): return .5
        return 0.0
    raise ValueError(kind)

def parse_verdict(raw: str, kind: str) -> dict:
    out=get_obj(raw)
    required={
        'A':['changed_during_pair','crosses_solid_side_in_pair','bottom_below_rim_before',
             'bottom_below_rim_after','clearance_above_rim_observed',
             'legal_opening_path_supported','ordinary_occlusion_possible'],
        'B':['gripper_visible','target_visible','gripper_outside_before','gripper_inside_solid_after','embedded_or_fused_now','ordinary_grasp_or_occlusion_possible'],
        'C':['candidate_more_suspicious','object_more_truncated','solid_boundary_evidence_stronger','outside_segment_visible','partial_containment','direction_toward_solid','normal_opening_or_occlusion_possible'],
    }[kind]
    for key in required:out[key]=normalize_yes(out.get(key,'UNCLEAR'))
    if kind=='A':
        valid_states={'OUTSIDE_SIDE','INSIDE_CAVITY','ABOVE_OPENING','STRADDLING_RIM','OCCLUDED','UNCLEAR'}
        for key in ('position_before','position_after'):
            val=str(out.get(key,'UNCLEAR')).strip().upper()
            out[key]=val if val in valid_states else 'UNCLEAR'
        directions={'LATERAL_INTO_WALL','UP_OVER_RIM','DOWN_THROUGH_OPENING','OTHER','NONE','UNCLEAR'}
        val=str(out.get('motion_direction','UNCLEAR')).strip().upper()
        out['motion_direction']=val if val in directions else 'UNCLEAR'
    if kind=='C':
        loc=str(out.get('occlusion_location','UNCLEAR')).upper()
        out['occlusion_location']=loc if loc in {'SOLID_MATERIAL','OPEN_CAVITY','GRIPPER','OTHER','UNCLEAR'} else 'UNCLEAR'
    out['observation']=str(out.get('observation',''))[:500]
    out['signal']=truth_score(out,kind)
    return out

def audit_a_temporal_continuity(records: list[dict]) -> dict:
    """Reconcile repeated frame states from adjacent pairwise VLM calls.

    A stream saying OUTSIDE -> INSIDE for EVERY pair is self-contradictory:
    the shared frame cannot simultaneously be INSIDE (previous pair's after)
    and OUTSIDE (next pair's before). Mark such candidates unverified.
    This does not prove negatives: a real event may require manual review.
    """
    items={int(r['frame']):r for r in records if r.get('status')=='ok'}
    conflicts=set()
    comparable=0
    for t,left in items.items():
        right=items.get(t+1)
        if right is None:continue
        a=left['verdict'].get('position_after','UNCLEAR')
        b=right['verdict'].get('position_before','UNCLEAR')
        if a not in {'OUTSIDE_SIDE','INSIDE_CAVITY'} or b not in {'OUTSIDE_SIDE','INSIDE_CAVITY'}:
            continue
        comparable+=1
        if a!=b:
            conflicts.update((t,t+1))
    strong=0
    for t,r in items.items():
        v=r['verdict']
        raw=float(v.get('raw_transition_signal',v.get('signal',0)))
        v['raw_transition_signal']=raw
        v['shared_frame_conflict']=t in conflicts
        # A disputed frame state cannot support a confident Type-A transition.
        v['signal']=round(min(raw,0.25) if t in conflicts else raw,4)
        if v['signal']>=.65:strong+=1
    return {'pairs_comparable':comparable,
            'pairs_in_conflict':len([t for t in items if t in conflicts]),
            'strong_consistent_candidates':strong}

def list_images(folder: Path):
    images=sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
    if not images:raise ValueError('No images in frame folder: '+str(folder))
    return images

def video_info(video: Path):
    if video.is_dir():
        files=list_images(video)
        im=Image.open(files[0]);width,height=im.size
        return len(files),width,height,0.0
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError(f'Cannot open video: {video}')
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if n<2:raise ValueError('Video must have >=2 frames')
    return n,width,height,fps

class Frames:
    def __init__(self,path: Path):
        self.files=list_images(path) if path.is_dir() else None
        self.cap=None if self.files is not None else cv2.VideoCapture(str(path))
        self.cache={}
    def frame(self,t):
        t=int(t)
        if t not in self.cache:
            if self.files is not None:
                self.cache[t]=Image.open(self.files[t]).convert('RGB')
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,t)
                ok,bgr=self.cap.read()
                if not ok:raise ValueError(f'Cannot decode f{t}')
                self.cache[t]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            if len(self.cache)>18:
                first=next(iter(self.cache))
                self.cache.pop(first)
        return self.cache[t]
    def close(self):
        if self.cap is not None:self.cap.release()

def fit_image(im:Image.Image,max_side:int) ->Image.Image:
    if max(im.size)>max_side:
        factor=max_side/max(im.size)
        return im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))))
    return im

def label(im:Image.Image,t:int,max_side:int,box=None):
    if box is not None: im=im.crop(tuple(int(x) for x in box))
    im=fit_image(im,max_side)
    output=Image.new('RGB',(im.width,im.height+25),(0,0,0));output.paste(im,(0,25))
    ImageDraw.Draw(output).text((5,6),f'ORIGINAL VIDEO FRAME {t}',fill=(255,255,255))
    return output

def expand_box(box,w,h,scale=1.4):
    x1,y1,x2,y2=[float(v) for v in box]
    if not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or x2<=x1 or y2<=y1:return None
    cx=(x1+x2)/2;cy=(y1+y2)/2
    bw=max(90,(x2-x1)*scale);bh=max(90,(y2-y1)*scale)
    xa=max(0,int(cx-bw/2));ya=max(0,int(cy-bh/2));xb=min(w,int(cx+bw/2));yb=min(h,int(cy+bh/2))
    return [xa,ya,xb,yb] if xb-xa>=32 and yb-ya>=32 else None

def parse_box(s):
    if not s:return None
    val=[int(x.strip()) for x in s.split(',')]
    if len(val)!=4:raise ValueError('ROI must be x1,y1,x2,y2')
    return val

@lru_cache(maxsize=2)
def cached_mask_array(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        if 'object_masks' in z:
            names=z['object_names'].tolist()
            key='link7' if 'link7' in names else 'gripper' if 'gripper' in names else None
            if key is None:raise ValueError('mask archive has no link7/gripper channel')
            return z['object_masks'][:,names.index(key)].astype(bool)
        if 'masks' in z:return z['masks'].astype(bool)
        raise ValueError('No masks/object_masks in archive')

def maybe_mask_box(path:Path,t:int,w:int,h:int):
    """Find bounding box in an existing Wilson gripper mask; load archive once."""
    import numpy as np
    m=cached_mask_array(str(path))[t]
    if m.shape!=(h,w):raise ValueError(f'Bad gripper mask size {m.shape}, expected {(h,w)}')
    ys,xs=np.where(m)
    if not len(xs):return None
    return [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]

def load_boxes(path:Path):
    obj=read_json(path)
    if isinstance(obj,dict):obj=obj.get('frames',obj.get('records',obj))
    if isinstance(obj,dict):
        out={int(k):v for k,v in obj.items() if str(k).isdigit()}
    else:
        out={int(v['frame']):v for v in obj}
    final={}
    for t,x in out.items():
        if isinstance(x,dict):x=x.get('bbox') or x.get('box_xyxy')
        if x is not None and len(x)==4:final[t]=x
    return final

def interpolate_box(boxes,t):
    if not boxes:return None
    keys=sorted(boxes)
    left=max((k for k in keys if k<=t),default=None)
    right=min((k for k in keys if k>=t),default=None)
    if left is None:return boxes[right]
    if right is None:return boxes[left]
    if left==right:return boxes[left]
    alpha=(t-left)/(right-left)
    return [(1-alpha)*x+alpha*y for x,y in zip(boxes[left],boxes[right])]

def scout_box(client,frame:Image.Image,t:int,image_side:int):
    img=fit_image(frame,image_side)
    response=client.ask([img],SCOUT_PROMPT,system_prompt=SYSTEM_SHARED)
    val=get_obj(response['answer']).get('box_xyxy')
    if not isinstance(val,list) or len(val)!=4:return None
    box=[float(q) for q in val]
    if not all(math.isfinite(q) for q in box) or not (box[0]<box[2] and box[1]<box[3]):return None
    sx=frame.width/img.width;sy=frame.height/img.height
    return [box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy]

def strong_runs(records,kind):
    """Consecutive source-frame evidence; no gap-filling for missed verdicts."""
    positive=sorted(int(x['frame']) for x in records if x.get('status')=='ok' and x['verdict']['signal']>=.65)
    runs=[]
    for t in positive:
        if runs and t==runs[-1][-1]+1:runs[-1].append(t)
        else:runs.append([t])
    return [{'start':x[0], 'end':x[-1], 'length':len(x)} for x in runs]

def report_stage(stage_path:Path,kind:str):
    if not stage_path.is_file():return {'agent':kind,'complete':False,'reason':'No specialist output'}
    records=read_json(stage_path).get('rows',[])
    good=[r for r in records if r.get('status')=='ok']
    candidates=sorted(good,key=lambda r:r['verdict']['signal'],reverse=True)
    runs=strong_runs(good,kind)
    return {'agent':kind,'complete':all(r.get('status')=='ok' for r in records) and bool(records),
            'n_queries':len(records),'n_ok':len(good),'peak':float(candidates[0]['verdict']['signal']) if candidates else 0,
            'top':[{ 'frame':x['frame'],'prev':x.get('prev'), 'signal':x['verdict']['signal'],
                     'observation':x['verdict'].get('observation',''),
                     'box_xyxy':x.get('box_xyxy'),'reference':x.get('reference'),
                     'primitives':{k:v for k,v in x['verdict'].items() if k not in ('observation','signal')}} for x in candidates[:6]],
            'runs':runs[:12], 'longest_run':max([x['length'] for x in runs],default=0)}

def run_agent(client,agent:str,frames:Frames,case_dir:Path,start:int,end:int,reference:int,
              image_side:int,scout_side:int,b_roi,c_roi,b_masks,b_boxes,scout_stride,max_tokens,video_size):
    w,h=video_size
    if agent in ('A','B'): targets=list(range(max(1,start),end+1))
    else: targets=[t for t in range(start,end+1) if t!=reference]
    if not targets:raise ValueError(f'Empty frame set for agent {agent}')
    path=case_dir/f'{agent}_queries.json'
    old=read_json(path).get('rows',[]) if path.exists() else []
    done={x['frame']:x for x in old if x.get('status')=='ok'}
    scout_file=case_dir/'B_scout_boxes.json'
    scouts=load_boxes(scout_file) if scout_file.is_file() else {}
    known_boxes=load_boxes(b_boxes) if b_boxes else {}
    for index,t in enumerate(targets):
        if t in done:continue
        item={'agent':agent,'frame':t,'status':'failed'}
        try:
            if agent=='A':
                prev=t-1
                images=[label(frames.frame(prev),prev,image_side),label(frames.frame(t),t,image_side)]
                prompt=A_PROMPT.format(prev=prev,curr=t)
                item['prev']=prev
            elif agent=='B':
                prev=t-1
                if b_roi is not None:
                    box=expand_box(b_roi,w,h,1.0);box_source='manual_roi'
                elif b_masks:
                    a=maybe_mask_box(b_masks,prev,w,h)
                    b=maybe_mask_box(b_masks,t,w,h)
                    coords=[q for q in (a,b) if q]
                    union=[min(q[0] for q in coords), min(q[1] for q in coords),max(q[2] for q in coords),max(q[3] for q in coords)] if coords else None
                    box=expand_box(union,w,h,1.8) if union else None
                    box_source='wilson_gripper_mask'
                elif known_boxes:
                    raw=interpolate_box(known_boxes,t)
                    box=expand_box(raw,w,h,1.65) if raw is not None else None
                    box_source='existing_qwen_gripper_box'
                else:
                    anchor=(t//scout_stride)*scout_stride
                    if anchor not in scouts:
                        scouts[anchor]=scout_box(client,frames.frame(anchor),anchor,scout_side)
                        write_json(scout_file,{'frames':[{'frame':q,'bbox':b} for q,b in sorted(scouts.items())]})
                    raw=scouts.get(anchor)
                    box=expand_box(raw,w,h,1.9) if raw is not None else None
                    box_source='qwen_gripper_scout'
                if box is None:
                    raise ValueError(f'Unable to localize gripper ROI f{t}; cannot silently use whole frame')
                images=[label(frames.frame(prev),prev,image_side,box),label(frames.frame(t),t,image_side,box)]
                prompt=B_PROMPT.format(prev=prev,curr=t)
                item.update(prev=prev,box_xyxy=box,box_source=box_source)
            else:
                box=expand_box(c_roi,w,h,1.0) if c_roi else None
                if box is None:
                    # Full frame when C ROI cannot be established, marked explicitly as fallback.
                    box_source='whole_frame_fallback'
                else:box_source='manual_local_roi'
                images=[label(frames.frame(reference),reference,image_side,box),label(frames.frame(t),t,image_side,box)]
                prompt=C_PROMPT.format(ref=reference,curr=t)
                item.update(reference=reference,box_xyxy=box,box_source=box_source)
            result=client.ask(images,prompt,system_prompt=SYSTEM_SHARED)
            item['verdict']=parse_verdict(result['answer'],agent)
            item['status']='ok'
            if item['verdict']['signal']>0 or (agent in ('B','C') and index%10==0):
                sheet=Image.new('RGB',(sum(i.width for i in images),max(i.height for i in images)),(0,0,0))
                off=0
                for im in images:sheet.paste(im,(off,0));off+=im.width
                fig=case_dir/'candidates'/f'{agent}_f{t:04d}.jpg';fig.parent.mkdir(parents=True,exist_ok=True)
                sheet.save(fig,quality=88)
                item['preview']=str(fig)
            print(f'{agent} f{t:04d} [{index+1}/{len(targets)}] '+('raw_pair_candidate=' if agent=='A' else 'signal=')+f'{item["verdict"]["signal"]:.2f} {item["verdict"].get("observation","")[:95]}',flush=True)
        except Exception as exc:
            item['error']=str(exc)
            print(f'{agent} f{t:04d} ERROR: {exc}',flush=True)
        done[t]=item
        write_json(path,{'agent':agent,'start':start,'end':end,'reference':reference,'rows':[done[k] for k in sorted(done)]})
    if agent=='A':
        data=read_json(path)
        audit=audit_a_temporal_continuity(data['rows'])
        data['temporal_audit']=audit
        write_json(path,data)
        print('A TEMPORAL CONSISTENCY:',json.dumps(audit),flush=True)
        print('NOTE: A per-pair log line is provisional. Final A signals are after shared-frame consistency audit.',flush=True)
    return report_stage(path,agent)

def pair_sheet(frames:Frames, a:int, b:int, image_side:int, *, title:str, box=None):
    """Show precise pair as one image; prevents D from losing frame identity."""
    left=label(frames.frame(a),a,image_side,box)
    right=label(frames.frame(b),b,image_side,box)
    header=36
    result=Image.new('RGB',(left.width+right.width,max(left.height,right.height)+header),(0,0,0))
    result.paste(left,(0,header));result.paste(right,(left.width,header))
    ImageDraw.Draw(result).text((9,10),title+' | LEFT=BEFORE RIGHT=AFTER',fill=(255,255,255))
    return result


def adjudicate(client,frames:Frames,case_dir:Path,start:int,end:int,image_side:int,force:bool):
    destination=case_dir/'D_final.json'
    if destination.exists() and not force:return read_json(destination)
    reports={k:report_stage(case_dir/f'{k}_queries.json',k) for k in 'ABC'}
    if any(not r['complete'] for r in reports.values()):
        missing=[f'{k}({r.get("n_ok",0)}/{r.get("n_queries",0)})' for k,r in reports.items() if not r['complete']]
        print('D skipped; specialists incomplete:',', '.join(missing),flush=True)
        return None

    # Positive findings get the full BEFORE/AFTER pairs. Zero-score specialists
    # do not waste the image budget and crowd out later important events.
    candidates=[]
    for k,limit in [('A',3),('B',2),('C',2)]:
        candidates.extend((k,x) for x in reports[k]['top'][:limit] if x['signal']>=(.65 if k=='A' else .01))
    candidates.sort(key=lambda item:(-item[1]['signal'], {'A':0,'B':1,'C':2}[item[0]],item[1]['frame']))
    candidates=candidates[:5]
    if not candidates:
        # Full case with no specialist positives still receives representative
        # original source pair baselines, never an empty visual request.
        ts=sorted(set([max(start,1),max(start,1)+(end-max(start,1))//2,end]))
        candidates=[('BASELINE',{'frame':t,'prev':max(0,t-1),'signal':0,'observation':'No specialist positive'}) for t in ts]

    images=[];manifest=[];times=set()
    def add(kind, a, b, role, *, box=None):
        if not(0<=a<=b<=end):return
        im=pair_sheet(frames,a,b,image_side,title=f'{kind} {role} f{a:04d} to f{b:04d}',box=box)
        images.append(im)
        target=case_dir/'D_evidence'/f'image_{len(images):02d}_{kind}_{role}_f{a:04d}_f{b:04d}.jpg'
        target.parent.mkdir(exist_ok=True,parents=True);im.save(target,quality=91)
        manifest.append({'image_index':len(images),'specialist':kind,'role':role,'source_frames':[a,b],
                         'crop_xyxy':box,'saved_image':str(target)})
        times.update([a,b])

    for k,x in candidates:
        t=int(x['frame'])
        if k=='C':
            ref=x.get('reference')
            if ref is not None and 0<=int(ref)<=end:
                add(k,int(ref),t,'normal_reference_vs_candidate')
            else:add(k,max(start,t-1),t,'candidate')
        else:
            prev=int(x.get('prev') if x.get('prev') is not None else t-1)
            add(k,prev,t,'main_candidate')
            # Immediate context for high-scoring A, to inspect top clearance path.
            if k=='A' and x['signal']>=.5:
                if prev-1>=0:add(k,prev-1,prev,'preceding_motion')
                if t+1<=end:add(k,t,t+1,'following_motion')
            if k=='B' and x.get('box_xyxy'):
                add(k,prev,t,'gripper_object_closeup',box=x['box_xyxy'])
        if len(images)>=12:break
    if not images:raise ValueError('No images prepared for D')

    evidence={'observed_frame_range':[start,end],'specialists':reports,
              'image_evidence_index':manifest,
              'focus':'Type A: actor bottom vs container rim BEFORE/AFTER; is there an above-rim path?'}
    user_prompt=D_PROMPT+'\nEVIDENCE INDEX AND SPECIALIST OBSERVATIONS:\n'+json.dumps(evidence,ensure_ascii=False)[:21500]
    response=client.ask(images,user_prompt,system_prompt=SYSTEM_SHARED)
    verdict=get_obj(response['answer'])
    if verdict.get('category') not in {'PENETRATION','NO_PENETRATION','UNCERTAIN'}:raise ValueError('Invalid D category')
    if verdict.get('subtype') not in {'A','B','C','NONE','UNCERTAIN'}:raise ValueError('Invalid D subtype')
    score=float(verdict.get('evidence_score'))
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid D score')
    verdict['evidence_score']=score
    result={'frame_range':[start,end],'key_original_frames':sorted(times),
            'evidence_images':manifest,'reports':reports,'verdict':verdict,
            'notes':'Visual hypothesis test, no GT inputs; scores are not calibrated probabilities.'}
    write_json(destination,result)
    print('D SOURCE FRAMES:',sorted(times),flush=True)
    print('FINAL D:',json.dumps(verdict,ensure_ascii=False),flush=True)
    return result


def run(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Run requires a GPU compute node. Use srun first.')
    catalog=read_json(args.catalog) if args.catalog.is_file() else {'cases':[]}
    chosen=next((c for c in catalog['cases'] if c['id']==args.case),None)
    video=args.video or (Path(chosen['video']) if chosen else None)
    if video is None:raise ValueError('Case not in catalog; supply --video /absolute/path/video.mp4')
    video=Path(video).resolve()
    if not video.exists():raise FileNotFoundError(video)
    n,w,h,fps=video_info(video)
    start=args.start if args.start is not None else 0
    end=args.end if args.end is not None else n-1
    if not 0<=start<=end<n:raise ValueError(f'Invalid range [{start},{end}] for {n} frames')
    reference=args.reference if args.reference is not None else 0
    if not 0<=reference<n:raise ValueError('Invalid reference frame')
    if args.b_masks and not args.b_masks.is_file():raise FileNotFoundError(args.b_masks)
    if args.b_boxes and not args.b_boxes.is_file():raise FileNotFoundError(args.b_boxes)
    c_roi=parse_box(args.c_roi);b_roi=parse_box(args.b_roi)
    case_dir=args.output/args.case
    case_dir.mkdir(parents=True,exist_ok=True)
    cfg={'case':args.case,'video':str(video),'n_frames':n,'resolution':[w,h],
         'frame_range':[start,end],'reference':reference,'image_side':args.image_side,
         'b_roi':b_roi,'c_roi':c_roi,'b_masks':str(args.b_masks) if args.b_masks else None,
         'b_boxes':str(args.b_boxes) if args.b_boxes else None, 'model':str(args.model),
         'prompt_version':'ABC-explicit-pair-change-and-temporal-audit-v3'}
    existing=case_dir/'config.json'
    if existing.exists() and read_json(existing)!=cfg and not args.force:
        raise RuntimeError('Different config exists for this output case. Choose --output another run folder, or --force (must clear old query files).')
    if existing.exists() and read_json(existing)!=cfg and args.force:
        for f in [*(case_dir/f'{a}_queries.json' for a in 'ABC'),case_dir/'D_final.json',case_dir/'B_scout_boxes.json']:
            f.unlink(missing_ok=True)
    write_json(existing,cfg)
    write_json(case_dir/'prompts.json',{'shared':SYSTEM_SHARED,'A':A_PROMPT,'B':B_PROMPT,'C':C_PROMPT,'D':D_PROMPT,'scout':SCOUT_PROMPT})
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
    print('HOST:',os.uname().nodename,'CASE:',args.case,'TOTAL:',n,'RANGE:',(start,end),'REFERENCE:',reference,flush=True)
    print('Loading Wilson Qwen once:',args.model,flush=True)
    client=VLMClient('penetration_abc',{'backend':'local_gpu','model':str(args.model),'max_tokens':args.max_tokens})
    print('Qwen ready',client.architecture,flush=True)
    frames=Frames(video)
    try:
        for agent in args.agents:
            print(f'=== AGENT {agent} BEGIN ===',flush=True)
            result=run_agent(client,agent,frames,case_dir,start,end,reference,args.image_side,args.scout_side,
                             b_roi,c_roi,args.b_masks,args.b_boxes,args.scout_stride,args.max_tokens,(w,h))
            print(f'=== AGENT {agent} END peak={result["peak"]:.2f} good={result["n_ok"]}/{result["n_queries"]} runs={result["runs"][:4]} ===',flush=True)
        if args.adjudicate:adjudicate(client,frames,case_dir,start,end,args.image_side,args.force)
    finally:frames.close()
    stages={k:report_stage(case_dir/f'{k}_queries.json',k) for k in 'ABC'}
    write_json(case_dir/'stages_summary.json',stages)
    print('RESULTS:',case_dir,flush=True)

def check(args):
    if args.action=='score':
        for case_dir in sorted(args.output.iterdir()):
            if case_dir.is_dir():
                print(case_dir.name, ' '.join(f'{k}: {report_stage(case_dir/f"{k}_queries.json",k).get("peak",0):.2f}' for k in 'ABC'))
                f=case_dir/'D_final.json'
                if f.is_file():print(' D:',read_json(f)['verdict'])
        return
    sys.exit(1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['run','score'])
    ap.add_argument('--case',help='catalog case ID or manual case identifier')
    ap.add_argument('--video',type=Path,help='absolute path to video OR pre-extracted frames folder if absent from Wilson catalog')
    ap.add_argument('--catalog',type=Path,default=CATALOG)
    ap.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    ap.add_argument('--output',type=Path,default=ROOT/'penetration_abc_v3')
    ap.add_argument('--start',type=int,help='first included source frame, default 0')
    ap.add_argument('--end',type=int,help='last included source frame, default last')
    ap.add_argument('--reference',type=int,help='Type-C reference frame, default 0')
    ap.add_argument('--agents',nargs='+',choices=['A','B','C'],default=['A','B','C'])
    ap.add_argument('--adjudicate',action='store_true')
    ap.add_argument('--b-roi',help='B explicit ROI x1,y1,x2,y2 in original video pixel coordinates')
    ap.add_argument('--b-masks',type=Path,help='Wilson mask_join npz with link7 channel')
    ap.add_argument('--b-boxes',type=Path,help='Existing gripper bbox JSON per source frame')
    ap.add_argument('--c-roi',help='C explicit ROI x1,y1,x2,y2, e.g. 220,70,760,430 for LVP0051')
    ap.add_argument('--scout-stride',type=int,default=5)
    ap.add_argument('--scout-side',type=int,default=640)
    ap.add_argument('--image-side',type=int,default=768)
    ap.add_argument('--max-tokens',type=int,default=520)
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()
    if args.action=='score':return check(args)
    if not args.case:ap.error('run requires --case')
    if not 256<=args.image_side<=1400 or args.scout_stride<1:ap.error('bad image-side or scout-stride')
    run(args)
if __name__=='__main__':main()
