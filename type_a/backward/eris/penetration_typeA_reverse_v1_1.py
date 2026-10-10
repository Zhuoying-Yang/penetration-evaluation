#!/usr/bin/env python3
"""Experimental, label-blind gripper/solid-panel crossing (both directions).

ERIS H200 runner. Uses existing penetration_abc_agents_v3 Frames/label/get_obj and
Wilson VLMClient. Multi-frame Agent0 -> all-frame 3-frame Qwen observations ->
GT-blind candidate selection (top evidence + temporal coverage) -> Gemini audit.

Does NOT change frozen Type A or Type C. NOT a calibrated 3D penetration probability.
Manual abnormal frame intervals and GT are never provided to either VLM.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
QWEN_MODEL = Path('/PHShome/zy992/Wilson/dependency/models/qwen_model')
PROTOCOL = 'experimental_typeA_reverse_gripper_drawer_v1_1_20261010'

SYSTEM = ("You are a visual observation agent. Do not infer abnormality from task intentions. "
          "Use only images provided. 2D overlap is not proof of physical penetration. "
          "Return exactly one valid JSON object, with no markdown.")

AGENT0 = '''OBJECT IDENTIFICATION ONLY. These FIVE pictures are uniformly sampled from a generated robot manipulation video.
Original task (may be absent): {task}
Identify up to TWO visually grounded potential interactions between a ROBOT GRIPPER or ROBOT ARM END-EFFECTOR and a SOLID ENVIRONMENT BOUNDARY, such as a drawer front/side panel, cabinet wall, bin wall, or solid door face.
Choose the moving ROBOT PART as actor, and the ENVIRONMENT object as target. State which specific SOLID region is relevant. The gripper must be visible somewhere in the given frames; do not guess unseen actors. Do not judge whether any crossing or error happened. Do not use labels, video IDs, or expectations about which frames are anomalous.
Prefer a genuine SOLID panel, not an open drawer cavity, legal upper opening, empty space, tabletop support surface, or a manipulable object being carried. If no grounded pair can be identified, return an empty list.
Return ONLY JSON:
{{"pairs":[{{"actor":"short robot moving part name","target":"short environment name","solid_boundary":"specific solid panel/wall","pair_type":"ROBOT_ENVIRONMENT","expected_interaction":"neutral description"}}]}}'''

# Agent0 fallback: neutral scene inventory, not a penetration/crossing query.
# No GT labels, suspect frame IDs, anomaly hints, or case-specific actor/panel names.
AGENT0_INVENTORY = '''SCENE INVENTORY ONLY. These SEVEN images are uniformly sampled across one robot-manipulation video.
Original task (may be absent): {task}
Your job is to name VISIBLE scene elements, independently of whether they interact or move correctly.
(1) Is any ROBOT GRIPPER / END EFFECTOR visible in at least one image?
(2) Is any DRAWER / CABINET / BIN / DOOR / similar environment fixture visible in at least one image?
(3) Is a specific SOLID PANEL/WALL/FACE of that fixture visually identifiable?
If robot and fixture are visible, do not require them to touch or overlap. The goal is a neutral
tracking pair, NOT a diagnosis of penetration. A drawer may be open. Its solid front or side panel
still counts as a solid panel; empty cavity and openings do not. Do not include carried objects
as the environment or assume a panel you cannot distinguish. Use NO or UNCLEAR if unsupported.
Return ONLY JSON, no prose:
{{"robot_visible":"YES|NO|UNCLEAR", "fixture_visible":"YES|NO|UNCLEAR", "solid_panel_visible":"YES|NO|UNCLEAR",
  "robot_name":"short visible robot end-effector name or empty string",
  "fixture_name":"short visible fixture name or empty string",
  "solid_panel_name":"specific visible solid fixture panel/wall or empty string",
  "observation":"brief appearance-only justification; no physical error judgment"}}'''

SCOUT = '''ROI SCOUT only. In this one frame, find a region that contains BOTH the named robot end-effector and the nearby named environment solid panel. The rectangle should cover the end-effector tips AND enough of the solid panel to see its actual edges and legal opening. If both cannot be located together, use null. Do not judge any physical error.
ACTOR: {actor}
TARGET: {target}
SOLID PANEL: {boundary}
Return JSON ONLY: {{"roi_norm":[x1,y1,x2,y2] or null,"reason":"short"}}'''

QWEN_PROMPT = '''ROLE: THREE-FRAME SOLID PANEL PATH OBSERVER (not a contact or grasp classifier).
You are given consecutive SOURCE video frames {a}, {b}, {c}, ordered chronologically.
Images 1,2,3: FULL frames {a},{b},{c}. Images 4,5,6: AUTOMATIC CONTEXT CROPS from these SAME frames, in matching order. Crop locations can change; do not mistake camera/crop changes for physical motion. All images display source frame IDs.
Actor: {actor}
Environment target: {target}
Solid material of interest: {boundary}

For EACH of the three frames, independently describe the location of the SAME PHYSICAL gripper jaw/end-effector relative to the solid drawer/cabinet/bin panel:
- INSIDE: visibly on the INTERIOR side of the specified solid panel.
- OUTSIDE: visibly on its EXTERIOR side.
- OPENING: above/through a legitimate aperture or doorway, not solid material.
- OCCLUDED: hidden from view, cannot resolve its position.
- UNCLEAR: ambiguous depth or actor identity.

BOTH directions count as potential changes: INSIDE->OUTSIDE or OUTSIDE->INSIDE. Track the actual gripper body/tip, not a carried cube or banana. A gripper vanishing on one side and appearing on the other, WITHOUT a visible path across a specific SOLID PANEL, can instead be teleportation, occlusion, or a generation artifact: do not call it proven physical traversal. Distinguish crossing of a solid face from movement up and OVER a drawer rim, through the drawer's open top/front space, or around its side. A single 2D overlap can be depth-order occlusion. If in doubt say UNCLEAR.

Check whether a SIDE CHANGE is visibly supported across these three frames, AND independently whether that movement is tied to the SPECIFIC solid panel (not merely two positions in the image). Describe the gripper position in each frame and the strongest contradictory explanation. Do NOT assume that any event exists.

Return EXACT JSON, all categorical fields use YES/NO/UNCLEAR; states use INSIDE/OUTSIDE/OPENING/OCCLUDED/UNCLEAR:
{{"same_gripper_identifiable":"YES|NO|UNCLEAR","solid_panel_visible":"YES|NO|UNCLEAR",
 "state_f0":"INSIDE|OUTSIDE|OPENING|OCCLUDED|UNCLEAR",
 "state_f1":"INSIDE|OUTSIDE|OPENING|OCCLUDED|UNCLEAR",
 "state_f2":"INSIDE|OUTSIDE|OPENING|OCCLUDED|UNCLEAR",
 "inside_outside_side_change":"YES|NO|UNCLEAR",
 "specific_solid_panel_crossing_evidence":"YES|NO|UNCLEAR",
 "legal_opening_route_supported":"YES|NO|UNCLEAR",
 "ordinary_occlusion_or_camera_explains":"YES|NO|UNCLEAR",
 "teleportation_without_observed_wall_path_possible":"YES|NO|UNCLEAR",
 "observation":"Brief per-frame evidence, reference the source frame numbers",
 "alternative_explanation":"Best explanation besides solid-wall crossing"}}'''

GEMINI_PROMPT = '''INDEPENDENT THREE-FRAME GRIPPER / SOLID-ENVIRONMENT CROSSING VERIFICATION.
Images 1,2,3 = full SOURCE frames {a},{b},{c} in order; images 4,5,6 = automatic crops of corresponding frames. Ignore any inference from which windows were selected; you see NO ground-truth labels, Qwen answers, or known anomaly times.
Actor: {actor}
Environment: {target}
Specified solid region: {boundary}

Track the same robot jaw/tip across 3 frames. Assess either inside-to-outside OR outside-to-inside change across the SPECIFIED SOLID MATERIAL. Look for a before/after positional change with the intervening solid panel still physically blocking a legal path, rather than normal movement through a drawer opening, lifting over a rim, going around an edge, ordinary 2D occlusion, camera movement, or gripper simply disappearing/teleporting without evidence of crossing the wall. A large apparent jump across the panel is a suspicious candidate, not proof of 3D material traversal. Do not force YES when uncertain; identify what visual evidence is observable and what remains unknown.

Return EXACT JSON: All category fields YES/NO/UNCLEAR, except direction one of INSIDE_TO_OUTSIDE, OUTSIDE_TO_INSIDE, NONE, UNCLEAR.
{{"same_gripper_identified":"YES|NO|UNCLEAR", "solid_panel_identified":"YES|NO|UNCLEAR",
 "side_change_observed":"YES|NO|UNCLEAR", "direction":"INSIDE_TO_OUTSIDE|OUTSIDE_TO_INSIDE|NONE|UNCLEAR",
 "solid_wall_crossing_supported":"YES|NO|UNCLEAR",
 "legal_opening_path_supported":"YES|NO|UNCLEAR",
 "ordinary_occlusion_explains":"YES|NO|UNCLEAR",
 "teleportation_without_wall_path_possible":"YES|NO|UNCLEAR",
 "supporting_evidence":"Describe specific observations in frames {a},{b},{c}",
 "contradicting_evidence":"Describe alternatives and uncertainty",
 "reason":"Concise evidence-grounded conclusion"}}'''

Q_FIELDS = ('same_gripper_identifiable','solid_panel_visible','inside_outside_side_change',
            'specific_solid_panel_crossing_evidence','legal_opening_route_supported',
            'ordinary_occlusion_or_camera_explains','teleportation_without_observed_wall_path_possible')
G_FIELDS = ('same_gripper_identified','solid_panel_identified','side_change_observed','solid_wall_crossing_supported',
            'legal_opening_path_supported','ordinary_occlusion_explains','teleportation_without_wall_path_possible')
STATES = {'INSIDE','OUTSIDE','OPENING','OCCLUDED','UNCLEAR'}
BOOL = {'YES','NO','UNCLEAR'}
DIRECTIONS = {'INSIDE_TO_OUTSIDE','OUTSIDE_TO_INSIDE','NONE','UNCLEAR'}


def hash_obj(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def store(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def sanitize(x):
    return ' '.join(re.sub(r"[^\w \-.,/()']",' ',str(x)).split())[:120]


def render(template, pair, **kw):
    return template.format(actor=sanitize(pair['actor']),target=sanitize(pair['target']),
                           boundary=sanitize(pair['solid_boundary']), **kw)


def normalize(verdict, fields):
    d=dict(verdict)
    for k in fields:
        value=str(d.get(k,'UNCLEAR')).strip().upper()
        d[k]=value if value in BOOL else 'UNCLEAR'
    for k in ('state_f0','state_f1','state_f2'):
        if k in d or fields==Q_FIELDS:
            v=str(d.get(k,'UNCLEAR')).strip().upper()
            d[k]=v if v in STATES else 'UNCLEAR'
    if fields==G_FIELDS:
        v=str(d.get('direction','UNCLEAR')).strip().upper()
        d['direction']=v if v in DIRECTIONS else 'UNCLEAR'
    for k in ('observation','alternative_explanation','supporting_evidence','contradicting_evidence','reason'):
        if k in d:d[k]=str(d[k])[:1500]
    return d


def qwen_score(v):
    """GT-independent proposal score. Not a probability. Ambiguous transitions retained."""
    if v['same_gripper_identifiable']=='NO' or v['solid_panel_visible']=='NO':return 0.0
    states=[v['state_f0'],v['state_f1'],v['state_f2']]
    seen={s for s in states if s in {'INSIDE','OUTSIDE'}}
    flip=('INSIDE' in seen and 'OUTSIDE' in seen)
    change=v['inside_outside_side_change']=='YES'
    if not change and not flip:return 0.0
    if v['legal_opening_route_supported']=='YES':return 0.10
    if v['specific_solid_panel_crossing_evidence']=='YES':
        if v['ordinary_occlusion_or_camera_explains']=='NO' and v['teleportation_without_observed_wall_path_possible']=='NO':
            return 0.95
        return 0.60 if change else 0.45
    if change or flip:
        return 0.45 if v['ordinary_occlusion_or_camera_explains']!='YES' else 0.25
    return 0.0


def gemini_score(v):
    """Evidence signal, not calibrated or 3D-proven; ambiguity preserved."""
    if v['same_gripper_identified']!='YES' or v['solid_panel_identified']!='YES':return 0.0
    if v['side_change_observed']!='YES':return 0.0
    if v['legal_opening_path_supported']=='YES' or v['ordinary_occlusion_explains']=='YES':return 0.0
    if v['solid_wall_crossing_supported']=='YES':
        if v['teleportation_without_wall_path_possible']=='NO' and v['legal_opening_path_supported']=='NO' and v['ordinary_occlusion_explains']=='NO':
            return 0.95
        return 0.65
    if v['solid_wall_crossing_supported']=='UNCLEAR':return 0.35
    return 0.0


def uniform_indices(n, count):
    if n<=0:return []
    if count<=1:return [n//2]
    return sorted(set(round(i*(n-1)/(count-1)) for i in range(min(n,count))))


def candidate_selection(rows, max_audits):
    """Select high scoring windows PLUS uniform probes even if all Qwen scores 0.

    This policy is fixed and cannot use event annotations. Spread across full timeline.
    """
    rows=sorted([r for r in rows if r.get('status')=='ok'],key=lambda r:r['frame'])
    if not rows:return []
    chosen=[];seen=set()
    probes=min(len(rows),max(2,max_audits//2))
    for k in uniform_indices(len(rows),probes):
        r=rows[k];t=r['frame']
        if t not in seen:
            chosen.append((r,'uniform_probe'));seen.add(t)
    remaining=sorted((r for r in rows if r['frame'] not in seen),
                     key=lambda r:(-float(r['signal']),r['frame']))
    for r in remaining:
        if len(chosen)>=max_audits:break
        if r['signal']<=0:break
        if any(abs(r['frame']-x['frame'])<3 for x,_ in chosen):continue
        chosen.append((r,'qwen_ranked'));seen.add(r['frame'])
    # If Qwen is entirely uninformative, still populate automatic coverage checks.
    while len(chosen)<min(max_audits,len(rows)):
        rest=[r for r in rows if r['frame'] not in seen]
        if not rest:break
        pick=max(rest,key=lambda r:(min(abs(r['frame']-z['frame']) for z,_ in chosen),
                                    float(r['signal']),-r['frame']))
        chosen.append((pick,'coverage_fill'));seen.add(pick['frame'])
    return sorted(chosen,key=lambda x:x[0]['frame'])


def good_roi(v):
    if not isinstance(v,(list,tuple)) or len(v)!=4:return None
    try:
        a,b,c,d=[float(x) for x in v]
    except (ValueError,TypeError):return None
    if all(math.isfinite(x) for x in [a,b,c,d]) and 0<=a<c<=1 and 0<=b<d<=1 and c-a>=0.04 and d-b>=0.04:
        return [a,b,c,d]
    return None


def to_box(roi,w,h):
    if roi is None:return None
    a,b,c,d=roi
    # generous context to preserve legitimate opening / panel edges
    pad=0.11
    return [max(0,int((a-pad)*w)),max(0,int((b-pad)*h)),
            min(w,int((c+pad)*w)),min(h,int((d+pad)*h))]


def triplet_images(v3,frames,w,h,ts,roi_for):
    full=[v3.label(frames.frame(t),t,720) for t in ts]
    boxes=[to_box(roi_for(t),w,h) for t in ts]
    crops=[v3.label(frames.frame(t),t,864,box) for t,box in zip(ts,boxes)]
    return full+crops,boxes


def preview(images,out):
    from PIL import Image,ImageDraw
    W,H=690,450
    canvas=Image.new('RGB',(3*W,2*H),'white')
    d=ImageDraw.Draw(canvas)
    for i,original in enumerate(images):
        a=original.copy();a.thumbnail((W-12,H-35))
        x=(i%3)*W;y=(i//3)*H
        d.text((x+9,y+6),('SOURCE FULL ' if i<3 else 'AUTO CROP ')+str(i%3+1),fill='black')
        canvas.paste(a,(x+6,y+30))
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out,quality=88)


def load_gemini():
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
    from robot.preprocessing.link7_persistent.interface.config import role_config
    env=WS/'.env.vlm'
    if env.is_file():load_env_file(env)
    if not os.environ.get('VLM2_API_KEY'):
        raise RuntimeError('VLM2_API_KEY unavailable. Qwen results are cached; rerun when Gemini is configured.')
    os.environ.setdefault('PDI_VLM_ROLE_OVERRIDES','{"vlm2":{"api_base":"https://api.302.ai/v1"}}')
    cfg=role_config('vlm2')
    if cfg.get('backend')!='cloud_api':raise RuntimeError('VLM2 cloud_api not configured')
    cfg.update(model='gemini-3.8-flash',max_tokens=4096,temperature=0,reasoning_effort=None)
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
    return VLMClient('reverse_crossing_gemini_v1',cfg)


def filter_neutral_pairs(answer):
    """Screen only actor/panel identities; this does not make a penetration assertion."""
    pairs=[]
    for x in (answer.get('pairs') or [])[:3]:
        if not isinstance(x,dict):
            continue
        p={k:sanitize(x.get(k,'')) for k in
           ('actor','target','solid_boundary','pair_type','expected_interaction')}
        if not all(p.get(k) for k in ('actor','target','solid_boundary')):
            continue
        if not re.search(r'gripper|finger|jaw|claw|pincer|end.effector|robot.arm|robot.hand|robotic.manipulator',p['actor'],re.I):
            continue
        if re.search(r'opening|cavity|air|empty space|table ?top|table surface',p['solid_boundary'],re.I):
            continue
        pairs.append(p)
    return pairs[:2]


def pair_from_inventory(inv):
    """Require visual evidence of all three entities, with no inferred crossing."""
    def yes(k):return str(inv.get(k,'')).strip().upper()=='YES'
    if not all(yes(k) for k in ('robot_visible','fixture_visible','solid_panel_visible')):
        return []
    actor=sanitize(inv.get('robot_name',''))
    target=sanitize(inv.get('fixture_name',''))
    boundary=sanitize(inv.get('solid_panel_name',''))
    if not (actor and target and boundary):return []
    if re.search(r'opening|cavity|air|empty space|table ?top|table surface',boundary,re.I):return []
    if not re.search(r'gripper|finger|jaw|claw|pincer|end.effector|robot.arm|robot.hand|robotic.manipulator',actor,re.I):
        # Don't reject a confirmed visible robot end-effector because a VLM called it a hand/effector.
        actor='robot end-effector ('+actor[:60]+')'
    return [{'actor':actor,'target':target,'solid_boundary':boundary,
             'pair_type':'ROBOT_ENVIRONMENT',
             'expected_interaction':'robot end-effector is present near the environment fixture; interaction not asserted'}]


def agent0_pairs(qclient,v3,frames,n,task,case_dir):
    path=case_dir/'agent0_multiframe.json'
    ix=uniform_indices(n,5)
    prompt=AGENT0.format(task=task or 'NOT PROVIDED; use images alone')
    fallback_ix=uniform_indices(n,7)
    fallback_prompt=AGENT0_INVENTORY.format(task=task or 'NOT PROVIDED; use images alone')
    if path.is_file():
        x=load(path)
        if (x.get('prompt')!=prompt or x.get('source_frames')!=ix or
            x.get('fallback_prompt')!=fallback_prompt or x.get('fallback_source_frames')!=fallback_ix):
            raise RuntimeError('Agent0 cache mismatch; use a NEW output folder')
        pairs=x['pairs']
        print('AGENT0 CACHED:',x.get('selection_method'),'PAIRS:',json.dumps(pairs,ensure_ascii=False),flush=True)
        return pairs

    images=[v3.label(frames.frame(t),t,896) for t in ix]
    raw=qclient.ask(images,prompt,system_prompt=SYSTEM)['answer']
    try:
        answer=v3.get_obj(raw)
        primary=filter_neutral_pairs(answer)
        primary_error=None
    except Exception as e:
        primary=[]
        primary_error=f'{type(e).__name__}: {e}'
    pairs=primary
    fallback_raw=None
    fallback_inv=None
    fallback_error=None
    method='primary_5frames' if pairs else 'no_pair'
    if not pairs:
        print('AGENT0 PRIMARY EMPTY; running neutral 7-frame scene inventory',flush=True)
        fallback_images=[v3.label(frames.frame(t),t,896) for t in fallback_ix]
        fallback_raw=qclient.ask(fallback_images,fallback_prompt,system_prompt=SYSTEM)['answer']
        try:
            fallback_inv=v3.get_obj(fallback_raw)
            pairs=pair_from_inventory(fallback_inv)
            method='scene_inventory_7frames' if pairs else 'no_pair'
        except Exception as e:
            fallback_error=f'{type(e).__name__}: {e}'

    store(path,{'source_frames':ix,'prompt':prompt,'raw':raw,
                'primary_error':primary_error, 'primary_pairs':primary,
                'fallback_source_frames':fallback_ix,'fallback_prompt':fallback_prompt,
                'fallback_raw':fallback_raw,'fallback_inventory':fallback_inv,
                'fallback_error':fallback_error,'selection_method':method,'pairs':pairs,
                'gt_frame_or_label_used':False})
    print('AGENT0 METHOD:',method,'PAIRS:',json.dumps(pairs,ensure_ascii=False),flush=True)
    return pairs


def scout_box(qclient,v3,frames,t,pair,side=896):
    prompt=render(SCOUT,pair)
    raw=qclient.ask([v3.label(frames.frame(t),t,side)],prompt,system_prompt=SYSTEM)['answer']
    parsed=v3.get_obj(raw)
    return {'roi_norm':good_roi(parsed.get('roi_norm')),'reason':str(parsed.get('reason',''))[:200]}


def run_case(qclient,v3,video,case,task,args):
    if not video.is_file():raise FileNotFoundError(video)
    n,w,h,fps=v3.video_info(video)
    if n<4:raise ValueError('Need >=4 frames')
    dst=args.output/case
    dst.mkdir(parents=True,exist_ok=True)
    config={'protocol':PROTOCOL,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'v3_sha256':hashlib.sha256(Path(v3.__file__).read_bytes()).hexdigest(),
            'video':str(video.resolve()),'video_size':video.stat().st_size,
            'video_mtime_ns':video.stat().st_mtime_ns,'task':task,
            'n_frames':n,'qwen_model':str(args.model),'max_tokens':args.max_tokens,
            'roi_stride':args.roi_stride,'max_audits':args.max_audits,
            'qwen_prompt_sha':hash_obj(QWEN_PROMPT),'gemini_prompt_sha':hash_obj(GEMINI_PROMPT),
            'agent0_sha':hash_obj(AGENT0),
            'agent0_inventory_sha':hash_obj(AGENT0_INVENTORY),
            'selection':'qwen_ranked_plus_uniform_coverage_v1'}
    ch=hash_obj(config)
    cfg=dst/'run_config.json'
    if cfg.is_file():
        if load(cfg)['hash']!=ch:raise RuntimeError('Config differs; use NEW output folder, not cached results')
    else:store(cfg,{'hash':ch,'config':config})
    summ=dst/'summary.json'
    if summ.is_file() and load(summ).get('status')=='complete':
        print('COMPLETE CACHED:',case,flush=True);return
    frames=v3.Frames(video)
    try:
        print(f'\n===== {case} full frames={n} =====',flush=True)
        pairs=agent0_pairs(qclient,v3,frames,n,task,dst)
        if getattr(args,'agent0_only',False):
            store(dst/'agent0_only_summary.json',
                  {'case':case,'n_frames':n,'pairs':pairs,
                   'status':'pair_identified' if pairs else 'no_pair_identified',
                   'not_a_penetration_score':True,'config_hash':ch})
            print('AGENT0-ONLY:',case,'pairs=',len(pairs),'(no triplet Qwen/Gemini run)',flush=True)
            return
        if not pairs:
            store(summ,{'status':'not_applicable','case':case,'reason':'no_visible_gripper_solid_environment_pair',
                        'video_signal':None,'config_hash':ch})
            print('NO PAIR: not-applicable (NOT a negative)',flush=True)
            return
        all_rows=[]
        for pid,pair in enumerate(pairs):
            pdir=dst/'pairs'/f'p{pid}'
            qpath=pdir/'triplet_queries.json'
            cached=load(qpath) if qpath.is_file() else {'config_hash':ch,'rows':[]}
            if cached.get('config_hash')!=ch:raise RuntimeError('Qwen cache configuration mismatch')
            done={int(r['frame']):r for r in cached['rows'] if r.get('status')=='ok'}
            scout_path=pdir/'roi_scout.json'
            scouts=load(scout_path) if scout_path.is_file() else {'config_hash':ch,'anchors':{}}
            if scouts.get('config_hash')!=ch:raise RuntimeError('ROI cache mismatch')
            # Entire video, fixed temporal stride 1, no GT/event-time filtering.
            anchors=list(range(0,n,args.roi_stride))
            if n-1 not in anchors:anchors.append(n-1)
            for anchor in anchors:
                if str(anchor) in scouts['anchors']:continue
                try:val=scout_box(qclient,v3,frames,anchor,pair)
                except Exception as e:
                    val={'roi_norm':None,'reason':f'scout_failed_{type(e).__name__}'}
                scouts['anchors'][str(anchor)]=val
                store(scout_path,scouts)
                print(f'SCOUT p{pid} f{anchor:04d} roi={val["roi_norm"]}',flush=True)
            def roi_for(t):
                nearest=min(anchors,key=lambda a:(abs(a-t),a))
                return scouts['anchors'][str(nearest)]['roi_norm']
            targets=range(2,n)
            for i,t in enumerate(targets,1):
                if t in done:continue
                ts=[t-2,t-1,t]
                rec={'frame':t,'source_frames':ts,'status':'failed'}
                try:
                    images,boxes=triplet_images(v3,frames,w,h,ts,roi_for)
                    prompt=render(QWEN_PROMPT,pair,a=ts[0],b=ts[1],c=ts[2])
                    answer=qclient.ask(images,prompt,system_prompt=SYSTEM)['answer']
                    verdict=normalize(v3.get_obj(answer),Q_FIELDS)
                    rec.update(status='ok',verdict=verdict,signal=qwen_score(verdict),
                               raw=answer,roi_boxes=boxes)
                except Exception as e:
                    rec['error']=f'{type(e).__name__}: {e}'
                done[t]=rec
                store(qpath,{'config_hash':ch,'pair':pair,
                             'rows':[done[t0] for t0 in sorted(done)]})
                print(f'TRIPLET p{pid} f{t:04d} [{i}/{n-2}] score={rec.get("signal",0):.2f} '
                      f'{rec.get("verdict",{}).get("observation",rec.get("error",""))[:110]}',flush=True)
            rows=[done[t] for t in sorted(done)]
            if {int(r['frame']) for r in rows if r['status']=='ok'}!=set(targets):
                raise RuntimeError('Incomplete source triplet scan. Same command resumes failed frames.')
            csvpath=pdir/'triplet_scores.csv'
            with csvpath.open('w',newline='') as f:
                cols=['frame','f0','f1','f2','qwen_signal','state_f0','state_f1','state_f2',*Q_FIELDS,'observation']
                writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
                for r in rows:
                    v=r['verdict'];ts=r['source_frames']
                    writer.writerow({'frame':r['frame'],'f0':ts[0],'f1':ts[1],'f2':ts[2],
                        'qwen_signal':r['signal'],**{k:v.get(k,'') for k in [*cols[5:-1]]},
                        'observation':v.get('observation','')})
            all_rows.append((pid,pair,rows,roi_for))
        # Use global shared Gemini budget; no GT or manual frame timestamp.
        merged=[]
        for pid,pair,rows,_ in all_rows:
            for r in rows:merged.append({'pair_id':pid,'pair':pair,**r})
        # Avoid different ROI state closures captured after loop; recreate per pair at audit time.
        grouped=[]
        # Seed with Qwen-ranked + uniform probes per pair, then global cap.
        slots=max(2,args.max_audits//len(all_rows))
        for pid,pair,rows,_ in all_rows:
            for r,reason in candidate_selection(rows,slots):
                grouped.append({'pair_id':pid,'pair':pair,'frame':r['frame'],
                                'qwen_signal':r['signal'],'selection_reason':reason})
        grouped=sorted(grouped,key=lambda r:(0 if r['selection_reason']=='qwen_ranked' else 1,
                                         -r['qwen_signal'],r['frame'],r['pair_id']))[:args.max_audits]
        grouped.sort(key=lambda r:(r['frame'],r['pair_id']))
        store(dst/'selected_candidates.json',{'policy':'qwen_ranked_plus_uniform_coverage_v1',
             'no_manual_frame_windows':True,'selected':grouped})
        qmax=max(r['signal'] for _,_,rows,_ in all_rows for r in rows)
        store(dst/'qwen_summary.json',{'case':case,'n_frames':n,'pairs':pairs,
              'n_triplets':sum(len(x[2]) for x in all_rows),'qwen_peak':qmax,
              'n_qwen_positive':sum(r['signal']>=0.65 for _,_,rows,_ in all_rows for r in rows),
              'n_audit_candidates':len(grouped),'candidate_frames':[(x['pair_id'],x['frame']) for x in grouped]})
        if args.no_gemini:
            store(summ,{'status':'qwen_only','case':case,'video_signal':None,
                        'qwen_peak':qmax,'n_candidates':len(grouped),'config_hash':ch})
            print('QWEN ONLY; rerun same command WITHOUT --no-gemini for independent Gemini audits',flush=True)
            return
        gclient=load_gemini()
        audits=[]
        for j,item in enumerate(grouped,1):
            pid=item['pair_id'];pair=item['pair'];t=item['frame']
            out=dst/'gemini'/f'p{pid}_f{t:04d}.json'
            if out.is_file():
                result=load(out)
                if result['config_hash']!=ch:raise RuntimeError('Gemini audit cache mismatch')
            else:
                anchors_record=load(dst/'pairs'/f'p{pid}'/'roi_scout.json')['anchors']
                an=sorted(map(int,anchors_record))
                def roi_for(t0):
                    return anchors_record[str(min(an,key=lambda a:(abs(a-t0),a)))]['roi_norm']
                ts=[t-2,t-1,t]
                imgs,boxes=triplet_images(v3,frames,w,h,ts,roi_for)
                image_path=dst/'previews'/f'p{pid}_f{t:04d}.jpg'
                preview(imgs,image_path)
                prompt=render(GEMINI_PROMPT,pair,a=ts[0],b=ts[1],c=ts[2])
                response=gclient.ask(imgs,prompt,system_prompt=SYSTEM)['answer']
                v=normalize(v3.get_obj(response),G_FIELDS)
                result={'case':case,'pair_id':pid,'frames':ts,'selection_reason':item['selection_reason'],
                        'qwen_signal':item['qwen_signal'],'gemini_signal':gemini_score(v),
                        'verdict':v,'raw':response,'preview':str(image_path),'roi_boxes':boxes,'config_hash':ch}
                store(out,result)
            audits.append(result)
            print(f'GEMINI [{j}/{len(grouped)}] p{pid} f{t:04d} score={result["gemini_signal"]:.2f} '
                  f'{result["verdict"].get("reason","")[:110]}',flush=True)
        winner=max(audits,key=lambda x:x['gemini_signal'])
        store(summ,{'status':'complete','case':case,'n_frames':n,
             'video_signal':winner['gemini_signal'],'best_frame':winner['frames'][-1],
             'best_pair_id':winner['pair_id'],'best_selection_reason':winner['selection_reason'],
             'qwen_peak':qmax,'n_gemini':len(audits),
             'score_semantics':'experimental visual evidence, not validated 3D penetration probability',
             'config_hash':ch})
        print('FINAL',case,'score=',winner['gemini_signal'],
              'best_t=',winner['frames'][-1],flush=True)
    finally:
        frames.close()


def self_test():
    p={'actor':'robot gripper','target':'wooden drawer','solid_boundary':'solid front panel'}
    for s in [SCOUT,QWEN_PROMPT,GEMINI_PROMPT]:
        render(s,p,**({'a':2,'b':3,'c':4} if s!=SCOUT else {}))
    sample={k:'NO' for k in Q_FIELDS}
    sample.update(same_gripper_identifiable='YES',solid_panel_visible='YES',inside_outside_side_change='YES',
                  specific_solid_panel_crossing_evidence='YES',
                  legal_opening_route_supported='NO',ordinary_occlusion_or_camera_explains='NO',
                  teleportation_without_observed_wall_path_possible='NO',
                  state_f0='INSIDE',state_f1='INSIDE',state_f2='OUTSIDE')
    assert qwen_score(sample)==0.95
    sample['specific_solid_panel_crossing_evidence']='UNCLEAR'
    assert qwen_score(sample)==0.45
    g={k:'NO' for k in G_FIELDS}
    g.update(same_gripper_identified='YES',solid_panel_identified='YES',side_change_observed='YES',
             solid_wall_crossing_supported='YES',legal_opening_path_supported='NO',
             ordinary_occlusion_explains='NO',teleportation_without_wall_path_possible='NO')
    assert gemini_score(g)==0.95
    g['solid_wall_crossing_supported']='UNCLEAR'
    assert gemini_score(g)==0.35
    rows=[{'frame':i,'signal':0.,'status':'ok'} for i in range(2,32)]
    sel=candidate_selection(rows,8)
    assert len(sel)==8 and all(x[1] in {'uniform_probe','coverage_fill'} for x in sel)
    assert good_roi([.2,.2,.7,.7]) and good_roi([.8,.2,.2,.7]) is None
    assert uniform_indices(93,5)==[0,23,46,69,92]
    inv={'robot_visible':'YES','fixture_visible':'YES','solid_panel_visible':'YES',
         'robot_name':'robot gripper','fixture_name':'open drawer','solid_panel_name':'drawer front solid wall'}
    assert len(pair_from_inventory(inv))==1
    inv['solid_panel_visible']='UNCLEAR'
    assert pair_from_inventory(inv)==[]
    print('SELF-TEST PASS (prompts, scoring, bidirectional states, zero-Qwen fallback, ROI)')


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--self-test',action='store_true')
    a.add_argument('--video-dir',type=Path,default=Path('/scratch/z/zy992/zhuoying/physact/cosmos3/export_robowm68_3seeds/seed103'))
    a.add_argument('--cases',default='0053,0055,0056,0057',help='comma-separated 4-digit video IDs, never GT labels')
    a.add_argument('--prefix',default='Cosmos3_seed103_',help='filename label prefix only')
    a.add_argument('--output',type=Path,default=ROOT/'typeA_reverse_v1_seed103_20261009')
    a.add_argument('--model',type=Path,default=QWEN_MODEL)
    a.add_argument('--max-tokens',type=int,default=850)
    a.add_argument('--roi-stride',type=int,default=12)
    a.add_argument('--max-audits',type=int,default=8)
    a.add_argument('--no-gemini',action='store_true',help='only Qwen; rerun without this to audit cached Qwen windows')
    a.add_argument('--agent0-only',action='store_true',help='run neutral pair discovery only; no full video triplets')
    args=a.parse_args()
    if args.self_test:return self_test()
    if not (4<=args.roi_stride<=100 and 2<=args.max_audits<=24):a.error('Invalid --roi-stride / --max-audits')
    ids=[c.strip() for c in args.cases.split(',') if c.strip()]
    if not ids or len(set(ids))!=len(ids) or not all(re.fullmatch(r'\d{4}',x) for x in ids):
        a.error('--cases must be unique comma-separated four-digit video IDs')
    args.output=args.output.resolve()
    if 'typeA_reverse_v1' not in args.output.name:
        a.error('Output folder must contain typeA_reverse_v1 (protect frozen results)')
    if args.output==ROOT or args.output==ROOT.parent:a.error('Output must not be project root')
    runs=[(args.video_dir/f'{x}.mp4',args.prefix+x,'') for x in ids]
    missing=[str(p) for p,_,_ in runs if not p.is_file()]
    if missing:raise FileNotFoundError('Missing videos (nothing launched): '+', '.join(missing))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen requires allocated GPU. Start srun H200; do NOT run on ERIS login node.')
    import penetration_abc_agents_v3 as v3
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
    qclient=VLMClient('reverse_gripper_qwen_v1',{'backend':'local_gpu','model':str(args.model),'max_tokens':args.max_tokens})
    failed=[]
    for video,case,task in runs:
        try:run_case(qclient,v3,video,case,task,args)
        except Exception as e:
            failed.append(case)
            print('FAILED',case,type(e).__name__,str(e),flush=True)
            traceback.print_exc()
    if failed:raise SystemExit('Failed cases; partial caches preserved: '+', '.join(failed))


if __name__=='__main__':main()
