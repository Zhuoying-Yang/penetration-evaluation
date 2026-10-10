#!/usr/bin/env python3
"""Transparent drawer V2: evidence-first visual audit with existing local Qwen.

Development diagnostics only for manually selected windows. Original Stage 1 and
existing V1 files are never modified. Uses exactly the V1 six unannotated images
(1 full context + 5 chronological fixed crops) and saves raw model text.

Examples:
 python reverse_transparent_vlm_v2.py self-test
 python reverse_transparent_vlm_v2.py run --video-base /path/to/seed103 \
   --spec 0055:74,76,78,80,82 --spec 0056:148,150,152,154,156 \
   --spec 0057:174,177,180,183,186 --output /path/to/output
 python reverse_transparent_vlm_v2.py report --output /path/to/output \
   --old-output /path/to/transparent_vlm_verifier_v1

Not a calibrated penetration detector, not GT-blind selection, no 3D guarantee.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

# This import does NOT initialize or load any model.
import reverse_transparent_vlm_verifier_v1 as base

VERSION = "transparent-v2-evidence-first-20261010"
SYSTEM = (
    "You audit visual evidence in short robot manipulation clips. Never infer a "
    "physical outcome just from the task or expected answer. Do not assume "
    "a solid wall is penetrated because its image projection overlaps the "
    "gripper. Do not assume motion is physically legal merely because an "
    "open top exists somewhere. An observed path matters. If transparent "
    "surfaces prevent a determination, report UNCLEAR. Respond with one JSON "
    "object only, no markdown."
)
PROMPT = '''Six unannotated images are provided: image 1 is a full-scene middle-frame
reference; images 2–6 are chronological crops of source frames {frames}. The
five snapshots may skip intervening frames. Source timestamps are frame IDs,
not durations; do not invent what happens between observed frames.

Objects: the BLACK robot gripper tips/jaws; a CLEAR/TRANSPARENT drawer with
SOLID plastic front and side panels, a top rim and an OPEN TOP, and an opaque
beige drawer housing. The drawer may slide while the gripper stays nearly still.

STEP 1 — VISUAL FACTS, not a physics verdict:
- Describe what the BLACK LOWEST gripper jaw/tip does in the first, middle,
  and last crop, relative to the moving drawer. Locate the same solid panel.
- Does the solid panel move horizontally relative to the gripper? Does its
  boundary appear to pass across the gripper's 2D image position?
- Do NOT treat a front/back order switch through transparent plastic as
  established 3D interpenetration; it can be ordinary occlusion.
- Explicitly look for gripper elevation above the panel rim at or before
  the apparent crossing. Distinguish actual observed movement from what
  COULD theoretically happen through an open top.

STEP 2 — COMPETING MECHANICAL HYPOTHESES:
A. VALID route: observed gripper clearance over the rim; OR visually
   established clearance/separation such that moving solid material misses
   the gripper. Specify WHICH sampled frames support it. Merely saying
   'the drawer has an open top' or 'the drawer slides around the gripper'
   without visible separation/clearance is NOT evidence of a valid route.
B. INVALID route: the SAME solid wall changes sides relative to a gripper
   portion that stays below the opening, with specific evidence that their
   OCCUPIED 3D regions intersect rather than just overlap in 2D.
C. INDETERMINATE: the apparent image-space crossing is visible but actual
   3D depth, visibility, or the skipped frames leave the path ambiguous.
A normal sliding mechanism is not itself evidence for or against a collision.

STEP 3 — NEUTRAL EVIDENCE RATING (NOT ground truth):
0.0–0.2 obvious normal motion with a concrete visible normal-path explanation;
0.2–0.4 weak concern, transparency/occlusion likely explains it;
0.4–0.6 concerning wall-relative relationship but 3D passage unresolved;
0.6–0.9 strong specific evidence of solid-wall inconsistency with no observed
valid alternative; 0.9–1.0 exceptionally direct evidence. Do NOT inflate a
score because an opening-clearance path was not shown; skipped frames do not
prove absence of legal motion. Do NOT force all uncertain videos to 0.5.

Return EXACTLY one JSON object with these keys:
{{
 "first_middle_last_gripper":"observable tip motion in the three sampled stages",
 "first_middle_last_wall":"observable SAME panel motion in the stages",
 "panel_vs_gripper_image_crossing":"YES|NO|UNCLEAR",
 "gripper_above_rim_at_crossing_observed":"YES|NO|UNCLEAR",
 "actual_separation_from_wall_observed":"YES|NO|UNCLEAR",
 "true_3d_intersection_observed":"YES|NO|UNCLEAR",
 "legal_path_visually_supported":"YES|NO|UNCLEAR",
 "legal_path_evidence_frames":"which supplied frames actually show clearance or separation, or NONE",
 "illegal_path_evidence_frames":"which supplied frames show a physically impossible intersection, or NONE",
 "depth_and_transparency_ambiguity":"YES|NO|UNCLEAR",
 "boundary_mark_on_gripper":"YES|NO|UNCLEAR",
 "normal_motion_explanation":"specific observed alternative, or UNCLEAR",
 "possible_failure_explanation":"specific visual evidence, or UNCLEAR",
 "visual_inconsistency_score":0.0,
 "reason":"2-4 short sentences contrasting observed evidence and remaining uncertainty"
}}'''
REQUIRED = (
    "first_middle_last_gripper", "first_middle_last_wall",
    "panel_vs_gripper_image_crossing", "gripper_above_rim_at_crossing_observed",
    "actual_separation_from_wall_observed", "true_3d_intersection_observed",
    "legal_path_visually_supported", "legal_path_evidence_frames",
    "illegal_path_evidence_frames", "depth_and_transparency_ambiguity",
    "boundary_mark_on_gripper", "normal_motion_explanation",
    "possible_failure_explanation", "visual_inconsistency_score", "reason",
)
ENUM_KEYS=(
    "panel_vs_gripper_image_crossing", "gripper_above_rim_at_crossing_observed",
    "actual_separation_from_wall_observed", "true_3d_intersection_observed",
    "legal_path_visually_supported", "depth_and_transparency_ambiguity",
    "boundary_mark_on_gripper",
)

def parse_result(raw):
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{', raw):
        try:
            obj,_=decoder.raw_decode(raw[m.start():])
        except (json.JSONDecodeError,ValueError):
            continue
        if isinstance(obj,dict) and 'visual_inconsistency_score' in obj:
            return obj
    raise ValueError('Could not find a complete JSON verdict in raw model text')


def validate(obj):
    missing=[k for k in REQUIRED if k not in obj]
    if missing:raise ValueError(f'Missing verdict fields: {missing}')
    for k in ENUM_KEYS:
        if obj[k] not in ('YES','NO','UNCLEAR'):
            raise ValueError(f'Bad enum: {k}={obj[k]}')
    v=obj['visual_inconsistency_score']
    if isinstance(v,bool):raise ValueError('Boolean score invalid')
    score=float(v)
    if not (math.isfinite(score) and 0<=score<=1):raise ValueError('Score must be between 0 and 1')
    obj['visual_inconsistency_score']=round(score,4)
    return obj


def unsupported_legal_claim(obj):
    # Flag a *logical* inconsistency. Do not force a positive penetration score.
    return (obj['legal_path_visually_supported']=='YES'
            and obj['gripper_above_rim_at_crossing_observed']!='YES'
            and obj['actual_separation_from_wall_observed']!='YES')


def parse_spec(s):
    if ':' not in s:raise ValueError('Specify CASE:frame1,...,frame5, e.g. 0056:148,150,152,154,156')
    cid,raw=s.split(':',1)
    if not re.fullmatch(r'\d{4}',cid):raise ValueError('Four digit case required')
    return cid,base.frames_parse(raw)


def hash_cfg(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:12]


def run(args):
    specs=[parse_spec(s) for s in args.spec]
    if args.backend=='qwen' and not args.preview_only:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('GPU node required for Qwen backend')
        sys.path.insert(0,str(base.WS))
        from robot.preprocessing.link7_persistent.vlm_client import VLMClient
        client=VLMClient('drawer_v2_geometry_audit',{
            'backend':'local_gpu','model':str(args.model),'max_tokens':args.max_tokens})
        print('LOCAL QWEN:',getattr(client,'architecture','unknown'),flush=True)
    elif args.backend=='gemini' and not args.preview_only:
        api_base,key=base.cloud_auth(args.env_file)
    roi=base.roi_parse(args.roi_norm)
    prompt_hash=hash_cfg({'system':SYSTEM,'prompt':PROMPT})
    output=Path(args.output)
    for cid,frs in specs:
        video=Path(args.video_base)/f'{cid}.mp4'
        n,w,h=base.get_video(video)
        if not 0<=frs[0]<frs[-1]<n:raise ValueError(f'Invalid frames for {video}')
        tag=f'f{frs[0]}_{frs[-1]}'
        dest=output/cid/tag
        dest.mkdir(parents=True,exist_ok=True)
        prompt=PROMPT.format(frames=', '.join(f'f{i}' for i in frs))
        config={'version':VERSION,'backend':args.backend,'case':cid,
                'frames':frs,'video':str(video),'roi_norm':roi,
                'full_side':args.full_side,'crop_side':args.crop_side,
                'prompt_hash':prompt_hash,'max_tokens':args.max_tokens,
                'model':str(args.model if args.backend=='qwen' else args.gemini_model),
                'selection':'MANUAL_DIAGNOSTIC_NOT_GT_BLIND'}
        name=f'{args.backend}_{hash_cfg(config)}'
        result=dest/f'{name}.json'
        rawfile=dest/f'{name}.txt'
        if result.exists():
            prev=json.loads(result.read_text())
            if prev.get('status')=='ok':
                print('CACHED:',cid,'score',prev['verdict']['visual_inconsistency_score'])
                continue
        imgs,rect=base.images_for(video,frs,roi,args.full_side,args.crop_side)
        base.contact_sheet(imgs).save(dest/'preview.jpg',quality=90)
        (dest/'prompt_v2.txt').write_text(SYSTEM+'\n\n'+prompt)
        base.write_json(dest/'input.json',{'config':config,'roi_pixels':rect})
        print(f'{cid}: frames={frs} roi={rect} images={len(imgs)}',flush=True)
        if args.preview_only:continue
        began=time.monotonic()
        try:
            if args.backend=='qwen':
                answer=client.ask(imgs,prompt,system_prompt=SYSTEM)
                raw=str(answer.get('answer',''))
                meta={}
            else:
                raw,meta=base.cloud(imgs,prompt,api_base,key,args.gemini_model,
                                    args.max_tokens,args.timeout)
            rawfile.write_text(raw)
            obj=validate(parse_result(raw))
            flagged=unsupported_legal_claim(obj)
            payload={'status':'ok','config':config,'verdict':obj,'metadata':meta,
                     'unsupported_legal_path_claim':flagged,
                     'elapsed_seconds':round(time.monotonic()-began,2)}
            print(' SCORE:',obj['visual_inconsistency_score'],
                  'image-crossing:',obj['panel_vs_gripper_image_crossing'],
                  'legal:',obj['legal_path_visually_supported'],
                  'unsupported-legal:',flagged,
                  'seconds:',payload['elapsed_seconds'],flush=True)
        except Exception as e:
            payload={'status':'error','config':config,
                     'error':f'{type(e).__name__}: {e}',
                     'elapsed_seconds':round(time.monotonic()-began,2)}
            print(' ERROR:',payload['error'],flush=True)
        base.write_json(result,payload)
    report(output,args.old_output)


def old_for_case(oldroot,cid,frs):
    if not oldroot:return None
    folder=Path(oldroot)/cid/f'f{frs[0]}_{frs[-1]}'
    for f in sorted(folder.glob('qwen_*.json')):
        try:
            d=json.loads(f.read_text())
            if d.get('status')=='ok' and d.get('config',{}).get('frames')==frs:
                return d['verdict']
        except (OSError,ValueError,KeyError):continue
    return None


def report(out,old_output):
    out=Path(out);rows=[]
    for p in sorted(out.glob('*/f*/*_*.json')):
        try:d=json.loads(p.read_text())
        except (OSError,ValueError):continue
        if d.get('status')!='ok' or 'verdict' not in d:continue
        v=d['verdict'];cfg=d['config'];old=old_for_case(old_output,cfg['case'],cfg['frames'])
        rows.append({'case':cfg['case'],'backend':cfg['backend'],
                     'frames':','.join(map(str,cfg['frames'])),
                     'old_score':old.get('visual_inconsistency_score') if old else '',
                     'v2_score':v['visual_inconsistency_score'],
                     'image_crossing':v['panel_vs_gripper_image_crossing'],
                     'legal_observed':v['legal_path_visually_supported'],
                     'separation_observed':v['actual_separation_from_wall_observed'],
                     'rim_clearance_observed':v['gripper_above_rim_at_crossing_observed'],
                     '3d_intersection_observed':v['true_3d_intersection_observed'],
                     'unsupported_legal_claim':d['unsupported_legal_path_claim'],
                     'reason':v['reason'],'result_file':str(p)})
    if rows:
        path=out/'comparison_v2.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print('COMPARISON:',path)
        for r in rows:print(f" {r['case']} {r['backend']} old={r['old_score']} new={r['v2_score']} "
                          f"legal={r['legal_observed']} crossing={r['image_crossing']} "
                          f"unsupported_legal={r['unsupported_legal_claim']}")
    else:
        print('No valid scored responses yet.')


def selftest():
    assert parse_spec('0056:148,150,152,154,156')==('0056',[148,150,152,154,156])
    example={k:'UNCLEAR' for k in REQUIRED};example['visual_inconsistency_score']=.4
    example['legal_path_visually_supported']='YES'
    example['actual_separation_from_wall_observed']='NO'
    example['gripper_above_rim_at_crossing_observed']='NO'
    assert unsupported_legal_claim(validate(example))
    example['actual_separation_from_wall_observed']='YES'
    assert not unsupported_legal_claim(example)
    raw='```json\n'+json.dumps(example)+'\n```'
    assert parse_result(raw)['visual_inconsistency_score']==.4
    assert 'Merely saying' in PROMPT
    print('SELF-TEST PASS: frame selection, complete JSON, evidence-logic check')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('self-test')
    p=sub.add_parser('run');p.add_argument('--video-base',type=Path,required=True)
    p.add_argument('--spec',action='append',required=True,help='CASE:f1,f2,f3,f4,f5. Repeatable.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-output',type=Path)
    p.add_argument('--backend',choices=['qwen','gemini'],default='qwen')
    p.add_argument('--model',type=Path,default=base.MODEL)
    p.add_argument('--gemini-model',default='gemini-3.8-flash')
    p.add_argument('--env-file',type=Path,default=base.WS/'.env.vlm')
    p.add_argument('--max-tokens',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=240)
    p.add_argument('--roi-norm',default='0.28,0.26,0.76,0.90')
    p.add_argument('--full-side',type=int,default=900)
    p.add_argument('--crop-side',type=int,default=800)
    p.add_argument('--preview-only',action='store_true')
    rep=sub.add_parser('report');rep.add_argument('--output',type=Path,required=True)
    rep.add_argument('--old-output',type=Path)
    args=ap.parse_args()
    if args.action=='self-test':selftest()
    elif args.action=='run':run(args)
    else:report(args.output,args.old_output)

if __name__=='__main__':main()
