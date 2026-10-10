#!/usr/bin/env python3
"""Transparent-drawer reverse Type-A temporal VLM verifier (research diagnostic).

Keep frozen Qwen proposal stage unchanged. Second-stage verifier can be LOCAL Qwen
or cloud Gemini, with IDENTICAL six unannotated images and IDENTICAL prompt.

Examples:
  python reverse_transparent_vlm_verifier_v1.py preview --video 0056.mp4 --frames 148,150,152,154,156 --output out/0056
  python reverse_transparent_vlm_verifier_v1.py run --video 0056.mp4 --frames 148,150,152,154,156 --backend qwen --output out/0056
  python reverse_transparent_vlm_verifier_v1.py run --video 0056.mp4 --frames 148,150,152,154,156 --backend gemini --output out/0056
  python reverse_transparent_vlm_verifier_v1.py run --video 0056.mp4 --source A_queries.json --backend qwen --topk 5 --output out/0056
  python reverse_transparent_vlm_verifier_v1.py report --output out
  python reverse_transparent_vlm_verifier_v1.py self-test

Manual --frames mode is for diagnostic prompt development, NEVER formal AUROC.
--source mode selects candidate windows AUTOMATICALLY from an existing FULL-video
Qwen reverse Type-A scan (no manual event selection). The original stage is
never modified, and no prompt receives Qwen's score, label, or human GT.

Scores are visual evidence ratings, NOT calibrated probabilities or 3D proof.
"""
from __future__ import annotations

import argparse, base64, csv, hashlib, io, json, math, os, re, sys, time
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

WS=Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
MODEL=Path('/PHShome/zy992/Wilson/dependency/models/qwen_model')
VERSION='transparent-moving-wall-verifier-v1-20261010'

SYSTEM=('You are a cautious evaluator of visual evidence in robot manipulation videos. '
        'Describe what is OBSERVED, distinguish solid-wall crossing from legal opening passage, '
        'and state uncertainty explicitly. Do not infer outcomes from task expectations. '
        'Return only one valid JSON object.')

PROMPT='''These SIX images show FIVE chronological video frames of the SAME scene:
Image 1: FULL SCENE at the MIDDLE sampled frame (context only).
Images 2-6: SAME fixed crop of source frames {frames}, in that order.
The sampled frames may skip intermediate frames. Treat the images as snapshots, not continuous proof.

Physical entities: the black robot gripper/end-effector, and a TRANSPARENT SLIDING DRAWER
with solid plastic FRONT/SIDE WALLS, an OPEN TOP, a rim, and an opaque beige housing.
This is an ordinary robot video until visual evidence indicates otherwise.

IMPORTANT REVISION FROM A STATIONARY-WALL VERIFIER:
The drawer itself may MOVE horizontally while the gripper moves very little. A SOLID
DRAWER WALL passing across a gripper fixed in space can be as physically suspicious
as a gripper passing through a stationary wall. Judge RELATIVE gripper/wall motion;
do NOT reject a candidate merely because the gripper barely moves in image coordinates.

Use the following audit steps before assigning a score:
1. Locate the gripper jaw/tip and the SAME specific front/side panel in early, middle,
   late crops. Distinguish its real solid boundary from table grid seen THROUGH plastic.
2. Describe independent gripper and drawer motion. Is the drawer sliding? Is its front
   solid wall approaching, passing, or moving away from the gripper? Is the gripper itself rising?
3. Examine whether the gripper/panel's relative FRONT-BACK or INSIDE-OUTSIDE relation
   visibly CHANGES. Projected overlap, apparent 2D side switching, or seeing the gripper
   through transparent plastic is NOT by itself evidence of 3D interpenetration.
4. Test a legal path: gripper can exit ABOVE the rim or via a real unobstructed opening;
   a sliding drawer can legally move around a gripper. Identify VISIBLE support for such
   a path, not merely the existence of an open top. Unseen actions in skipped frames
   remain UNKNOWN; absence of a visible legal path is NOT proof of a solid-wall crossing.
5. Inspect whether a drawer-like edge or line appears ON the gripper, is truncated at
   a solid boundary, or changes depth ordering. Separate true persistent geometry
   artifacts from reflections, transparency, compression, and normal occlusion.
6. Explicitly state the strongest evidence AGAINST and FOR a solid-wall penetration.
   If the geometry is ambiguous, use UNCLEAR and assign moderate or low score.

Assign a CONTINUOUS 0..1 VISUAL INCONSISTENCY EVIDENCE score:
0.00: clear normal motion; 0.10-0.30: weak specific concern or likely ordinary occlusion;
0.40-0.60: visually specific suspicious wall-relative change but ambiguity remains;
0.70-0.90: strong observable solid-wall inconsistency without supported legal passage;
1.00: exceptional direct visual evidence. Generic doubt alone should NOT produce 0.5+.
Do NOT output score 0 just because the gripper is nearly stationary; do NOT output
high score solely because the drawer moves or gripper is behind transparent plastic.
The score is NOT a probability, proof of 3D collision, or a ground-truth classification.

Return EXACTLY one JSON object; no Markdown:
{{
"gripper_motion":"STATIONARY|LATERAL|VERTICAL|MIXED|UNCLEAR",
"drawer_motion":"STATIONARY|SLIDING_LATERAL|MIXED|UNCLEAR",
"gripper_position_early":"INSIDE|OUTSIDE|ABOVE_OPENING|STRADDLING|UNCLEAR",
"gripper_position_late":"INSIDE|OUTSIDE|ABOVE_OPENING|STRADDLING|UNCLEAR",
"relative_side_change":"YES|NO|UNCLEAR",
"solid_wall_sweeps_past_gripper":"YES|NO|UNCLEAR",
"gripper_clears_rim":"YES|NO|UNCLEAR",
"visible_legal_path":"YES|NO|UNCLEAR",
"penetration_visual_evidence":"STRONG|MODERATE|WEAK|NONE|UNCLEAR",
"boundary_mark_artifact":"YES|NO|UNCLEAR",
"early_middle_late_observations":"early: ...; middle: ...; late: ...",
"evidence_for":"concrete observed visible evidence or none",
"evidence_against":"plausible ordinary geometry, transparent depth ambiguity, or legal passage",
"visual_inconsistency_score":0.0,
"reason":"2-4 sentences describing the moving wall relative to gripper and limitations"
}}'''


def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def signature(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()[:16]

def parse_json(raw):
    dec=json.JSONDecoder()
    for m in re.finditer(r'\{',str(raw)):
        try:
            val,_=dec.raw_decode(str(raw)[m.start():])
            if isinstance(val,dict) and 'visual_inconsistency_score' in val:return val
        except ValueError:pass
    raise ValueError('No complete JSON object with visual_inconsistency_score; raw saved')

def check_verdict(o):
    v=o.get('visual_inconsistency_score')
    if isinstance(v,bool): raise ValueError('Boolean score')
    try: score=float(v)
    except (TypeError,ValueError):raise ValueError('Missing numeric visual_inconsistency_score')
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Score outside 0..1')
    o['visual_inconsistency_score']=round(score,4)
    for fld in ('drawer_motion','gripper_motion','visible_legal_path','solid_wall_sweeps_past_gripper'):
        if fld not in o:raise ValueError('Missing field: '+fld)
    return o

def frames_parse(s):
    f=[int(t.strip()) for t in s.split(',')]
    if len(f)!=5 or f!=sorted(set(f)) or min(f)<0:raise ValueError('Need EXACTLY five ascending distinct frames')
    return f

def from_source(path,n,topk,exclusion):
    d=json.loads(Path(path).read_text());rows=d.get('rows',[])
    if d.get('start') not in (0,1) or int(d.get('end',-1))!=n-1:
        raise ValueError('Auto selection requires a FULL VIDEO Qwen scan: start 0/1 and end n-1')
    covered={int(r['frame']) for r in rows if r.get('status')=='ok'}
    if covered!=set(range(1,n)):
        raise ValueError(f'Incomplete Qwen cache: {len(covered)} / {n-1} pairs')
    candidates=[]
    for r in rows:
        v=r.get('verdict') or {}
        raw=float(v.get('raw_transition_signal',v.get('signal',0)))
        if raw>0:candidates.append((raw,int(r['frame'])))
    candidates.sort(key=lambda p:(-p[0],p[1]))
    kept=[]
    for score,center in candidates:
        if any(abs(center-q['center'])<=exclusion for q in kept):continue
        frames=window(center,n)
        if any(tuple(frames)==tuple(q['frames']) for q in kept):continue
        kept.append({'center':center,'frames':frames,'qwen_raw':score})
        if len(kept)>=topk:break
    return kept,len(candidates)

def window(center,n):
    # Cover 20 source frames around pair-based proposal; five ordered samples.
    # Endpoint shifts at video boundary; never insert human-selected frames.
    if n<5:raise ValueError('Video too short')
    first=max(0,min(n-21,center-10)) if n>=21 else 0
    last=min(n-1,first+20)
    return [int(round(first+(last-first)*i/4)) for i in range(5)]

def get_video(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError(f'Cannot open {path}')
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
    return n,w,h

def images_for(video,frames,roi,full_side,crop_side):
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,frames[0]);idxset=set(frames);images={}
    try:
        for t in range(frames[0],frames[-1]+1):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'Decode failed f{t}')
            if t in idxset:images[t]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    if len(images)!=5:raise RuntimeError('Missing frame(s)')
    w,h=images[frames[0]].size;x1,y1,x2,y2=roi
    rect=(round(x1*w),round(y1*h),round(x2*w),round(y2*h))
    if rect[2]-rect[0]<50 or rect[3]-rect[1]<50:raise ValueError('ROI too small')
    full=images[frames[2]].copy();full.thumbnail((full_side,full_side))
    result=[with_title(full,f'FULL CONTEXT f{frames[2]}')]
    for i,t in enumerate(frames):
        crop=images[t].crop(rect);scale=crop_side/max(crop.size)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
        result.append(with_title(crop,f'FRAME {t}  ({i+1}/5 chronological)'))
    return result,rect

def with_title(img,label):
    panel=Image.new('RGB',(img.width,img.height+28),'white');panel.paste(img,(0,28));ImageDraw.Draw(panel).text((8,8),label,fill='black');return panel

def contact_sheet(images):
    cw,ch=485,405;out=Image.new('RGB',(cw*3,ch*2),'#ededed')
    for i,im in enumerate(images):
        tile=im.copy();tile.thumbnail((cw-12,ch-12));out.paste(tile,((i%3)*cw+6,(i//3)*ch+6))
    return out

def roi_parse(t):
    a=tuple(map(float,t.split(',')))
    if len(a)!=4 or not all(math.isfinite(x) for x in a) or not (0<=a[0]<a[2]<=1 and 0<=a[1]<a[3]<=1):
        raise ValueError('Bad --roi-norm')
    return a

def qwen(images,prompt,model,max_tokens):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Qwen requires allocated GPU node')
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
    c=VLMClient('transparent_drawer_verifier',{'backend':'local_gpu','model':str(model),'max_tokens':max_tokens})
    print('QWEN loaded:',getattr(c,'architecture','unknown'),flush=True)
    def infer():
        answer=c.ask(images,prompt,system_prompt=SYSTEM)
        return str(answer.get('answer',''))
    return infer

def cloud_auth(envfile):
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
    from robot.preprocessing.link7_persistent.interface.config import role_config
    load_env_file(Path(envfile));cfg=role_config('vlm2')
    key=os.environ.get(cfg.get('api_key_env','VLM2_API_KEY'))
    if not key:raise RuntimeError('Missing Wilson VLM2 credentials; no secrets printed')
    return cfg['api_base'].rstrip('/'),key

def cloud(images,prompt,base,key,model,max_tokens,read_timeout):
    import requests
    content=[{'type':'text','text':prompt}];sizes=[]
    for im in images:
        buf=io.BytesIO();im.save(buf,'JPEG',quality=80,optimize=True);raw=buf.getvalue()
        sizes.append(round(len(raw)/1024,1));content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode('ascii')}})
    print('GEMINI sending 6 images KB:',sizes,'read_timeout',read_timeout,flush=True)
    payload={'model':model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':content}],
             'temperature':0,'max_tokens':max_tokens}
    # Use long CONNECT timeout to tolerate unreliable TLS upload, 0 retries.
    resp=requests.post(base+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload,
                       timeout=(60,read_timeout));resp.raise_for_status()
    data=resp.json();choice=data['choices'][0];msg=choice['message']['content']
    if isinstance(msg,list):msg='\n'.join(z.get('text','') for z in msg if isinstance(z,dict))
    return str(msg),{'finish_reason':choice.get('finish_reason'),'usage':data.get('usage'),'reported_model':data.get('model')}

def do_run(args):
    n,w,h=get_video(args.video);roi=roi_parse(args.roi_norm)
    if args.frames and args.source:raise ValueError('Choose EITHER --frames or --source, not both')
    if args.source:
        selected,count=from_source(args.source,n,args.topk,args.nms_radius)
        print('AUTO: positive Qwen proposals',count,'selected',len(selected),flush=True)
        if not selected:
            print('NO POSITIVE QWEN PROPOSALS: candidate recall may be 0. No verifier call.');return
        method='auto_full_video_qwen_candidates'
    else:
        if not args.frames:raise ValueError('Need --frames or --source')
        selected=[{'center':None,'frames':frames_parse(args.frames),'qwen_raw':None}]
        method='manual_window_DIAGNOSTIC_not_full_video'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.backend=='gemini' and not args.preview_only:base,key=cloud_auth(args.env_file)
    for k,sel in enumerate(selected):
        frames=sel['frames'];assert 0<=frames[0]<frames[-1]<n
        tag='f'+str(frames[0])+'_'+str(frames[-1])
        wd=out/tag;wd.mkdir(parents=True,exist_ok=True)
        prompt=PROMPT.format(frames=', '.join('f'+str(t) for t in frames))
        config={'version':VERSION,'method':method,'backend':args.backend,'video':str(Path(args.video).resolve()),
                'frames':frames,'roi_norm':roi,'full_side':args.full_side,'crop_side':args.crop_side,
                'model':str(args.model if args.backend=='qwen' else args.gemini_model),
                'max_tokens':args.max_tokens,'prompt_hash':signature(PROMPT),'qwen_center':sel['center']}
        digest=signature(config);target=wd/f'{args.backend}_{digest}.json';rawpath=wd/f'{args.backend}_{digest}.txt'
        if target.exists():
            old=json.loads(target.read_text())
            if old.get('status')=='ok':print('CACHED:',target,'score',old['verdict']['visual_inconsistency_score']);continue
        images,rect=images_for(args.video,frames,roi,args.full_side,args.crop_side)
        contact_sheet(images).save(wd/'preview.jpg',quality=91)
        (wd/'prompt.txt').write_text(SYSTEM+'\n\n'+prompt)
        write_json(wd/'window.json',{'frames':frames,'method':method,'original_qwen_center':sel['center'],
                                    'original_qwen_raw':sel['qwen_raw'],'roi_pixels':rect,
                                    'note':'Qwen candidate metadata never sent to verifier'})
        print(f'[{k+1}/{len(selected)}] {args.backend} source frames {frames} ROI {rect} preview {wd/"preview.jpg"}',flush=True)
        if args.preview_only:continue
        t0=time.monotonic();metadata={};raw=''
        try:
            if args.backend=='qwen':
                fn=qwen_engine;raw=fn(images,prompt)
            else:raw,metadata=cloud(images,prompt,base,key,args.gemini_model,args.max_tokens,args.timeout)
            rawpath.write_text(raw) # persist raw BEFORE parsing (no costly repeat on parse error)
            verdict=check_verdict(parse_json(raw));status='ok'
            print('  SCORE',verdict['visual_inconsistency_score'],'wall_sweep',verdict['solid_wall_sweeps_past_gripper'],
                  'legal',verdict['visible_legal_path'],'elapsed',round(time.monotonic()-t0,1),flush=True)
        except Exception as exc:
            status='error';verdict=None;metadata['error']=f'{type(exc).__name__}: {exc}'
            print('  ERROR',metadata['error'][:400],flush=True)
            if raw and not rawpath.exists():rawpath.write_text(raw)
        write_json(target,{'status':status,'config':config,'verdict':verdict,'metadata':metadata,
                           'raw_path':str(rawpath) if rawpath.exists() else None,
                           'elapsed_seconds':round(time.monotonic()-t0,2)})
    report(out)

def report(out):
    out=Path(out);rows=[]
    for p in sorted(out.glob('f*/[qg]*_*.json')):
        try:d=json.loads(p.read_text())
        except (ValueError,OSError):continue
        if d.get('status')!='ok' or 'verdict' not in d:continue
        v=d['verdict'];cfg=d['config']
        rows.append({'window':p.parent.name,'backend':cfg['backend'],'frames':','.join(map(str,cfg['frames'])),
                     'score':v['visual_inconsistency_score'],'wall_sweep':v.get('solid_wall_sweeps_past_gripper'),
                     'legal_path':v.get('visible_legal_path'),'relative_change':v.get('relative_side_change'),
                     'reason':v.get('reason',''),'result_file':str(p)})
    if rows:
        with (out/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('SUMMARY:',out/'comparison.csv')
        for r in rows:print(' ',r['window'],r['backend'],'score',r['score'],'wall',r['wall_sweep'],'legal',r['legal_path'])
    else:print('No complete parsed verifier scores yet.')

def selftest():
    assert frames_parse('148,150,152,154,156')==[148,150,152,154,156]
    assert window(154,189)==[144,149,154,159,164]
    o=parse_json('```json\n{"visual_inconsistency_score":0.45,"drawer_motion":"SLIDING_LATERAL","gripper_motion":"STATIONARY","visible_legal_path":"UNCLEAR","solid_wall_sweeps_past_gripper":"UNCLEAR"}\n```')
    assert check_verdict(o)['visual_inconsistency_score']==.45
    assert 'DRAWER WALL passing across a gripper' in PROMPT
    print('SELF TEST PASSED')

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    for a in ('preview','run'):
        p=sub.add_parser(a)
        p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        sel=p.add_mutually_exclusive_group(required=True)
        sel.add_argument('--frames',help='Five manual indices: DIAGNOSTIC ONLY')
        sel.add_argument('--source',type=Path,help='FULL Qwen scan JSON for automatic proposals')
        p.add_argument('--backend',choices=['qwen','gemini'],default='qwen')
        p.add_argument('--topk',type=int,default=5);p.add_argument('--nms-radius',type=int,default=10)
        p.add_argument('--roi-norm',default='0.28,0.26,0.76,0.90')
        p.add_argument('--full-side',type=int,default=900);p.add_argument('--crop-side',type=int,default=800)
        p.add_argument('--model',type=Path,default=MODEL);p.add_argument('--gemini-model',default='gemini-3.8-flash')
        p.add_argument('--max-tokens',type=int,default=1800);p.add_argument('--timeout',type=float,default=240)
        p.add_argument('--env-file',type=Path,default=WS/'.env.vlm')
        p.set_defaults(preview_only=(a=='preview'))
    sub.add_parser('self-test')
    r=sub.add_parser('report');r.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.action=='self-test':selftest();return
    if args.action=='report':report(args.output);return
    if args.max_tokens<256 or args.topk<1 or args.nms_radius<0:raise ValueError('Bad tokens/topk/nms')
    global qwen_engine
    if args.action=='run' and args.backend=='qwen':
        # Lazy once-only initialization after preview and parsing available frames;
        # use one Qwen model instance across all candidate windows.
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Local Qwen inference needs GPU allocation')
        sys.path.insert(0,str(WS))
        from robot.preprocessing.link7_persistent.vlm_client import VLMClient
        client=VLMClient('transparent_drawer_verifier',{'backend':'local_gpu','model':str(args.model),'max_tokens':args.max_tokens})
        print('LOCAL QWEN:',getattr(client,'architecture','unknown'),flush=True)
        def qwen_engine(images,prompt):
            return str(client.ask(images,prompt,system_prompt=SYSTEM).get('answer',''))
    do_run(args)

if __name__=='__main__':main()
