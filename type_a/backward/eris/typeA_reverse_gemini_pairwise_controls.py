#!/usr/bin/env python3
"""Reverse Type-A Gemini diagnostic: EXACTLY two adjacent source frames per request.

Preserves GitHub's image preparation (FULL before/after + SAME-ROI CROP
before/after as FOUR separate images in ONE API request), system message,
Gemini model, score parser. Only the user prompt changes to inside -> outside.

Use on ERIS CPU compute node. Qwen cache is read only. Focus frame numbers are
Qwen AFTER-frame indices; negative controls come from Qwen-zero windows.
Development diagnostic only. Never mix with full-video blinded AUROC data.
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, math, os, random, sys, time
from pathlib import Path

ROOT = Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
VIDEO = Path('/scratch/z/zy992/zhuoying/physact/cosmos3/export_robowm68_3seeds/seed103/0053.mp4')
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
DEFAULT_SOURCE = ROOT/'typeA_reverse_v3_probe_seed103_0053_f135_188'/'Cosmos3_seed103_0053'/'A_queries.json'

# Independent judgment: never tell Gemini the model's score, label, or selection group.
# Exactly two ORIGINAL consecutive source frames; no hand-selected timeline description.
REVERSE_PROMPT = '''Four images are provided in this exact order:
1. FULL BEFORE, original source frame {before}
2. FULL AFTER, original source frame {after} (immediately consecutive)
3. CROP BEFORE of frame {before}
4. CROP AFTER of frame {after} (EXACT SAME spatial crop)

MOVING TARGET: the robot gripper/end-effector, especially the same physical black jaws and fingertips, NOT the yellow cube.
ENVIRONMENT: the drawer-like container's SOLID FRONT PANEL and solid side walls. Its genuine top/front OPENING is not solid material.

You are independently checking a possible REVERSE Type-A error: INSIDE/BEHIND the solid drawer front panel -> OUTSIDE/IN FRONT of it THROUGH solid material.
This is a question, NOT a claim that a crossing occurred. Evaluate only these TWO supplied frames. Do not infer unseen earlier or later frames.

First locate the SAME gripper tips and the SAME solid panel in each frame. Describe their observed relative positions separately (INSIDE/BEHIND, OUTSIDE/IN FRONT, AT_OPENING/ABOVE_RIM, OCCLUDED, or UNCLEAR). Next compare the precise before-to-after change.

Potential violation: the SAME gripper emerges from the drawer cavity to the exterior through the visible solid front panel/side wall, with no physically plausible route through or around a real opening. A legal exit over the rim or through an opening is NOT penetration. A 2D overlap, transparency, unchanged pose, uncertain depth, or ordinary occlusion is NOT proof. Do not assume an opening route occurred merely because an opening exists elsewhere; require observable support for that route.

Important: One adjacent-frame pair can show only a small part of a transition. Do not fabricate large motion, but do not automatically reject a localized solid-panel crossing just because its pixel displacement is small. Say UNCLEAR when the two frames do not establish which side the gripper occupies. If both frames show the same state, say NO for observed side change.

Give a continuous visual-inconsistency score in [0,1], NOT a calibrated probability:
0.00 = no observed abnormality; 0.10-0.30 = weak or plausibly normal;
0.40-0.60 = substantial unresolved evidence; 0.70-0.90 = strong, grounded solid-wall exit evidence;
0.95-1.00 = exceptionally clear impossible material crossing.
A high score MUST be supported by specific, contrasting visible evidence for the same gripper and the actual solid panel in these two images. Low scores are appropriate when the evidence does not support the hypothesis.

Return exactly one valid JSON object with these keys:
{{
  "target_identified": "YES|NO|UNCLEAR",
  "solid_panel_identified": "YES|NO|UNCLEAR",
  "gripper_state_before": "INSIDE|OUTSIDE|AT_OPENING|OCCLUDED|UNCLEAR",
  "gripper_state_after": "INSIDE|OUTSIDE|AT_OPENING|OCCLUDED|UNCLEAR",
  "gripper_tip_before": "concrete visual observation",
  "gripper_tip_after": "concrete visual observation",
  "inside_to_outside_change": "YES|NO|UNCLEAR",
  "crosses_solid_panel": "YES|NO|UNCLEAR",
  "legal_opening_route_supported": "YES|NO|UNCLEAR",
  "ordinary_occlusion_or_perspective_possible": "YES|NO|UNCLEAR",
  "visual_inconsistency_score": 0.0,
  "reason": "2-3 sentences comparing exactly frames {before} and {after} and explaining the score"
}}
'''

def dump(path:Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def rows_from_qwen(path):
    doc=json.loads(path.read_text())
    out={}
    for r in doc.get('rows',[]):
        if r.get('status')!='ok':continue
        t=int(r['frame']); b=int(r.get('prev',t-1))
        if b!=t-1:raise ValueError(f'Non-adjacent Qwen frame: {b}->{t}')
        v=r.get('verdict') or {}
        raw=float(v.get('raw_transition_signal',v.get('signal',0)))
        if not math.isfinite(raw) or not 0<=raw<=1:raise ValueError(f'Bad Qwen raw f{t}: {raw}')
        out[t]={'before':b,'after':t,'qwen_raw':raw,'qwen_post':float(v.get('signal',0)),
                'qwen_conflict':bool(v.get('shared_frame_conflict',False))}
    if not out:raise ValueError(f'No completed Qwen frames in {path}')
    return out

def select_pairs(qwen, focus_ends, n_controls, seed, buffer):
    focus=[]
    for t in focus_ends:
        if t not in qwen:raise ValueError(f'Qwen source missing f{t}; available range {min(qwen)}..{max(qwen)}')
        focus.append({**qwen[t],'group':'focus'})
    # No annotated timing or class is sent to Gemini; controls chosen only by
    # Qwen zero score, distant from ANY Qwen-positive after-frame.
    active=[t for t,r in qwen.items() if r['qwen_raw']>0]
    eligible=[t for t,r in qwen.items() if r['qwen_raw']==0 and
              all(abs(t-u)>buffer for u in active) and t not in focus_ends]
    if len(eligible)<n_controls:
        raise ValueError(f'Only {len(eligible)} Qwen-zero controls outside ±{buffer} positive frames; '
                         f'reduce --controls or --control-buffer. Eligible={eligible}')
    controls=random.Random(seed).sample(sorted(eligible), n_controls)
    controls=[{**qwen[t],'group':'qwen_zero_control'} for t in controls]
    # Alternate focus/control so the first 2 API calls test both behaviors.
    # No group labels, timing hints, or Qwen predictions are ever in the prompt.
    chosen=[]
    for i in range(max(len(focus),len(controls))):
        if i<len(focus):chosen.append(focus[i])
        if i<len(controls):chosen.append(controls[i])
    return chosen,len(eligible)

def load_client_credentials(env_file):
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
    from robot.preprocessing.link7_persistent.interface.config import role_config
    load_env_file(env_file)
    conf=role_config('vlm2')
    if conf.get('backend')!='cloud_api':raise RuntimeError('Wilson vlm2 must use cloud_api')
    var=conf.get('api_key_env','VLM2_API_KEY')
    key=os.environ.get(var)
    if not key:raise RuntimeError(f'{var} not configured in {env_file}')
    return conf,key

def ask_gemini(images, prompt, system, key, base, model, max_tokens, timeout):
    import requests
    parts=[{'type':'text','text':prompt}]
    kb=[]
    for im in images:
        data=io.BytesIO()
        im.convert('RGB').save(data,format='JPEG',quality=85,optimize=True)
        jpg=data.getvalue();kb.append(len(jpg)//1024)
        parts.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(jpg).decode('ascii')}})
    payload={'model':model,'messages':[{'role':'system','content':system},
                {'role':'user','content':parts}], 'temperature':0,'max_tokens':max_tokens}
    url=base.rstrip('/')+'/chat/completions'
    t0=time.perf_counter()
    print(f'  API model={model} images=4 KB={kb} max_tokens={max_tokens} timeout={timeout}s (no retries)',flush=True)
    try:
        res=requests.post(url,json=payload,headers={'Authorization':'Bearer '+key},timeout=(10,timeout))
    except requests.exceptions.RequestException as e:
        print(f'  NETWORK FAILURE after {time.perf_counter()-t0:.2f}s: {type(e).__name__}: {e}',flush=True)
        raise
    sec=time.perf_counter()-t0
    print(f'  HTTP={res.status_code} elapsed={sec:.2f}s',flush=True)
    if not res.ok: raise RuntimeError(f'Provider HTTP {res.status_code}: {res.text[:600]}')
    msg=res.json()['choices'][0]['message']['content']
    raw=msg if isinstance(msg,str) else '\n'.join(m.get('text','') for m in msg if isinstance(m,dict))
    return raw,sec,res.json().get('usage'),kb

def test():
    q={t:{'before':t-1,'after':t,'qwen_raw':(1. if t in (154,155,156) else 0.),
          'qwen_post':0.25 if t in (154,155,156) else 0.,'qwen_conflict':t in (154,155,156)}
       for t in range(135,189)}
    chosen,n=select_pairs(q,[154,155,156],3,43,3)
    assert n>3 and len(chosen)==6
    assert sum(r['group']=='focus' for r in chosen)==3
    assert all(abs(r['after']-154)>3 for r in chosen if r['group']!='focus')
    assert all(r['after']-r['before']==1 for r in chosen)
    assert 'INSIDE/BEHIND' in REVERSE_PROMPT
    assert 'OUTSIDE/IN FRONT' in REVERSE_PROMPT
    assert 'frame 153' in REVERSE_PROMPT.format(before=153,after=154)
    print('SELF-TEST PASS (adjacent pairs, Qwen proposal, control selection, neutral reverse prompt)')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--video',type=Path,default=VIDEO)
    p.add_argument('--output',type=Path,default=ROOT/'reverse_pairwise_gemini_0053_v1')
    p.add_argument('--focus-ends',default='154,155,156',help='Qwen AFTER frames, e.g. 154 means ORIGINAL f153->f154')
    p.add_argument('--controls',type=int,default=3)
    p.add_argument('--control-buffer',type=int,default=3)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--roi-norm',default=None,help='Optional fixed normalized ROI x1,y1,x2,y2; default FULL-FRAME fallback exactly as previous diagnostic')
    p.add_argument('--model',default='gemini-3.8-flash')
    p.add_argument('--full-side',type=int,default=960)
    p.add_argument('--crop-side',type=int,default=1000)
    p.add_argument('--max-tokens',type=int,default=900)
    p.add_argument('--timeout',type=float,default=85.)
    p.add_argument('--max-calls',type=int,default=0,help='Only run first N selected pair requests (0=all); resume later with same --output and --max-calls 0')
    p.add_argument('--api-base',default=None)
    p.add_argument('--env-file',type=Path,default=WS/'.env.vlm')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:return test()
    if args.controls<0 or args.control_buffer<0 or args.timeout<=0 or args.max_calls<0: p.error('Invalid controls/buffer/timeout/max-calls')
    focus=[int(x.strip()) for x in args.focus_ends.split(',') if x.strip()]
    if not focus or len(set(focus))!=len(focus):p.error('Specify unique --focus-ends')
    q=rows_from_qwen(args.source)
    chosen,n=select_pairs(q,focus,args.controls,args.seed,args.control_buffer)
    print('QWEN SOURCE:',args.source,flush=True)
    print('VIDEO:',args.video,flush=True)
    print(f'VALID QWEN-ZERO CONTROL POOL: {n} (buffer={args.control_buffer})',flush=True)
    for c in chosen:
        print(f"  f{c['before']:04d}->f{c['after']:04d} {c['group']} "
              f"Qraw={c['qwen_raw']:.2f} Qpost={c['qwen_post']:.2f} conflict={c['qwen_conflict']}",flush=True)
    if args.dry_run:
        print('REQUESTS THIS RUN:',len(chosen) if args.max_calls==0 else min(len(chosen),args.max_calls))
        print('DRY-RUN COMPLETE: no API calls or output changes')
        return
    if not args.video.is_file():raise FileNotFoundError(args.video)
    sys.path.insert(0,str(args.root))
    import penetration_A_v3_crop_gemini_score as audit
    roi=audit.parse_roi(args.roi_norm) if args.roi_norm else (0,0,1,1)
    cfg={
        'version':'reverse-gemini-pairwise-controls-v1',
        'source':str(args.source.resolve()),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'video':str(args.video.resolve()),'video_size':args.video.stat().st_size,
        'pairs':chosen,'roi_norm':list(roi),'full_side':args.full_side,'crop_side':args.crop_side,
        'model':args.model,'max_tokens':args.max_tokens,'system':audit.SYSTEM,
        'prompt':REVERSE_PROMPT,'api_base':args.api_base,
        'scope':'diagnostic-only manually selected focus frames, Qwen-derived zero controls',
    }
    config_path=args.output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=cfg:
        raise RuntimeError('Run config differs; use new --output directory')
    dump(config_path,cfg)
    conf, key=load_client_credentials(args.env_file)
    base=args.api_base or conf.get('api_base')
    if not str(base).startswith('https://'):raise RuntimeError(f'Invalid API base: {base}')
    print('DIRECT HTTPS API:',base, '(key hidden)',flush=True)
    fr=audit.Frames(args.video)
    results=[]
    selected_for_run=chosen[:args.max_calls] if args.max_calls else chosen
    try:
        for i,c in enumerate(selected_for_run,1):
            b,t=c['before'],c['after']
            record=args.output/f'pair_f{b:04d}_f{t:04d}.json'
            if record.is_file():
                d=json.loads(record.read_text());print(f'[{i}/{len(selected_for_run)}] CACHED f{b}->{t}: {d["score"]:.3f}',flush=True)
            else:
                print(f'[{i}/{len(selected_for_run)}] Evaluating f{b}->{t} ...',flush=True)
                images, box=audit.prepared_images(fr,b,t,roi,args.full_side,args.crop_side)
                if len(images)!=4:raise RuntimeError('Expected four original full+crop images')
                imgfile=args.output/f'visual_f{b:04d}_f{t:04d}.jpg'
                imgfile.parent.mkdir(parents=True,exist_ok=True)
                audit.contact_sheet(images).save(imgfile,quality=88)
                raw,seconds,usage,kb=ask_gemini(images,REVERSE_PROMPT.format(before=b,after=t),
                    audit.SYSTEM,key,base,args.model,args.max_tokens,args.timeout)
                v=audit.parse_json_object(raw)
                s=audit.score_of(v)
                identity=str(v.get('target_identified','UNCLEAR')).upper()
                gated=s if identity=='YES' else 0.
                d={**c,'verdict':v,'raw_response':raw,'score':s,'gated_score':gated,
                   'latency_s':round(seconds,3),'usage':usage,'image_kb':kb,
                   'visual':str(imgfile),'crop_box_pixels':list(box)}
                dump(record,d)
                print(f'  SCORE={s:.3f} gated={gated:.3f} '
                      f'change={v.get("inside_to_outside_change")} solid={v.get("crosses_solid_panel")}',flush=True)
                print(f'  REASON: {v.get("reason","")}',flush=True)
            results.append(d)
            dump(args.output/'interim_summary.json',make_summary(results,chosen,n))
    finally:fr.close()
    summary=make_summary(results,chosen,n)
    dump(args.output/'diagnostic_summary.json',summary)
    with (args.output/'scores.csv').open('w',newline='') as f:
        cols=['before','after','group','qwen_raw','qwen_post','score','gated_score','latency_s','inside_to_outside_change','crosses_solid_panel','reason']
        wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader()
        for r in results:
            v=r.get('verdict') or {}
            wr.writerow({k:(r.get(k) if k not in ('inside_to_outside_change','crosses_solid_panel','reason') else v.get(k)) for k in cols})
    print('DONE; results:',args.output/'scores.csv',flush=True)
    for grp in ('focus','qwen_zero_control'):
        g=[x for x in results if x['group']==grp]
        print(grp, 'n=',len(g),'max=',max((x['gated_score'] for x in g),default=0.),
              'scores=',[x['gated_score'] for x in g],flush=True)

def make_summary(done,selected,n_controls):
    return {'diagnostic_only':True,'auroc_eligible':False,'control_pool':n_controls,
        'selection':[{'before':r['before'],'after':r['after'],'group':r['group'],
                      'qwen_raw':r['qwen_raw'],'qwen_post':r['qwen_post']} for r in selected],
        'finished':len(done),'total':len(selected),'complete':len(done)==len(selected),
        'focus_scores':[r['gated_score'] for r in done if r['group']=='focus'],
        'control_scores':[r['gated_score'] for r in done if r['group']=='qwen_zero_control'],
        'note':'Controls are Qwen-zero proposals, NOT verified negatives; the same reverse prompt and image policy were used for every pair.'}

if __name__=='__main__':main()
