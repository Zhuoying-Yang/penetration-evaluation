#!/usr/bin/env python3
"""Frozen, label-blind Type-C Qwen verification from *completed* V3-C proposals.

Stage 1 (external): Agent0 (5 uniform frames) -> V3 C fixed ref-frame scan.
Stage 2 (this script): 8 objective proposal timestamps -> one Qwen MULTIFRAME
visual audit per timestamp. Outputs separate solid-panel and object-shortening
signals. No Gemini, no ground-truth labels at inference, no hand-marked frames.

Use --dry-run before loading the model. C_queries.json must be complete.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
from pathlib import Path

ROOT=Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
WS=Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
MODEL='/PHShome/zy992/Wilson/dependency/models/qwen_model'
VERSION='typeC_qwen_temporal_verify_v1_20261009'

PROMPT='''You are an INDEPENDENT video-frame verifier for generated robot manipulation videos.
You see {n} ORIGINAL SOURCE FRAMES, labeled with frame indices in chronological order:
{indices}. They include a reference and context from BEFORE and AFTER a Qwen
candidate frame {t}. You must use the actual visual frames, not any upstream claim.

Track ONLY manipulated actor: {actor}.
Nearby container/environment: {target}.
Potential SOLID boundary: {boundary}.
These are identity hints only; they DO NOT imply a physical violation.

Analyze TWO SEPARATE error types. Be particularly careful:
A. SOLID-PANEL INTERSECTION: Does a continuous visible part of the SAME actor
appear to pass THROUGH or become embedded in a SOLID panel, rather than legally
extend through a container/drawer OPENING? Mere partial inside/outside, 2D
silhouette overlap, occlusion by a foreground drawer edge, or a gripper over
that edge is NOT positive evidence. To say YES, locate the apparent interface
at the SOLID panel, not in the opening.
B. OBJECT-SHAPE LOSS: Does the same actor lose a substantial physical segment,
become abnormally truncated, or abruptly become shorter through time, in a way
NOT explained by perspective, bending/rotation, ordinary occlusion by a panel,
normal passage through an opening, gripper coverage, or camera motion? It is
NOT sufficient that the projected visible banana length becomes shorter when
part enters a drawer. Need genuinely unexplained geometry change.

Look at the consecutive local context frames to distinguish stable true cues
from single-frame hallucinations. If evidence is ambiguous, answer UNCLEAR.
Do not infer real 3D contact from simple pixel overlap. Never use the object's
identity, dataset name, or assumed GT to guess an abnormality.

Return ONLY valid JSON containing ALL these keys:
{{"actor_identified":"YES|NO|UNCLEAR",
"solid_boundary_identified":"YES|NO|UNCLEAR",
"outside_segment_visible":"YES|NO|UNCLEAR",
"solid_material_intersection":"YES|NO|UNCLEAR",
"legal_opening_explains":"YES|NO|UNCLEAR",
"normal_occlusion_or_pose_explains":"YES|NO|UNCLEAR",
"substantial_shape_loss":"YES|NO|UNCLEAR",
"shape_loss_unexplained":"YES|NO|UNCLEAR",
"persistent_across_context":"YES|NO|UNCLEAR",
"assessment":"SOLID_EMBEDDING|ABNORMAL_SHAPE_LOSS|BOTH|LEGAL_OPENING|ORDINARY_OCCLUSION|NO_EVENT|UNCERTAIN",
"reason":"short frame-number-specific visible evidence, including the interface with SOLID material vs opening and what explains any apparent shortening"}}
No confidence number needed: deterministic scoring is applied externally.
'''

YES={'YES','NO','UNCLEAR'}
CATS={'SOLID_EMBEDDING','ABNORMAL_SHAPE_LOSS','BOTH','LEGAL_OPENING','ORDINARY_OCCLUSION','NO_EVENT','UNCERTAIN'}
FLAGS=('actor_identified','solid_boundary_identified','outside_segment_visible',
       'solid_material_intersection','legal_opening_explains',
       'normal_occlusion_or_pose_explains','substantial_shape_loss',
       'shape_loss_unexplained','persistent_across_context')

def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def save(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
def load(path):return json.loads(Path(path).read_text())
def safe(s):return ' '.join(re.sub(r'[^\w\s.,()/\'-]',' ',str(s)).split())[:100]

def select_frames(times,budget):
    """Same label-blind representative sampler as prior Gemini C audit."""
    times=sorted(set(int(x) for x in times));runs=[]
    for t in times:
        if not runs or t>runs[-1][-1]+1:runs.append([t])
        else:runs[-1].append(t)
    runs.sort(key=lambda r:(-len(r),r[0]))
    result=[]
    def add(t):
        if t not in result and len(result)<budget:result.append(t)
    for run in runs[:max(1,budget//2)]:add(run[len(run)//2])
    if runs:add(runs[0][0]);add(runs[0][-1])
    while len(result)<min(budget,len(times)):
        t=max((t for t in times if t not in result),
              key=lambda t:(min(abs(t-x) for x in result) if result else 10**8,-t))
        add(t)
    return sorted(result)

def context_indices(ref,t,n):
    """Chronological fixed context; six images, including intact ref and future frames."""
    return sorted(set([ref,max(0,t-8),max(0,t-4),t,
                       min(n-1,t+4),min(n-1,t+8)]))

def get_json(raw):
    raw=re.sub(r'<think>.*?</think>','',str(raw),flags=re.S)
    dec=json.JSONDecoder()
    for i,c in enumerate(raw):
        if c=='{':
            try:
                d,_=dec.raw_decode(raw[i:])
                if isinstance(d,dict):return d
            except (ValueError,json.JSONDecodeError):pass
    raise ValueError('No JSON object returned: '+raw[:250])

def normalized(d):
    for k in FLAGS:
        if k not in d:raise ValueError('Missing required Qwen output flag '+k)
        val=str(d[k]).strip().upper()
        if val not in YES:raise ValueError(f'Invalid {k}: {d[k]}')
        d[k]=val
    category=str(d.get('assessment','')).strip().upper()
    if category not in CATS:raise ValueError('Invalid assessment: '+category)
    d['assessment']=category
    d['reason']=str(d.get('reason',''))[:1000]
    if len(d['reason'])<10:raise ValueError('Missing reason')
    return d

def scores(d):
    """Deterministic, conservative, independent of the model's numeric confidence.

    Only strong visible evidence gets >= 0.75. Normal occlusion/opening
    explicitly vetoes both metrics. Shape loss is reported separately from
    solid-panel intrusion to avoid conflating persistency with penetration.
    """
    yes=lambda k:d[k]=='YES'
    no=lambda k:d[k]=='NO'
    if not yes('actor_identified'):return dict(solid=0.0,shape=0.0,combined=0.0,reason='actor_unverified')
    if d['assessment'] in ('LEGAL_OPENING','ORDINARY_OCCLUSION','NO_EVENT'):
        return dict(solid=0.0,shape=0.0,combined=0.0,reason='model_reports_legal_or_normal')
    # For solid entry, opening and occlusion are independent vetoes.
    solid=0.0
    if yes('solid_boundary_identified') and yes('outside_segment_visible'):
        if d['assessment'] in ('SOLID_EMBEDDING','BOTH') and yes('solid_material_intersection') and no('legal_opening_explains') and no('normal_occlusion_or_pose_explains'):
            solid=.95 if yes('persistent_across_context') else .75
        elif d['solid_material_intersection']=='UNCLEAR' and no('legal_opening_explains') and no('normal_occlusion_or_pose_explains'):
            solid=.35
    shape=0.0
    if d['assessment'] in ('ABNORMAL_SHAPE_LOSS','BOTH') and yes('substantial_shape_loss') and yes('shape_loss_unexplained') and no('normal_occlusion_or_pose_explains') and no('legal_opening_explains'):
        shape=.90 if yes('persistent_across_context') else .75
    elif d['substantial_shape_loss']=='UNCLEAR' and d['shape_loss_unexplained']=='UNCLEAR' and no('normal_occlusion_or_pose_explains'):
        shape=.25
    return dict(solid=solid,shape=shape,combined=max(solid,shape),
                reason=('supported' if max(solid,shape)>=.75 else 'unverified_or_explained'))

def prepare(qdir,a0file,video,case,threshold,max_audits):
    qdir=Path(qdir);video=Path(video)
    cfg=load(qdir/'run_config.json')
    if cfg.get('case')!=case or Path(cfg['video']).resolve()!=video.resolve():
        raise ValueError('case/video not same as cached V3 C run')
    n=int(cfg.get('end',0))+1 # may be subset, actual full n verified separately
    import cv2
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot read video: '+str(video))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if int(cfg.get('end',-1))!=n-1 or int(cfg.get('start',-1))!=1:
        raise ValueError(f'Need FULL stage-1 scan 1..{n-1}, got {cfg.get("start")}-{cfg.get("end")}')
    data=load(qdir/'C_queries.json')
    rows=data.get('rows',[])
    ok={int(r['frame']):r for r in rows if r.get('status')=='ok'}
    required=set(range(1,n))-{int(cfg['reference'])}
    if set(ok)!=required:
        raise ValueError(f'Incomplete C_queries: got {len(ok)}/{len(required)}, missing {sorted(required-set(ok))[:8]}')
    a0=load(a0file)
    pair=a0.get('selected')
    if pair is None or cfg.get('agent0')!=pair:raise ValueError('Agent0 selected pair is missing/mismatched')
    qualifying=[t for t,r in ok.items() if float(r['verdict'].get('signal',0))>=threshold]
    return cfg,pair,n,qualifying,select_frames(qualifying,max_audits)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--video',required=True,type=Path)
    p.add_argument('--qwen-dir',required=True,type=Path)
    p.add_argument('--agent0-json',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-audits',type=int,default=8)
    p.add_argument('--min-qwen-signal',type=float,default=.75)
    p.add_argument('--image-side',type=int,default=896)
    p.add_argument('--max-tokens',type=int,default=750)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not 1<=a.max_audits<=20:p.error('--max-audits must be 1..20')
    if not 0<a.min_qwen_signal<=1:p.error('--min-qwen-signal must be 0..1')
    cfg,pair,n,qualifying,selected=prepare(a.qwen_dir,a.agent0_json,a.video,a.case,a.min_qwen_signal,a.max_audits)
    print('CASE:',a.case,'frame_count:',n,'Qwen Stage1 positive frames:',len(qualifying),'selected:',selected,flush=True)
    print('MONITORED:',json.dumps(pair,ensure_ascii=False),flush=True)
    if a.dry_run:return
    out=a.output/a.case
    out.mkdir(parents=True,exist_ok=True)
    config={'protocol':VERSION,'case':a.case,'video':str(a.video.resolve()),
            'qwen_dir':str(a.qwen_dir.resolve()),'stage1_config_sha256':sha(json.dumps(cfg,sort_keys=True)),
            'agent0':pair,'selected':selected,'max_audits':a.max_audits,
            'min_qwen_signal':a.min_qwen_signal,'model':MODEL,
            'image_side':a.image_side,'max_tokens':a.max_tokens,'prompt_sha256':sha(PROMPT)}
    cp=out/'run_config.json'
    if cp.is_file() and load(cp)!=config:raise RuntimeError('Verifier config changed, choose a NEW --output directory')
    save(cp,config)
    if not selected:
        save(out/'summary.json',{'protocol':VERSION,'case':a.case,'status':'complete',
             'qwen_candidate_count':0,'audits':0,'solid_score':0,'shape_score':0,
             'combined_score':0,'selected_frames':[],'note':'No Stage1 candidates above threshold'})
        print('FINAL:',a.case,'solid=0.0 shape=0.0 combined=0.0 (no Stage1 candidates)',flush=True)
        return
    # Save and reuse individual successful audits; any failed record counts as incomplete.
    import torch
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU: run on allocated H200')
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
    from penetration_agent0_v3C_multiframe import load_v3
    v3=load_v3(ROOT/'penetration_abc_agents_v3.py')
    client=VLMClient('penetration_typeC_qwen_temporal_verify',{
        'backend':'local_gpu','model':MODEL,'max_tokens':a.max_tokens})
    frames=v3.Frames(a.video)
    try:
        for i,t in enumerate(selected,1):
            record_path=out/'audits'/f'f{t:04d}.json'
            if record_path.is_file() and load(record_path).get('status')=='ok':
                r=load(record_path);print('CACHED',i,len(selected),t,r['scores'],flush=True);continue
            idx=context_indices(int(cfg['reference']),t,n)
            prompt=PROMPT.format(n=len(idx),indices=', '.join(map(str,idx)),t=t,
                   actor=safe(pair['actor']),target=safe(pair['target']),boundary=safe(pair['solid_boundary']))
            images=[v3.label(frames.frame(j),j,a.image_side) for j in idx]
            rec={'case':a.case,'frame':t,'source_frames':idx,'stage1_score':float(
                next(r['verdict']['signal'] for r in load(a.qwen_dir/'C_queries.json')['rows'] if r.get('frame')==t)),
                'status':'failed'}
            try:
                ans=client.ask(images,prompt,
                    system_prompt='Assess visually supported motion and object continuity. Avoid presumed labels; JSON only.')
                rec['raw_answer']=str(ans.get('answer',''))
                rec['verdict']=normalized(get_json(rec['raw_answer']))
                rec['scores']=scores(rec['verdict'])
                rec['status']='ok'
                # Four-or-six-frame contact-sheet for human audit, not used in inference.
                from PIL import Image,ImageDraw
                w=max(im.width for im in images);h=max(im.height for im in images)
                cols=3;rows=(len(images)+cols-1)//cols
                sheet=Image.new('RGB',(cols*w,rows*h),(245,245,245))
                for j,im in enumerate(images):sheet.paste(im,((j%cols)*w,(j//cols)*h))
                preview=out/'previews'/f'f{t:04d}.jpg';preview.parent.mkdir(parents=True,exist_ok=True)
                sheet.save(preview,quality=85)
                rec['preview']=str(preview)
                print(f'VERIFY {i}/{len(selected)} f{t:04d} solid={rec["scores"]["solid"]:.2f} shape={rec["scores"]["shape"]:.2f} '
                      f'assessment={rec["verdict"]["assessment"]} reason={rec["verdict"]["reason"][:120]}',flush=True)
            except Exception as e:
                rec['error']=str(e)
                print(f'VERIFY ERROR f{t:04d}: {e}',flush=True)
            save(record_path,rec)
        records=[load(out/'audits'/f'f{t:04d}.json') for t in selected]
        successes=[r for r in records if r.get('status')=='ok']
        status='complete' if len(successes)==len(selected) else 'incomplete'
        sm={'protocol':VERSION,'case':a.case,'status':status,'qwen_candidate_count':len(qualifying),
            'selected_frames':selected,'audits':len(successes),
            'solid_score':max([r['scores']['solid'] for r in successes],default=0.0),
            'shape_score':max([r['scores']['shape'] for r in successes],default=0.0),
            'combined_score':max([r['scores']['combined'] for r in successes],default=0.0),
            'failed_frames':[r['frame'] for r in records if r.get('status')!='ok'],
            'notes':'Separate uncalibrated evidence metrics; combined includes shape persistency (not penetration-specific).'}
        save(out/'summary.json',sm)
        print('FINAL:',a.case,json.dumps(sm,ensure_ascii=False),flush=True)
        if status!='complete':sys.exit(2)
    finally:frames.close()

if __name__=='__main__':main()
