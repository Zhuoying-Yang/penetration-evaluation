#!/usr/bin/env python3
"""Fixed-policy Type-A benchmark: Agent0 -> V3-A Qwen -> Gemini -> AUROC.

Run on an ERIS GPU allocation. No ground-truth labels or hand-marked event frames
are given to the VLM agents. GT is used ONLY by the offline report command.
Only appropriate for V3-A's open-container side-wall / rim geometry.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
MODEL = Path('/PHShome/zy992/Wilson/dependency/models/qwen_model')
MANIFEST_DEFAULT = Path(__file__).with_name('penetration_typeA_manifest.csv')
PROTOCOL = 'agent0_literal_v3A_post065_gemini_crop_v1_20261009'

# Fixed and label-blind. No cup/bin failure example or known anomalous frame.
AGENT0 = '''You inspect the FIRST frame of a robot manipulation video.
Original task instruction (if supplied): {task}
Identify up to three physical interaction pairs potentially relevant to a
SOLID CONTAINER WALL CROSSING evaluation. The primary pair should be the
moving manipulated object versus a container/environment solid boundary,
if such a relationship can be identified. A robot gripper is not necessarily
the manipulated object. Be faithful to the image and task; do not invent
entities not supported by them. A legitimate opening is different from a
solid wall. Do not decide whether any penetration occurs, and do not infer
one from the instructions. Do not use any later frames.
Return ONLY a JSON object, with pairs ordered from most relevant to least:
{{"pairs":[{{"actor":"short moving object name", "target":"short other object name",
"solid_boundary":"specific solid part of target", "pair_type":"OBJECT_ENVIRONMENT|ROBOT_OBJECT|OTHER",
"expected_interaction":"neutral description"}}]}}
If uncertain, use an empty pairs list rather than invent a target.'''


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_manifest(path):
    rows = list(csv.DictReader(Path(path).open(newline='')))
    if not rows: raise ValueError('Empty manifest')
    seen=set()
    for row in rows:
        name=row['case_id'].strip()
        if not name or name in seen: raise ValueError('Missing or duplicate case_id: ' + name)
        seen.add(name)
        if row['label'].strip() not in ('0', '1'): raise ValueError('Bad label: '+name)
        row['case_id']=name
        row['label']=int(row['label'])
        row['video']=str(Path(row['video']).expanduser())
        row['task']=row.get('task','').strip()
    return rows


def validate_manifest(manifest):
    rows=read_manifest(manifest)
    print('DATASET:', len(rows), 'videos:', sum(r['label']==1 for r in rows),
          'positives,', sum(r['label']==0 for r in rows), 'negatives')
    missing=[]
    try:
        import cv2
    except ImportError:
        cv2 = None
    for r in rows:
        video=Path(r['video'])
        if not video.is_file():
            missing.append(r['case_id'])
            print('MISSING VIDEO:',r['case_id'],video)
        else:
            frames='?'
            if cv2 is not None:
                cap=cv2.VideoCapture(str(video))
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            print(f"  {r['case_id']:23s} label={r['label']} frames={frames} task={'yes' if r['task'] else 'no'}")
    if missing: print('STOP: video files missing; do not start a partial benchmark.')
    return not missing


def choose_pair(output):
    """A fixed, label-blind rule, used identically on positives and negatives."""
    pairs=[]
    if not isinstance(output,dict): return None, []
    for item in output.get('pairs',[])[:3]:
        if not isinstance(item,dict): continue
        p={k:str(item.get(k,'')).strip() for k in ('actor','target','solid_boundary','pair_type','expected_interaction')}
        if all(p[k] for k in ('actor','target','solid_boundary')):
            pairs.append(p)
    ranked = [p for p in pairs if p['pair_type'].upper()=='OBJECT_ENVIRONMENT']
    return (ranked[0] if ranked else (pairs[0] if pairs else None)),pairs


def generate_prompt(v3, pair):
    from penetration_agent0_v3A_literal import compile_prompt
    return compile_prompt(v3.A_PROMPT, pair, 'literal')


def normalize_roi(x):
    if not isinstance(x, list) or len(x)!=4: return None
    try: x=[float(n) for n in x]
    except (ValueError, TypeError): return None
    if not all(math.isfinite(n) for n in x):return None
    a,b,c,d=x
    if 0<=a<c<=1 and 0<=b<d<=1 and c-a >= .08 and d-b >= .08:
        pad=.07
        return (max(0,a-pad),max(0,b-pad),min(1,c+pad),min(1,d+pad))
    return None


def scout_roi(v3, client, frames, t, pair, image_side=1024):
    """Optional Qwen crop localization. Never use manually annotated ROI/GT frames."""
    prompt=(
        'Locate the area containing BOTH the monitored actor and the relevant\n'
        'solid environmental barrier near their interaction, in THIS image.\n'
        f"ACTOR: {pair['actor']}\nTARGET: {pair['target']}\n"
        f"BOUNDARY: {pair['solid_boundary']}\n"
        'Return normalized [x1,y1,x2,y2] coordinates from 0 to 1 covering BOTH\n'
        'entities and their interface; avoid an extremely tight object-only crop.\n'
        'If one entity is not visible or you cannot locate both, return null.\n'
        'JSON only: {"roi_norm":[0.1,0.1,0.8,0.8] or null,"observation":"..."}'
    )
    result=client.ask([v3.label(frames.frame(t),t,image_side)],prompt,system_prompt=v3.SYSTEM_SHARED)
    data=v3.get_obj(result['answer'])
    return normalize_roi(data.get('roi_norm'))


def agent0_select(v3, client, frames, task, path):
    prompt=AGENT0.format(task=task or 'NOT AVAILABLE; use initial frame only')
    if path.is_file():
        result=json.loads(path.read_text())
        if result.get('agent0_prompt') != prompt:
            raise RuntimeError('Agent 0 prompt mismatch in cached output; new output directory required')
        return result['selected'],result['pairs']
    answer=client.ask([v3.label(frames.frame(0),0,1024)],prompt,system_prompt=v3.SYSTEM_SHARED)
    parsed=v3.get_obj(answer['answer'])
    chosen,pairs=choose_pair(parsed)
    dump(path,{'agent0_prompt':prompt,'agent0_raw':answer['answer'], 'pairs':pairs,'selected':chosen})
    print('AGENT0:',json.dumps(chosen,ensure_ascii=False),flush=True)
    return chosen,pairs


def get_complete_a(v3, path, n_frames):
    if not path.is_file(): return None
    x=json.loads(path.read_text())
    rows=x.get('rows',[])
    good={int(r['frame']) for r in rows if r.get('status')=='ok'}
    if good!=set(range(1,n_frames)):
        print('INCOMPLETE QWEN:',len(good),'/',n_frames-1,flush=True)
        return None
    return x


def candidates_from_post(rows, threshold, cap):
    c=[]
    for r in rows:
        v=r.get('verdict',{})
        sig=float(v.get('signal',0))
        if sig>=threshold:
            c.append({'before':int(r['prev']),'after':int(r['frame']),
                      'post_score':sig,'raw_score':float(v.get('raw_transition_signal',sig)),
                      'observation':v.get('observation','')})
    # One candidate per nonoverlapping temporal neighborhood; same rule for all.
    chosen=[]
    for x in sorted(c,key=lambda x:(-x['post_score'],-x['raw_score'],x['after'])):
        if any(abs(x['after']-y['after'])<=2 for y in chosen):continue
        chosen.append(x)
        if len(chosen)>=cap:break
    return sorted(chosen,key=lambda x:x['after']),c


def universal_gemini_prompt(audit_module):
    """The original V3-A Gemini crop prompt, with ONLY actor-identity neutralization.

    This replacement is CONSTANT across all benchmark cases; the original
    says a gripper can never be the tracked actor, which excludes legitimate
    finger-through-cup sidewall Type-A violations by construction.
    """
    original = audit_module.PROMPT
    phrase = 'The robot/gripper is not the target. Ignore all unrelated objects.'
    if phrase not in original:
        raise RuntimeError('Unexpected auditor prompt version; review fixed adaptation')
    return original.replace(phrase,
        'The named moving target may be a gripper finger, a robot part, or an object carried by the robot. '
        'Track ONLY the named moving target and the named solid barrier; ignore unrelated objects.')


def run_one(row,args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen requires GPU compute node; do not run on the login node')
    import penetration_abc_agents_v3 as v3
    import penetration_A_v3_crop_gemini_score as audit_module
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
    video=Path(row['video'])
    if not video.is_file():raise FileNotFoundError(video)
    n,w,h,fps=v3.video_info(video)
    if n<2:raise ValueError('Video too short')
    case_dir=args.output/row['case_id']
    case_dir.mkdir(parents=True,exist_ok=True)
    model_cfg={'protocol':PROTOCOL,'video':str(video.resolve()),'video_size':video.stat().st_size,
               'video_mtime_ns':video.stat().st_mtime_ns,'frame_range':[0,n-1],
               'task':row['task'],'qwen_model':str(args.qwen_model),
               'v3_sha':hashlib.sha256(Path(v3.__file__).read_bytes()).hexdigest(),
               'auditor_sha':hashlib.sha256(Path(audit_module.__file__).read_bytes()).hexdigest(),
               'max_tokens':args.max_tokens,'image_side':args.image_side,
               'max_monitored_pairs':args.max_monitored_pairs,
               'min_qwen_post':args.min_qwen_post,'max_audits':args.max_audits,
               'gemini_model':args.gemini_model,'gemini_max_tokens':args.gemini_max_tokens,
               'gemini_prompt':universal_gemini_prompt(audit_module),
               'roi_policy':'qwen_pair_scout_with_full_frame_fallback'}
    cfg_hash=sha(model_cfg)
    cfgpath=case_dir/'run_config.json'
    if cfgpath.is_file():
        old=json.loads(cfgpath.read_text())
        if old['hash']!=cfg_hash:
            raise RuntimeError('Different settings in same result folder; use a new --output')
    else:dump(cfgpath,{'hash':cfg_hash,'config':model_cfg})
    summpath=case_dir/'summary.json'
    if summpath.is_file():
        summ=json.loads(summpath.read_text())
        if summ.get('status')=='complete' and summ.get('config_hash')==cfg_hash:
            print('COMPLETE CACHED:',row['case_id'],flush=True)
            return summ
    print('LOADING QWEN:',row['case_id'],flush=True)
    qclient=VLMClient('agent0_typeA_fixed_v3',{'backend':'local_gpu',
                     'model':str(args.qwen_model),'max_tokens':args.max_tokens})
    frames=v3.Frames(video)
    try:
        primary,all_pairs=agent0_select(v3,qclient,frames,row['task'],case_dir/'agent0.json')
        # The same deterministic selection policy for every video. Evaluate
        # up to N pairs, including robot-object AND object-environment.
        # Selection does not use human GT or event frame information.
        ranked=sorted(all_pairs,key=lambda p: 0 if p['pair_type'].upper()=='OBJECT_ENVIRONMENT' else 1)
        monitored=ranked[:args.max_monitored_pairs]
        dump(case_dir/'monitored_pairs.json',{'pairs':monitored,
             'max_pairs':args.max_monitored_pairs,'selection_rule':'object-environment first, stable order'})
        print('MONITORED PAIRS:',json.dumps(monitored,ensure_ascii=False),flush=True)
        if not monitored:
            summary={'status':'complete','case':row['case_id'],'n_frames':n,
                     'agent0_pairs':[],'agent0_no_pair':True,'n_raw':0,'n_retained':0,
                     'n_selected':0,'n_audited':0,'qwen_peak_raw':0.0,
                     'qwen_peak_post':0.0,'video_score':0.0,'score_rule':PROTOCOL,
                     'config_hash':cfg_hash}
            dump(summpath,summary)
            return summary
        peak_raw=peak_post=0.0
        raw_n=0
        all_candidates=[]
        # Per-pair V3 query caches mean interrupted scans can resume correctly.
        original_v3_A_prompt=v3.A_PROMPT
        for pid,pair in enumerate(monitored):
            pair_dir=case_dir/'pairs'/f'p{pid}'
            pair_dir.mkdir(parents=True,exist_ok=True)
            # Do not accumulate scene locks across multiple Agent-0 pairs.
            v3.A_PROMPT=original_v3_A_prompt
            v3.A_PROMPT=generate_prompt(v3,pair)
            # Save exact rendered prompt; V3's original core and scoring stay fixed.
            dump(pair_dir/'prompts.json',{'A':v3.A_PROMPT,'agent0':AGENT0,
                                         'gemini':universal_gemini_prompt(audit_module)})
            print(f'QWEN A PAIR {pid+1}/{len(monitored)}: {pair["actor"]} vs {pair["target"]}',flush=True)
            v3.run_agent(qclient,'A',frames,pair_dir,0,n-1,0,args.image_side,640,
                         None,None,None,None,5,args.max_tokens,(w,h))
            a=get_complete_a(v3,pair_dir/'A_queries.json',n)
            if a is None:
                raise RuntimeError(f'Incomplete Qwen A scan for pair {pid}; rerun to retry')
            vrows=a['rows']
            peak_raw=max(peak_raw,max((float(r['verdict'].get('raw_transition_signal',0)) for r in vrows),default=0))
            peak_post=max(peak_post,max((float(r['verdict'].get('signal',0)) for r in vrows),default=0))
            raw_n+=sum(float(r['verdict'].get('raw_transition_signal',0))>=args.min_qwen_post for r in vrows)
            retained,_=candidates_from_post(vrows,args.min_qwen_post,n)
            all_candidates.extend([{**x,'pair_id':pid} for x in retained])
        # Common top-K budget across all scene pairs (avoids varying API volume
        # by how many pairs Agent 0 happens to identify).
        ranked_c=sorted(all_candidates,key=lambda x:(-x['post_score'],-x['raw_score'],x['after'],x['pair_id']))
        selected=[]
        for c in ranked_c:
            if any(c['pair_id']==s['pair_id'] and abs(c['after']-s['after'])<=2 for s in selected):
                continue
            selected.append(c)
            if len(selected)>=args.max_audits:break
        selected=sorted(selected,key=lambda c:(c['after'],c['pair_id']))
        dump(case_dir/'candidates.json',{'min_qwen_post':args.min_qwen_post,
                   'max_audits':args.max_audits,'all_retained':all_candidates,
                   'selected':selected})
        print('QWEN DONE:',row['case_id'],'peaks raw/post:',peak_raw,peak_post,
              'retained=',len(all_candidates),'selected=',len(selected),flush=True)
        if not selected:
            summary={'status':'complete','case':row['case_id'],'n_frames':n,
                     'agent0_pairs':monitored,'agent0_no_pair':False,'n_raw':raw_n,
                     'n_retained':len(all_candidates),'n_selected':0,'n_audited':0,
                     'qwen_peak_raw':peak_raw,'qwen_peak_post':peak_post,
                     'video_score':0.0,'score_rule':PROTOCOL,'config_hash':cfg_hash}
            dump(summpath,summary)
            return summary
        if not os.environ.get('VLM2_API_KEY'):
            raise RuntimeError('VLM2_API_KEY missing: required for Gemini candidate verification')
        os.environ.setdefault('PDI_VLM_ROLE_OVERRIDES','{"vlm2":{"api_base":"https://api.302.ai/v1"}}')
        from robot.preprocessing.link7_persistent.interface.config import role_config
        from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
        env_file=WS/'.env.vlm'
        if env_file.is_file():load_env_file(env_file)
        gconfig=role_config('vlm2')
        if gconfig.get('backend')!='cloud_api':
            raise RuntimeError('Wilson VLM2 backend is not cloud_api')
        gconfig['model']=args.gemini_model
        gconfig['max_tokens']=args.gemini_max_tokens
        gconfig['temperature']=0
        gconfig['reasoning_effort']=None
        gclient=VLMClient('agent0_typeA_gemini',gconfig)
        gframes=audit_module.Frames(video)
        result_scores=[]
        try:
            for i,c in enumerate(selected,1):
                b,t,pid=c['before'],c['after'],c['pair_id']
                pair=monitored[pid]
                record_path=case_dir/'gemini'/f'p{pid}_f{b:04d}_f{t:04d}.json'
                if record_path.is_file():
                    record=json.loads(record_path.read_text())
                    if record.get('config_hash')!=cfg_hash:
                        raise RuntimeError('Stale Gemini cache; new output needed')
                    verdict=record['verdict']
                    score=float(record['score'])
                    print('GEMINI CACHED',pid,b,t,score,flush=True)
                else:
                    try:
                        roi=scout_roi(v3,qclient,frames,t,pair,args.image_side)
                    except Exception as exc:
                        print('ROI SCOUT FALLBACK:',type(exc).__name__,str(exc)[:90],flush=True)
                        roi=None
                    roi_source='qwen_roi_scout' if roi is not None else 'full_frame_fallback'
                    images,box=audit_module.prepared_images(gframes,b,t,roi or (0,0,1,1),960,1000)
                    preview=case_dir/'gemini'/f'visual_p{pid}_f{b:04d}_f{t:04d}.jpg'
                    preview.parent.mkdir(exist_ok=True,parents=True)
                    audit_module.contact_sheet(images).save(preview,quality=92)
                    prompt=universal_gemini_prompt(audit_module).format(
                           before=b,after=t,actor=pair['actor'],container=pair['target'])
                    response=gclient.ask(images,prompt,system_prompt=audit_module.SYSTEM)
                    verdict=audit_module.parse_json_object(response['answer'])
                    initial_verdict=verdict
                    identity=verdict.get('target_identified')
                    if identity not in ('YES','NO','UNCLEAR'):
                        raise ValueError(f'Bad Gemini target_identified: {identity!r}')

                    if identity != 'YES' and roi is not None:
                        full_images, full_box = audit_module.prepared_images(
                            gframes,b,t,(0,0,1,1),960,1000)
                        retry_response=gclient.ask(
                            full_images,prompt,system_prompt=audit_module.SYSTEM)
                        verdict=audit_module.parse_json_object(retry_response['answer'])
                        identity=verdict.get('target_identified')
                        if identity not in ('YES','NO','UNCLEAR'):
                            raise ValueError(f'Bad Gemini retry identity: {identity!r}')
                        roi_source += '+full_frame_identity_retry'

                    score=audit_module.score_of(verdict) if identity=='YES' else 0.0
                    status='verified' if identity=='YES' else 'actor_unverified_rejected'
                    print(f'IDENTITY {status} p{pid} f{b}->{t} first={initial_verdict.get("target_identified")} final={identity}',flush=True)
                    record={'case':row['case_id'],'pair_id':pid,'frames':[b,t],
                            'identity_status':status, 'initial_gemini_verdict':initial_verdict,
                            'qwen':c,'agent0':pair,'roi_source':roi_source,
                            'roi_norm':roi,'roi_pixels':box,'visual':str(preview),
                            'verdict':verdict,'score':score,'config_hash':cfg_hash}
                    dump(record_path,record)
                    print(f'GEMINI {i}/{len(selected)} p{pid} f{b}->{t} score={score:.3f} '
                          f"reason={str(verdict.get('reason',''))[:120]}",flush=True)
                result_scores.append({'pair_id':pid,'pair':[b,t],
                                      'score':score,'record':str(record_path)})
        finally:gframes.close()
        if len(result_scores)!=len(selected):raise RuntimeError('Incomplete Gemini auditing')
        best=max(result_scores,key=lambda r:r['score'])
        summary={'status':'complete','case':row['case_id'],'n_frames':n,
                 'agent0_pairs':monitored,'agent0_no_pair':False,'n_raw':raw_n,
                 'n_retained':len(all_candidates),'n_selected':len(selected),
                 'n_audited':len(result_scores),'qwen_peak_raw':peak_raw,
                 'qwen_peak_post':peak_post,'video_score':best['score'],
                 'best_pair':best['pair'],'best_pair_id':best['pair_id'],
                 'score_rule':PROTOCOL,'config_hash':cfg_hash}
        dump(summpath,summary)
        print('FINAL:',row['case_id'],'score',summary['video_score'],flush=True)
        return summary
    finally:
        frames.close()
        del qclient
        gc.collect()


def auc_pairwise(labels,scores):
    positive=[float(s) for l,s in zip(labels,scores) if l==1]
    negative=[float(s) for l,s in zip(labels,scores) if l==0]
    if not positive or not negative:return None
    wins=sum(1.0 if a>b else (.5 if a==b else 0.0)
             for a in positive for b in negative)
    return wins/(len(positive)*len(negative))


def bootstrap_auc(data,key,n=1000,seed=20261009):
    pos=[r for r in data if r['label']==1]
    neg=[r for r in data if r['label']==0]
    if not pos or not neg:return None
    rand=random.Random(seed)
    a=[]
    for _ in range(n):
        sampled=[rand.choice(pos) for _ in pos]+[rand.choice(neg) for _ in neg]
        a.append(auc_pairwise([r['label'] for r in sampled],[r[key] for r in sampled]))
    a.sort()
    return [round(a[int(.025*(n-1))],4),round(a[int(.975*(n-1))],4)]


def report(args):
    manifest=read_manifest(args.manifest)
    rows=[];missing=[]
    for r in manifest:
        p=args.output/r['case_id']/'summary.json'
        if not p.is_file():
            missing.append(r['case_id']);continue
        result=json.loads(p.read_text())
        if result.get('status')!='complete' or any(result.get(k) is None for k in
             ('qwen_peak_raw','qwen_peak_post','video_score')):
            missing.append(r['case_id']);continue
        rows.append({'case_id':r['case_id'],'label':r['label'], 'mechanism':r.get('mechanism',''),
                     'qwen_peak_raw':result['qwen_peak_raw'],
                     'qwen_peak_post':result['qwen_peak_post'],
                     'video_score':result['video_score'],
                     'n_retained':result['n_retained'],
                     'n_audited':result['n_audited'],
                     'agent0_no_pair':result.get('agent0_no_pair',False),
                     'best_pair':str(result.get('best_pair',''))})
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'scores.csv').open('w',newline='') as f:
        fields=['case_id','label','mechanism','qwen_peak_raw','qwen_peak_post','video_score',
                'n_retained','n_audited','agent0_no_pair','best_pair']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    fields={'qwen_raw_auc':'qwen_peak_raw','qwen_post_auc':'qwen_peak_post',
            'end_to_end_gemini_auc':'video_score'}
    summary={'protocol':PROTOCOL,'requested':len(manifest),'completed':len(rows),
             'n_positive':sum(x['label']==1 for x in rows),
             'n_negative':sum(x['label']==0 for x in rows),'missing_cases':missing,
             'benchmark_complete':not bool(missing)}
    for name,key in fields.items():
        value=auc_pairwise([r['label'] for r in rows],[r[key] for r in rows])
        summary[name]=round(value,4) if value is not None else None
        summary[name+'_stratified_bootstrap95']=bootstrap_auc(rows,key) if value is not None else None
    # Predeclared analysis: narrow open-container/wall mechanism versus
    # extended A wall crossing (including the finger-through-cup positive).
    core = [r for r in rows if r['mechanism']!='A_gripper_cup_wall']
    summary['core_A_container_n_positive']=sum(r['label']==1 for r in core)
    summary['core_A_container_n_negative']=sum(r['label']==0 for r in core)
    summary['core_A_container_gemini_auc']=auc_pairwise(
        [r['label'] for r in core],[r['video_score'] for r in core])
    summary['core_A_container_qwen_post_auc']=auc_pairwise(
        [r['label'] for r in core],[r['qwen_peak_post'] for r in core])
    dump(args.output/'auroc_summary.json',summary)
    print('\n================ BENCHMARK REPORT ================')
    for r in rows:
        print(f"{r['case_id']:24s} GT={r['label']} Qwen_post={r['qwen_peak_post']:.3f} "
              f"Gemini_video={r['video_score']:.3f} audits={r['n_audited']}")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if missing:print('WARNING: PARTIAL AUROC ONLY — DO NOT REPORT AS COMPLETE BENCHMARK')
    if summary['n_positive']<5:print('WARNING: VERY FEW POSITIVES; AUROC confidence interval unstable')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('check','run-one','batch','report'))
    p.add_argument('--manifest',type=Path,default=MANIFEST_DEFAULT)
    p.add_argument('--output',type=Path,default=ROOT/'typeA_agent0_literal_qwen_gemini_20261009')
    p.add_argument('--case',default='',help='required by run-one; batch optionally comma-separated IDs')
    p.add_argument('--qwen-model',type=Path,default=MODEL)
    p.add_argument('--image-side',type=int,default=1024)
    p.add_argument('--max-tokens',type=int,default=520)
    p.add_argument('--min-qwen-post',type=float,default=.65)
    p.add_argument('--max-audits',type=int,default=12)
    p.add_argument('--max-monitored-pairs',type=int,default=2)
    p.add_argument('--gemini-model',default='gemini-3.8-flash')
    p.add_argument('--gemini-max-tokens',type=int,default=4096)
    args=p.parse_args()
    rows=read_manifest(args.manifest)
    if args.action=='check':
        if not validate_manifest(args.manifest):sys.exit(2)
        return
    if args.action=='report':return report(args)
    if not 0<=args.min_qwen_post<=1 or args.max_audits<1 or not 1<=args.max_monitored_pairs<=3: p.error('Invalid filter or max-audits')
    chosen=set(args.case.split(',')) if args.case else set()
    if args.action=='run-one':
        if len(chosen)!=1:p.error('run-one requires --case ID')
        r=next((r for r in rows if r['case_id'] in chosen),None)
        if r is None:p.error('Case absent from manifest')
        return run_one(r,args)
    for r in rows:
        if chosen and r['case_id'] not in chosen:continue
        command=[sys.executable,str(Path(__file__).resolve()),'run-one',
                 '--manifest',str(args.manifest),'--output',str(args.output),
                 '--case',r['case_id'],'--qwen-model',str(args.qwen_model),
                 '--image-side',str(args.image_side),'--max-tokens',str(args.max_tokens),
                 '--min-qwen-post',str(args.min_qwen_post),
                 '--max-audits',str(args.max_audits),
                 '--max-monitored-pairs',str(args.max_monitored_pairs),'--gemini-model',args.gemini_model,
                 '--gemini-max-tokens',str(args.gemini_max_tokens)]
        print('\n======== CASE',r['case_id'],'========',flush=True)
        ret=subprocess.run(command).returncode
        if ret:
            print('FAILED:',r['case_id'],'exit',ret,'(continuing)',flush=True)
    return report(args)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('FATAL:',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.exit(1)
