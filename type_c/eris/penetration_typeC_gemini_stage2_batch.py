#!/usr/bin/env python3
"""Label-blind Gemini review of cached Qwen Type-C stage-1 candidates.

Never runs Qwen, never gives GT labels to Gemini. Outputs separate evidence
axes for solid boundary crossing, unsupported interface truncation, and abrupt
position discontinuity. Scores are uncalibrated visual evidence, not 3D proof.

Use --dry-run before spending API calls; --report-only uses cached summaries.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
DEFAULT_MANIFEST = ROOT / 'penetration_typeC_pilot_manifest.csv'
DEFAULT_QWEN = ROOT / 'typeC_qwen_multiframe_benchmark_v1_20261009' / 'qwen'
DEFAULT_OUT = ROOT / 'typeC_gemini_after_qwen_v1_20261009'
PROTOCOL = 'typeC_cached_qwen_gemini_multiframe_v1_20261009'

# Identical for all video cases; actor, target and boundary are identification
# only. Do not mention GT, known positives, specific examples, or known events.
SYSTEM = """You are an independent visual auditor of synthetic robot-manipulation videos.
Judge visible evidence in ordered SOURCE FRAMES, not task success. A Qwen detector
flagged these timestamps as candidates, which is NOT proof of an error. Avoid
confirmation bias. The provided entity names identify what to inspect, not a
claim of penetration. A legal opening and an object occluded behind a front
panel are different from crossing the solid panel. Do not infer unobserved 3D
trajectories. Return a single JSON object only."""

PROMPT = """Examine the SAME moving actor and environment boundary in these {count}
chronologically ordered ORIGINAL video frames: {frame_list}. The first image is
an identity/appearance reference; the rest show the local temporal sequence
around candidate frame {candidate}.

Moving actor: {actor}
Container/environment: {target}
Solid boundary to inspect: {boundary}

The frames are OBSERVATIONS, not 3D reconstructions. A drawer/bin can have an
EMPTY OPENING and SOLID FRONT, SIDE, RIM or WALL. Determine what the actor
ACTUALLY touches/overlaps visually at the candidate time. The mere presence
of an opening elsewhere does not establish that the actor traveled through it;
conversely, overlap in image coordinates or ordinary occlusion is NOT enough
to prove solid penetration. Use uncertain scores when the path is not visible.

Evaluate THREE DISTINCT kinds of visual evidence:
1. SOLID PENETRATION: a segment of the actor visibly intersects/crosses an
   identified SOLID boundary in a way that cannot be explained by an opening,
   perspective or normal foreground occlusion.
2. INTERFACE TRUNCATION: the same actor becomes abnormally shortened,
   abruptly cut, or loses a visible part AT THE CONTAINER INTERFACE, beyond
   what motion, pose, foreshortening, gripper covering, or a foreground panel
   can account for. Judge visible geometry over neighboring frames; do not
   compare raw 2D lengths from different poses as if they were rigid.
3. ABRUPT DISCONTINUITY: the actor disappears/reappears or relocates across
   the boundary without a supported connecting trajectory. A temporal jump
   is NOT, by itself, solid penetration; keep this evidence separate.

For each, produce a continuous 0..1 *uncalibrated EVIDENCE* score. 0 = no
support; 0.1..0.3 = weak or normal alternative; 0.4..0.6 = ambiguous;
0.7..0.9 = strong visible evidence; 1 = exceptionally convincing. A legal
opening must LOWER solid-penetration evidence ONLY when it plausibly explains
THE ACTUAL visible interface in these images. Do not use blanket veto rules.
If the actor or boundary cannot be tracked, assign low or uncertain scores
based on actual visual evidence, and explain the limitation.

Return ONLY this JSON object; all fields required:
{{
 "actor_identified": "YES|NO|UNCLEAR",
 "boundary_identified": "YES|NO|UNCLEAR",
 "interface_location": "SOLID_PANEL|OPEN_CAVITY|OTHER|UNCLEAR",
 "solid_crossing_visible": "YES|NO|UNCLEAR",
 "legal_opening_explanation": "YES|NO|UNCLEAR",
 "normal_occlusion_explanation": "YES|NO|UNCLEAR",
 "unsupported_interface_truncation": "YES|NO|UNCLEAR",
 "abrupt_actor_discontinuity": "YES|NO|UNCLEAR",
 "solid_penetration_score": 0.0,
 "interface_truncation_score": 0.0,
 "temporal_discontinuity_score": 0.0,
 "observed_evidence": "Visible position, contour, panel/opening, and temporal changes, using frame numbers",
 "alternative_explanation": "Specific legal opening, occlusion, pose, or missing evidence, where applicable",
 "reason": "Concise justification separating solid penetration, truncation and temporal jump"
}}
"""
FIELDS = ('solid_penetration_score','interface_truncation_score','temporal_discontinuity_score')
YESNO = {'YES','NO','UNCLEAR'}


def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def sha(data):
    if not isinstance(data,bytes): data=str(data).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def parse_obj(answer):
    answer=re.sub(r'<think>.*?</think>','',answer,flags=re.S)
    dec=json.JSONDecoder()
    for m in re.finditer(r'\{',answer):
        try:
            obj,_=dec.raw_decode(answer[m.start():])
            if isinstance(obj,dict):return obj
        except json.JSONDecodeError:pass
    raise ValueError('No complete JSON returned: '+repr(answer[:260]))


def validate(obj):
    for k in ('actor_identified','boundary_identified','solid_crossing_visible',
              'legal_opening_explanation','normal_occlusion_explanation',
              'unsupported_interface_truncation','abrupt_actor_discontinuity'):
        if obj.get(k) not in YESNO:raise ValueError(f'Bad/missing {k}: {obj.get(k)!r}')
    if obj.get('interface_location') not in ('SOLID_PANEL','OPEN_CAVITY','OTHER','UNCLEAR'):
        raise ValueError('Missing/invalid interface_location')
    for k in ('observed_evidence','alternative_explanation','reason'):
        if not isinstance(obj.get(k),str) or not obj[k].strip():
            raise ValueError('Missing grounded explanation: '+k)
    scores={}
    for k in FIELDS:
        v=obj.get(k)
        if isinstance(v,bool):raise ValueError(k+' is boolean')
        x=float(v)
        if not math.isfinite(x) or not 0<=x<=1:raise ValueError(k+' out of range')
        scores[k]=x
    return scores


def runs(vals):
    result=[]
    for t in sorted(set(vals)):
        if not result or t>result[-1][-1]+1:result.append([t])
        else:result[-1].append(t)
    return result


def select_frames(vals,budget):
    """Deterministic, label-blind temporal diversity across high Qwen candidates."""
    vals=sorted(set(vals))
    if not vals or budget<=0:return []
    ordered=sorted(runs(vals),key=lambda r:(-len(r),r[0]))
    chosen=[]
    def add(v):
        if v not in chosen and len(chosen)<budget:chosen.append(v)
    for run in ordered[:max(1,budget//2)]:add(run[len(run)//2])
    if ordered:
        add(ordered[0][0]);add(ordered[0][-1])
    while len(chosen)<min(len(vals),budget):
        avail=(v for v in vals if v not in chosen)
        add(max(avail,key=lambda v:(min(abs(v-p) for p in chosen) if chosen else 10**9,-v)))
    return sorted(chosen)


def window_indices(t,n,reference):
    # Seven visible sequential frames; dedup near ends. The reference is first.
    return sorted(set([reference,max(0,t-12),max(0,t-6),max(0,t-2),t,
                       min(n-1,t+6),min(n-1,t+12)]))


def get_images(video,indices,side):
    import cv2
    from PIL import Image,ImageDraw
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise OSError('Cannot open '+str(video))
    images=[]
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
            ok,img=cap.read()
            if not ok:raise OSError(f'Cannot read frame {idx} of {video}')
            im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            im.thumbnail((side,side),Image.Resampling.LANCZOS)
            canvas=Image.new('RGB',(im.width,im.height+40),'white')
            canvas.paste(im,(0,40))
            ImageDraw.Draw(canvas).text((12,13),f'SOURCE FRAME {idx}',fill='black')
            images.append(canvas)
    finally:cap.release()
    return images


def save_contact_sheet(images,path):
    from PIL import Image
    cols=3;w=max(i.width for i in images);h=max(i.height for i in images)
    sheet=Image.new('RGB',(cols*w,math.ceil(len(images)/cols)*h),'white')
    for j,im in enumerate(images):sheet.paste(im,((j%cols)*w,(j//cols)*h))
    path.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(path,quality=88)


def get_client(args):
    if not os.environ.get('VLM2_API_KEY'):
        env=WS/'.env.vlm'
        if env.is_file():
            sys.path.insert(0,str(WS))
            from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
            load_env_file(env)
    if not os.environ.get('VLM2_API_KEY'):
        raise RuntimeError('VLM2_API_KEY not found (checked env and existing Wilson .env.vlm). Do not paste keys into chat.')
    from openai import OpenAI
    return OpenAI(api_key=os.environ['VLM2_API_KEY'],base_url=args.api_base,
                  timeout=180.0,max_retries=2)


def read_manifest(path):
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty manifest')
    if not {'case_id','video','label'}.issubset(rows[0]):raise ValueError('Required: case_id,video,label')
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Duplicate cases')
    return rows


def locate(row,qwen_root):
    case=row['case_id']
    qdir=Path(row.get('reuse_qwen_dir') or qwen_root/case)
    agent0=Path(row.get('reuse_agent0_json') or qdir/'agent0_multiframe.json')
    return qdir,agent0


def scan_info(row,qwen_root,threshold):
    case=row['case_id'];video=Path(row['video']).resolve()
    qdir,a0path=locate(row,qwen_root)
    if not video.is_file():raise FileNotFoundError('Missing video: '+str(video))
    for p in (qdir/'C_queries.json',qdir/'run_config.json',a0path):
        if not p.is_file():raise FileNotFoundError('Missing cached stage1 file: '+str(p))
    cfg=json.loads((qdir/'run_config.json').read_text())
    a0=json.loads(a0path.read_text()).get('selected')
    queries=json.loads((qdir/'C_queries.json').read_text())
    if cfg.get('case')!=case or Path(cfg['video']).resolve()!=video:
        raise ValueError('Stage1 cache case/video mismatch '+case)
    if cfg.get('agent0')!=a0:raise ValueError('Agent0 differs from Qwen stage1 '+case)
    start=int(cfg['start']);end=int(cfg['end']);ref=int(cfg['reference'])
    rows=queries['rows'];oks=[r for r in rows if r.get('status')=='ok']
    seen=[int(r['frame']) for r in oks]
    expected=set(range(start,end+1))-{ref}
    if len(seen)!=len(set(seen)) or set(seen)!=expected:
        raise ValueError(f'Incomplete Stage1 Qwen for {case}: {len(set(seen))}/{len(expected)}')
    if start>1 or ref!=0:raise ValueError('Protocol expected start <=1 and frame0 reference')
    import cv2
    cap=cv2.VideoCapture(str(video));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if end!=n-1:raise ValueError('Qwen did not scan full video for '+case)
    cand=[int(r['frame']) for r in oks if float((r.get('verdict') or {}).get('signal',0))>=threshold]
    if a0 is None and cand:raise ValueError('Candidate scores but no selected actor '+case)
    return dict(case=case,video=video,qdir=qdir,a0=a0,qwen_hash=sha((qdir/'C_queries.json').read_bytes()),
                agent0_hash=sha(a0path.read_bytes()),n=n,ref=ref,qualifying=cand)


def evaluate_one(row,a,client):
    info=scan_info(row,a.qwen_root,a.min_qwen_signal)
    case=info['case'];cands=select_frames(info['qualifying'],a.max_audits)
    out=a.output/case
    config={'protocol':PROTOCOL,'case':case,'video':str(info['video']),
            'qwen_hash':info['qwen_hash'],'agent0_hash':info['agent0_hash'],
            'chosen_frames':cands,'threshold':a.min_qwen_signal,
            'max_audits':a.max_audits,'prompt_hash':sha(PROMPT+SYSTEM),
            'model':a.model,'api_base':a.api_base,'image_side':a.image_side,
            'max_tokens':a.max_tokens,'frame_offsets':[-12,-6,-2,0,6,12]}
    cf=out/'config.json'
    if cf.exists() and json.loads(cf.read_text())!=config:
        raise ValueError(f'{case}: Cached config mismatch. Change --output for a new protocol.')
    if a.dry_run:
        print(f'WOULD GEMINI: {case} actor={info["a0"]} qwen_candidates={len(info["qualifying"])} frames={cands}',flush=True)
        return
    dump(cf,config)
    if not cands:
        summary={'case':case,'status':'complete','candidate_count':0,'audited':0,
                 'selected_frames':[],**{k:0.0 for k in FIELDS},'typeC_exploratory_score':0.0,
                 'note':'No Qwen candidate above fixed threshold'}
        dump(out/'summary.json',summary);return summary
    if client is None:client=get_client(a)
    all_records=[];fails=[]
    print(f'CASE {case}: {len(info["qualifying"])} stage1 candidates -> {len(cands)} Gemini audits',flush=True)
    for j,t in enumerate(cands,1):
        record_path=out/'audits'/f'f{t:04d}.json'
        if record_path.is_file():
            rec=json.loads(record_path.read_text())
            if rec.get('config_hash')!=sha(json.dumps(config,sort_keys=True)):
                raise ValueError('Stale audit cache '+str(record_path))
            validate(rec['verdict'])
            print(f'CACHED {case} f{t:04d}',flush=True)
        else:
            indices=window_indices(t,info['n'],info['ref'])
            try:
                images=get_images(info['video'],indices,a.image_side)
                preview=out/'previews'/f'f{t:04d}.jpg'
                save_contact_sheet(images,preview)
                prompt=PROMPT.format(count=len(indices),frame_list=', '.join(map(str,indices)),
                    candidate=t,actor=info['a0']['actor'],target=info['a0']['target'],
                    boundary=info['a0']['solid_boundary'])
                content=[{'type':'text','text':SYSTEM+'\n\n'+prompt}]
                for im in images:
                    b=io.BytesIO();im.save(b,format='JPEG',quality=86)
                    content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+
                                        base64.b64encode(b.getvalue()).decode('ascii'),'detail':'high'}})
                answer='';last_error=None
                for attempt in range(2):
                    r=client.chat.completions.create(model=a.model,
                        messages=[{'role':'user','content':content}],
                        temperature=0,max_tokens=a.max_tokens)
                    answer=r.choices[0].message.content or ''
                    try:
                        verdict=parse_obj(answer)
                        scores=validate(verdict)
                        last_error=None;break
                    except Exception as exc:
                        last_error=exc
                        print(f'JSON RETRY {case} f{t:04d}: {str(exc)[:140]}',flush=True)
                if last_error:raise last_error
                rec={'case':case,'frame':t,'source_frames':indices,'qwen_signal_candidate':True,
                    'scores':scores,'verdict':verdict,'raw_response':answer,
                    'preview':str(preview),
                    'config_hash':sha(json.dumps(config,sort_keys=True))}
                dump(record_path,rec)
                print(f'GEMINI {case} {j}/{len(cands)} f{t:04d} '
                      f'solid={scores[FIELDS[0]]:.2f} trunc={scores[FIELDS[1]]:.2f} '
                      f'jump={scores[FIELDS[2]]:.2f} '
                      f'interface={verdict["interface_location"]}',flush=True)
            except Exception as exc:
                fails.append({'frame':t,'error':f'{type(exc).__name__}: {str(exc)[:450]}'})
                print(f'GEMINI FAILED {case} f{t:04d}: {type(exc).__name__}: {str(exc)[:200]}',flush=True)
                continue
        all_records.append(rec)
    complete=len(all_records)==len(cands) and not fails
    peaks={k:max((r['scores'][k] for r in all_records),default=None) for k in FIELDS}
    summary={'case':case,'status':'complete' if complete else 'incomplete',
       'candidate_count':len(info['qualifying']),'audited':len(all_records),
       'selected_frames':cands,'failures':fails,
       **{k:(peaks[k] if complete else None) for k in FIELDS},
       'typeC_exploratory_score':(max(peaks[FIELDS[0]],peaks[FIELDS[1]]) if complete else None),
       'note':'Separate uncalibrated visual evidence axes. TypeC score=max(solid,truncation) is exploratory, not 3D proof.'}
    dump(out/'summary.json',summary)
    print('SUMMARY',case,summary['status'],{k:summary.get(k) for k in FIELDS},flush=True)
    return summary


def auroc(pairs):
    pos=[s for y,s in pairs if y==1];neg=[s for y,s in pairs if y==0]
    if not pos or not neg:return None
    return round(sum(float(x>y)+0.5*float(x==y) for x in pos for y in neg)/(len(pos)*len(neg)),5)


def report(rows,out):
    fields=['case_id','label','mechanism','status','solid_penetration_score',
           'interface_truncation_score','temporal_discontinuity_score',
           'typeC_exploratory_score','audited','candidate_count']
    result=[]
    for row in rows:
        path=out/row['case_id']/'summary.json'
        s=json.loads(path.read_text()) if path.is_file() else {}
        status=s.get('status','not_run')
        item={'case_id':row['case_id'],'label':row['label'],
              'mechanism':row.get('mechanism',''),'status':status,
              'audited':s.get('audited',''),'candidate_count':s.get('candidate_count','')}
        for k in list(FIELDS)+['typeC_exploratory_score']:
            item[k]=s.get(k) if status=='complete' else ''
        result.append(item)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'scores.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(result)
    complete=[r for r in result if r['status']=='complete']
    stats={'requested':len(result),'completed':len(complete),
           'benchmark_complete':len(complete)==len(result),
           'failures':[r['case_id'] for r in result if r['status']!='complete'],
           'n_positives_complete':sum(r['label']=='1' for r in complete),
           'n_negatives_complete':sum(r['label']=='0' for r in complete)}
    for key in list(FIELDS)+['typeC_exploratory_score']:
        stats[key+'_auroc']=auroc([(int(r['label']),float(r[key])) for r in complete])
    stats['note']='Tiny developer pilot: two positive cases of differing mechanisms, four negatives; scores uncalibrated, AUC unreliable for generalization.'
    dump(out/'auroc_summary.json',stats)
    print('\n=== GEMINI AFTER QWEN, OFFLINE GT REPORT ===')
    for r in result:
        print('  ',r['case_id'],r['label'],r['status'],
              'solid=',r['solid_penetration_score'],
              'trunc=',r['interface_truncation_score'],
              'jump=',r['temporal_discontinuity_score'],flush=True)
    print(json.dumps(stats,indent=2),flush=True)
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--qwen-root',type=Path,default=DEFAULT_QWEN)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--case',default=None)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--max-audits',type=int,default=8)
    p.add_argument('--min-qwen-signal',type=float,default=.75)
    p.add_argument('--image-side',type=int,default=1152)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--api-base',default='https://api.302.ai/v1')
    p.add_argument('--model',default='gemini-3.8-flash')
    a=p.parse_args()
    if not 1<=a.max_audits<=20:p.error('max-audits must be 1..20')
    manifest=read_manifest(a.manifest)
    if a.report_only:report(manifest,a.output);return
    todo=[r for r in manifest if a.case is None or a.case==r['case_id']]
    if not todo:p.error('Unrecognized --case')
    client=None
    for row in todo:
        try:
            s=evaluate_one(row,a,client)
            # API client creation in evaluate_one is lazily scoped, no persistent
            # local model weights, so tiny remote client setup per case is fine.
            if s and s['status']!='complete':print('WARNING incomplete case:',row['case_id'],flush=True)
        except Exception as exc:
            print('CASE ERROR:',row['case_id'],type(exc).__name__,str(exc)[:500],flush=True)
    if not a.dry_run:
        report(manifest,a.output)
        if any(not (a.output/r['case_id']/'summary.json').is_file() or
               json.loads((a.output/r['case_id']/'summary.json').read_text()).get('status')!='complete'
               for r in todo):sys.exit(2)

if __name__=='__main__':main()
