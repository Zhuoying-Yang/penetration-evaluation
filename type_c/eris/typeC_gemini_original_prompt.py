#!/usr/bin/env python3
"""Gemini on ORIGINAL cached V3-C contrastive prompt and ORIGINAL two-frame protocol.

No new verification prompt. No GT labels supplied to model. Reuses completed
Agent0 -> Qwen C scan, the original prompt and original scoring function.
This tests prompt transfer (Qwen vs Gemini), not geometric physical truth.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
MODEL = 'gemini-3.8-flash'
VERSION = 'original-v3c-prompt-gemini-pairwise-v1'


def sha(x):
    if not isinstance(x, bytes): x = x.encode('utf-8')
    return hashlib.sha256(x).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def load_v3(path):
    spec = importlib.util.spec_from_file_location('original_v3c_for_gemini', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_runs(times):
    runs = []
    for t in sorted(set(times)):
        if not runs or t > runs[-1][-1]+1:
            runs.append([t])
        else:
            runs[-1].append(t)
    return runs


def choose_candidates(rows, limit, threshold):
    """Label-blind: detect starts of candidate runs, prioritize early onset and
    temporally distributed proposals rather than long-run midpoint only.
    """
    good = {int(r['frame']): r for r in rows if r.get('status') == 'ok'}
    scores = {t: float(r.get('verdict',{}).get('signal',0)) for t,r in good.items()}
    all_pos = sorted(t for t,s in scores.items() if s >= threshold)
    if not all_pos: return [], 0
    runs = candidate_runs(all_pos)
    choices=[]
    def add(t):
        if t in scores and t not in choices and len(choices)<limit: choices.append(t)
    # Earliest onset for any high-signal run; beginning/middle/end of longest run.
    add(all_pos[0])
    for r in sorted(runs, key=lambda r: (-len(r),r[0]))[:min(5,len(runs))]:
        add(r[0]); add(r[len(r)//2]); add(r[-1])
    # Distribute remaining budget across the full candidate time span.
    while len(choices) < min(limit, len(all_pos)):
        new = max((t for t in all_pos if t not in choices),
                  key=lambda t:(min(abs(t-x) for x in choices),-t))
        add(new)
    return sorted(choices),len(all_pos)


def read_manifest(path):
    with open(path,newline='') as f: rows=list(csv.DictReader(f))
    if not rows or not {'case_id','video'}.issubset(rows[0]):
        raise ValueError('manifest requires case_id,video')
    return rows


def inspect(row,args):
    case=row['case_id']; video=Path(row['video'])
    qdir=Path(row.get('reuse_qwen_dir') or (args.qwen_root/case))
    needed=[qdir/'C_queries.json',qdir/'run_config.json',qdir/'rendered_prompt.json']
    if not video.is_file():raise FileNotFoundError(str(video))
    for path in needed:
        if not path.is_file(): raise FileNotFoundError('Missing ORIGINAL Qwen scan: '+str(path))
    cfg=json.loads((qdir/'run_config.json').read_text())
    prompt_dict=json.loads((qdir/'rendered_prompt.json').read_text())
    prompt=prompt_dict['C']
    if sha(prompt)!=cfg['prompt_sha256']:
        raise ValueError('Original C prompt hash MISMATCH; will not substitute a prompt')
    if cfg['case']!=case or Path(cfg['video']).resolve()!=video.resolve():
        raise ValueError('Qwen cache/video mismatch for '+case)
    reference=int(cfg['reference']);start=int(cfg['start']);end=int(cfg['end'])
    if reference != 0:raise ValueError('Expected original fixed first frame reference')
    queries=json.loads((qdir/'C_queries.json').read_text())['rows']
    oks=[r for r in queries if r.get('status')=='ok']
    seen={int(r['frame']) for r in oks}
    if len(seen)!=len(oks) or seen!=(set(range(start,end+1))-{reference}):
        raise ValueError(f'Incomplete Qwen scan for {case}: {len(seen)}/{end-start+1}')
    for r in oks:
        if r.get('box_xyxy') is not None:
            raise ValueError('Unexpected Qwen ROI crop in '+case+'; image protocol differs')
    selected,num=choose_candidates(oks,args.max_audits,args.min_signal)
    qwen_scores={int(r['frame']):float(r['verdict']['signal']) for r in oks}
    return {'qwen_scores':qwen_scores,'case':case,'video':video,'qdir':qdir,'config':cfg,'prompt':prompt,
            'prompt_sha':sha(prompt),'reference':reference,'candidates':selected,
            'total_candidates':num,'queries_sha':sha((qdir/'C_queries.json').read_bytes())}


def client_for(args):
    if not os.getenv('VLM2_API_KEY'):
        env=WS/'.env.vlm'
        if env.is_file():
            sys.path.insert(0,str(WS))
            from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
            load_env_file(env)
    if not os.getenv('VLM2_API_KEY'):
        raise RuntimeError('Missing VLM2_API_KEY; use your existing secure env. Do not post key.')
    from openai import OpenAI
    return OpenAI(api_key=os.environ['VLM2_API_KEY'],base_url=args.api_base,
                  timeout=180.0,max_retries=2)


def as_payload(im):
    f=BytesIO(); im.save(f,format='JPEG',quality=90)
    return {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+
             base64.b64encode(f.getvalue()).decode('ascii'),'detail':'high'}}


def run_one(row,args,v3,client):
    info=inspect(row,args)
    case=info['case']; out=args.output/case; selected=info['candidates']
    cfg={'version':VERSION,'case':case,'qwen_queries_sha':info['queries_sha'],
         'original_prompt_sha':info['prompt_sha'],'original_system_sha':sha(v3.SYSTEM_SHARED),
         'frames':selected,'model':args.model,'api_base':args.api_base,
         'image_side':int(info['config']['image_side']),
         'max_tokens':args.max_tokens,'min_signal':args.min_signal}
    cfgpath=out/'config.json'
    if cfgpath.is_file() and json.loads(cfgpath.read_text())!=cfg:
        raise RuntimeError('Output config mismatch. Change --output, do not mix cached judgments.')
    print('CASE:',case,'Qwen qualifying:',info['total_candidates'],
          'SELECTED:',selected,'original prompt SHA:',info['prompt_sha'][:16],flush=True)
    if args.dry_run: return
    save(cfgpath,cfg)
    if not selected:
        save(out/'summary.json',{'case':case,'status':'complete','audited':0,
                 'qwen_candidates':0,'video_signal':0.,'note':'No Qwen proposals.'})
        return
    frames=v3.Frames(info['video'])
    scored=[];errors=[]
    try:
        for j,t in enumerate(selected,1):
            cached=out/'audits'/f'f{t:04d}.json'
            if cached.is_file():
                record=json.loads(cached.read_text())
                if record.get('config_hash')!=sha(json.dumps(cfg,sort_keys=True)):
                    raise ValueError('Cache hash mismatch '+str(cached))
                verdict=record['verdict']
                print('CACHED',case,t,verdict.get('signal'),flush=True)
            else:
                ref=info['reference']
                # Exact original Qwen input construction: two source frames labelled
                # by V3 at the ORIGINAL image resolution and with the ORIGINAL prompt.
                images=[v3.label(frames.frame(i),i,int(info['config']['image_side']))
                        for i in (ref,t)]
                sheet=out/'previews'/f'f{t:04d}.jpg';sheet.parent.mkdir(parents=True,exist_ok=True)
                from PIL import Image
                combined=Image.new('RGB',(images[0].width+images[1].width,
                                          max(images[0].height,images[1].height)))
                combined.paste(images[0],(0,0));combined.paste(images[1],(images[0].width,0))
                combined.save(sheet,quality=90)
                # Original system and original C prompt with original frame placeholders.
                prompt=info['prompt'].format(ref=ref,curr=t)
                try:
                    response=client.chat.completions.create(
                        model=args.model, temperature=0,max_tokens=args.max_tokens,
                        messages=[{'role':'system','content':v3.SYSTEM_SHARED},
                                  {'role':'user','content':[{'type':'text','text':prompt}]+
                                   [as_payload(im) for im in images]}])
                    raw=response.choices[0].message.content or ''
                    verdict=v3.parse_verdict(raw,'C')
                    record={'case':case,'frame':t,'reference':ref,
                            'original_prompt_sha':info['prompt_sha'],'verdict':verdict,
                            'raw':raw,'preview':str(sheet),
                            'config_hash':sha(json.dumps(cfg,sort_keys=True))}
                    save(cached,record)
                    print(f'GEMINI {case} {j}/{len(selected)} f{t:04d} signal={verdict["signal"]:.2f} '
                          f'{verdict.get("observation", "")[:110]}',flush=True)
                except Exception as e:
                    errors.append({'frame':t,'error':str(e)[:300]})
                    print('GEMINI ERROR',case,t,type(e).__name__,str(e)[:300],flush=True)
                    continue
            scored.append({'frame':t,'qwen_signal':info['qwen_scores'][t],
                           'gemini_signal':verdict['signal'],'signal':verdict['signal'],
                           'occlusion_location':verdict.get('occlusion_location'),
                           'observation':verdict.get('observation','')})
    finally:frames.close()
    status='complete' if not errors and len(scored)==len(selected) else 'incomplete'
    summary={'case':case,'status':status,'qwen_candidates':info['total_candidates'],
             'selected_frames':selected,'audited':len(scored),'per_frame':scored,
             'video_signal':max((x['signal'] for x in scored),default=None) if status=='complete' else None,
             'errors':errors,
             'note':'Original Type C visual candidate score, not a calibrated penetration probability.'}
    save(out/'summary.json',summary)
    print('SUMMARY',case,'status',status,'video_signal',summary['video_signal'],flush=True)
    if errors:raise RuntimeError('Some Gemini calls failed; cache is resumable.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'penetration_typeC_pilot_manifest.csv')
    p.add_argument('--qwen-root',type=Path,default=ROOT/'typeC_qwen_multiframe_benchmark_v1_20261009'/'qwen')
    p.add_argument('--v3',type=Path,default=ROOT/'penetration_abc_agents_v3.py')
    p.add_argument('--output',type=Path,default=ROOT/'typeC_gemini_original_prompt_v1_20261009')
    p.add_argument('--case',default='Cosmos3_seed101_0048',help='Specific case or ALL')
    p.add_argument('--model',default=MODEL)
    p.add_argument('--api-base',default='https://api.302.ai/v1')
    p.add_argument('--max-audits',type=int,default=12)
    p.add_argument('--min-signal',type=float,default=.75)
    p.add_argument('--max-tokens',type=int,default=1300)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.max_audits<1 or a.max_audits>50:p.error('--max-audits must be 1..50')
    rows=read_manifest(a.manifest)
    rows=[r for r in rows if a.case=='ALL' or r['case_id']==a.case]
    if not rows:p.error('Case not found in manifest '+a.case)
    v3=load_v3(a.v3)
    cli=None
    if not a.dry_run:cli=client_for(a)
    failures=[]
    for row in rows:
        try:run_one(row,a,v3,cli)
        except Exception as e:
            failures.append((row['case_id'],str(e)))
            print('CASE FAILED',row['case_id'],type(e).__name__,str(e)[:300],flush=True)
    if failures:
        print('FAILURES:',failures,flush=True)
        sys.exit(2)

if __name__=='__main__':main()
