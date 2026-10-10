#!/usr/bin/env python3
"""Run reproducible Agent0->V3 Type C->Qwen temporal verification across a CSV.

Manifest: case_id,label,video,reuse_qwen_dir,reuse_agent0_json
Label is used for REPORT ONLY; never passed to Qwen or the detector.
Run on an H200 Slurm allocation. Resumable by individual cases / query caches.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
ROOT=Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
DEFAULT_ROOT=ROOT/'typeC_qwen_multiframe_benchmark_v1_20261009'

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)

def rows_from_manifest(path):
    with open(path,newline='') as f:rows=list(csv.DictReader(f))
    required={'case_id','label','video'}
    if not rows or not required.issubset(rows[0]):raise ValueError('Manifest needs case_id,label,video')
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Duplicate case_id')
    for r in rows:
        if r['label'] not in ('0','1'):raise ValueError('Invalid GT label for '+r['case_id'])
        if not r['case_id'].replace('_','').isalnum():raise ValueError('Unsafe case id '+r['case_id'])
    return rows

def auroc(pairs):
    pos=[float(score) for gt,score in pairs if gt==1]
    neg=[float(score) for gt,score in pairs if gt==0]
    if not pos or not neg:return None
    return round(sum(1 if x>y else 0.5 if x==y else 0 for x in pos for y in neg)/(len(pos)*len(neg)),5)

def report(rows,out):
    collected=[]
    for row in rows:
        case=row['case_id'];s=out/'verify'/case/'summary.json'
        item={'case_id':case,'label':int(row['label']),
              'mechanism':row.get('mechanism',''),'status':'not_run',
              'solid_score':'','shape_score':'','combined_score':'','audits':'',
              'n_candidates':''}
        if s.is_file():
            d=json.loads(s.read_text());item['status']=d.get('status','unknown')
            if item['status'] in ('complete','no_supported_pair'):
                item.update(solid_score=d['solid_score'],shape_score=d['shape_score'],
                            combined_score=d['combined_score'],audits=d.get('audits',0),
                            n_candidates=d.get('qwen_candidate_count',0))
        collected.append(item)
    out.mkdir(parents=True,exist_ok=True)
    csvpath=out/'scores.csv'
    with open(csvpath,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(collected[0]));w.writeheader();w.writerows(collected)
    eligible=[r for r in collected if r['status'] in ('complete','no_supported_pair')]
    stats={'requested':len(collected),'completed':len(eligible),
           'benchmark_complete':len(eligible)==len(collected),
           'positive_complete':sum(x['label']==1 for x in eligible),
           'negative_complete':sum(x['label']==0 for x in eligible),
           'failures':[x['case_id'] for x in collected if x not in eligible],
           'agent0_abstentions':[x['case_id'] for x in eligible if x['status']=='no_supported_pair'],
           'solid_panel_auroc':auroc([(r['label'],r['solid_score']) for r in eligible]),
           'shape_loss_auroc':auroc([(r['label'],r['shape_score']) for r in eligible]),
           'combined_auroc':auroc([(r['label'],r['combined_score']) for r in eligible]),
           'note':'Tiny pilot, mixed generator domains; scores are uncalibrated. No claims on generalization.'}
    save(out/'auroc_summary.json',stats)
    print('\n=== TYPE C QWEN BENCHMARK REPORT ===')
    for r in collected:
        print(f"{r['case_id']:26s} GT={r['label']} status={r['status']:19s} solid={r['solid_score']} shape={r['shape_score']} combined={r['combined_score']}")
    print(json.dumps(stats,indent=2))
    return stats

def run(cmd,log):
    print('EXEC:', ' '.join(map(str,cmd)),flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    # Stream as we go and preserve transcripts, without tee/PIPE deadlocks.
    with open(log,'a',buffering=1) as f:
        p=subprocess.Popen([str(x) for x in cmd],stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line);f.write(line)
        return p.wait()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--case',default=None,help='Run only one case for quick pilot')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-audits',type=int,default=8)
    p.add_argument('--image-side',type=int,default=1024)
    p.add_argument('--max-tokens',type=int,default=520)
    a=p.parse_args()
    rows=rows_from_manifest(a.manifest)
    chosen=[r for r in rows if a.case is None or r['case_id']==a.case]
    if not chosen:p.error('Case not found in manifest: '+str(a.case))
    if a.report_only:
        report(rows,a.output);return
    a.output.mkdir(parents=True,exist_ok=True)
    base=ROOT/'penetration_agent0_v3C_multiframe.py'
    verify=ROOT/'penetration_typeC_qwen_verify.py'
    if not a.dry_run:
        for f in (base,verify):
            if not f.exists():raise FileNotFoundError('Expected on ERIS: '+str(f))
    for r in chosen:
        case=r['case_id'];video=Path(r['video'])
        qdir=Path(r['reuse_qwen_dir']) if r.get('reuse_qwen_dir') else a.output/'qwen'/case
        a0=Path(r['reuse_agent0_json']) if r.get('reuse_agent0_json') else qdir/'agent0_multiframe.json'
        smpath=a.output/'verify'/case/'summary.json'
        print('\n===== CASE',case,'===== GT is REPORT-ONLY',flush=True)
        if not video.is_file():
            print('MISSING VIDEO:',video,'(case incomplete)',flush=True)
            continue
        if smpath.is_file() and json.loads(smpath.read_text()).get('status') in ('complete','no_supported_pair'):
            print('COMPLETE CACHED:',case,flush=True);continue
        stage1_complete=(qdir/'C_queries.json').is_file() and (qdir/'typeC_summary.json').is_file()
        if not stage1_complete:
            # Explicit reuse path must be complete; never run scans into an unrelated reused cache.
            if r.get('reuse_qwen_dir'):
                print('REUSE C QUERY INCOMPLETE:',qdir,flush=True);continue
            cmd=[a.python,'-u',base,'--case',case,'--video',video,'--output',a.output/'qwen',
                 '--agent0-samples','5','--reference','0','--start','1',
                 '--image-side',str(a.image_side),'--max-tokens',str(a.max_tokens)]
            if a.dry_run:
                print('WOULD SCAN:', ' '.join(map(str,cmd)),flush=True);continue
            rc=run(cmd,a.output/'logs'/f'{case}_scan.log')
            if rc!=0:
                # If Agent0 explicitly abstained, count as E2E score zero and flag it.
                if a0.is_file() and json.loads(a0.read_text()).get('selected') is None:
                    save(smpath,{'case':case,'status':'no_supported_pair','solid_score':0,
                         'shape_score':0,'combined_score':0,'audits':0,
                         'qwen_candidate_count':0,'note':'Agent0 abstained; conservative end-to-end zero'})
                    print('AGENT0 ABSTAINED:',case,flush=True)
                else:print('SCAN ERROR:',case,'exit',rc,flush=True)
                continue
        cmd=[a.python,'-u',verify,'--case',case,'--video',video,
             '--qwen-dir',qdir,'--agent0-json',a0,'--output',a.output/'verify',
             '--max-audits',str(a.max_audits)]
        if a.dry_run:
            print('WOULD VERIFY:', ' '.join(map(str,cmd)),flush=True);continue
        rc=run(cmd,a.output/'logs'/f'{case}_verify.log')
        if rc!=0:print('VERIFY FAILED:',case,'exit',rc,flush=True)
    if not a.dry_run:report(rows,a.output)

if __name__=='__main__':main()
