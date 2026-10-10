#!/usr/bin/env python3
"""Type A reverse Gemini soft-score V2: original two-source-frame format.

For each adjacent frame pair send FOUR images in ONE request: full before,
full after, same-ROI crop before, same-ROI crop after. Use the original Type-A
continuous scoring philosophy; adapt only actor identity and IN->OUT direction.

Modes:
  diagnostic: explicit source-frame focus ends and Qwen-zero controls (development only)
  topk: rank Qwen *raw* candidates across a complete full-video scan, NMS,
        top 12, Gemini independently judges each; final=max(qraw*G).

Never send Qwen confidence, control membership, labels or annotated frame ranges
into the Gemini prompt. This script never changes original frozen GitHub files.

On ERIS CPU compute node; Gemini uses the existing Wilson cloud credentials.
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
import random
import sys
import time
from pathlib import Path

ROOT = Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
DEFAULT_VIDEO = Path('/scratch/z/zy992/zhuoying/physact/cosmos3/export_robowm68_3seeds/seed103/0053.mp4')
DEFAULT_SOURCE = ROOT/'typeA_reverse_v3_probe_seed103_0053_f135_188'/'Cosmos3_seed103_0053'/'A_queries.json'
VERSION = 'reverse_gemini_soft_original_philosophy_v2_20261010'

# Faithful to the original GitHub Type-A Gemini prompt structure and score
# semantics. Adaptations: robot gripper is the named moving target; drawer front
# is the relevant solid wall; legal exit vs legal top entry; visual un-occlusion.
# No exemplar, ground-truth label, event frame, Qwen decision or score.
PROMPT = '''Four images are provided in this exact order:
1. FULL BEFORE (source frame {before})
2. FULL AFTER (source frame {after}, the next consecutive frame)
3. CROP BEFORE (magnified ROI of frame {before})
4. CROP AFTER (the EXACT SAME spatial ROI of frame {after})

MOVING TARGET: the same robot gripper / end-effector, especially its jaws and fingertips (NOT any held cube).
ENVIRONMENT WITH SOLID WALLS: the drawer/container, especially its SOLID FRONT PANEL; a real open cavity or rim clearance is NOT solid material.
The robot GRIPPER IS the named moving target. Ignore unrelated objects.

IMPORTANT: Your job is NOT first to classify the gripper OUTSIDE or INSIDE.
Do NOT output score zero merely because the gripper is near or partly hidden by
 the drawer in both frames. Instead observe exactly what is visible:

(A) In EACH image identify the SAME gripper fingertips, their visible boundary,
    the drawer's front SOLID panel, upper edge/rim, and genuine opening.
(B) Compare the SAME gripper jaw/tip and silhouette in both crops.
    Has a portion reappeared from behind a solid panel or become less occluded?
    Does the visible fingertip move relative to the panel? Has it moved high
    enough to CLEAR the rim or pass through a genuine opening?
(C) Consider the gripper's visible tip relative to the FRONT PANEL and its rim
    in BOTH frames; do not confuse 2D image coordinates with true 3D depth.
(D) If the gripper visibly exits ABOVE/AROUND the rim or through an actual
    opening, that is a LEGAL exit and should receive a LOW inconsistency score.
(E) Normal occlusion, transparency, or perspective may explain apparent
    reappearance or projection across the panel. If so lower the score and
    explain why. But a gripper emerging in FRONT of a solid panel while its
    tip is still below the rim can be suspicious, even if you cannot assign
    unambiguous INSIDE/OUTSIDE depth states from these two frames.
(F) Do not infer penetration merely because the gripper overlaps the drawer.
    Do not assume it used an opening solely because an opening exists elsewhere.

Return a CONTINUOUS VISUAL INCONSISTENCY SCORE in [0,1]:
0.00 = clear no abnormality; 0.1-0.3 = likely normal; 0.4-0.6 = unresolved;
0.7-0.9 = substantial physical-inconsistency evidence; 1.0 = exceptionally strong evidence.
This score expresses visual support for abnormal gripper-solid-panel interaction,
NOT the probability that a physical collision was mathematically proven.
Neither a strict INSIDE->OUTSIDE transition nor a hard binary crossing verdict is required.
Do not automatically reward a proposal; ground the score in what you SEE.

Return exactly one JSON object with keys:
{{
  "target_identified": "YES|NO|UNCLEAR",
  "solid_panel_identified": "YES|NO|UNCLEAR",
  "target_lower_part_before": "brief grounded visible observation of gripper tip",
  "target_lower_part_after": "brief grounded visible observation of gripper tip",
  "rim_relation_before": "BELOW|ABOVE|STRADDLING|UNCLEAR",
  "rim_relation_after": "BELOW|ABOVE|STRADDLING|UNCLEAR",
  "lower_contour_change": "INCREASING_OCCLUSION|DECREASING_OCCLUSION|SIMILAR|OTHER|UNCLEAR",
  "legal_exit_evidence": "CLEAR|PLAUSIBLE|NOT_OBSERVED|UNCLEAR",
  "ordinary_occlusion_evidence": "CLEAR|PLAUSIBLE|NOT_OBSERVED|UNCLEAR",
  "inside_to_outside_change": "YES|NO|UNCLEAR",
  "through_solid_panel_evidence": "YES|NO|UNCLEAR",
  "movement_evidence": "specific observed changes, or unclear",
  "visual_inconsistency_score": 0.0,
  "reason": "2-4 concise sentences: actual observed boundary changes, why normal or suspicious, and limitations"
}}'''


def save_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sha_json(o):
    return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def qwen_rows(source):
    d=json.loads(Path(source).read_text())
    if d.get('agent') not in (None,'A'):raise ValueError('Source is not Type-A Qwen cache')
    rows={}
    for r in d.get('rows',[]):
        if r.get('status')!='ok':continue
        t=int(r['frame']); b=int(r.get('prev',t-1))
        if b!=t-1:raise ValueError(f'Non-adjacent Qwen record: f{b}->f{t}')
        if t in rows:raise ValueError(f'Duplicate Qwen frame f{t}')
        v=r.get('verdict') or {}
        raw=float(v.get('raw_transition_signal',v.get('signal',0)))
        post=float(v.get('signal',0))
        if not (math.isfinite(raw) and 0<=raw<=1 and math.isfinite(post) and 0<=post<=1):
            raise ValueError(f'Invalid Qwen score f{t}')
        rows[t]={'before':b,'after':t,'qwen_raw':raw,'qwen_post':post,
                 'qwen_conflict':bool(v.get('shared_frame_conflict',False))}
    if not rows:raise ValueError('No completed Type-A Qwen records')
    return d,rows


def select_diagnostic(rows,focus_ends,controls,buffer,seed):
    f=[]
    for t in focus_ends:
        if t not in rows:raise ValueError(f'Focus frame f{t} missing from Qwen cache')
        f.append({**rows[t],'group':'focus'})
    positive=[t for t,r in rows.items() if r['qwen_raw']>0]
    eligible=[t for t,r in rows.items() if r['qwen_raw']==0 and t not in focus_ends
              and all(abs(t-u)>buffer for u in positive)]
    if controls>len(eligible):raise ValueError(f'{controls} controls requested but only {len(eligible)} eligible')
    selected_controls=random.Random(seed).sample(sorted(eligible),controls)
    c=[{**rows[t],'group':'qwen_zero_control'} for t in selected_controls]
    chosen=[]
    for i in range(max(len(f),len(c))):
        if i<len(f):chosen.append(f[i])
        if i<len(c):chosen.append(c[i])
    return chosen,{'qwen_zero_pool':len(eligible)}


def select_topk(rows,topk,nms_radius):
    # Rank all >0 proposals, not a chosen numeric threshold; 0 * Gemini = 0.
    positives=sorted((r for r in rows.values() if r['qwen_raw']>0),
                     key=lambda r:(-r['qwen_raw'], r['after']))
    kept=[]
    for r in positives:
        if any(abs(r['after']-k['after'])<=nms_radius for k in kept):continue
        kept.append({**r,'group':'topk'})
        if len(kept)==topk:break
    return sorted(kept,key=lambda x:x['after']),{'positive_qwen_windows':len(positives)}


def require_full_video(source_doc,rows,video_n):
    frames=set(rows)
    if (source_doc.get('start') not in (0,1) or
        source_doc.get('end')!=video_n-1 or
        frames != set(range(1,video_n))):
        raise ValueError('Top-k formal mode REQUIRES complete full-video Qwen cache covering pairs f0->1 through f(n-2)->(n-1). No hand-selected event window is allowed.')


def credentials(env_file):
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
    from robot.preprocessing.link7_persistent.interface.config import role_config
    load_env_file(env_file)
    conf=role_config('vlm2')
    if conf.get('backend')!='cloud_api':raise RuntimeError('Wilson vlm2 backend must be cloud_api')
    keyvar=conf.get('api_key_env','VLM2_API_KEY')
    key=os.environ.get(keyvar)
    if not key:raise RuntimeError(f'{keyvar} not configured in {env_file}')
    return conf,key


def send(images,prompt,system,key,base,model,max_tokens,timeout):
    import requests
    content=[{'type':'text','text':prompt}]
    kilobytes=[]
    for im in images:
        buf=io.BytesIO();im.convert('RGB').save(buf,'JPEG',quality=85,optimize=True)
        dat=buf.getvalue();kilobytes.append(round(len(dat)/1024,1))
        content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(dat).decode('ascii')}})
    if len(images)!=4:raise RuntimeError('Expected original 4 images from exactly 2 source frames')
    payload={'model':model,'messages':[{'role':'system','content':system},
        {'role':'user','content':content}], 'temperature':0,'max_tokens':max_tokens}
    start=time.monotonic()
    print(f'  SEND model={model} images=4 size_KB={kilobytes} max_tokens={max_tokens} timeout={timeout}s retries=0',flush=True)
    try:
        reply=requests.post(str(base).rstrip('/')+'/chat/completions',json=payload,
                            headers={'Authorization':'Bearer '+key},timeout=(10,timeout))
    except requests.RequestException as e:
        print('  NETWORK ERROR:',str(e)[:280],flush=True)
        raise
    elapsed=time.monotonic()-start
    print(f'  HTTP={reply.status_code} elapsed={elapsed:.2f}s',flush=True)
    if not reply.ok:raise RuntimeError(f'Provider HTTP {reply.status_code}: {reply.text[:500]}')
    result=reply.json()
    msg=result['choices'][0]['message']['content']
    raw=msg if isinstance(msg,str) else '\n'.join(x.get('text','') for x in msg if isinstance(x,dict))
    finish_reason=result.get('choices',[{}])[0].get('finish_reason')
    print(f'  RESPONSE chars={len(raw)} finish_reason={finish_reason!r}',flush=True)
    if finish_reason in ('length','max_tokens'):
        print('  WARNING: provider reports response truncated at output-token limit',flush=True)
    return raw,elapsed,result.get('usage'),kilobytes,finish_reason


def summary(mode,all_selected,done,meta,complete):
    valid=[r for r in done if r.get('status')=='ok']
    max_g=max((r['gemini_identity_gated_score'] for r in valid),default=0.0)
    max_raw=max((r['raw_weighted'] for r in valid),default=0.0)
    max_post=max((r['post_weighted'] for r in valid),default=0.0)
    return {'version':VERSION,'mode':mode,'complete':complete,
            'diagnostic_only':mode=='diagnostic',
            'scored_window_count':len(valid),'selected_window_count':len(all_selected),
            'selection':all_selected,'selection_metadata':meta,
            'audited_max_gemini':max_g,
            'audited_max_raw_weighted':max_raw,
            'audited_max_post_weighted':max_post,
            'video_score':max_raw if complete and mode=='topk' else None,
            'note':'All scores are visual-inconsistency indicators, not calibrated physical penetration probabilities. Diagnostic scores are not complete-video AUROC eligible.'}


def self_test():
    q={t:{'before':t-1,'after':t,'qwen_raw':1. if t in [154,155,156,160,161,166,185] else 0.,
          'qwen_post':0.25 if t in [154,155,156] else 0.,'qwen_conflict':t in [154,155,156]}
       for t in range(135,189)}
    diag,meta=select_diagnostic(q,[154,155,156],3,3,43)
    assert len(diag)==6 and sum(r['group']=='focus' for r in diag)==3
    assert sum(r['group']=='qwen_zero_control' for r in diag)==3
    assert meta['qwen_zero_pool']>3
    sel,meta2=select_topk(q,12,2)
    assert [r['after'] for r in sel]==[154,160,166,185],sel
    assert meta2['positive_qwen_windows']==7
    assert not any('focus' in PROMPT for _ in [0])
    assert 'Do NOT output score zero merely because' in PROMPT
    assert 'Neither a strict INSIDE->OUTSIDE transition nor a hard binary crossing verdict is required' in PROMPT
    assert '0.1-0.3 = likely normal' in PROMPT
    assert '{before}' in PROMPT and '{after}' in PROMPT
    assert sha_json({'a':[1,2]})==sha_json({'a':(1,2)})
    try:
        require_full_video({'start':135,'end':188},q,189)
    except ValueError:pass
    else:raise AssertionError('Partial scan incorrectly accepted as full video')
    full={t:{'before':t-1,'after':t,'qwen_raw':0.,'qwen_post':0.,'qwen_conflict':False} for t in range(1,189)}
    require_full_video({'start':0,'end':188},full,189)
    print('SELF-TEST PASS: original-soft-style reverse prompt, diagnostic/controls, all-positive topk/NMS, full-video guard and normalized config hashing')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('diagnostic','topk'),default='diagnostic')
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--video',type=Path,default=DEFAULT_VIDEO)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=ROOT/'typeA_reverse_gemini_soft_v2_0053')
    p.add_argument('--focus-ends',default='154,155,156',help='DIAGNOSTIC ONLY: Qwen after-frame labels; e.g. 154 means f153->154')
    p.add_argument('--controls',type=int,default=3)
    p.add_argument('--control-buffer',type=int,default=3)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--topk',type=int,default=12)
    p.add_argument('--nms-radius',type=int,default=2)
    p.add_argument('--roi-norm',default=None,help='Diagnostic manual/fixed ROI normalized x1,y1,x2,y2 (not allowed in topk mode)')
    p.add_argument('--model',default='gemini-3.8-flash')
    p.add_argument('--max-tokens',type=int,default=900)
    p.add_argument('--timeout',type=float,default=85.)
    p.add_argument('--full-side',type=int,default=960)
    p.add_argument('--crop-side',type=int,default=1000)
    p.add_argument('--api-base',default=None)
    p.add_argument('--env-file',type=Path,default=WS/'.env.vlm')
    p.add_argument('--max-calls',type=int,default=0,help='0=all; >0 cap requests for staged run, same output allows resuming without paying twice')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:return self_test()
    if args.topk<1 or args.nms_radius<0 or args.controls<0 or args.control_buffer<0 or args.max_calls<0 or args.timeout<=0:
        p.error('Invalid selection, controls, timeout or call count')
    if args.mode=='topk' and args.roi_norm is not None:
        p.error('Manually specified ROI forbidden in full-video topk mode. Automatic scene ROI can be added as separately validated variant.')
    d,rows=qwen_rows(args.source)
    if args.mode=='diagnostic':
        focus=[int(x) for x in args.focus_ends.split(',') if x.strip()]
        if not focus or len(set(focus))!=len(focus):p.error('Provide unique diagnostic --focus-ends')
        chosen,meta=select_diagnostic(rows,focus,args.controls,args.control_buffer,args.seed)
    else:
        # Full video verified before *any* Gemini call (after metadata video access).
        import cv2
        cap=cv2.VideoCapture(str(args.video)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        if n<2:raise ValueError('Video unavailable or fewer than two frames')
        require_full_video(d,rows,n)
        chosen,meta=select_topk(rows,args.topk,args.nms_radius)
    print('SOURCE:',args.source,flush=True)
    print('VIDEO:',args.video,flush=True)
    print('SOURCE WINDOW:',d.get('start'),d.get('end'), 'QWEN ROWS:',len(rows),flush=True)
    print('MODE:',args.mode,'TOPK' if args.mode=='topk' else 'DEVELOPMENT DIAGNOSTIC',flush=True)
    print('SELECTION:',len(chosen),'windows; meta:',meta,flush=True)
    for c in chosen:
        print(f" f{c['before']:04d}->f{c['after']:04d} {c['group']} raw={c['qwen_raw']:.2f} post={c['qwen_post']:.2f} conflict={c['qwen_conflict']}",flush=True)
    if args.dry_run:
        print('DRY RUN COMPLETE (no API calls or output writes)');return
    if not args.video.exists():raise FileNotFoundError(args.video)
    sys.path.insert(0,str(args.root))
    import penetration_A_v3_crop_gemini_score as audit
    roi=audit.parse_roi(args.roi_norm) if args.roi_norm is not None else (0,0,1,1)
    # Use JSON-round-trip canonicalization (tuple/list issue fixed).
    cfg={'version':VERSION,'mode':args.mode,'source':str(args.source.resolve()),
         'source_sha':hashlib.sha256(args.source.read_bytes()).hexdigest(),
         'video':str(args.video.resolve()),'video_size':args.video.stat().st_size,
         'selection':chosen,'metadata':meta,'roi_norm':list(roi),
         'full_side':args.full_side,'crop_side':args.crop_side,'max_tokens':args.max_tokens,
         'model':args.model,'prompt':PROMPT,'system':audit.SYSTEM,'api_base':args.api_base,
         'nms_radius':args.nms_radius if args.mode=='topk' else None,
         'topk':args.topk if args.mode=='topk' else None}
    cfgfile=args.output/'run_config.json'
    if cfgfile.exists():
        previous=json.loads(cfgfile.read_text())
        if previous.get('hash')!=sha_json(cfg):
            raise RuntimeError('Saved output config differs; use new --output, or restore original settings. Cached Gemini calls untouched.')
    else:save_json(cfgfile,{'hash':sha_json(cfg),'config':cfg})
    if not chosen:
        out=summary(args.mode,chosen,[],meta,True)
        save_json(args.output/'diagnostic_summary.json',out)
        print('NO NONZERO QWEN CANDIDATES, video_score=0 (full Qwen scan verified).')
        return
    conf,key=credentials(args.env_file)
    base=args.api_base or conf.get('api_base')
    if not str(base).startswith('https://'):raise RuntimeError(f'Expected HTTPS API base, got {base}')
    print('DIRECT API:',base,'model:',args.model,'(credentials hidden)',flush=True)
    frames=audit.Frames(args.video)
    chosen_run=chosen[:args.max_calls] if args.max_calls else chosen
    results=[]
    try:
        for idx,c in enumerate(chosen_run,1):
            b,t=c['before'],c['after']; path=args.output/f'gemini_f{b:04d}_f{t:04d}.json'
            if path.is_file():
                saved=json.loads(path.read_text())
                if saved.get('config_hash')!=sha_json(cfg):raise RuntimeError('Cached verdict config mismatch: '+str(path))
                print(f'[{idx}/{len(chosen_run)}] CACHED f{b}->{t} G={saved["gemini_identity_gated_score"]:.3f}',flush=True)
                item=saved
            else:
                print(f'[{idx}/{len(chosen_run)}] VERIFY f{b}->{t} ...',flush=True)
                images, box=audit.prepared_images(frames,b,t,roi,args.full_side,args.crop_side)
                preview=args.output/f'images_f{b:04d}_f{t:04d}.jpg'
                preview.parent.mkdir(parents=True,exist_ok=True)
                audit.contact_sheet(images).save(preview,quality=88)
                raw,secs,usage,im_kb,finish_reason=send(images,PROMPT.format(before=b,after=t),
                              audit.SYSTEM,key,base,args.model,args.max_tokens,args.timeout)
                # Persist successful HTTP response BEFORE parsing: never waste an API call.
                rawfile=args.output/f'api_response_f{b:04d}_f{t:04d}.json'
                save_json(rawfile,{'before':b,'after':t,'model':args.model,
                                  'max_tokens':args.max_tokens,'finish_reason':finish_reason,
                                  'elapsed_s':round(secs,3),'usage':usage,
                                  'response_text':raw})
                try:
                    verdict=audit.parse_json_object(raw)
                except Exception as parse_error:
                    print(f'  PARSE ERROR: {parse_error.__class__.__name__}; '
                          f'raw saved at {rawfile}; finish_reason={finish_reason!r}',flush=True)
                    raise RuntimeError(f'Gemini response could not be parsed. Full raw saved to {rawfile}') from parse_error
                g=audit.score_of(verdict)
                identity=str(verdict.get('target_identified','UNCLEAR')).strip().upper()
                solid=str(verdict.get('solid_panel_identified','UNCLEAR')).strip().upper()
                gated=g if identity=='YES' else 0.0
                item={**c,'status':'ok','config_hash':sha_json(cfg),'verdict':verdict,
                     'gemini_raw_score':g,'gemini_identity_gated_score':gated,
                     'raw_weighted':round(c['qwen_raw']*gated,5),
                     'post_weighted':round(c['qwen_post']*gated,5),
                     'target_identified':identity,'solid_panel_identified':solid,
                     'raw_response':raw,'finish_reason':finish_reason,'latency_s':round(secs,3),'usage':usage,
                     'image_kb':im_kb,'roi_box':list(box),'image_preview':str(preview)}
                save_json(path,item)
                print(f'  G={g:.3f} identity={identity} panel={solid} RAWxG={item["raw_weighted"]:.3f} '
                      f"change={verdict.get('inside_to_outside_change','?')} solid={verdict.get('through_solid_panel_evidence','?')}",flush=True)
                print('  REASON:',verdict.get('reason',''),flush=True)
            results.append(item)
            save_json(args.output/'interim_summary.json',summary(args.mode,chosen,results,meta,len(results)==len(chosen)))
    finally:frames.close()
    complete=len(results)==len(chosen)
    summ=summary(args.mode,chosen,results,meta,complete)
    save_json(args.output/'diagnostic_summary.json' if args.mode=='diagnostic' else args.output/'video_summary.json',summ)
    with (args.output/'scores.csv').open('w',newline='') as f:
        columns=['before','after','group','qwen_raw','qwen_post','qwen_conflict','gemini_raw_score',
                 'gemini_identity_gated_score','raw_weighted','post_weighted','latency_s',
                 'target_identified','solid_panel_identified','inside_to_outside_change',
                 'through_solid_panel_evidence','reason']
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for item in results:
            verdict=item['verdict']
            w.writerow({k:(verdict.get(k,'') if k in ('inside_to_outside_change','through_solid_panel_evidence','reason')
                        else item.get(k,'')) for k in columns})
    print('DONE',len(results),'/',len(chosen),'score_csv:',args.output/'scores.csv',flush=True)
    print('MAX Gemini:',summ['audited_max_gemini'],'MAX raw*Gemini:',summ['audited_max_raw_weighted'],flush=True)
    if not complete:print('INCOMPLETE: resume with identical args and omit --max-calls',flush=True)
    if args.mode=='diagnostic':
        for group in ['focus','qwen_zero_control']:
            ss=[r['gemini_identity_gated_score'] for r in results if r['group']==group]
            print(group,'scores=',ss,'max=',max(ss,default=0.),flush=True)
    else:print('FULL VIDEO SCORE:',summ['video_score'],flush=True)

if __name__=='__main__':main()
