#!/usr/bin/env python3
"""Offline, label-blind Type C Gemini validation from cached Qwen V3-C proposals.

Does not run Qwen or select candidates using GT. Audits fixed representative
reference-vs-current frames. Checks solid material vs a legal open drawer.
"""
from __future__ import annotations
import argparse
import base64
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re

PROMPT = '''You are an INDEPENDENT visual auditor of generated robot manipulation video frames.
The four images are ORIGINAL source frames, each labeled with its frame number, in time order:
reference frame {ref}, local context frames {a}, {b}, and candidate frame {t}.

Monitor exactly one actor: {actor}.
Solid object: {target}. Candidate solid boundary: {boundary}.
The names are object IDENTIFICATION only, not a claim that any failure occurs.

TASK: Determine whether the actor visibly INTERSECTS SOLID MATERIAL, rather than
legitimately extending through an EMPTY OPENING or simply becoming hidden behind
an intact foreground panel. The last frame may show part of the actor inside and
part outside; that alone DOES NOT prove penetration. A drawer has open cavities,
solid front and side panels. Check which part of the drawer actually meets the
banana/actor, using only visible pixels. Also distinguish foreshortening, normal
occlusion, gripper occlusion, viewpoint changes, bending of a deformable object,
and visual truncation. The first frame is for appearance identity, not proof of
constant pose or length; do not compare absolute projected lengths naively.

Describe WHERE the outside portion terminates, whether it points into solid
material, whether a legitimate passage through the open cavity explains it,
and what would be required to infer a true impossible intersection. If the
camera view cannot decide, respond UNCLEAR; do not assume penetration.

Return ONLY a JSON object with these keys:
{{"actor_identified":"YES|NO|UNCLEAR",
"target_identified":"YES|NO|UNCLEAR",
"solid_boundary_visible":"YES|NO|UNCLEAR",
"outside_segment_visible":"YES|NO|UNCLEAR",
"appears_cut_at_solid_material":"YES|NO|UNCLEAR",
"open_cavity_path_supported":"YES|NO|UNCLEAR",
"ordinary_occlusion_possible":"YES|NO|UNCLEAR",
"truncation_vs_reference":"YES|NO|UNCLEAR",
"assessment":"SOLID_PENETRATION|LEGAL_OPENING|ORDINARY_OCCLUSION|UNCERTAIN|NO_RELEVANT_INTERACTION",
"visual_inconsistency_score":0.0,
"reason":"specific visible evidence, including the solid panel vs opening distinction"}}
Score from 0 to 1 as UNCALIBRATED evidence strength, not a probability:
0-0.15 normal geometry/legal opening; 0.2-0.45 ambiguous occlusion;
0.5-0.7 suspicious solid-boundary overlap; 0.75-1 strongly evidenced penetration.
A mere part-inside/part-outside configuration MUST NOT receive a high score
unless the intersection is specifically at SOLID MATERIAL.
'''


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def json_from_response(response):
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.S)
    for m in re.finditer(r'\{', response):
        try:
            data,_ = json.JSONDecoder().raw_decode(response[m.start():])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    raise ValueError('Gemini response contained no JSON: ' + response[:250])


def find_runs(times):
    runs=[]
    for t in sorted(set(times)):
        if not runs or t > runs[-1][-1] + 1:
            runs.append([t])
        else:
            runs[-1].append(t)
    return runs


def select_frames(times, budget):
    """Fixed rule independent of labels or case names; spans sustained + early events."""
    if not times or budget <= 0:
        return []
    times=sorted(set(times))
    runs=sorted(find_runs(times), key=lambda r:(-len(r),r[0]))
    selected=[]
    def add(t):
        if t not in selected and len(selected)<budget:
            selected.append(t)
    # Audit mids of top runs, avoiding one long high-score streak monopolizing budget.
    for run in runs[:min(len(runs), max(1,budget//2))]:
        add(run[len(run)//2])
    # The longest sustained run may change its appearance from beginning to end.
    if runs:
        add(runs[0][0]);add(runs[0][-1])
    # Remaining budget: quantiles of ALL candidate timestamps, including isolated events.
    if budget>len(selected):
        while len(selected)<min(budget,len(times)):
            t=max((t for t in times if t not in selected),
                  key=lambda t:(min(abs(t-s) for s in selected) if selected else 10**9,-t))
            add(t)
    return sorted(selected)


def get_frame_images(video, indices, side):
    import cv2
    from PIL import Image, ImageDraw
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('Could not open video: '+str(video))
    images=[]
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ok,raw=cap.read()
            if not ok:raise RuntimeError('Cannot read source frame '+str(idx))
            x=Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))
            x.thumbnail((side,side),Image.Resampling.LANCZOS)
            result=Image.new('RGB',(x.width,x.height+42),'white')
            result.paste(x,(0,42))
            ImageDraw.Draw(result).text((10,12),f'SOURCE FRAME {idx}',fill='black')
            images.append(result)
    finally:
        cap.release()
    return images


def make_sheet(images):
    from PIL import Image
    w=max(x.width for x in images);h=max(x.height for x in images)
    sheet=Image.new('RGB',(w*2,h*2),'white')
    for i,x in enumerate(images):sheet.paste(x,((i%2)*w,(i//2)*h))
    return sheet


def audited_score(d):
    needed=['actor_identified','target_identified','solid_boundary_visible',
            'appears_cut_at_solid_material','open_cavity_path_supported',
            'ordinary_occlusion_possible','assessment','visual_inconsistency_score']
    if any(k not in d for k in needed):
        raise ValueError('Gemini missing required fields: '+str([k for k in needed if k not in d]))
    raw=float(d['visual_inconsistency_score'])
    if not 0 <= raw <= 1:
        raise ValueError('Invalid score outside 0..1')
    if d['actor_identified']!='YES' or d['target_identified']!='YES':
        return 0.0,'identity_unverified'
    if d['solid_boundary_visible']=='NO':
        return 0.0,'solid_boundary_unavailable'
    if d['solid_boundary_visible']=='UNCLEAR':
        return min(raw,0.45),'boundary_uncertain'
    # Observed legal-opening explanation with no evidence of solid contact.
    if d['open_cavity_path_supported']=='YES' and d['appears_cut_at_solid_material']=='NO':
        return min(raw,0.15),'legal_opening_no_solid_intersection'
    if d['assessment']=='NO_RELEVANT_INTERACTION':
        return min(raw,0.15),'no_relevant_interaction'
    if d['assessment']=='LEGAL_OPENING' and d['appears_cut_at_solid_material']!='YES':
        return min(raw,0.15),'legal_opening'
    return raw,'audited'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--qwen-dir',type=Path,required=True,help='Completed Type C case output with C_queries.json')
    p.add_argument('--agent0-json',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='gemini-3.8-flash')
    p.add_argument('--api-base',default='https://api.302.ai/v1')
    p.add_argument('--max-audits',type=int,default=6)
    p.add_argument('--min-qwen-signal',type=float,default=.75)
    p.add_argument('--image-side',type=int,default=960)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not a.video.is_file(): p.error('Missing video: '+str(a.video))
    cfile=a.qwen_dir/'C_queries.json'
    if not cfile.is_file(): p.error('Missing Qwen C_queries.json: '+str(cfile))
    if not a.agent0_json.is_file():p.error('Missing Agent0: '+str(a.agent0_json))
    if not 1<=a.max_audits<=20:p.error('max-audits must be 1..20')
    q=json.loads(cfile.read_text())
    metadata=json.loads((a.qwen_dir/'run_config.json').read_text())
    if metadata.get('case')!=a.case or Path(metadata['video']).resolve()!=a.video.resolve():
        raise ValueError('Case/video mismatch with saved Qwen run')
    rows=q.get('rows',[])
    ref=int(metadata['reference']);start=int(metadata['start']);end=int(metadata['end'])
    seen={int(r['frame']) for r in rows if r.get('status')=='ok'}
    expected=set(range(start,end+1))-{ref}
    if seen!=expected:
        raise ValueError(f'Incomplete Qwen C scan: got {len(seen)}/{len(expected)}; do not audit partial run')
    selected_a0=json.loads(a.agent0_json.read_text())['selected']
    if selected_a0!=metadata['agent0']:
        raise ValueError('Agent0 selection differs from saved Qwen config')
    qualifying=[int(r['frame']) for r in rows if r.get('status')=='ok' and
                float(r['verdict'].get('signal',0))>=a.min_qwen_signal]
    chosen=select_frames(qualifying,a.max_audits)
    print('CASE:',a.case,'reference:',ref, 'QWEN positive candidate frames:',len(qualifying))
    print('SELECTED FOR GEMINI:',chosen)
    if a.dry_run:
        return
    if not os.environ.get('VLM2_API_KEY'):
        raise RuntimeError('VLM2_API_KEY missing. Export your API token before running.')
    from openai import OpenAI
    client=OpenAI(api_key=os.environ['VLM2_API_KEY'],base_url=a.api_base,timeout=180.0,max_retries=2)
    out=a.output/a.case
    out.mkdir(parents=True,exist_ok=True)
    configuration={'case':a.case,'video':str(a.video.resolve()),'qwen_config':metadata,
         'qwen_file_sha256':hashlib.sha256(cfile.read_bytes()).hexdigest(),
         'agent0':selected_a0,'prompt_sha256':sha(PROMPT),'model':a.model,'api_base':a.api_base,
         'min_qwen_signal':a.min_qwen_signal,'max_audits':a.max_audits,
         'image_side':a.image_side,'max_tokens':a.max_tokens,
         'selection_policy':'longest_runs_midpoints_start_end_then_farthest_timestamp_v1'}
    cfg=out/'audit_config.json'
    if cfg.is_file() and json.loads(cfg.read_text())!=configuration:
        raise RuntimeError('Configuration differs: use another --output to prevent stale Gemini cache')
    save(cfg,configuration)
    save(out/'selected_candidates.json',{'frames':chosen,'n_qualifying':len(qualifying),
        'qwen_threshold':a.min_qwen_signal})
    if not chosen:
        save(out/'audit_summary.json',{'case':a.case,'status':'complete','n_candidates':0,
            'n_audited':0,'video_score':0.0})
        print('No Qwen candidates; 0 score under fixed screening rule.')
        return
    scores=[];failures=[]
    from PIL import Image
    for j,t in enumerate(chosen,1):
        f=out/'gemini'/f'C_f{t:04d}.json'
        if f.is_file():
            record=json.loads(f.read_text())
            print(f'GEMINI CACHED f{t:04d}: {record["score"]:.3f}',flush=True)
        else:
            indices=[ref,max(0,t-2),max(0,t-1),t]
            images=get_frame_images(a.video,indices,a.image_side)
            preview=out/'gemini'/f'C_f{t:04d}_context.jpg'
            preview.parent.mkdir(parents=True,exist_ok=True)
            make_sheet(images).save(preview,quality=92)
            prompt=PROMPT.format(ref=ref,a=indices[1],b=indices[2],t=t,
                actor=selected_a0['actor'],target=selected_a0['target'],boundary=selected_a0['solid_boundary'])
            content=[{'type':'text','text':prompt}]
            for im in images:
                buf=BytesIO();im.save(buf,format='JPEG',quality=88)
                base=base64.b64encode(buf.getvalue()).decode('ascii')
                content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base,'detail':'high'}})
            try:
                resp=client.chat.completions.create(model=a.model,
                    messages=[{'role':'user','content':content}],
                    temperature=0,max_tokens=a.max_tokens)
                answer=resp.choices[0].message.content or ''
                verdict=json_from_response(answer)
                score,status=audited_score(verdict)
                record={'case':a.case,'frame':t,'source_frames':indices,'score':score,
                    'status':status,'verdict':verdict,'raw_response':answer,
                    'visual':str(preview),'config_sha256':sha(json.dumps(configuration,sort_keys=True))}
                save(f,record)
                print(f'GEMINI {j}/{len(chosen)} f{t:04d} score={score:.3f} {status} '
                      f"assessment={verdict.get('assessment')} reason={verdict.get('reason','')[:125]}",flush=True)
            except Exception as exc:
                failures.append({'frame':t,'error':f'{type(exc).__name__}: {exc}'})
                print('GEMINI ERROR f',t,':',type(exc).__name__,str(exc)[:300],flush=True)
                continue
        scores.append({'frame':t,'score':float(record['score']),'status':record['status'],
            'assessment':record['verdict'].get('assessment')})
    summary={'case':a.case,'status':'complete' if not failures and len(scores)==len(chosen) else 'incomplete',
       'n_qwen_candidates':len(qualifying),'n_selected':len(chosen),'n_audited':len(scores),
       'failed':failures,'selected_frame_scores':scores,
       'video_score':max((r['score'] for r in scores),default=None) if not failures else None,
       'score_scope':'fixed-sample Qwen candidate audit; not full-video ground-truth penetration probability'}
    save(out/'audit_summary.json',summary)
    print('AUDIT SUMMARY:',json.dumps(summary,indent=2),flush=True)
    if failures:raise RuntimeError('Gemini audits incomplete; cached successful requests preserved')

if __name__=='__main__':
    main()
