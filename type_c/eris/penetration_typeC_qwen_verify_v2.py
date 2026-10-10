#!/usr/bin/env python3
"""Type-C v2: neutral visibility/occlusion test + independent anomaly test.

Reuses complete Type-C Qwen stage-1 cache. NO labels/predetermined event frames are
passed to the VLM. A high score requires positive visual support from both calls.
Uncertain evidence -> low/zero. Keep results separate from Type-C v1.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import penetration_typeC_qwen_verify as v1

ROOT=v1.ROOT; WS=v1.WS; MODEL=v1.MODEL
VERSION='typeC_qwen_neutral_gate_v2_20261009'

NEUTRAL='''Describe the visible OBJECT/SCENE RELATIONSHIP without deciding whether any physics violation exists.
Here are {n} ORIGINAL frames in CHRONOLOGICAL order, with frame IDs {indices}; candidate around f{t}.
Manipulated object: {actor}. Environment: {target}. Boundary to inspect: {boundary}.

Focus on the ACTUAL location where the object seems to become partly hidden or shorter.
Treat a visible opening as an opening, NOT as solid wall; a container rim can occlude
an object that legitimately descends through its open top. A drawer front edge can
occlude an object entering the drawer through its opening. Changing visible/projected
length alone is NOT proof that physical length changed. Gripper coverage, perspective
and bends can hide a tip. Be precise about whether the object INTERSECTS solid pixels
at an impossible location OR merely disappears behind a normal foreground edge.

Return ONLY JSON with:
{{"actor_identified":"YES|NO|UNCLEAR",
"plausible_opening_at_interface":"YES|NO|UNCLEAR",
"ordinary_occlusion_or_pose_plausible":"YES|NO|UNCLEAR",
"specific_solid_material_intersection":"YES|NO|UNCLEAR",
"shape_change_visible_without_occluder":"YES|NO|UNCLEAR",
"interface_location":"short description or UNKNOWN, name solid part versus opening",
"object_segments":"which parts remain visible; what becomes hidden",
"observations":"specific frame-number-based explanation of normal or abnormal geometry"}}
Do not assume violation. When there is any reasonable normal explanation choose YES
for the corresponding plausible explanation. When evidence is insufficient choose UNCLEAR.
'''

VIOLATION=v1.PROMPT+'''
ADDITIONAL CAUTION: In this benchmark NORMAL insertion of a cube/cup/banana through
an OPEN top or open drawer cavity is very common. The container rim or drawer face
normally occludes parts of an object. Mark legal_opening_explains YES whenever that
explains the observed location; do not award solid intersection simply because the
object is partly inside and partly outside. For shape loss, require evidence that a
material segment is gone *even when it should remain unoccluded*, not merely a shorter
visible projection. If you cannot localize the impossible interface, answer UNCLEAR.
'''

NF=('actor_identified','plausible_opening_at_interface','ordinary_occlusion_or_pose_plausible',
    'specific_solid_material_intersection','shape_change_visible_without_occluder')

def neutral_validate(x):
    for k in NF:
        if k not in x:raise ValueError('Missing neutral '+k)
        x[k]=str(x[k]).strip().upper()
        if x[k] not in v1.YES:raise ValueError('Bad neutral '+k+': '+str(x[k]))
    for k in ('interface_location','object_segments','observations'):
        x[k]=str(x.get(k,''))[:1500]
        if len(x[k])<5:raise ValueError('Missing textual neutral '+k)
    return x

def gate_score(neutral,violation):
    """Independent, conservative second pass; no numeric model confidence."""
    base=v1.scores(violation)
    if neutral['actor_identified']!='YES':
        return dict(solid=0.,shape=0.,combined=0.,reason='neutral_actor_unverified',base=base)
    if neutral['plausible_opening_at_interface']=='YES':
        return dict(solid=0.,shape=0.,combined=0.,reason='legal_opening_veto',base=base)
    if neutral['ordinary_occlusion_or_pose_plausible']=='YES':
        return dict(solid=0.,shape=0.,combined=0.,reason='normal_visibility_veto',base=base)
    solid=base['solid'] if (neutral['specific_solid_material_intersection']=='YES'
          and neutral['plausible_opening_at_interface']=='NO'
          and neutral['ordinary_occlusion_or_pose_plausible']=='NO') else 0.
    shape=base['shape'] if (neutral['shape_change_visible_without_occluder']=='YES'
          and neutral['plausible_opening_at_interface']=='NO'
          and neutral['ordinary_occlusion_or_pose_plausible']=='NO') else 0.
    reason='double_confirmed' if max(solid,shape)>=.75 else 'no_independent_confirmation_or_uncertain'
    return dict(solid=solid,shape=shape,combined=max(solid,shape),reason=reason,base=base)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',required=True)
    ap.add_argument('--video',type=Path,required=True)
    ap.add_argument('--qwen-dir',type=Path,required=True)
    ap.add_argument('--agent0-json',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-audits',type=int,default=8)
    ap.add_argument('--min-qwen-signal',type=float,default=.75)
    ap.add_argument('--image-side',type=int,default=896)
    ap.add_argument('--max-tokens',type=int,default=1600)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    if not 1<=a.max_audits<=20:ap.error('--max-audits 1..20')
    cfg,pair,n,qualifying,selected=v1.prepare(a.qwen_dir,a.agent0_json,a.video,a.case,a.min_qwen_signal,a.max_audits)
    print('CASE:',a.case,'frames:',n,'stage1 candidates:',len(qualifying),'selected:',selected,flush=True)
    print('MONITORED:',json.dumps(pair,ensure_ascii=False),flush=True)
    if a.dry_run:return
    out=a.output/a.case;out.mkdir(parents=True,exist_ok=True)
    conf=dict(protocol=VERSION,case=a.case,video=str(a.video.resolve()),
        qwen_dir=str(a.qwen_dir.resolve()),stage1_sha=v1.sha(json.dumps(cfg,sort_keys=True)),
        agent0=pair,selected=selected,model=MODEL,min_qwen_signal=a.min_qwen_signal,
        image_side=a.image_side,max_tokens=a.max_tokens,neutral_prompt_sha=v1.sha(NEUTRAL),
        violation_prompt_sha=v1.sha(VIOLATION))
    cp=out/'run_config.json'
    if cp.exists() and v1.load(cp)!=conf:raise RuntimeError('Changed config: select another --output')
    v1.save(cp,conf)
    if not selected:
        v1.save(out/'summary.json',dict(protocol=VERSION,case=a.case,status='complete',
            qwen_candidate_count=0,selected_frames=[],audits=0,solid_score=0.,
            shape_score=0.,combined_score=0.,failed_frames=[]))
        return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Run on H200 allocation')
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
    from penetration_agent0_v3C_multiframe import load_v3
    v3=load_v3(ROOT/'penetration_abc_agents_v3.py')
    client=VLMClient('typeC_qwen_verifier_v2',{'backend':'local_gpu','model':MODEL,'max_tokens':a.max_tokens})
    frames=v3.Frames(a.video)
    try:
        for j,t in enumerate(selected,1):
            dst=out/'audits'/f'f{t:04d}.json'
            if dst.exists() and v1.load(dst).get('status')=='ok':
                print('CACHED',j,'/',len(selected),'frame',t,flush=True);continue
            ix=v1.context_indices(int(cfg['reference']),t,n)
            args=dict(n=len(ix),indices=', '.join(map(str,ix)),t=t,
                actor=v1.safe(pair['actor']),target=v1.safe(pair['target']),
                boundary=v1.safe(pair['solid_boundary']))
            ims=[v3.label(frames.frame(k),k,a.image_side) for k in ix]
            rec=dict(case=a.case,frame=t,source_frames=ix,status='failed')
            try:
                def ask_once(prompt,validator):
                    last=None
                    for attempt in range(2):
                        r=client.ask(ims,prompt,system_prompt='Ground all claims in the shown frames. Output one complete JSON object only.')
                        raw=str(r.get('answer',''))
                        try:return validator(v1.get_json(raw)),raw
                        except Exception as e:last=str(e);print('JSON RETRY:',t,attempt+1,last[:100],flush=True)
                    raise ValueError('Qwen returned malformed JSON twice: '+str(last))
                neutral,raw_n=ask_once(NEUTRAL.format(**args),neutral_validate)
                violation,raw_v=ask_once(VIOLATION.format(**args),v1.normalized)
                rec.update(neutral=neutral,violation=violation,raw_neutral=raw_n,
                    raw_violation=raw_v,scores=gate_score(neutral,violation),status='ok')
                print(f'VERIFY {j}/{len(selected)} f{t:04d} SOLID={rec["scores"]["solid"]:.2f} '
                      f'SHAPE={rec["scores"]["shape"]:.2f} gate={rec["scores"]["reason"]} '
                      f'opening={neutral["plausible_opening_at_interface"]} '
                      f'occlusion={neutral["ordinary_occlusion_or_pose_plausible"]}',flush=True)
            except Exception as e:
                rec['error']=str(e)
                print('FAILED',t,str(e)[:200],flush=True)
            v1.save(dst,rec)
        records=[v1.load(out/'audits'/f'f{t:04d}.json') for t in selected]
        success=[r for r in records if r.get('status')=='ok']
        sm=dict(protocol=VERSION,case=a.case,status='complete' if len(success)==len(selected) else 'incomplete',
            qwen_candidate_count=len(qualifying),selected_frames=selected,audits=len(success),
            solid_score=max((r['scores']['solid'] for r in success),default=0.),
            shape_score=max((r['scores']['shape'] for r in success),default=0.),
            combined_score=max((r['scores']['combined'] for r in success),default=0.),
            failed_frames=[r['frame'] for r in records if r.get('status')!='ok'],
            notes='Conservative neutral counter-explanation gate. Uncalibrated; no GT in inference.')
        v1.save(out/'summary.json',sm)
        print('FINAL:',a.case,json.dumps(sm,ensure_ascii=False),flush=True)
        if sm['status']!='complete':sys.exit(2)
    finally:frames.close()

if __name__=='__main__':main()
