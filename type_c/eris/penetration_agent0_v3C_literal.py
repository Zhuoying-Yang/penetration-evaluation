#!/usr/bin/env python3
"""Agent 0 (first frame) -> minimally name-conditioned ORIGINAL V3 Type C detector.

Development probe: no GT labels, manual event frames or hand-set crops are used.
Preserves original V3 C image protocol (reference vs candidate), JSON parsing,
truth_score, and candidate sheets. Does not run Gemini; use a separate fixed
C auditor after confirming the Qwen proposal stage.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
MODEL = '/PHShome/zy992/Wilson/dependency/models/qwen_model'

AGENT0_PROMPT = '''You inspect only the FIRST frame of a robot manipulation video.
Original task, if supplied: {task}
Identify up to THREE visually grounded physical interaction pairs for evaluating
whether a manipulated object becomes truncated or occluded by a SOLID boundary
of a drawer, cabinet, box or other surrounding object. Include the carried
object and its environmental boundary, not just the robot gripper. A legal
opening or an open drawer cavity is NOT solid material.
Do not decide penetration, do not guess the future, and do not invent objects.
If the relevant object is not visible, return an empty pairs list.
Return ONLY JSON with pairs in descending scene relevance:
{{"pairs":[{{"actor":"short exact object name","target":"short surrounding object name",
"solid_boundary":"specific solid panel or edge; not an opening",
"pair_type":"OBJECT_ENVIRONMENT|ROBOT_OBJECT|OTHER",
"expected_interaction":"neutral phrase"}}]}}
'''

C_ID_SENTENCE = ('Compare ONLY the same visible manipulated object (for example, a banana) '
                 'and the drawer/bin/other container material.')


def safe(x):
    s = re.sub(r'[^\w\s.,()/\'-]', ' ', str(x))
    return ' '.join(s.split())[:110]


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_v3(path: Path):
    spec = importlib.util.spec_from_file_location('v3_for_agent0_typec', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def select_pair(obj):
    choices=[]
    for p in obj.get('pairs', []):
        if not isinstance(p, dict) or not all(safe(p.get(k,'')) for k in ('actor','target','solid_boundary')):
            continue
        choices.append(p)
    # Scene-independent preference: manipulable object vs drawer/cabinet/container.
    def key(p):
        actor=safe(p['actor']).lower(); target=safe(p['target']).lower()
        environment=any(k in target for k in ('drawer','cabinet','container','bin','bucket','box','shelf'))
        robotic=any(k in actor for k in ('gripper','robot arm','finger','robot hand'))
        return (p.get('pair_type') == 'OBJECT_ENVIRONMENT', environment, not robotic)
    if not choices:
        return None
    return sorted(choices, key=key, reverse=True)[0]


def compile_type_c(core: str, pair: dict):
    """Three literal replacements, no prefix and no change to detection criteria."""
    assert core.count(C_ID_SENTENCE) == 1, 'Original C identification sentence changed'
    actor = safe(pair['actor']); target = safe(pair['target']); boundary = safe(pair['solid_boundary'])
    assert actor and target and boundary
    prompt = core.replace(
        C_ID_SENTENCE,
        f'Compare ONLY the same visible manipulated object ({actor}) and the {target} material.'
    )
    old='at a solid drawer edge/panel'
    assert prompt.count(old) == 1, 'Original C solid-boundary phrase changed'
    prompt=prompt.replace(old, f'at the {boundary}')
    return prompt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', default='Cosmos3_seed101_0048')
    ap.add_argument('--video', type=Path, default=Path('/scratch/z/zy992/zhuoying/physact/cosmos3/export_robowm68_3seeds/seed101/0048.mp4'))
    ap.add_argument('--v3-script', type=Path, default=ROOT/'penetration_abc_agents_v3.py')
    ap.add_argument('--agent0-json', type=Path, default=None, help='Optional previous Agent 0 JSON; otherwise compute/caches first frame only')
    ap.add_argument('--task', default='', help='Original task only, never GT labels or failure descriptions')
    ap.add_argument('--output', type=Path, default=ROOT/'penetration_agent0_v3C_literal_20261009')
    ap.add_argument('--reference', type=int, default=0, help='Fixed reference frame, default first frame')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=None, help='Default full video')
    ap.add_argument('--image-side', type=int, default=1024)
    ap.add_argument('--max-tokens', type=int, default=520)
    ap.add_argument('--agent0-only', action='store_true')
    args=ap.parse_args()
    if not args.video.is_file(): ap.error('Missing video: '+str(args.video))
    if not args.v3_script.is_file(): ap.error('Missing V3: '+str(args.v3_script))
    v3=load_v3(args.v3_script)
    n,w,h,fps=v3.video_info(args.video)
    end=n-1 if args.end is None else args.end
    if not (0 <= args.reference < n and 0 <= args.start <= end < n):
        ap.error(f'Invalid reference/range: ref={args.reference} start={args.start} end={end} total={n}')
    outdir=args.output/args.case
    outdir.mkdir(parents=True, exist_ok=True)
    source_a0=args.agent0_json or (outdir/'agent0.json')
    source_has_a0=source_a0.is_file()
    selected=None
    if source_has_a0:
        data=json.loads(source_a0.read_text())
        selected=data.get('selected') or select_pair(data)
        if selected is None:
            raise RuntimeError('Saved Agent 0 has no supported pair: '+str(source_a0))
    # A0-only run is also cached; full run should not repeat it.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU: run inside the current H200 allocation')
    sys.path.insert(0, str(WS))
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
    client=VLMClient('penetration_agent0_v3c_literal', {
        'backend': 'local_gpu', 'model': MODEL, 'max_tokens': args.max_tokens})
    frames=v3.Frames(args.video)
    try:
        if selected is None:
            a0_prompt=AGENT0_PROMPT.format(task=args.task or 'NOT PROVIDED')
            image=v3.label(frames.frame(0),0,args.image_side)
            ans=client.ask([image],a0_prompt,
                system_prompt='Identify physical entities from this image only. Do not infer future physical errors.')
            try:
                raw=v3.get_obj(ans['answer'])
            except Exception:
                save(outdir/'agent0_unparsed.json', {'answer':str(ans.get('answer',''))})
                raise
            selected=select_pair(raw)
            save(source_a0, {'pairs':raw.get('pairs',[]), 'selected':selected,
                'frame':0, 'prompt':a0_prompt, 'raw_answer':ans.get('answer','')})
        print('CASE:',args.case,'total_frames:',n,flush=True)
        print('AGENT0:',json.dumps(selected,ensure_ascii=False),flush=True)
        if selected is None:
            raise RuntimeError('Agent 0 selected no supported pair (object-selection failure)')
        prompt=compile_type_c(v3.C_PROMPT,selected)
        metadata={
            'protocol':'agent0_literal_original_v3C_reference_fullframe_v1',
            'case':args.case,'video':str(args.video.resolve()),
            'video_size':args.video.stat().st_size,
            'agent0':selected,'reference':args.reference,'start':args.start,'end':end,
            'v3_sha256':hashlib.sha256(args.v3_script.read_bytes()).hexdigest(),
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'model':MODEL,'image_side':args.image_side,'max_tokens':args.max_tokens}
        config=outdir/'run_config.json'
        if config.is_file() and json.loads(config.read_text()) != metadata:
            raise RuntimeError('Configuration changed; use a different --output to avoid stale query cache')
        save(config,metadata)
        save(outdir/'rendered_prompt.json',{'C':prompt,'agent0_pair':selected})
        print('PROMPT SHA256:',metadata['prompt_sha256'],flush=True)
        print('OUTPUT:',outdir,flush=True)
        if args.agent0_only:
            print('Agent0 only: no C detector run',flush=True)
            return
        v3.C_PROMPT=prompt
        result=v3.run_agent(client,'C',frames,outdir,args.start,end,args.reference,
            args.image_side,640,None,None,None,None,5,args.max_tokens,(w,h))
        rows=json.loads((outdir/'C_queries.json').read_text())['rows']
        valid=[r for r in rows if r.get('status')=='ok']
        candidates=[{'frame':int(r['frame']),'score':float(r['verdict']['signal']),
                     'verdict':r['verdict']}
                    for r in valid if float(r['verdict'].get('signal',0))>=0.5]
        save(outdir/'typeC_summary.json', {
            'case':args.case,'total_frames':n,'n_ok':len(valid),
            'n_expected':(end-args.start+1)-(1 if args.start <= args.reference <= end else 0),
            'peak':result['peak'], 'candidates':candidates})
        print('C DONE: n_ok=',len(valid),'peak=',result['peak'],
              'candidates >=0.5:',len(candidates),flush=True)
        print('TOP CANDIDATES:',[(r['frame'],r['score']) for r in candidates[:20]],flush=True)
    finally:
        frames.close()

if __name__=='__main__':
    main()
