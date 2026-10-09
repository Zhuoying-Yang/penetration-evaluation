#!/usr/bin/env python3
"""Controlled Agent-0 / V3-A prompt ablation, GPU Qwen only (no Gemini).

Keeps the exact original V3 A model calls, scoring and temporal consistency.
Changes only the scene-identification sentence using an existing Agent-0 result.

No ground truth or failure-frame numbers are included in any model prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
MODEL = '/PHShome/zy992/Wilson/dependency/models/qwen_model'


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def clean(text: str, length: int = 120) -> str:
    # Agent-0 fields are data, not executable instructions. Remove braces/newlines.
    x = re.sub(r'[^\w \-.,/()\']', ' ', str(text), flags=re.UNICODE)
    return ' '.join(x.split())[:length]


def read_agent0(path: Path) -> dict:
    obj = json.loads(path.read_text())
    pair = obj.get('selected')
    if not pair:
        pairs = obj.get('pairs') or []
        if isinstance(pairs, list) and pairs:
            pair = next((p for p in pairs if p.get('pair_type') == 'OBJECT_ENVIRONMENT'), pairs[0])
    if not pair:
        raise ValueError('No Agent-0 selected pair found in ' + str(path))
    for k in ('actor', 'target', 'solid_boundary'):
        if not pair.get(k):
            raise ValueError('Agent-0 pair missing '+k)
    return pair


def compile_prompt(v3_core: str, pair: dict, mode: str) -> str:
    if mode == 'vanilla':
        return v3_core
    if mode == 'previous_lock':
        # Same fixed string as the batch's generate_prompt() for comparison.
        actor = str(pair['actor']).replace('\n', ' ')[:110]
        target = str(pair['target']).replace('\n', ' ')[:110]
        barrier = str(pair['solid_boundary']).replace('\n', ' ')[:140]
        return (
            'SCENE OBJECT LOCK (selected from frame 0 by a fixed, label-blind Agent 0):\n'
            f'MOVING ACTOR: {actor}\nOTHER PHYSICAL OBJECT: {target}\n'
            f'SOLID BARRIER: {barrier}\n'
            'Track ONLY this actor relative to this barrier in the source frame pair.\n'
            'The identification is not a penetration claim. If visibility or depth is\n'
            'insufficient, use UNCLEAR. Do not replace either physical entity.\n\n'
        ) + v3_core
    if mode == 'literal':
        actor = clean(pair['actor'])
        target = clean(pair['target'])

        if not actor or not target:
            raise ValueError('Empty Agent 0 object name')

        prompt = re.sub(
            r'\bcup\b', lambda _: actor,
            v3_core, flags=re.IGNORECASE
        )
        prompt = re.sub(
            r'\bbin\b', lambda _: target,
            prompt, flags=re.IGNORECASE
        )
        return prompt

    if mode not in ('minimal', 'slot'):
        raise ValueError(mode)
    old = ('Identify the MOVING ACTOR (the object held by the gripper if present) '
           'and the OTHER object\'s SOLID BARRIER (e.g. the large bin side wall). '
           'Do not confuse a carried cup with the large bin, or the gripper with the cup.')
    if v3_core.count(old) != 1:
        raise RuntimeError('V3-A identification sentence changed; stop and inspect the original prompt')
    actor = clean(pair['actor'])
    target = clean(pair['target'])
    barrier = clean(pair['solid_boundary'], 140)
    if mode == 'minimal':
        new = (
            'Identify the MOVING ACTOR (Agent 0 suggested: ' + actor + ') and the OTHER '
            "object's SOLID BARRIER (Agent 0 suggested: " + barrier + ' of ' + target + '). '
            'These are object-identity suggestions, NOT claims about penetration or the '
            "actor's position. Verify them in BOTH images; use UNCLEAR if unsupported."
        )
        return v3_core.replace(old, new)
    new = (
        'Agent 0 suggested the following potentially relevant entities from the FIRST frame: '
        f'MOVING ACTOR = {actor}; OTHER OBJECT = {target}; SOLID BARRIER = {barrier}. '
        'Verify these proposed identities against BOTH images. Independently locate the same actor '
        'relative to this solid boundary before and after. This suggestion is NOT evidence of '
        'penetration or of any particular position. If the proposed identities cannot be verified '
        'in the two frames, use UNCLEAR. Apply the unchanged Type-A crossing criteria below.'
    )
    return v3_core.replace(old, new)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', required=True)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--agent0-json', type=Path, required=True)
    ap.add_argument('--v3-script', type=Path, default=ROOT / 'penetration_abc_agents_v3.py')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=('literal', 'minimal', 'slot', 'vanilla', 'previous_lock'), default='minimal')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    ap.add_argument('--image-side', type=int, default=1024)
    ap.add_argument('--max-tokens', type=int, default=520)
    ap.add_argument('--inspect-after', type=int, default=None,
                    help='Offline log inspection ONLY, not passed to the model prompt')
    ap.add_argument('--dry-run', action='store_true', help='Print/save prompt; do not load Qwen')
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location('penetration_abc_agents_v3_ablation', args.v3_script)
    v3 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v3
    spec.loader.exec_module(v3)

    pair = read_agent0(args.agent0_json)
    prompt = compile_prompt(v3.A_PROMPT, pair, args.mode)
    outdir = args.output / args.case / args.mode
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.video.is_file() and not args.dry_run:
        raise FileNotFoundError(args.video)
    metadata = {
        'case': args.case, 'mode': args.mode, 'video': str(args.video),
        'agent0_source': str(args.agent0_json), 'agent0_pair': pair,
        'v3_sha256': hashlib.sha256(args.v3_script.read_bytes()).hexdigest(),
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'start': args.start, 'end': args.end, 'image_side': args.image_side,
        'max_tokens': args.max_tokens, 'qwen_model': MODEL,
    }
    old_meta_path = outdir / 'ablation_config.json'
    if old_meta_path.exists() and json.loads(old_meta_path.read_text()) != metadata:
        raise RuntimeError('Settings or prompt changed; use a new --output to avoid stale Qwen cache')
    save_json(old_meta_path, metadata)
    save_json(outdir / 'rendered_prompt.json', {'A': prompt, 'agent0_pair': pair})
    print('CASE:', args.case, '| MODE:', args.mode)
    print('AGENT0:', json.dumps(pair, ensure_ascii=False))
    print('PROMPT SHA256:', metadata['prompt_sha256'])
    print('EXACT PROMPT SAVED:', outdir / 'rendered_prompt.json')
    if args.dry_run:
        print('DRY RUN: no Qwen calls')
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU; enter your Slurm H200 allocation first')
    sys.path.insert(0, str(WS))
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient

    n, w, h, fps = v3.video_info(args.video)
    end = n - 1 if args.end < 0 else args.end
    if not (0 <= args.start < end < n):
        raise ValueError(f'Invalid frame range {args.start}..{end} for {n} frames')
    frames = v3.Frames(args.video)
    try:
        client = VLMClient('penetration_agent0_v3A_ablation', {
            'backend': 'local_gpu', 'model': MODEL, 'max_tokens': args.max_tokens})
        v3.A_PROMPT = prompt
        v3.run_agent(client, 'A', frames, outdir, args.start, end, 0,
                     args.image_side, 640, None, None, None, None, 5,
                     args.max_tokens, (w, h))
        records = json.loads((outdir / 'A_queries.json').read_text())['rows']
        good = [r for r in records if r.get('status') == 'ok']
        rawhits = [(r['frame'], float(r['verdict'].get('raw_transition_signal', 0)))
                   for r in good if float(r['verdict'].get('raw_transition_signal', 0)) >= 0.65]
        posthits = [(r['frame'], float(r['verdict'].get('signal', 0)))
                    for r in good if float(r['verdict'].get('signal', 0)) >= 0.65]
        summary = {'case': args.case, 'mode': args.mode, 'good': len(good),
                   'expected': end-max(1, args.start)+1,
                   'raw_candidates': rawhits, 'post_candidates': posthits,
                   'raw_peak': max((s for _, s in rawhits), default=0.0),
                   'post_peak': max((s for _, s in posthits), default=0.0)}
        save_json(outdir / 'ablation_summary.json', summary)
        print('ABLATION SUMMARY:', json.dumps(summary, ensure_ascii=False))
        if args.inspect_after is not None:
            row = next((r for r in good if r['frame'] == args.inspect_after), None)
            print('INSPECT AFTER FRAME', args.inspect_after, ':',
                  json.dumps(row.get('verdict') if row else None, ensure_ascii=False))
    finally:
        frames.close()


if __name__ == '__main__':
    main()
