#!/usr/bin/env python3
"""Experimental label-blind Type A Qwen candidate detector: restored two-frame V3-A.

- Agent 0 discovers manipulated-object / solid-boundary pairs in uniformly sampled video.
- Qwen sees ONLY source frames t-1 and t, 1024px; NO ROI scout or crops.
- V3-A pairwise prompt and temporal audit are reused. The minimum necessary
  modifications are named-actor grounding and support for BOTH side directions.
- Scores are visual candidate signals, NOT 3D proof or calibrated probabilities.
- Original frozen V3-A and Backward v1.1 files are NOT modified.
- No GT, expected labels, annotated event frames or manual ROI reach the VLM.

Run on ERIS allocated GPU with Qwen environment; see adjacent README.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

# Repository root when stored at type_a/backward/eris/ within the git checkout.
_HERE = Path(__file__).resolve()
PROJECT = _HERE.parents[3] if _HERE.parent.name == 'eris' else _HERE.parent
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
MODEL = Path('/PHShome/zy992/Wilson/dependency/models/qwen_model')
VIDEO_DIR = Path('/scratch/z/zy992/zhuoying/physact/cosmos3/export_robowm68_3seeds/seed103')
PROTOCOL = 'typeA_backward_twoframe_generic_qwen_v1_20261011'

SYSTEM_AGENT0 = ("Identify physical entities only from images. Do not infer penetrations or anomalies. "
                 "Do not use knowledge of labels or suspect frame times. Return valid JSON only.")

AGENT0 = '''OBJECT-PAIR IDENTIFICATION ONLY. These FIVE uniformly sampled images are from a generated robot manipulation video.
Original task (if provided): {task}
Identify up to TWO VISUALLY GROUNDED actor/solid-boundary pairs that could be relevant to object-environment crossing analysis.
The ACTOR can be an object manipulated by the robot (cube, banana, cup, tool, etc.) OR a robot gripper/end-effector/arm part.
The TARGET is a different physical object or environment fixture with a recognizable SOLID barrier, such as a drawer front panel, side wall, cabinet door, bin wall, or cup wall.
Prefer a manipulated OBJECT versus the environmental barrier as the first pair when visible; also include the gripper versus a solid barrier as a second pair when visually grounded. This ranking is fixed for every video and is NOT a judgment that any violation occurred.
Do NOT replace an object with a gripper just because a robot is present. Do NOT choose empty cavity, opening, air, or table support as the SOLID barrier. Do not hallucinate actors, boundaries, crossing events, or frame times.
Return JSON ONLY:
{{"pairs":[{{"actor":"short visible actor name","target":"short physical object or fixture name","solid_boundary":"specific visible solid wall or panel","pair_type":"OBJECT_ENVIRONMENT|ROBOT_ENVIRONMENT|ROBOT_OBJECT|OTHER","expected_interaction":"neutral description, no anomaly claim"}}]}}'''

INVENTORY = '''SCENE INVENTORY ONLY. Seven images are uniformly sampled from a robot manipulation video.
Original task (if provided): {task}
Select ONE visually grounded manipulated object or robot part, and ONE different physical object with a visible SOLID WALL/PANEL (not an opening, a plate, or a supporting tabletop).
Actors include cubes, cups, bananas, tools, robotic grippers, and end-effectors. Prefer a manipulated object if identifiable. Do NOT infer penetration or any error.
If the pair cannot be visually supported, use NO or UNCLEAR. Return JSON ONLY:
{{"actor_visible":"YES|NO|UNCLEAR","fixture_visible":"YES|NO|UNCLEAR","solid_panel_visible":"YES|NO|UNCLEAR",
"actor_name":"short actor name or empty","fixture_name":"short target object name or empty",
"solid_panel_name":"specific solid wall/panel or empty","pair_type":"OBJECT_ENVIRONMENT|ROBOT_ENVIRONMENT|ROBOT_OBJECT|OTHER",
"observation":"neutral appearance-only description"}}'''

OLD_ID = ('Identify the MOVING ACTOR (the object held by the gripper if present) '
          "and the OTHER object's SOLID BARRIER (e.g. the large bin side wall). "
          'Do not confuse a carried cup with the large bin, or the gripper with the cup.')
OLD_DIRECTION = ('3. A suspicious Type-A candidate is an OUTSIDE_SIDE -> INSIDE_CAVITY change '
                 'across a SOLID SIDE region, while actor\'s bottom appears below the rim, '
                 'with no visible above-rim route. The presence of a legal opening elsewhere '
                 'does not excuse a side-wall crossing. However, do NOT invent unobserved '
                 '3D heights; a 2D overlap may be normal occlusion, so use UNCLEAR when warranted.')


def render_a_prompt(v3, pair):
    """Keep the original two-frame V3-A structure; adapt actor and direction only."""
    original = v3.A_PROMPT
    if original.count(OLD_ID) != 1 or original.count(OLD_DIRECTION) != 1:
        raise RuntimeError('V3-A original prompt changed; review before running!')
    actor, target, boundary = (clean(pair[k]) for k in ('actor', 'target', 'solid_boundary'))
    result = original.replace(OLD_ID,
        f'AGENT 0 TRACKING PAIR (entity identification only, NOT evidence of an error): '
        f'ACTOR = {actor}; OTHER OBJECT = {target}; SPECIFIC SOLID BARRIER = {boundary}. '
        'Track ONLY this named physical actor, whether manipulated object or gripper. '
        'Do NOT switch to a similarly colored stationary object or to a gripper if '
        'the actor is a carried object. Verify the actor and the SAME solid panel in both frames.')
    result = result.replace(OLD_DIRECTION,
        '3. A suspicious candidate is EITHER OUTSIDE_SIDE -> INSIDE_CAVITY OR '
        'INSIDE_CAVITY -> OUTSIDE_SIDE across the SPECIFIED SOLID material. '
        'The actor may move relative to the wall, OR the solid wall may sweep past '
        'a nearly stationary actor. Distinguish solid-material crossing from a '
        'legal route over a rim, through an opening, or around a panel edge. '
        'A 2D overlap, transparency, or visual disappearance alone cannot prove 3D collision. '
        'If the path is uncertain, answer UNCLEAR.')
    # As in the frozen Forward actor-literal mode, generalize historical 'cup/bin' exemplars.
    result = re.sub(r'\bcup\b', lambda _: actor, result, flags=re.IGNORECASE)
    result = re.sub(r'\bbin\b', lambda _: target, result, flags=re.IGNORECASE)
    # Preserve the field structure; add a reverse-direction token.
    result = result.replace('LATERAL_INTO_WALL|UP_OVER_RIM',
                            'LATERAL_INTO_WALL|LATERAL_OUT_OF_WALL|UP_OVER_RIM')
    # Remove misleading single-object assumptions while retaining the scoring rubric.
    result = result.replace('A low cup bottom relative to a bin rim',
                            'A low actor bottom relative to a container rim')
    return result


def clean(x):
    return ' '.join(re.sub(r"[^\w\s.,/()'-]", ' ', str(x)).split())[:110]


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def sample_indices(n, k):
    if n < 1 or k < 1:
        return []
    count = min(n, k)
    if count == 1:
        return [n // 2]
    return sorted(set(round(i * (n - 1) / (count - 1)) for i in range(count)))


def valid_pairs(raw):
    allowed = []
    if not isinstance(raw, dict):
        return allowed
    for item in (raw.get('pairs') or [])[:3]:
        if not isinstance(item, dict):
            continue
        p = {k: clean(item.get(k, '')) for k in
             ('actor', 'target', 'solid_boundary', 'pair_type', 'expected_interaction')}
        if not all(p[k] for k in ('actor', 'target', 'solid_boundary')):
            continue
        if p['actor'].lower() == p['target'].lower():
            continue
        if re.search(r'opening|empty cavity|air gap|table ?top|table surface', p['solid_boundary'], re.I):
            continue
        allowed.append(p)
    allowed.sort(key=lambda p: (p['pair_type'].upper() != 'OBJECT_ENVIRONMENT',
                                p['pair_type'].upper() != 'ROBOT_ENVIRONMENT'))
    return allowed[:2]


def inventory_pair(inv):
    if not isinstance(inv, dict) or not all(str(inv.get(k, '')).upper() == 'YES'
        for k in ('actor_visible', 'fixture_visible', 'solid_panel_visible')):
        return []
    x = {k: clean(inv.get(v, '')) for k, v in
         [('actor', 'actor_name'), ('target', 'fixture_name'), ('solid_boundary', 'solid_panel_name')]}
    if not all(x.values()) or x['actor'].lower() == x['target'].lower():
        return []
    if re.search(r'opening|empty cavity|table ?top|table surface|air gap', x['solid_boundary'], re.I):
        return []
    x['pair_type'] = clean(inv.get('pair_type', 'OBJECT_ENVIRONMENT'))
    x['expected_interaction'] = 'neutral scene-inventory pair'
    return [x]


def identify_pairs(client, v3, frames, n, output):
    path = output / 'agent0_multiframe.json'
    ix5, ix7 = sample_indices(n, 5), sample_indices(n, 7)
    if path.is_file():
        d = load(path)
        if d.get('frames5') != ix5 or d.get('frames7') != ix7 or d.get('protocol') != PROTOCOL:
            raise RuntimeError('Agent0 configuration mismatch; use a new output folder')
        return d['pairs']
    raw5 = client.ask([v3.label(frames.frame(t), t, 896) for t in ix5],
                      AGENT0.format(task='NOT PROVIDED'), system_prompt=SYSTEM_AGENT0)['answer']
    try:
        primary = valid_pairs(v3.get_obj(raw5))
    except Exception:
        primary = []
    raw7, inv = None, None
    pairs = primary
    if not pairs:
        raw7 = client.ask([v3.label(frames.frame(t), t, 896) for t in ix7],
                          INVENTORY.format(task='NOT PROVIDED'), system_prompt=SYSTEM_AGENT0)['answer']
        try:
            inv = v3.get_obj(raw7)
            pairs = inventory_pair(inv)
        except Exception:
            pairs = []
    save(path, {'protocol': PROTOCOL, 'frames5': ix5, 'frames7': ix7,
                'prompt5': AGENT0, 'prompt7': INVENTORY,
                'raw5': raw5, 'raw7': raw7, 'inventory': inv,
                'selection_method': 'primary' if primary else 'inventory' if pairs else 'none',
                'pairs': pairs, 'gt_used': False})
    return pairs


def score_both_directions(v, kind):
    if kind != 'A':
        raise ValueError('This experiment supports only Type A')
    yes = lambda k: str(v.get(k, 'UNCLEAR')).upper() == 'YES'
    no = lambda k: str(v.get(k, 'UNCLEAR')).upper() == 'NO'
    before, after = (str(v.get(k, 'UNCLEAR')).upper()
                     for k in ('position_before', 'position_after'))
    direction = str(v.get('motion_direction', 'UNCLEAR')).upper()
    into = before == 'OUTSIDE_SIDE' and after == 'INSIDE_CAVITY'
    outof = before == 'INSIDE_CAVITY' and after == 'OUTSIDE_SIDE'
    if not (into or outof) or not (yes('changed_during_pair') and yes('crosses_solid_side_in_pair')):
        return 0.0
    if into and direction != 'LATERAL_INTO_WALL':
        return 0.0
    if outof and direction != 'LATERAL_OUT_OF_WALL':
        return 0.0
    if yes('legal_opening_path_supported'):
        return 0.0
    if yes('ordinary_occlusion_possible'):
        return 0.30
    below = yes('bottom_below_rim_before') and yes('bottom_below_rim_after')
    if below and no('clearance_above_rim_observed') and no('ordinary_occlusion_possible'):
        return 1.0
    if below and not yes('clearance_above_rim_observed'):
        return 0.70
    return 0.50


def patch_v3(v3):
    """Retain original V3-A image construction and continuity audit; parse reverse direction."""
    orig_parse = v3.parse_verdict
    def parse(raw, kind):
        if kind != 'A':
            return orig_parse(raw, kind)
        out = v3.get_obj(raw)
        for k in ('changed_during_pair', 'crosses_solid_side_in_pair',
                  'bottom_below_rim_before', 'bottom_below_rim_after',
                  'clearance_above_rim_observed', 'legal_opening_path_supported',
                  'ordinary_occlusion_possible'):
            out[k] = v3.normalize_yes(out.get(k, 'UNCLEAR'))
        for k in ('position_before', 'position_after'):
            z = str(out.get(k, 'UNCLEAR')).strip().upper()
            out[k] = z if z in {'OUTSIDE_SIDE', 'INSIDE_CAVITY', 'ABOVE_OPENING',
                                'STRADDLING_RIM', 'OCCLUDED', 'UNCLEAR'} else 'UNCLEAR'
        z = str(out.get('motion_direction', 'UNCLEAR')).strip().upper()
        out['motion_direction'] = z if z in {'LATERAL_INTO_WALL', 'LATERAL_OUT_OF_WALL',
            'UP_OVER_RIM', 'DOWN_THROUGH_OPENING', 'OTHER', 'NONE', 'UNCLEAR'} else 'UNCLEAR'
        out['observation'] = str(out.get('observation', ''))[:500]
        out['signal'] = score_both_directions(out, 'A')
        return out
    v3.parse_verdict = parse


def select_candidates(pair_rows, max_audits):
    """Blind raw top-K NMS; keep post scores and flag shared-frame inconsistencies."""
    all_rows = []
    for pid, pair, rows in pair_rows:
        for r in rows:
            if r.get('status') != 'ok':
                continue
            v = r['verdict']
            all_rows.append({'pair_id': pid, 'actor': pair['actor'],
                             'frame': int(r['frame']),
                             'raw': float(v.get('raw_transition_signal', v.get('signal', 0))),
                             'post': float(v.get('signal', 0)),
                             'shared_frame_conflict': bool(v.get('shared_frame_conflict', False))})
    # Prefer genuine raw proposals; retain high-recall events even if the
    # original temporal consistency audit lowered their POST scores.
    ranked = sorted((r for r in all_rows if r['raw'] > 0),
                    key=lambda r: (-r['raw'], r['frame'], r['pair_id']))
    chosen = []
    def add(c):
        if any(c['pair_id'] == p['pair_id'] and abs(c['frame']-p['frame']) <= 2
               for p in chosen):
            return False
        chosen.append(c)
        return True
    for candidate in ranked:
        add(candidate)
        if len(chosen) >= max_audits:
            break
    if len(chosen) < max_audits:
        # Coverage fallback, fixed and GT-blind. In the all-zero case do not
        # accidentally choose only the first seconds of the recording.
        temporal = sorted(all_rows, key=lambda r: (r['frame'], r['pair_id']))
        sampled = [temporal[i] for i in sample_indices(len(temporal), max_audits * 4)]
        for candidate in sampled:
            add(candidate)
            if len(chosen) >= max_audits:
                break
    return sorted(chosen, key=lambda r: (r['frame'], r['pair_id']))


def run_case(client, v3, video, case, args):
    n, w, h, _ = v3.video_info(video)
    if n < 3:
        raise ValueError('Need >=3 frames')
    dest = args.output / case
    dest.mkdir(parents=True, exist_ok=True)
    cfg = {'protocol': PROTOCOL,
           'source_sha': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'v3_sha': hashlib.sha256(Path(v3.__file__).read_bytes()).hexdigest(),
           'video': str(video.resolve()), 'video_size': video.stat().st_size,
           'video_mtime_ns': video.stat().st_mtime_ns, 'n_frames': n,
           'model': str(args.model), 'max_tokens': args.max_tokens,
           'image_side': 1024, 'agent0_prompt': digest(AGENT0),
           'candidate_rule': 'raw_score_topk_nms2_with_post_audit',
           'max_audits': args.max_audits}
    key = digest(cfg)
    cfgpath = dest / 'run_config.json'
    if cfgpath.exists() and load(cfgpath)['hash'] != key:
        raise RuntimeError('Different config in output directory: select a new --output folder')
    save(cfgpath, {'hash': key, 'config': cfg})
    summ = dest / 'summary.json'
    if summ.exists() and load(summ).get('config_hash') == key and load(summ).get('status') == 'qwen_complete':
        print('CACHED:', case, flush=True)
        return
    frames = v3.Frames(video)
    try:
        pairs = identify_pairs(client, v3, frames, n, dest)
        print('AGENT0', case, json.dumps(pairs, ensure_ascii=False), flush=True)
        if args.agent0_only:
            return
        if not pairs:
            save(summ, {'case': case, 'status': 'no_supported_pair', 'video_score': None,
                        'not_a_negative_prediction': True, 'config_hash': key})
            return
        combined = []
        default_prompt = v3.A_PROMPT
        for pid, pair in enumerate(pairs):
            pdir = dest / 'pairs' / f'p{pid}'
            pdir.mkdir(parents=True, exist_ok=True)
            v3.A_PROMPT = render_a_prompt(v3, pair)
            save(pdir / 'prompts.json', {'agent0_pair': pair, 'A': v3.A_PROMPT,
                                         'system': v3.SYSTEM_SHARED})
            print(f'PAIR {pid} {pair["actor"]} -> {pair["solid_boundary"]}', flush=True)
            print('TWO FRAMES (t-1, t), FULL 1024px, no ROI scout', flush=True)
            try:
                result = v3.run_agent(client, 'A', frames, pdir, 0, n-1, 0, 1024, 640,
                                      None, None, None, None, 5, args.max_tokens, (w, h))
            finally:
                v3.A_PROMPT = default_prompt
            record = load(pdir / 'A_queries.json')
            rows = record['rows']
            ok_frames = {int(r['frame']) for r in rows if r.get('status') == 'ok'}
            if ok_frames != set(range(1, n)):
                raise RuntimeError(f'Incomplete scan for {case} pair {pid}; rerun to resume')
            combined.append((pid, pair, rows))
            print('PAIR DONE', pid, 'post_peak', result.get('peak'), flush=True)
        selected = select_candidates(combined, args.max_audits)
        save(dest / 'selected_candidates.json', {
            'method': 'Qwen raw top-K NMS ±2, across all pairs',
            'ground_truth_used': False, 'selected': selected})
        raw_peak = max((float(r['verdict'].get('raw_transition_signal', 0))
                        for _, _, rows in combined for r in rows), default=0.)
        post_peak = max((float(r['verdict'].get('signal', 0))
                         for _, _, rows in combined for r in rows), default=0.)
        save(summ, {'case': case, 'status': 'qwen_complete', 'n_frames': n,
                    'pairs': pairs, 'raw_peak': raw_peak, 'post_peak': post_peak,
                    'n_selected': len(selected), 'candidate_frames': selected,
                    'video_score': None,
                    'semantics': 'Qwen candidates only, not confirmed penetration or Sol-verified',
                    'config_hash': key})
        print('QWEN COMPLETE', case, 'RAW', raw_peak, 'POST', post_peak,
              'TOP CANDIDATES', [(x['pair_id'],x['frame']) for x in selected], flush=True)
    finally:
        frames.close()


def self_test():
    # All checks run without importing GPU models or cv2.
    class Fake:
        A_PROMPT = ('ROLE A. ' + OLD_ID + '\n' + OLD_DIRECTION + '\n'
                    'motion_direction: LATERAL_INTO_WALL|UP_OVER_RIM')
    p = {'actor': 'yellow cube', 'target': 'drawer', 'solid_boundary': 'drawer front solid wall'}
    s = render_a_prompt(Fake(), p)
    assert 'yellow cube' in s and 'drawer front solid wall' in s
    assert 'LATERAL_OUT_OF_WALL' in s and 'INSIDE_CAVITY -> OUTSIDE_SIDE' in s
    assert 'THREE-FRAME' not in s and 'ROI SCOUT' not in s
    assert valid_pairs({'pairs': [{**p, 'pair_type': 'OBJECT_ENVIRONMENT'}]})[0]['actor'] == 'yellow cube'
    assert inventory_pair({'actor_visible': 'YES', 'fixture_visible': 'YES',
                           'solid_panel_visible': 'YES', 'actor_name': 'robot gripper',
                           'fixture_name': 'drawer', 'solid_panel_name': 'drawer side wall'})
    v = dict(position_before='INSIDE_CAVITY', position_after='OUTSIDE_SIDE',
             changed_during_pair='YES', crosses_solid_side_in_pair='YES',
             motion_direction='LATERAL_OUT_OF_WALL', legal_opening_path_supported='NO',
             ordinary_occlusion_possible='NO', bottom_below_rim_before='YES',
             bottom_below_rim_after='YES', clearance_above_rim_observed='NO')
    assert score_both_directions(v, 'A') == 1.0
    v['position_before'], v['position_after'] = 'OUTSIDE_SIDE', 'INSIDE_CAVITY'
    v['motion_direction'] = 'LATERAL_INTO_WALL'
    assert score_both_directions(v, 'A') == 1.0
    v['legal_opening_path_supported'] = 'YES'
    assert score_both_directions(v, 'A') == 0.0
    assert sample_indices(189, 5) == [0, 47, 94, 141, 188]
    assert select_candidates([(0, p, [{'status':'ok','frame':9,'verdict':{'signal':.25,
        'raw_transition_signal':1.,'shared_frame_conflict':True}},
        {'status':'ok','frame':10,'verdict':{'signal':0.,'raw_transition_signal':0.}}])], 5)[0]['frame'] == 9
    print('SELF-TEST PASS: V3 two-frame prompt, generic actor, both crossing directions, NMS')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--agent0-only', action='store_true')
    ap.add_argument('--video-dir', type=Path, default=VIDEO_DIR)
    ap.add_argument('--cases', default='0053,0054,0062')
    ap.add_argument('--prefix', default='Cosmos3_seed103_')
    ap.add_argument('--output', type=Path, default=PROJECT/'typeA_backward_twoframe_generic_20261011')
    ap.add_argument('--model', type=Path, default=MODEL)
    ap.add_argument('--max-tokens', type=int, default=520)
    ap.add_argument('--max-audits', type=int, default=12)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.max_audits < 1:
        ap.error('max-audits must be positive')
    ids = [x.strip() for x in args.cases.split(',') if x.strip()]
    if not ids or not all(re.fullmatch(r'\d{4}', x) for x in ids) or len(ids) != len(set(ids)):
        ap.error('cases must be unique 4-digit comma-separated IDs')
    args.output = args.output.resolve()
    if args.output == PROJECT or 'twoframe_generic' not in args.output.name:
        ap.error('Use a dedicated NEW output folder containing twoframe_generic')
    runs = [(args.video_dir / f'{cid}.mp4', args.prefix + cid) for cid in ids]
    missing = [str(video) for video, _ in runs if not video.is_file()]
    if missing:
        raise FileNotFoundError('Missing video files: ' + ', '.join(missing))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run on allocated ERIS GPU compute node, not login node')
    sys.path.insert(0, str(PROJECT))
    import penetration_abc_agents_v3 as v3
    patch_v3(v3)
    sys.path.insert(0, str(WS))
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
    client = VLMClient('typeA_backward_twoframe_generic_v1', {
        'backend': 'local_gpu', 'model': str(args.model), 'max_tokens': args.max_tokens})
    errors = []
    for video, case in runs:
        try:
            run_case(client, v3, video, case, args)
        except Exception as exc:
            errors.append(case)
            print('FAILED', case, str(exc), flush=True)
            traceback.print_exc()
    if errors:
        raise SystemExit('Incomplete cases: ' + ', '.join(errors))


if __name__ == '__main__':
    main()
