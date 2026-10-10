#!/usr/bin/env python3
"""Rewrite manifests for a label-blind *trajectory validity* test, without touching images.

Run on Mac before the existing sol_transparent_moreframes_v2.py judge command.
No API calls, server, video decoding, or credentials required.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

VERSION = 'transparent-trajectory-prompt-v3-20261010'
SYSTEM = """You are a visual physical-plausibility auditor for robot videos.
Judge whether the OBSERVED CONTINUOUS ACTION is physically possible, not whether two
objects have large image overlap or large movement. Evaluate legal actions and anomalies
symmetrically, without presuming any positive or negative case label. A transparent
panel can reveal background detail, create reflections and ambiguous apparent edges.
Use only the supplied frames and be candid about unresolved 3D depth. Cite actual
source frame IDs. Return a single JSON object only."""

PROMPT = """TASK: Evaluate the physical VALIDITY OF THE GRIPPER'S MOTION PATH relative to the
solid transparent drawer walls. Do not use the amount of visible image-plane motion,
overlap, or crossing as a shortcut for the score.

INPUT ORDER:
- Image 1 is a FULL-SCENE CONTEXT image at source frame f{middle}.
- The next {n} BROAD CROPS are the same spatial region, ordered chronologically:
  {frame_list}.
{focus_line}
They are sampled STILL FRAMES, not a continuous video. The crop moves neither
with the gripper nor with the drawer. The gripper, drawer, and camera may move
independently. Background objects visible through clear plastic are NOT panel texture.

QUESTION: From start to finish, is the motion PHYSICALLY VALID?
The gripper may start INSIDE the drawer and exit it, or start OUTSIDE and enter.
A visually obvious inside->outside or outside->inside transition can be FULLY LEGAL.
The legal route can be through the OPEN TOP, after the gripper is raised above
the rim and then displaced, or via a real free gap. The actual path matters.
A gripper and drawer moving relative to one another does NOT by itself imply
penetration; a panel shifting behind/in front of a gripper in the image may be
ordinary 3D separation.

Conversely, if the SAME black gripper jaw remains BELOW the relevant solid panel
rim and appears to directly pass THROUGH the SAME solid side/front wall without
a real opening, that is a penetration candidate. The WALL MOVING through a fixed
gripper is physically equivalent to the gripper moving through a fixed wall.
Look for repeated physically unsupported side changes or 'bouncing' back and
forth across the SAME solid boundary while the gripper does not clear the rim.
But DO NOT call this bouncing solely because a transparent edge oscillates,
reflection shifts, or camera projection changes. It must be the SAME gripper
feature versus the SAME physical wall edge over multiple consecutive frames.

REQUIRED METHOD (in this exact order):
A. Identify the BLACK GRIPPER'S LOWEST JAW/TIP and the drawer's SOLID wall and
   upper rim; distinguish both from background patterns visible through the wall.
B. Provide a chronological timeline of 3–6 important source frames, including
   evidence BEFORE, DURING, and AFTER the apparent transition. For each,
   separately record the gripper's position relative to the top rim, the
   panel's motion, and any visible clearance or separation.
C. Identify any apparent inside/outside or side-of-wall transition. Is it a
   smooth and PHYSICALLY ALLOWED movement via an observed opening/clearance,
   a likely projection-only change, an unsupported through-wall passage,
   or genuinely unresolved? A legal transition MUST NOT be penalized just
   because it is visually large or obvious. An unsupported wall crossing
   MUST NOT be dismissed merely because the drawer has an open top.
D. Specifically check for a persistent *wall-edge straddling or repeated flip*
   of the SAME jaw without a plausible trajectory; cite the source frames
   and say whether this is genuine or may be transparency/refraction.
E. Compare BOTH a concrete normal-motion explanation and a concrete anomalous
   explanation. State which explanation better fits the COMPLETE sequence.

IMPORTANT CAUTIONS:
- NO invented 3D depth or unshown intermediate movement. Lack of observed
  clearance is not conclusive proof of collision. Do not assume the physical
  start-side is known if transparency prevents identifying it.
- 2D crossing can occur legally. If the gripper clearly lifts above the
  rim BEFORE clearing the panel, that supports a normal exit/entry.
- If evidence is ambiguous, report UNCLEAR rather than pretending that a
  legitimate 3D route or solid intersection is visible.
- Output a graded action-invalidity evidence score in [0,1], NOT a calibrated
  probability of physical collision. Use the complete trajectory, not one
  dramatic transition:
  0.00–0.20 = clearly coherent/legal action or no suspicious path transition;
  0.20–0.40 = mostly plausible with minor unresolved appearance;
  0.40–0.60 = genuinely ambiguous; neither legitimate nor illegal route proven;
  0.60–0.80 = repeated or specific unsupported solid-boundary passage cues;
  0.80–1.00 = exceptionally compelling persistent physically impossible motion.
  A high score needs direct, frame-specific contradiction with legal motion;
  image-space overlap or failure to see depth is NOT enough by itself.

Return exactly JSON; use actual f-number references in evidence:
{{
  "timeline": [
    {{"frame":"f###","gripper_vs_rim":"...","wall_motion":"...","visible_clearance":"YES|NO|UNCLEAR","evidence":"..."}}
  ],
  "gripper_start_state":"INSIDE|OUTSIDE|UNCLEAR",
  "gripper_end_state":"INSIDE|OUTSIDE|UNCLEAR",
  "gripper_trajectory":"observed motion of the same jaw or tip",
  "drawer_wall_trajectory":"observed motion of the same solid wall",
  "observed_over_rim_or_free_gap_path":"YES|NO|UNCLEAR",
  "legal_path_support_frames":"specific source frame IDs and visible path, or NONE",
  "image_plane_boundary_crossing":"YES|NO|UNCLEAR",
  "unsupported_solid_wall_crossing":"YES|NO|UNCLEAR",
  "unsupported_crossing_support_frames":"source frame IDs and evidence, or NONE",
  "repeated_boundary_flips":"YES|NO|UNCLEAR",
  "repeated_flip_support_frames":"source frame IDs, or NONE",
  "trajectory_coherence":"COHERENT|INCONSISTENT|UNCLEAR",
  "depth_transparency_ambiguity":"YES|NO|UNCLEAR",
  "most_plausible_legal_explanation":"specific observed legal trajectory, or UNCLEAR",
  "most_plausible_failure_explanation":"specific observed impossible trajectory, or UNCLEAR",
  "visual_inconsistency_score":0.0,
  "reason":"Why is the observed physical ACTION coherent, inconsistent, or unresolved? Explain why mere 2D crossing does or does not matter."
}}"""


def manifest_paths(root: Path, ids: set[str]):
    paths = [root / 'manifest.json'] if (root / 'manifest.json').is_file() else sorted(root.glob('*/manifest.json'))
    results = []
    for p in paths:
        doc = json.loads(p.read_text(encoding='utf-8'))
        if str(doc.get('video_case', '')) in ids:
            results.append((p, doc))
    if not results:
        raise FileNotFoundError(f'No matching manifests in {root}, requested={sorted(ids)}')
    return results


def rewrite(src: Path, out: Path, cases: set[str], dry_run: bool = False):
    if src.resolve() == out.resolve():
        raise ValueError('Input and output directories must differ')
    for manifest, old in manifest_paths(src, cases):
        frames = old['source_frames']
        if len(frames) < 5 or sorted(set(frames)) != frames:
            raise ValueError(f'Invalid source frame order for {manifest}')
        names = old['images']
        if len(names) < len(frames) + 1:
            raise ValueError(f'Missing chronological crop images for {manifest}')
        if len(names) != len(set(names)):
            raise ValueError('Duplicate image names')
        for name in names:
            if Path(name).name != name:
                raise ValueError(f'Unsafe image path in {manifest}: {name}')
            if not (manifest.parent / name).is_file():
                raise FileNotFoundError(manifest.parent / name)
        n = len(frames)
        additional = len(names) - (n + 1)
        focus_line = (
            f'- The last {additional} images are ENLARGED RE-CROPS of already-listed source frames '
            '(read their f-number labels); they are NOT new independent observations. '
            'Use them to inspect details alongside corresponding broad crops.\n'
            if additional else ''
        )
        prompt = PROMPT.format(
            middle=frames[n//2], n=n,
            frame_list=', '.join('f'+str(x) for x in frames),
            focus_line=focus_line,
        )
        new = dict(old)
        new.update({
            'version': VERSION,
            'system_prompt': SYSTEM,
            'user_prompt': prompt,
            'prompt_hash': hashlib.sha256((SYSTEM+prompt).encode()).hexdigest(),
            'prior_prompt_hash': old.get('prompt_hash'),
            'prompt_aim': 'Blind legal-motion vs unsupported-through-wall comparison, not case-label-based',
        })
        dest = out / str(old['case'])
        print(f'{old["case"]}: {n} chronological frames + {additional} focus views = {len(names)} images -> {dest}')
        if dry_run:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(manifest.parent / name, dest / name)
        (dest / 'manifest.json').write_text(json.dumps(new, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        if (manifest.parent / 'focus_preview.jpg').is_file():
            shutil.copy2(manifest.parent / 'focus_preview.jpg', dest / 'focus_preview.jpg')
        if (manifest.parent / 'preview.jpg').is_file():
            shutil.copy2(manifest.parent / 'preview.jpg', dest / 'preview.jpg')
    if not dry_run:
        print(f'READY: {out} (same JPEG bytes; new prompts only)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=lambda x: Path(x).expanduser())
    p.add_argument('--output', required=True, type=lambda x: Path(x).expanduser())
    p.add_argument('--cases', default='0055,0056')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    ids = {x.strip() for x in a.cases.split(',')}
    if not ids or not all(re.fullmatch(r'\d{4}', x) for x in ids):
        p.error('--cases must be comma separated four-digit case IDs')
    rewrite(a.input, a.output, ids, a.dry_run)

if __name__ == '__main__':
    main()
