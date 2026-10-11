#!/usr/bin/env python3
"""V3 minimal generalization: gripper + drawer, with no appearance-specific wording.

Reuses identical images, frame windows, JSON response schema, and scoring rubric.
This is a manual *role-description ablation*, NOT an Agent0 integration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

VERSION = 'trajectory-v3-generic-gripper-drawer-20261010'


def replace_exact(text: str, old: str, new: str, case: str) -> str:
    n = text.count(old)
    if n != 1:
        raise ValueError(f'{case}: expected exactly 1 occurrence of {old!r}, found {n}. '
                         'Use the original V3 input folder, not an already-modified prompt.')
    return text.replace(old, new, 1)


PROMPT_EDITS = [
    (
        'TASK: Evaluate the physical VALIDITY OF THE GRIPPER\'S MOTION PATH relative to the\nsolid transparent drawer walls.',
        'TASK: Evaluate the physical VALIDITY OF THE ROBOT GRIPPER\'S MOTION PATH relative to the\nsolid wall or panel of the drawer.'
    ),
    (
        'Background objects visible through clear plastic are NOT panel texture.',
        'Do not mistake background patterns, reflections, or apparent edges for the actual solid wall.'
    ),
    (
        'The legal route can be through the OPEN TOP, after the gripper is raised above\nthe rim and then displaced, or via a real free gap.',
        'A legal route can pass through an opening, over or around a real solid edge,\nor through a genuine free gap. The path must be visible or supported by the sequence.'
    ),
    ('the SAME black gripper jaw remains BELOW the relevant solid panel',
     'the SAME gripper jaw or tip remains on the blocked side of the relevant solid panel'),
    ('because a transparent edge oscillates,', 'because an apparent wall edge oscillates,'),
    (
        "A. Identify the BLACK GRIPPER'S LOWEST JAW/TIP and the drawer's SOLID wall and\n   upper rim; distinguish both from background patterns visible through the wall.",
        'A. Track the SAME gripper jaw/finger/tip and the SAME solid drawer panel or\n   wall. Locate its physical edges and any genuine opening or free passage.'
    ),
    ('the gripper\'s position relative to the top rim, the',
     'the gripper\'s position relative to the relevant solid edge or opening, the'),
    ('because the drawer has an open top.',
     'merely because the drawer has an opening elsewhere.'),
    ('whether this is genuine or may be transparency/refraction.',
     'whether this is genuine or a projection, reflection, or occlusion artifact.'),
    ('if transparency prevents identifying it.',
     'if the supplied views do not identify it.'),
    (
        'If the gripper clearly lifts above the\n  rim BEFORE clearing the panel, that supports a normal exit/entry.',
        'If the gripper clearly follows a route through an opening or around a free\n  edge BEFORE appearing across the panel, that supports a normal exit/entry.'
    ),
]

SYSTEM_EDITS = [
    (
        'A transparent\npanel can reveal background detail, create reflections and ambiguous apparent edges.',
        'If a surface is transparent or reflective, background detail and reflections\ncan create ambiguous apparent edges.'
    ),
]


def run(src: Path, dst: Path, ids: set[str]):
    if src.resolve() == dst.resolve():
        raise ValueError('Input and output must differ.')
    if not src.is_dir():
        raise FileNotFoundError(src)
    found = set()
    for p in sorted(src.glob('*/manifest.json')):
        old = json.loads(p.read_text(encoding='utf-8'))
        case = str(old.get('video_case', ''))
        if case not in ids:
            continue
        if case in found:
            raise ValueError(f'Duplicate case {case}')
        found.add(case)
        if 'trajectory-prompt-v3' not in str(old.get('version', '')):
            raise ValueError(f'{p} is not the original V3 manifest ({old.get("version")!r}).')
        system = old['system_prompt']
        prompt = old['user_prompt']
        for before, after in SYSTEM_EDITS:
            system = replace_exact(system, before, after, case)
        for before, after in PROMPT_EDITS:
            prompt = replace_exact(prompt, before, after, case)
        if 'BLACK GRIPPER' in prompt or 'black gripper' in prompt or 'transparent drawer' in prompt.lower():
            raise ValueError(f'{case}: specific appearance wording remains')
        source_images = old['images']
        if len(source_images) != 19:
            raise ValueError(f'{case}: expected original 19 image inputs; got {len(source_images)}')
        target = dst / p.parent.name
        if target.exists():
            raise FileExistsError(f'{target} exists. Use a new --output directory.')
        target.mkdir(parents=True, exist_ok=False)
        for name in source_images:
            if Path(name).name != name:
                raise ValueError('Unsafe image path')
            shutil.copy2(p.parent/name, target/name)
        for extra in ('preview.jpg', 'focus_preview.jpg'):
            if (p.parent/extra).is_file():
                shutil.copy2(p.parent/extra, target/extra)
        new = dict(old)
        new.update(
            version=VERSION,
            system_prompt=system,
            user_prompt=prompt,
            prior_prompt_hash=old.get('prompt_hash'),
            prompt_hash=hashlib.sha256((system+prompt).encode()).hexdigest(),
            prompt_aim='V3 minimum object-description generalization; same frames, images, and scoring',
            actor_hint='robot gripper / end-effector',
            target_hint='drawer',
            agent0_used=False,
        )
        (target/'manifest.json').write_text(json.dumps(new, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f'{case}: copied {len(source_images)} unchanged images; new generic prompt -> {target}')
    missing = ids - found
    if missing:
        raise FileNotFoundError(f'Missing cases in source: {sorted(missing)}')
    print('READY. No API call, no Agent0, no model changes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', default='0055,0056,0057')
    a = parser.parse_args()
    run(a.input.expanduser(), a.output.expanduser(), set(a.cases.split(',')))


if __name__ == '__main__':
    main()
