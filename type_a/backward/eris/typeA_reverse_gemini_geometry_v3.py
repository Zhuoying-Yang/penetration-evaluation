#!/usr/bin/env python3
"""Reverse Type-A Gemini V3 prompt-only experiment.

Uses the existing V2 safe evaluator for all transport, 2-adjacent-frame
4-image formatting, source selection, NMS, scoring/gating and caches. Only
changes PROMPT and VERSION in memory; frozen GitHub source and V2 remain intact.

Focus: grounded visual evidence of gripper-solid-drawer intersection even when
there is negligible movement across adjacent frames. Not a detector of proven
3D physics; legal opening/ordinary transparency must reduce suspicion.

Examples:
  python typeA_reverse_gemini_geometry_v3.py --self-test
  python typeA_reverse_gemini_geometry_v3.py --mode diagnostic \\
    --source FULL_A_queries.json --video VIDEO.mp4 --focus-ends 186 --controls 1 \\
    --output UNIQUE_DIR --max-tokens 1500 --timeout 180
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).with_name('typeA_reverse_gemini_soft_v2_safe.py')
if not BASE.is_file():
    raise SystemExit(f'Missing V2 safe evaluator next to this script: {BASE}')

spec = importlib.util.spec_from_file_location('_typeA_reverse_soft_v2_safe_base', BASE)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

v2.VERSION = 'reverse_gemini_geometry_v3_prompt_only_20261010'

# Braces doubled for Python's str.format(before=..., after=...).
v2.PROMPT = '''Four images show EXACTLY TWO CONSECUTIVE original video frames in this order:
1. FULL BEFORE (frame {before})
2. FULL AFTER (frame {after})
3. CROP BEFORE (same region of frame {before}, possibly a full-frame fallback)
4. CROP AFTER (EXACT SAME region of frame {after})

MOVING TARGET: the SAME robot gripper/end-effector, especially its jaws and tips.
SOLID ENVIRONMENT: the drawer's solid FRONT or SIDE PANEL. An open top, drawer
opening, empty cavity, handle gap, or transparent surface is NOT automatically
proof of a solid-panel intersection. The gripper is the target, not a carried object.

TASK: Assign a CONTINUOUS visual-inconsistency score for evidence of the gripper
emerging from INSIDE/BEHIND the drawer through solid panel material to OUTSIDE/
IN FRONT, or being visibly embedded/straddling that same solid panel in an
impossible configuration. A suspicious event can last more than one frame:
therefore lack of measurable motion between THESE TWO adjacent frames does
NOT by itself prove that the visible geometry is physically valid. Conversely,
no displacement is NOT itself evidence of penetration.

OBSERVATION-FIRST CHECKLIST (do not skip):
A. Locate the SAME gripper tip/jaws and the ACTUAL solid front/side panel in
   BEFORE and AFTER separately. Distinguish panel material from rim/opening,
   transparent wall, handle, background, gripper shadow, and the drawer cavity.
B. Compare the gripper tip's contour and its relation to the SAME physical
   panel edge/surface in the two images. Is the tip continuously in free air,
   behind a translucent panel, or in contact with the front face? Or does a
   portion appear to emerge through/straddle solid material below the legal
   opening, with an incompatible contour, abnormal reappearance, or impossible
   depth/occlusion ordering? Report the CONCRETE visible feature and frame.
C. Does any change come from the DRAWER moving while the gripper stays fixed?
   Account for camera motion, transparency, perspective, ordinary occlusion,
   reflections, and physically permitted passage over/around the panel.
D. Distinguish (1) contact or projected overlap, (2) ambiguous solid-boundary
   geometry, and (3) strong directly visible intersection or impossible contour.
   Do not call a real physical wall crossing based solely on two apparent
   inside/outside labels, or because the top opening is not in these frames.
E. Record separately whether an INSIDE->OUTSIDE *transition* is actually visible
   between these frames and whether a local wall-intersection artifact is visible.
   The artifact may be visible with no state change across the pair.

SCORING (use your observations, not a numeric default or a Qwen proposal):
0.00: clear normal geometry, no specific visible anomaly.
0.10-0.30: weak / ambiguous concern tied to an identifiable contour or occlusion
           cue; do NOT assign nonzero just for generic uncertainty or contact.
0.40-0.60: specific suspicious wall-relative geometry, but opening/occlusion/
           transparency or 3D depth ambiguity prevents confident judgment.
0.70-0.90: strong visibly grounded evidence of gripper crossing/occupying solid
           panel material, inconsistent with an evident legal passage.
0.90-1.00: exceptional unmistakable direct evidence.
Normal exit via a genuine opening or over the rim is LOW. Two almost identical
frames can receive a HIGH score only if the solid-panel intersection itself is
clearly visible in BOTH or one of them; otherwise LOW or zero as warranted.
Do not hallucinate depth, unobserved motion, or frame history. If panel or
gripper identity cannot be established, do not make a confident claim.

Return exactly ONE valid JSON object (no markdown fencing), with keys:
{{
  "target_identified":"YES|NO|UNCLEAR",
  "solid_panel_identified":"YES|NO|UNCLEAR",
  "gripper_tip_before":"specific visible location and contour",
  "gripper_tip_after":"specific visible location and contour",
  "panel_relation_before":"FREE_SPACE|BEHIND_PANEL|IN_FRONT_OF_PANEL|AT_OPENING|SUSPECTED_INTERSECTION|OCCLUDED|UNCLEAR",
  "panel_relation_after":"FREE_SPACE|BEHIND_PANEL|IN_FRONT_OF_PANEL|AT_OPENING|SUSPECTED_INTERSECTION|OCCLUDED|UNCLEAR",
  "local_geometry_evidence":"one precise observed contour/occlusion fact, or NONE",
  "legal_exit_evidence":"CLEAR|PLAUSIBLE|NOT_OBSERVED|UNCLEAR",
  "ordinary_occlusion_evidence":"CLEAR|PLAUSIBLE|NOT_OBSERVED|UNCLEAR",
  "inside_to_outside_change":"YES|NO|UNCLEAR",
  "through_solid_panel_evidence":"YES|NO|UNCLEAR",
  "visual_inconsistency_score":0.0,
  "reason":"2-3 sentences grounded in the image pair, including alternative explanation"
}}'''


def self_test():
    p = v2.PROMPT.format(before=185, after=186)
    assert 'frame 185' in p and 'frame 186' in p
    assert 'no measurable motion' not in p or 'NO' in p
    assert 'lack of measurable motion' in p
    assert 'gripper' in p.lower() and 'solid' in p.lower()
    assert 'inside_to_outside_change' in p and 'visual_inconsistency_score' in p
    assert 'inside_to_outside_change' in p and 'through_solid_panel_evidence' in p
    assert v2.VERSION.startswith('reverse_gemini_geometry_v3')
    assert len(v2.select_topk({t:{'before':t-1,'after':t,'qwen_raw':1.,'qwen_post':.25,'qwen_conflict':True} for t in [99,102,105]},12,2)[0])==3
    print('PASS: V3 prompt rendering, schema, original two-frame V2 transport, new version, TopK selection unchanged')

if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
    else:
        v2.main()
