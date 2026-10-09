#!/usr/bin/env python3
"""Soft visual-inconsistency scoring of Qwen V3 Type-A candidate frame pairs.

Backend: Wilson VLM2 Gemini via 302.ai. Review all raw V3 positives,
using two full frames PLUS the same local object/container crop from both.
This is *not* a 3D-penetration proof or a calibrated probability.

Example (Eris):
  python penetration_A_v3_crop_gemini_score.py \
    --case COSMOS25_0006 --actor 'brown cup' --container 'beige trash bin' \
    --roi-norm 0.28,0.035,0.58,0.65
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import os
import re
import sys
from typing import Any

import cv2
from PIL import Image, ImageDraw

ROOT = Path('/scratch/z/zy992/zhuoying/physact/sam3_robowm/penetration_v1')
WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
MODEL = 'gemini-3.8-flash'
VERSION = 'typeA_wilson_gemini_crop_soft_v1_20261008'

SYSTEM = '''You are a cautious visual evidence scorer for robot manipulation videos.
Look only at the explicitly named moving target and solid container.
Observe visual facts first, then score possible physical inconsistency.
Do not invent objects or unseen intermediate movement. Do not treat the
existence of an open top as proof that the object used the opening.
Do not treat overlap or ordinary occlusion by itself as proof of penetration.
An image pair is sampled in time: unusual appearance can have alternative explanations.
Use the specified target identity throughout. Return a single JSON object only.'''

PROMPT = '''Four images are provided in this exact order:
1. FULL BEFORE (source frame {before})
2. FULL AFTER (source frame {after}, the next consecutive frame)
3. CROP BEFORE (magnified ROI of frame {before})
4. CROP AFTER (the EXACT SAME spatial ROI of frame {after})

MOVING TARGET: {actor}
CONTAINER WITH SOLID SIDES: {container}
The robot/gripper is not the target. Ignore all unrelated objects.

IMPORTANT: Your job is NOT first to classify the target OUTSIDE or INSIDE.
Do NOT output score zero merely because the target is near or partly hidden by
 the container in both frames. Instead observe exactly what is visible:

(A) In EACH image identify the named target, its visible lower boundary/tip,
    the container's top rim, solid side wall, and the open mouth.
(B) Compare the target's SAME lower portion and silhouette in the two crops.
    Has its lower part disappeared behind the solid wall or become cut off?
    Does the visible bottom move? Has the target moved upward sufficiently?
(C) Consider the cup/object bottom versus rim in BOTH frames, but do not
    confuse 2D image y coordinates with true 3D elevation.
(D) If the target is moved ABOVE the opening and lowered through the top,
    that is a LEGAL entry and should receive a LOW inconsistency score.
(E) Normal occlusion or perspective may also explain apparent truncation.
    If so lower the score, but explain why. A target shifting behind a wall
    while its lower part appears below the rim can be suspicious even if
    you cannot assign unambiguous 'outside'/'inside' labels.
(F) Do not infer a penetration merely because the target and container overlap.

Return a CONTINUOUS VISUAL INCONSISTENCY SCORE in [0,1]:
0.00 = clear no abnormality; 0.1-0.3 = likely normal; 0.4-0.6 = unresolved;
0.7-0.9 = substantial physical-inconsistency evidence; 1.0 = exceptionally strong evidence.
This score expresses visual support for abnormal target-solid-wall interaction,
NOT the probability that a physical collision was mathematically proven.
Neither a strict OUTSIDE->INSIDE transition nor hard binary verdict is required.
Do not automatically reward a proposal; ground the score in what you SEE.

Return exactly one JSON object with keys:
{{
  "target_identified": "YES|NO|UNCLEAR",
  "target_lower_part_before": "brief grounded visible observation",
  "target_lower_part_after": "brief grounded visible observation",
  "rim_relation_before": "BELOW|ABOVE|STRADDLING|UNCLEAR",
  "rim_relation_after": "BELOW|ABOVE|STRADDLING|UNCLEAR",
  "lower_contour_change": "INCREASING_OCCLUSION|DECREASING_OCCLUSION|SIMILAR|OTHER|UNCLEAR",
  "legal_top_entry_evidence": "CLEAR|PLAUSIBLE|NOT_OBSERVED|UNCLEAR",
  "ordinary_occlusion_evidence": "CLEAR|PLAUSIBLE|NOT_OBSERVED|UNCLEAR",
  "movement_evidence": "specific observed changes, or unclear",
  "visual_inconsistency_score": 0.0,
  "reason": "2-4 concise sentences: actual observed boundary changes, why normal or suspicious, and limitations"
}}'''


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def parse_json_object(answer: str) -> dict:
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{', answer):
        try:
            value, _ = decoder.raw_decode(answer[m.start():])
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            continue
    raise ValueError('No parseable JSON object in Gemini response: ' + repr(answer[:400]))


def score_of(value: dict) -> float:
    try:
        raw = value['visual_inconsistency_score']
        if isinstance(raw, bool):
            raise ValueError('Boolean is not a score')
        score = float(raw)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError('Gemini must return visual_inconsistency_score as a number') from e
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f'Score must be in [0,1], got {score}')
    return score


def get_candidates(source: Path, minimum: float):
    data = json.loads(source.read_text())
    if data.get('agent') not in (None, 'A'):
        raise ValueError('Expected V3 A_queries.json')
    seen = set()
    res = []
    for row in data.get('rows', []):
        if row.get('status') != 'ok':
            continue
        verdict = row.get('verdict') or {}
        raw = float(verdict.get('raw_transition_signal', verdict.get('signal', 0)))
        if raw < minimum:
            continue
        end = int(row['frame'])
        start = int(row.get('prev', end - 1))
        if end != start + 1 or start < 0:
            raise ValueError(f'Expected adjacent pair, got f{start}->f{end}')
        if (start, end) in seen:
            raise ValueError(f'Duplicate candidate pair: {start}->{end}')
        seen.add((start, end))
        res.append({'before': start, 'after': end, 'qwen_v3_raw': raw,
                    'qwen_v3_post': verdict.get('signal'),
                    'qwen_v3_observation': verdict.get('observation', '')})
    return sorted(res, key=lambda x: (x['before'], x['after']))


def scan_coverage(src: Path, video_frames: int) -> dict:
    """Distinguish a scored frame window from a full-video V3 evaluation."""
    data=json.loads(src.read_text())
    start=data.get('start')
    end=data.get('end')
    rows=data.get('rows',[])
    if isinstance(start,int) and isinstance(end,int):
        expected=max(0,end-max(1,start)+1)
        ok_frames={r.get('frame') for r in rows if r.get('status')=='ok'}
        complete_window=(len(ok_frames)==expected and
                         all(t in ok_frames for t in range(max(1,start),end+1)))
        whole_video=complete_window and start<=1 and end==video_frames-1
    else:
        complete_window=False
        whole_video=False
    return {'source_range':[start,end],'source_n_rows':len(rows),
            'source_window_complete':complete_window,
            'source_full_video_complete':whole_video,
            'video_n_frames':video_frames}


def find_video(case: str, catalog: Path):
    if not catalog.is_file():
        return None
    data = json.loads(catalog.read_text())
    found = next((c for c in data.get('cases', []) if c.get('id') == case), None)
    return Path(found['video']) if found else None


class Frames:
    def __init__(self, video: Path):
        self.video = video
        if video.is_dir():
            self.files = sorted(p for p in video.iterdir()
                                if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
            self.cap = None
            self.n = len(self.files)
        else:
            self.files = None
            self.cap = cv2.VideoCapture(str(video))
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open video: {video}')
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def read(self, index: int):
        if index < 0 or index >= self.n:
            raise IndexError(f'Frame {index} out of range for {self.video} (n={self.n})')
        if self.files is not None:
            return Image.open(self.files[index]).convert('RGB')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {index}')
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    def close(self):
        if self.cap:
            self.cap.release()


def parse_roi(raw: str):
    if not raw:
        return None
    xy = [float(x.strip()) for x in raw.split(',')]
    if len(xy) != 4 or not all(math.isfinite(x) for x in xy):
        raise ValueError('ROI must be x1,y1,x2,y2')
    a, b, c, d = xy
    if not (0 <= a < c <= 1 and 0 <= b < d <= 1):
        raise ValueError('--roi-norm values must be fractions in [0,1] with x1<x2 and y1<y2')
    return tuple(xy)


def crop_box(size, roi):
    w, h = size
    if roi is None:
        # Conservative fallback: show center, but use explicit ROI for a reliable comparison.
        roi = (.15, .03, .85, .80)
    x1, y1, x2, y2 = roi
    box = (max(0, round(x1*w)), max(0, round(y1*h)),
           min(w, round(x2*w)), min(h, round(y2*h)))
    if box[2]-box[0] < 32 or box[3]-box[1] < 32:
        raise ValueError('ROI too small; include both target and container rim/wall')
    return box


def resize_labeled(im, label, max_side, enlarge=False):
    im = im.copy()
    if enlarge:
        # PIL thumbnail NEVER upscales: the crop must actually be enlarged for the VLM.
        scale = max_side / max(im.size)
        im = im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),
                       Image.Resampling.LANCZOS)
    else:
        im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    out = Image.new('RGB', (im.width, im.height+32), 'white')
    out.paste(im,(0,32))
    ImageDraw.Draw(out).text((8,10),label,fill='black')
    return out


def contact_sheet(images):
    # [fullbefore, fullafter, cropbefore, cropafter]
    before, after, crop0, crop1 = images
    wide = max(before.width+after.width, crop0.width+crop1.width)
    high = max(before.height,after.height) + max(crop0.height,crop1.height)
    sheet=Image.new('RGB',(wide,high),'white')
    sheet.paste(before,(0,0))
    sheet.paste(after,(before.width,0))
    y=max(before.height,after.height)
    sheet.paste(crop0,(0,y))
    sheet.paste(crop1,(crop0.width,y))
    return sheet


def prepared_images(frames: Frames, before: int, after: int, roi,
                    full_side: int, crop_side: int):
    fr0,fr1=frames.read(before),frames.read(after)
    if fr0.size != fr1.size:
        raise ValueError('Frames differ in dimensions; cannot use the same ROI')
    box=crop_box(fr0.size,roi)
    full0=resize_labeled(fr0,f'FULL BEFORE f{before}',full_side)
    full1=resize_labeled(fr1,f'FULL AFTER f{after}',full_side)
    crop0=resize_labeled(fr0.crop(box),f'CROP BEFORE f{before} (same ROI)',crop_side,enlarge=True)
    crop1=resize_labeled(fr1.crop(box),f'CROP AFTER f{after} (same ROI)',crop_side,enlarge=True)
    return [full0,full1,crop0,crop1],box


def video_identity(video: Path):
    if video.is_dir():
        return {'path':str(video.resolve()), 'n_files':len(list(video.iterdir()))}
    st = video.stat()
    return {'path':str(video.resolve()), 'size':st.st_size,'mtime_ns':st.st_mtime_ns}


def config_hash(args, src: Path, vid: Path, roi):
    settings = {'version':VERSION, 'v3_file':str(src.resolve()),
                'v3_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
                'video':video_identity(vid), 'model':str(args.model),
                'actor':args.actor,'container':args.container,'min_v3_raw':args.min_v3_raw,
                'roi_norm':roi,'full_side':args.full_side,'crop_side':args.crop_side,
                'max_tokens':args.max_tokens,'system':SYSTEM,'prompt':PROMPT}
    h = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    return settings, h


def run(args, client=None):
    src = args.v3_json or args.v3_root/args.case/'A_queries.json'
    if not src.is_file():
        raise FileNotFoundError(f'Missing Qwen V3 proposals: {src}')
    cand_all=get_candidates(src,args.min_v3_raw)
    to_audit=cand_all
    if args.only_pair:
        b,a = [int(t) for t in args.only_pair.split(':')]
        to_audit=[c for c in cand_all if (c['before'],c['after'])==(b,a)]
        if not to_audit:
            raise ValueError(f'f{b}->f{a} is not a V3 candidate; run --dry-run first')
    print(f'V3 raw candidates: {len(cand_all)}; selected for this run: {len(to_audit)}',flush=True)
    for c in to_audit:
        print(f'  f{c["before"]:04d}->f{c["after"]:04d} raw={c["qwen_v3_raw"]:.2f}',flush=True)
    if args.dry_run:
        return {'candidates':cand_all}

    video=args.video or find_video(args.case,args.catalog)
    if video is None or not video.exists():
        raise FileNotFoundError('Video not found; pass --video explicitly')
    roi=parse_roi(args.roi_norm)
    config,run_hash=config_hash(args,src,video,roi)
    out=args.output/args.case
    out.mkdir(parents=True,exist_ok=True)
    cfg_file=out/'config.json'
    if cfg_file.exists():
        old=json.loads(cfg_file.read_text())
        if old.get('sha256')!=run_hash:
            raise RuntimeError(f'Output cache is for different settings: {out}. Use --output NEW_FOLDER or --force.')
    save_json(cfg_file, {'sha256':run_hash, 'settings':config})

    frames=Frames(video)
    coverage=scan_coverage(src,frames.n)
    if not coverage["source_full_video_complete"]:
        print("NOTE: V3 candidate source is NOT a complete full-video scan; the score below covers its scanned window only.",flush=True)
    results=[]
    try:
        for ix,c in enumerate(to_audit,1):
            b,a=c['before'],c['after']
            rec_path=out/f'score_f{b:04d}_f{a:04d}.json'
            if rec_path.is_file() and not args.force:
                record=json.loads(rec_path.read_text())
                if record.get('config_sha256')!=run_hash:
                    raise ValueError('Stale cache record, use --force or a new output directory')
                conf=score_of(record['auditor_verdict'])
                print(f'CACHED [{ix}/{len(to_audit)}] f{b}->f{a} score={conf:.3f}',flush=True)
            else:
                images,box=prepared_images(frames,b,a,roi,args.full_side,args.crop_side)
                img_path=out/f'visual_f{b:04d}_f{a:04d}.jpg'
                contact_sheet(images).save(img_path,quality=94)
                if args.preview_only:
                    print(f'PREVIEW f{b}->f{a}: {img_path} source ROI={box}',flush=True)
                    continue
                if client is None:
                    sys.path.insert(0, str(WS))
                    from robot.preprocessing.link7_persistent.interface.config import role_config
                    from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
                    from robot.preprocessing.link7_persistent.vlm_client import VLMClient
                    env_path = args.env_file or WS / '.env.vlm'
                    load_env_file(Path(env_path))
                    config = role_config('vlm2')
                    if config.get('backend') != 'cloud_api':
                        raise RuntimeError('Wilson VLM2 is not configured for cloud_api')
                    if args.model:
                        config['model'] = args.model
                    config['max_tokens'] = args.max_tokens
                    config['temperature'] = 0
                    # Gemini models are not sent OpenAI-specific reasoning_effort.
                    config['reasoning_effort'] = None
                    key_env = config['api_key_env']
                    if not os.environ.get(key_env):
                        raise RuntimeError(
                            f'{key_env} not configured. Wilson env file checked: {env_path}. '
                            f'If using that file, unset any stale {key_env} first.')
                    print('Wilson Gemini:', config['model'],
                          'API base:', config['api_base'],
                          'key variable:', key_env,
                          '(value hidden)', flush=True)
                    client=VLMClient('typeA_wilson_gemini_crop_auditor', config)
                prompt=PROMPT.format(before=b,after=a,actor=args.actor,container=args.container)
                response=client.ask(images,prompt,system_prompt=SYSTEM)
                verdict=parse_json_object(response['answer'])
                conf=score_of(verdict)
                record={'case':args.case,'pair':[b,a], 'qwen_v3':c,
                        'config_sha256':run_hash,'visual_path':str(img_path),
                        'crop_box_pixels':list(box), 'model':str(args.model),
                        'auditor_verdict':verdict,'raw_answer':response['answer'],
                        'visual_inconsistency_score':conf}
                save_json(rec_path,record)
                print(f'AUDIT [{ix}/{len(to_audit)}] f{b}->f{a} SCORE={conf:.3f} '
                      f'contour={verdict.get("lower_contour_change")} '
                      f'legal={verdict.get("legal_top_entry_evidence")}',flush=True)
                print('  BEFORE:',str(verdict.get('target_lower_part_before',''))[:240],flush=True)
                print('  AFTER :',str(verdict.get('target_lower_part_after',''))[:240],flush=True)
                print('  REASON:',str(verdict.get('reason',''))[:600],flush=True)
            results.append({'pair':[b,a], 'score':conf, 'record':str(rec_path)})
            best=max(results,key=lambda r:r['score'])
            summary={'case':args.case,'version':VERSION, 'n_v3_candidates':len(cand_all),
                     'n_audited_this_run':len(results),
                     'partial_selection':bool(args.only_pair),
                     'source_coverage':coverage,
                     'complete':not bool(args.only_pair) and len(results)==len(cand_all) and coverage['source_window_complete'],
                     'score_rule':'max unthresholded Gemini visual inconsistency score across all V3 raw candidates',
                     'pair_scores':results,'best_pair_so_far':best['pair'],
                     'best_score_so_far':best['score'],
                     'video_score':(best['score'] if not args.only_pair and len(results)==len(cand_all) and coverage['source_window_complete'] else None),
                     'score_scope':'full_video' if coverage['source_full_video_complete'] else 'V3_scanned_window_only'}
            save_json(out/'summary.json',summary)
        success=True
    finally:
        frames.close()
    if args.preview_only:
        print('PREVIEW ONLY. No inference; no scored summary.',flush=True)
        return None
    if not results and len(cand_all)==0:
        summary={'case':args.case,'version':VERSION,'n_v3_candidates':0,'n_audited_this_run':0,
                 'partial_selection':False, 'complete':coverage['source_window_complete'],
                 'source_coverage':coverage,
                 'video_score':0.0 if coverage['source_window_complete'] else None,
                 'score_rule':'no V3 candidate -> 0 (provided V3 itself completed successfully)',
                 'pair_scores':[]}
        save_json(out/'summary.json',summary)
    else:
        summary=json.loads((out/'summary.json').read_text())
    print('SUMMARY:',json.dumps({k:v for k,v in summary.items() if k!='pair_scores'},ensure_ascii=False),flush=True)
    print('OUTPUT:',out,flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--actor',required=True)
    p.add_argument('--container',required=True)
    p.add_argument('--v3-root',type=Path,default=ROOT/'penetration_abc_v3')
    p.add_argument('--v3-json',type=Path)
    p.add_argument('--video',type=Path)
    p.add_argument('--catalog',type=Path,default=ROOT/'wilson_batch_gt10_v1/catalog.json')
    p.add_argument('--model',default=MODEL,help='Wilson VLM2 model name (default gemini-3.8-flash)')
    p.add_argument('--env-file',type=Path,default=None,help='Wilson .env.vlm file (default workspace root .env.vlm)')
    p.add_argument('--output',type=Path,default=ROOT/'penetration_A_v3_crop_gemini_v1')
    p.add_argument('--roi-norm',default=None,help='x1,y1,x2,y2 as 0..1 fractions of original frame')
    p.add_argument('--full-side',type=int,default=960)
    p.add_argument('--crop-side',type=int,default=1000)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--min-v3-raw',type=float,default=.95)
    p.add_argument('--only-pair',help='Debug a single candidate (e.g. 22:23). Video score stays null.')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--preview-only',action='store_true',help='Save full/crop contact sheet, no model')
    p.add_argument('--force',action='store_true',help='Re-run inference for existing per-pair JSON')
    args=p.parse_args()
    if not 0 <= args.min_v3_raw <= 1:
        p.error('--min-v3-raw must be in [0,1]')
    if not (256<=args.full_side<=2048 and 256<=args.crop_side<=2048):
        p.error('Image sides must be 256..2048')
    run(args)

if __name__=='__main__':main()
