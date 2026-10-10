#!/usr/bin/env python3
"""Reverse Type-A TEMPORAL diagnostic: 5 ordered frames, 1 full scene + 5 same-ROI crops.

This is a development-only experiment using manually chosen frame windows.
No Qwen proposal or human label is included in the model input. The original
OUT->IN pipeline and existing reverse Type-A scripts are not modified.

Examples:
  python typeA_reverse_temporal_v4.py --video /path/0056.mp4 \
    --frames 148,150,152,154,156 --case 0056 --output /path/results/0056 --preview-only
  python typeA_reverse_temporal_v4.py --video /path/0056.mp4 \
    --frames 148,150,152,154,156 --case 0056 --output /path/results/0056
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

WS = Path('/PHShome/zy992/Wilson/deformationdetection/workspace-cosmos3-0019-masking-20261007')
VERSION = 'reverse_typeA_temporal_v4_fiveframe_dev_20261010'

SYSTEM = """You are a cautious visual evidence evaluator for robot-manipulation video frames.
Analyze visible spatial and temporal evidence, not the assumed task outcome.
Some sequences are physically normal. Others may contain anomalous geometry.
Never infer a solid-wall crossing from 2D overlap, transparent appearance, or
absence of evidence alone. You may answer UNCLEAR. Output exactly one JSON object."""

PROMPT = """You receive SIX images from the same video, in the following order:
1. FULL SCENE at source frame {middle} (spatial context only)
2-6. LOCAL CROPS at source frames {frames}, in chronological order.
The five local crops use the EXACT SAME rectangle in the original camera view.
Frames are sampled at intervals; they are NOT necessarily consecutive.

MOVING TARGET: the SAME robot gripper/end-effector, especially its black jaws/tips.
ENVIRONMENT: a transparent sliding drawer with an open top and solid plastic walls.

Please perform a frame-by-frame temporal comparison:
A. Identify the gripper tip position relative to the SAME specific solid drawer
   front/side panel and open rim in each of the FIVE local images. Explain any
   cases where transparency makes the 3D position unidentifiable.
B. Do the start and end states visibly change from inside the drawer to outside,
   or does the gripper remain in essentially the same 3D region? Distinguish
   changes in depth ordering from purely projected (2D) movement.
C. Check whether a legal exit path is VISIBLY supported: e.g. moving above the
   rim and over the opening, or moving around an open edge. Since frames are
   sampled, an unseen legal route cannot automatically be ruled out.
D. Separately inspect whether a new line/edge/texture resembling the drawer
   boundary appears on the gripper and subsequently MOVES WITH the gripper.
   Treat reflections through clear plastic, ordinary occlusion, and compression
   artifacts as alternative explanations. A boundary-like mark alone is NOT
   proof of physical penetration.
E. Decide whether the sampled sequence provides direct evidence of crossing
   SOLID wall material without following a legal opening path. Be conservative:
   if side ordering or the physical wall is uncertain, say UNCLEAR.

Return a CONTINUOUS visual inconsistency score 0.0-1.0, not a calibrated
probability: 0.0 = visibly normal, 0.1-0.3 = weak/likely normal, 0.4-0.6 =
ambiguous but suspicious, 0.7-0.9 = strong visual evidence, 1.0 = exceptionally
clear. Do not increase the score merely because the task asks about crossing.
Do not assume the video is positive; ground every conclusion in the images.

Return exactly one JSON object:
{{
 "target_identified": "YES|NO|UNCLEAR",
 "frame_observations": ["frame {f0}: ...", "frame {f1}: ...", "frame {f2}: ...", "frame {f3}: ...", "frame {f4}: ..."],
 "initial_gripper_state": "INSIDE|OUTSIDE|ABOVE_OPENING|STRADDLING|UNCLEAR",
 "final_gripper_state": "INSIDE|OUTSIDE|ABOVE_OPENING|STRADDLING|UNCLEAR",
 "visible_inside_to_outside_transition": "YES|NO|UNCLEAR",
 "legal_exit_path_observed": "YES|NO|UNCLEAR",
 "solid_wall_crossing_evidence": "YES|NO|UNCLEAR",
 "boundary_imprint_evidence": "YES|NO|UNCLEAR",
 "boundary_imprint_reason": "What changes, whether it follows gripper, and alternatives",
 "visual_inconsistency_score": 0.0,
 "reason": "2-4 sentences describing concrete evidence, possible lawful exit/occlusion, uncertainty"
}}"""


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_json(raw):
    dec = json.JSONDecoder()
    for m in re.finditer(r'\{', raw):
        try:
            value, _ = dec.raw_decode(raw[m.start():])
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            pass
    raise ValueError('No complete JSON object in response (raw response saved).')


def parse_roi(raw):
    vals = tuple(float(x.strip()) for x in raw.split(','))
    if len(vals) != 4 or not all(math.isfinite(x) for x in vals):
        raise ValueError('ROI must contain four finite numbers')
    x1, y1, x2, y2 = vals
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('ROI must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1')
    return vals


def labeled(im, tag, side, enlarge=False):
    im = im.copy()
    scale = float(side)/max(im.size)
    if enlarge or scale < 1:
        im = im.resize((max(1,round(im.width*scale)), max(1,round(im.height*scale))),
                       Image.Resampling.LANCZOS)
    result = Image.new('RGB', (im.width,im.height+32), 'white')
    result.paste(im, (0,32))
    ImageDraw.Draw(result).text((9,9),tag,fill='black')
    return result


def extract_images(video: Path, times, roi, full_side, crop_side):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video {video}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 1 or times[0] < 0 or times[-1] >= n:
        cap.release()
        raise ValueError(f'Frame indices outside video range 0..{n-1}')
    needed = set(times)
    decoded = {}
    try:
        # Seek once, then decode sequentially to avoid repeated slow H264 seeking.
        cap.set(cv2.CAP_PROP_POS_FRAMES,times[0])
        for idx in range(times[0],times[-1]+1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'Could not decode source frame {idx}')
            if idx in needed:
                decoded[idx] = Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                print(f'  extracted f{idx}', flush=True)
    finally:
        cap.release()
    if set(decoded) != needed:
        raise RuntimeError('Missing decoded frame(s)')
    whs = {im.size for im in decoded.values()}
    if len(whs) != 1:
        raise ValueError('Frame dimensions not constant')
    w,h = next(iter(whs))
    x1,y1,x2,y2 = roi
    rect = (round(w*x1),round(h*y1),round(w*x2),round(h*y2))
    if rect[2]-rect[0]<50 or rect[3]-rect[1]<50:
        raise ValueError('ROI too small')
    full = labeled(decoded[times[2]],f'FULL SCENE | f{times[2]}',full_side)
    crops = [labeled(decoded[t].crop(rect),f'CROP {i+1}/5 | f{t}',crop_side,True)
             for i,t in enumerate(times)]
    return [full] + crops, rect


def preview_sheet(images):
    # Layout is for local preview ONLY; the API receives images separately in order.
    width,height=520,430
    sheet=Image.new('RGB',(3*width,2*height),'#f3f3f3')
    for i,im in enumerate(images):
        tile=im.copy(); tile.thumbnail((width-14,height-14),Image.Resampling.LANCZOS)
        sheet.paste(tile, ((i%3)*width+7,(i//3)*height+7))
    return sheet


def credentials(envfile):
    sys.path.insert(0,str(WS))
    from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
    from robot.preprocessing.link7_persistent.interface.config import role_config
    load_env_file(envfile)
    cfg=role_config('vlm2')
    if cfg.get('backend')!='cloud_api':
        raise RuntimeError('Expected Wilson vlm2 cloud_api backend')
    keyname=cfg.get('api_key_env','VLM2_API_KEY')
    key=os.getenv(keyname)
    if not key:
        raise RuntimeError(f'Missing {keyname} in environment')
    return str(cfg['api_base']).rstrip('/'),key


def ask_direct(images, text, base, key, model, max_tokens, timeout):
    import requests
    content=[{'type':'text','text':text}]
    sizes=[]
    for im in images:
        data=io.BytesIO()
        im.save(data,'JPEG',quality=83,optimize=True)
        raw=data.getvalue()
        sizes.append(round(len(raw)/1024,1))
        content.append({'type':'image_url','image_url':{
            'url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode('ascii')}})
    payload={'model':model,
             'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':content}],
             'temperature':0,'max_tokens':max_tokens}
    print(f'  Direct API: {model} images={len(images)} KB={sizes}; '
          f'timeout={timeout}s; retries=0',flush=True)
    start=time.monotonic()
    try:
        resp=requests.post(base+'/chat/completions',
                           headers={'Authorization':'Bearer '+key},
                           json=payload,timeout=(60,timeout))
    except requests.RequestException as exc:
        print(f'  API ERROR after {time.monotonic()-start:.1f}s: '
              f'{type(exc).__name__}: {str(exc)[:250]}',flush=True)
        raise
    elapsed=time.monotonic()-start
    print(f'  HTTP={resp.status_code} elapsed={elapsed:.2f}s',flush=True)
    if resp.status_code!=200:
        raise RuntimeError(f'API HTTP {resp.status_code}: {resp.text[:450]}')
    data=resp.json()
    choice=data['choices'][0]
    result=choice['message']['content']
    if isinstance(result,list):
        result='\n'.join(x.get('text','') for x in result if isinstance(x,dict))
    if not isinstance(result,str):
        raise ValueError('Unexpected response content type')
    return {'raw':result,'elapsed_seconds':round(elapsed,3),
            'finish_reason':choice.get('finish_reason'),
            'usage':data.get('usage'),'model':data.get('model',model)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,help='For filesystem naming only, not in prompt')
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--frames',required=True,help='Five ascending indices, comma-separated')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--roi-norm',default='0.28,0.26,0.76,0.90')
    p.add_argument('--full-side',type=int,default=960)
    p.add_argument('--crop-side',type=int,default=1000)
    p.add_argument('--max-tokens',type=int,default=1500)
    p.add_argument('--timeout',type=float,default=240)
    p.add_argument('--model',default='gemini-3.8-flash')
    p.add_argument('--env-file',type=Path,default=WS/'.env.vlm')
    p.add_argument('--preview-only',action='store_true')
    args=p.parse_args()
    times=tuple(int(v.strip()) for v in args.frames.split(','))
    if len(times)!=5 or list(times)!=sorted(set(times)):
        p.error('Exactly five strictly ascending, distinct frame indices required')
    if args.full_side<256 or args.crop_side<256 or args.max_tokens<32 or args.timeout<=0:
        p.error('Bad size/tokens/timeout')
    if not args.video.is_file():
        p.error('Video file not found: '+str(args.video))
    roi=parse_roi(args.roi_norm)
    prompt=PROMPT.format(middle=times[2],frames=', '.join('f'+str(v) for v in times),
                         **{'f'+str(i):t for i,t in enumerate(times)})
    vstat=args.video.stat()
    cfg={'version':VERSION,'case':args.case,'video':str(args.video.resolve()),
         'video_size':vstat.st_size,'video_mtime_ns':vstat.st_mtime_ns,
         'frames':list(times),'roi_norm':list(roi),
         'full_side':args.full_side,'crop_side':args.crop_side,
         'model':args.model,'max_tokens':args.max_tokens,
         'system':SYSTEM,'prompt':prompt,'layout':'one_full_middle_then_five_fixed_ROI_crops'}
    h=digest(cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    configfile=args.output/'run_config.json'
    if configfile.is_file():
        previous=json.loads(configfile.read_text())
        if previous.get('sha256')!=h:
            raise RuntimeError('Output contains a DIFFERENT config. Use new --output folder.')
    atomic_json(configfile, {'sha256':h,'settings':cfg})
    (args.output/'prompt.txt').write_text(SYSTEM+'\n\n'+prompt+'\n')

    print('CASE:',args.case,'FRAMES:',times,flush=True)
    print('DEVELOPMENT DIAGNOSTIC (manual frame selection; not full-video evaluation)',flush=True)
    images,rect=extract_images(args.video,times,roi,args.full_side,args.crop_side)
    print('ROI PIXELS:',rect,'IMAGES: 1 full + 5 crops',flush=True)
    preview=args.output/'preview.jpg'
    preview_sheet(images).save(preview,quality=91)
    print('PREVIEW:',preview,flush=True)
    if args.preview_only:
        print('PREVIEW ONLY: zero API requests',flush=True)
        return

    verdict_file=args.output/'verdict.json'
    response_file=args.output/'api_response.json'
    if verdict_file.is_file():
        cached=json.loads(verdict_file.read_text())
        if cached.get('config_sha256')!=h:
            raise RuntimeError('Cached result config mismatch')
        print('CACHED SCORE:',cached['visual_inconsistency_score'],flush=True)
        print('CACHED REASON:',cached['verdict'].get('reason',''),flush=True)
        return
    if response_file.is_file():
        data=json.loads(response_file.read_text())
        if data.get('config_sha256')!=h:
            raise RuntimeError('Cached raw response config mismatch')
        print('Using SAVED HTTP response; no second API call',flush=True)
        result=data['response']
    else:
        base,key=credentials(args.env_file)
        print('API BASE:',base,'(key hidden)',flush=True)
        result=ask_direct(images,prompt,base,key,args.model,args.max_tokens,args.timeout)
        # Persist raw answer before attempting JSON parsing.
        atomic_json(response_file,{'config_sha256':h,'response':result})
        print('RAW SAVED:',response_file,flush=True)
    verdict=parse_json(result['raw'])
    raw_score=verdict.get('visual_inconsistency_score')
    if isinstance(raw_score,bool):
        raise ValueError('Score must be numeric, not bool')
    score=float(raw_score)
    if not math.isfinite(score) or not 0<=score<=1:
        raise ValueError('Visual inconsistency score must lie in [0,1]')
    record={'config_sha256':h,'case':args.case,'frames':list(times),
            'visual_inconsistency_score':score,'verdict':verdict,
            'api_elapsed_seconds':result['elapsed_seconds']}
    atomic_json(verdict_file,record)
    print('\n========== V4 TEMPORAL RESULT ==========',flush=True)
    print('SCORE:',score,flush=True)
    for k in ('initial_gripper_state','final_gripper_state',
              'visible_inside_to_outside_transition','legal_exit_path_observed',
              'solid_wall_crossing_evidence','boundary_imprint_evidence',
              'boundary_imprint_reason','reason'):
        print(k+':',verdict.get(k,'(missing)'),flush=True)
    print('OUTPUT:',args.output,flush=True)

if __name__=='__main__':
    main()
