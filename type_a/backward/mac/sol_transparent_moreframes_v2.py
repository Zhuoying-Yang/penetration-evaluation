#!/usr/bin/env python3
"""Prepare 9+ chronological robot-video frames on ERIS, judge later using lab Sol on Tailscale Mac.

ERIS: prepare (no key, no API, CPU only) --video-base ... --spec 0056:144,146,...
Mac:  judge --input downloaded_folder --output ... --env-file local.env --only-case 0056

Distinct case+frame-window IDs prevent accidental overwrite. Development manually
chosen clips are NOT full-video automated evaluation or confirmed frame-level GT.
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, math, os, re, time
from pathlib import Path

VERSION = 'sol-transparent-moreframes-v2-20261010'
SYSTEM = '''You are an independent, cautious visual evidence evaluator of robot-video physical plausibility.
Use only the supplied images. Do NOT know any human labels or expected answer. Do NOT assume an image-space overlap proves a 3D collision. Transparent plastic creates ambiguous occlusion and refraction. An open top alone does not establish a legal exit. A moving drawer wall can cross the stationary gripper just as the gripper could cross a stationary wall. Be specific about observable evidence and uncertainty. Return one JSON object only.'''
PROMPT = '''Evaluate the following ordered snapshots of a robot gripper and a CLEAR TRANSPARENT sliding drawer.
The first image is a FULL scene context at frame {middle}. The remaining {n} cropped images depict the SAME fixed image region at source frames {frames}, in chronological order. They are samples, NOT a continuous movie; unshown intermediate movements cannot be ruled out. Do not infer a physically invalid path simply because a legal action is absent from sampled frames.

Objects: black gripper jaw and tip; transparent drawer with SOLID front/side panels, upper rim, and OPEN TOP; surrounding beige drawer housing. The DRAWER may move while gripper remains approximately fixed. Track the SAME front/side panel and the SAME gripper portion through the sequence.

Determine separately:
1. Visually supported horizontal motion of the drawer wall relative to gripper, and any crossing of their positions in the IMAGE PLANE.
2. Whether the lowest gripper part is demonstrably ABOVE the moving rim at the time wall passes, or there is an observable gap/valid side route. Saying 'it has an open top' is insufficient without an actual path or clearance. Cite frame IDs for observed clearance.
3. Evidence that solid panel SWEPT THROUGH occupied gripper volume, versus normal transparency/occlusion, apparent depth-order change, or visual ambiguity. Image overlap alone is insufficient evidence.
4. Whether a drawer-boundary-like mark seems to persist on the moving gripper (only if visible and not a reflection or ordinary occlusion).
5. State what can and cannot be concluded from these sampled images. No guessed frames or invented 3D depth.

Scoring is visual evidence of an abnormal interaction, NOT a confirmed 3D collision probability:
0.0–0.2: clear normal mechanism with visible legal route or no specific anomaly;
0.2–0.4: minor visual concern or likely ordinary occlusion;
0.4–0.6: specific suspicious relative geometry but true passage still ambiguous;
0.6–0.9: strong visible evidence for physical inconsistency and no supported alternative;
0.9–1.0: exceptionally compelling direct evidence.
Do not force 0 for merely stationary gripper, nor force a high score just because the drawer moves. Use UNCLEAR when not resolved.

Return exactly JSON with the following keys:
{{
 "gripper_early_middle_late":"specific visible state of the gripper tip over these stages",
 "panel_early_middle_late":"specific visible motion of same solid drawer panel",
 "relative_wall_motion":"YES|NO|UNCLEAR",
 "image_plane_boundary_crossing":"YES|NO|UNCLEAR",
 "rim_cleared_during_crossing":"YES|NO|UNCLEAR",
 "actual_legal_path_observed":"YES|NO|UNCLEAR",
 "legal_path_support_frames":"frame IDs and what is visibly observed, or NONE",
 "strong_solid_intersection_evidence":"YES|NO|UNCLEAR",
 "solid_intersection_support_frames":"frame IDs and actual visible evidence, or NONE",
 "depth_transparency_ambiguity":"YES|NO|UNCLEAR",
 "boundary_mark_artifact":"YES|NO|UNCLEAR",
 "normal_explanation":"most credible normal interpretation grounded in the frames",
 "failure_explanation":"most credible anomalous interpretation grounded in the frames",
 "visual_inconsistency_score":0.0,
 "reason":"concise contrast of evidence for vs. against and remaining uncertainty"
}}'''

def save_json(path, data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8');tmp.replace(p)

def parse_spec(s):
    if ':' not in s: raise ValueError('Expected ID:frame0,frame1,...')
    cid,rr=s.split(':',1);cid=cid.strip()
    if not re.fullmatch(r'\d{4}',cid):raise ValueError('Case must be 4 digits, e.g. 0056')
    frames=[int(x.strip()) for x in rr.split(',')]
    if len(frames)<5 or len(frames)>16 or frames!=sorted(set(frames)) or frames[0]<0:
        raise ValueError('Use 5–16 DISTINCT increasing nonnegative frame IDs')
    return cid,frames

def crop_norm(s):
    vals=[float(x) for x in s.split(',')]
    if len(vals)!=4 or not(0<=vals[0]<vals[2]<=1 and 0<=vals[1]<vals[3]<=1):raise ValueError('Invalid normalized ROI')
    return vals

def load_env(path):
    if not path:return
    p=Path(path).expanduser()
    if not p.is_file():raise FileNotFoundError(f'Cannot find .env at {p}')
    for line in p.read_text(encoding='utf-8').splitlines():
        v=line.strip()
        if not v or v.startswith('#'):continue
        if v.startswith('export '):v=v[7:].strip()
        if '=' not in v:continue
        key,value=v.split('=',1);key=key.strip();value=value.strip()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',key):continue
        if len(value)>=2 and value[0]==value[-1] and value[0] in ('"',"'"):value=value[1:-1]
        os.environ.setdefault(key,value)

def extract_frames(video,frames):
    import cv2
    from PIL import Image
    c=cv2.VideoCapture(str(video))
    if not c.isOpened():raise RuntimeError(f'Cannot open {video}')
    n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));w=int(c.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames[-1]>=n:raise ValueError(f'{video}: requested frame {frames[-1]}, total {n}')
    c.set(cv2.CAP_PROP_POS_FRAMES,frames[0]);want=set(frames);output={}
    try:
        for f in range(frames[0],frames[-1]+1):
            ok,bgr=c.read()
            if not ok:raise RuntimeError(f'Could not decode frame {f} in {video}')
            if f in want:output[f]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    finally:c.release()
    if len(output)!=len(frames):raise RuntimeError('Not all frames extracted')
    return output,(w,h,n)

def resize_image(img,side):
    from PIL import Image
    out=img.copy();out.thumbnail((side,side),Image.Resampling.LANCZOS)
    return out

def labeled(img,label):
    from PIL import Image,ImageDraw
    pad=30
    out=Image.new('RGB',(img.width,img.height+pad),'white')
    out.paste(img,(0,pad));ImageDraw.Draw(out).text((9,8),label,fill='black')
    return out

def contact_sheet(images):
    from PIL import Image,ImageDraw
    cols=3;tile_w=360;tile_h=310
    rows=(len(images)+cols-1)//cols
    out=Image.new('RGB',(cols*tile_w,rows*tile_h),'#eeeeee')
    for i,(name,img) in enumerate(images):
        q=img.copy();q.thumbnail((tile_w-12,tile_h-42),Image.Resampling.LANCZOS)
        x=(i%cols)*tile_w+(tile_w-q.width)//2;y=(i//cols)*tile_h+32
        out.paste(q,(x,y));ImageDraw.Draw(out).text(((i%cols)*tile_w+8,(i//cols)*tile_h+9),name,fill='black')
    return out

def prepare(a):
    for spec in a.spec:
        cid,frames=parse_spec(spec);video=Path(a.video_base)/f'{cid}.mp4'
        ims,(w,h,n)=extract_frames(video,frames)
        x1,y1,x2,y2=crop_norm(a.roi_norm)
        rect=[round(x1*w),round(y1*h),round(x2*w),round(y2*h)]
        mid=frames[len(frames)//2]
        window=f'{cid}_f{frames[0]:04d}_{frames[-1]:04d}'
        folder=Path(a.output)/window;folder.mkdir(parents=True,exist_ok=True)
        files=[]
        full=labeled(resize_image(ims[mid],a.full_side),f'FULL context source f{mid}')
        images=[('00_full_context.jpg',full)]
        for i,frame in enumerate(frames,1):
            cr=resize_image(ims[frame].crop(tuple(rect)),a.crop_side)
            images.append((f'{i:02d}_crop_f{frame:04d}.jpg',labeled(cr,f'CROP f{frame} same spatial ROI')))
        for name,img in images:
            img.save(folder/name,format='JPEG',quality=a.quality,optimize=True)
            files.append(name)
        contact_sheet(images).save(folder/'preview.jpg',quality=88)
        prompt=PROMPT.format(middle=mid,n=len(frames),frames=', '.join('f'+str(k) for k in frames))
        save_json(folder/'manifest.json',{
            'version':VERSION,'case':window,'video_case':cid,'source_frames':frames,
            'video_filename':video.name,'frame_size':[w,h],'roi_pixels':rect,'images':files,
            'system_prompt':SYSTEM,'user_prompt':prompt,
            'prompt_hash':hashlib.sha256((SYSTEM+prompt).encode()).hexdigest(),
            'selection':'MANUAL_DIAGNOSTIC_NOT_GT_BLIND',
            'note':'Unannotated original images, not the hand-drawn panel overlay. Development selected windows; not full-video metrics.'})
        print(f'PREPARED {window}: {len(frames)} source frames; {len(files)} images; ROI={rect}; output={folder}',flush=True)
    print('PREPARE COMPLETE. Copy folder to Mac while VPN active; no API calls or credentials used.',flush=True)

def get_response_text(r):
    txt=getattr(r,'output_text',None)
    if isinstance(txt,str) and txt.strip():return txt
    vals=[]
    for block in getattr(r,'output',[]) or []:
        for item in getattr(block,'content',[]) or []:
            s=getattr(item,'text',None)
            if isinstance(s,str):vals.append(s)
    return '\n'.join(vals)

def parse_score(raw):
    dec=json.JSONDecoder()
    for m in re.finditer(r'\{',str(raw)):
        try:d,_=dec.raw_decode(raw[m.start():])
        except ValueError:continue
        if isinstance(d,dict) and 'visual_inconsistency_score' in d:
            x=d['visual_inconsistency_score']
            if isinstance(x,bool):raise ValueError('Boolean score invalid')
            score=float(x)
            if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid numeric score')
            d['visual_inconsistency_score']=score
            return d
    return None

def judge(a):
    root=Path(a.input).expanduser()
    dirs=[root] if (root/'manifest.json').exists() else sorted(p.parent for p in root.glob('*/manifest.json'))
    if a.only_case:
        dirs=[p for p in dirs if json.loads((p/'manifest.json').read_text())['video_case'] == a.only_case]
    if not dirs:raise FileNotFoundError('No input manifests for selected case')
    if not a.dry_run:
        load_env(a.env_file)
        key=os.environ.get('OPENAI_API_KEY')
        base=(a.api_base or os.environ.get('OPENAI_BASE_URL') or '').rstrip('/')
        if not key:raise RuntimeError('OPENAI_API_KEY missing from local Mac .env; never paste key into chat')
        if not base:raise RuntimeError('OPENAI_BASE_URL missing; pass --api-base or put into local .env')
        if not base.startswith(('http://','https://')):raise ValueError('Invalid API base')
        from openai import OpenAI
        client=OpenAI(api_key=key,base_url=base,timeout=a.timeout,max_retries=0)
        print('ENDPOINT:',base,'MODEL:',a.model,'TRANSPORT:',a.api_mode,flush=True)
    out=Path(a.output).expanduser();out.mkdir(parents=True,exist_ok=True);rows=[]
    for folder in dirs:
        cfg=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        dest=out/cfg['case'];dest.mkdir(parents=True,exist_ok=True)
        result_path=dest/'sol_result.json'
        if result_path.exists() and not a.force:
            try:
                old=json.loads(result_path.read_text(encoding='utf-8'))
                if old.get('status')=='ok' and old.get('model')==a.model and old.get('prompt_hash')==cfg['prompt_hash'] and old.get('api_mode')==a.api_mode:
                    print('CACHED',cfg['case'],'score=',old['score']);continue
            except Exception:pass
        sizes=[];encoded=[]
        for name in cfg['images']:
            p=folder/name
            data=p.read_bytes();sizes.append(round(len(data)/1024,1))
            encoded.append('data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii'))
        if a.dry_run:
            print('DRY RUN',cfg['case'],'frames',cfg['source_frames'],'images',len(encoded),'KB',sizes)
            continue
        print('REQUEST',cfg['case'],'model',a.model,'images',len(encoded),'KB',sizes,flush=True)
        t0=time.monotonic()
        try:
            if a.api_mode=='responses':
                user=[{'type':'input_text','text':cfg['user_prompt']}]
                user += [{'type':'input_image','image_url':uri,'detail':'high'} for uri in encoded]
                payload=[{'role':'system','content':[{'type':'input_text','text':cfg['system_prompt']}]},
                         {'role':'user','content':user}]
                r=client.responses.create(model=a.model,input=payload,max_output_tokens=a.max_output_tokens,store=False)
                raw=get_response_text(r)
            else:
                user=[{'type':'text','text':cfg['user_prompt']}]
                user += [{'type':'image_url','image_url':{'url':uri,'detail':'high'}} for uri in encoded]
                r=client.chat.completions.create(model=a.model,messages=[
                    {'role':'system','content':cfg['system_prompt']},
                    {'role':'user','content':user}],max_completion_tokens=a.max_output_tokens)
                raw=r.choices[0].message.content or ''
            save_json(dest/'raw_api_response.json',r.model_dump(mode='json',exclude_none=True))
            (dest/'raw_model_text.txt').write_text(raw,encoding='utf-8')
            verdict=parse_score(raw)
            doc={'status':'ok' if verdict else 'unparsed','case':cfg['case'],'model':a.model,
                 'api_mode':a.api_mode,'prompt_hash':cfg['prompt_hash'],
                 'source_frames':cfg['source_frames'],'score':verdict['visual_inconsistency_score'] if verdict else None,
                 'verdict':verdict,'elapsed_sec':round(time.monotonic()-t0,2)}
            save_json(result_path,doc)
            print('RESPONSE',cfg['case'],'status',doc['status'],'score',doc['score'],'sec',doc['elapsed_sec'],flush=True)
            rows.append({'case':cfg['case'],'score':doc['score'],'status':doc['status'],'model':a.model,'reason':(verdict or {}).get('reason','')})
        except Exception as e:
            err=f'{type(e).__name__}: {str(e)[:500]}'
            save_json(result_path,{'status':'error','case':cfg['case'],'model':a.model,'api_mode':a.api_mode,'prompt_hash':cfg['prompt_hash'],'error':err})
            print('ERROR',cfg['case'],err,flush=True)
            rows.append({'case':cfg['case'],'score':'','status':'error','model':a.model,'reason':err})
            if a.stop_on_error:break
    if rows:
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['case','score','status','model','reason']);w.writeheader();w.writerows(rows)
        print('SAVED:',out/'comparison.csv',flush=True)

def self_test(_):
    assert parse_spec('0056:148,150,152,154,156')[1]==[148,150,152,154,156]
    try:parse_spec('0056:1,2,3')
    except ValueError:pass
    else:raise AssertionError('short frame selection should fail')
    assert crop_norm('0.28,0.26,0.76,0.90')==[0.28,0.26,0.76,0.90]
    assert parse_score('{"visual_inconsistency_score":0.2}')['visual_inconsistency_score']==0.2
    print('SELF-TEST PASS',flush=True)

def main():
    a=argparse.ArgumentParser(description=__doc__);sub=a.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--video-base',required=True);p.add_argument('--spec',action='append',required=True)
    p.add_argument('--roi-norm',default='0.28,0.26,0.76,0.90');p.add_argument('--full-side',type=int,default=900)
    p.add_argument('--crop-side',type=int,default=800);p.add_argument('--quality',type=int,default=86)
    p.add_argument('--output',required=True);p.set_defaults(func=prepare)
    p=sub.add_parser('judge');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--env-file');p.add_argument('--api-base');p.add_argument('--model',default='gpt-5.6-sol')
    p.add_argument('--api-mode',choices=['responses','chat'],default='responses')
    p.add_argument('--max-output-tokens',type=int,default=3000);p.add_argument('--timeout',type=float,default=240)
    p.add_argument('--only-case',choices=['0055','0056','0057']);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--force',action='store_true');p.add_argument('--stop-on-error',action='store_true');p.set_defaults(func=judge)
    p=sub.add_parser('self-test');p.set_defaults(func=self_test)
    a.parse_args().func(a.parse_args())

if __name__=='__main__':main()
