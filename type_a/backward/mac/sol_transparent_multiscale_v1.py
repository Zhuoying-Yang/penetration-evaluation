#!/usr/bin/env python3
"""On Mac, create a label-free multi-scale input set from already-downloaded
Sol/Luna JPEGs. Use sol_transparent_moreframes_v2.py judge on the output.
No ERIS access, API credentials, OpenAI calls, or video dependencies here.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import argparse, json, shutil, hashlib, re

# These are developmental diagnostic windows, NOT a blinded evaluation.
FOCUS_FRAMES={
    '0055':[74,76,78,80,82],
    '0056':[148,151,154,156,160],
    '0057':[174,177,180,183,186],
}

# Fixed rectangle in original 615x461 *broad crop* coordinates, excluding the
# 30px text strip. Same crop within a case at every source frame; no GT corners.
REL_BOX=(0.165,0.078,0.615,0.695)
TOP_LABEL=30


def add_label(img, caption):
    out=Image.new('RGB',(img.width,img.height+TOP_LABEL),'white')
    out.paste(img,(0,TOP_LABEL))
    ImageDraw.Draw(out).text((8,7),caption,fill=(0,0,0))
    return out


def build(src_root, dst_root, cases, focus_side=850):
    summary=[]
    for cid in cases:
        candidates=sorted(p for p in src_root.glob(f'{cid}_*/manifest.json'))
        if len(candidates)!=1:
            raise RuntimeError(f'Expected exactly 1 manifest for {cid}, got {len(candidates)}: {candidates}')
        old_cfg=json.loads(candidates[0].read_text(encoding='utf-8'))
        old=candidates[0].parent
        out=dst_root/old.name
        if out.resolve()==old.resolve():
            raise RuntimeError('Output must be separate from the original inputs')
        out.mkdir(parents=True,exist_ok=True)

        # Copy existing full scene + broad temporal images UNMODIFIED.
        original_names=old_cfg['images']
        for name in original_names:
            source=old/name
            if not source.is_file():raise FileNotFoundError(source)
            shutil.copy2(source,out/name)

        focus_frames=FOCUS_FRAMES[cid]
        if not set(focus_frames).issubset(set(old_cfg['source_frames'])):
            raise RuntimeError(f'{cid} focus frames missing: {focus_frames}')
        focus_names=[]
        for j,frame in enumerate(focus_frames,1):
            match=[n for n in original_names if re.search(fr'_f0*{frame}\.jpg$',n)]
            if len(match)!=1:raise RuntimeError(f'Cannot find exactly one broad crop for f{frame}: {match}')
            with Image.open(old/match[0]) as s:
                s=s.convert('RGB')
                W,H=s.size
                usable_h=H-TOP_LABEL
                box=(round(W*REL_BOX[0]),round(TOP_LABEL+usable_h*REL_BOX[1]),
                     round(W*REL_BOX[2]),round(TOP_LABEL+usable_h*REL_BOX[3]))
                crop=s.crop(box)
                # Enlarging can improve salience; does NOT recreate lost detail.
                w,h=crop.size
                scale=focus_side/max(w,h)
                enlarged=crop.resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS)
                im=add_label(enlarged,f'FOCUS f{frame} | same fixed ROI | enlarged broad crop')
                name=f'{j+len(original_names):02d}_focus_f{frame:04d}.jpg'
                im.save(out/name,'JPEG',quality=92,optimize=True)
                focus_names.append(name)

        # The original prompt is kept; add an explicit, label-neutral paragraph
        # describing that these are re-crops of existing snapshots, NOT extra frames.
        appendix=(
            '\nADDITIONAL VISUAL AIDS: After the original full-scene and chronological broad crops, '
            f'there are {len(focus_names)} labeled FOCUS images, in increasing frame order at '
            + ', '.join('f'+str(x) for x in focus_frames) + '. Each FOCUS image is a '
            'magnified re-crop of an ALREADY PROVIDED frame, using the SAME fixed region '
            'around the gripper tip, drawer opening, and transparent panel boundary. '
            'They are not additional independent observations and contain no annotations '
            'or 3D depth information. Verify clearance using the broad frame and focus '
            'for the SAME timestamp. Do not infer penetration from contrast, overlap, '
            'magnification, or a boundary that could belong to the background.'
        )
        cfg=dict(old_cfg)
        cfg.update({
            'version':'sol-transparent-multiscale-v1',
            'images':original_names+focus_names,
            'user_prompt':old_cfg['user_prompt']+appendix,
            'derived_from':str(old),
            'focus_frames':focus_frames,
            'focus_relative_box':list(REL_BOX),
            'focus_note':'Cropped from JPEG broad inputs, no new optical detail; no hand annotations',
        })
        cfg['prompt_hash']=hashlib.sha256((cfg['system_prompt']+cfg['user_prompt']).encode()).hexdigest()
        (out/'manifest.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

        # Easy-to-inspect row of the five *new* focus images.
        ims=[]
        for name in focus_names:
            im=Image.open(out/name).convert('RGB')
            im.thumbnail((340,360),Image.Resampling.LANCZOS)
            ims.append(im.copy())
        collage=Image.new('RGB',(max(i.width for i in ims)*len(ims),max(i.height for i in ims)),'white')
        x=0
        for im in ims:
            collage.paste(im,(x,0));x+=max(i.width for i in ims)
        collage.save(out/'focus_preview.jpg',quality=90)
        summary.append((cid,out,len(cfg['images']),focus_frames))
        print(f'{cid}: {len(cfg["images"])} total images ({len(original_names)} originals + {len(focus_names)} focus), output={out}',flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='~/Downloads/sol_transparent_inputs_dense')
    p.add_argument('--output',default='~/Downloads/sol_transparent_inputs_multiscale')
    p.add_argument('--cases',default='0055,0056,0057')
    a=p.parse_args()
    cases=[x.strip() for x in a.cases.split(',')]
    if any(x not in FOCUS_FRAMES for x in cases):p.error('Only cases 0055,0056,0057 supported')
    build(Path(a.input).expanduser(),Path(a.output).expanduser(),cases)

if __name__=='__main__':main()
