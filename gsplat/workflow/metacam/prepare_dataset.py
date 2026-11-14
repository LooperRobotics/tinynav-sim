import json,sys,os,struct
from pathlib import Path
import numpy as np
from PIL import Image
p=Path('/scene');mode=sys.argv[1];m=json.load(open(p/'nerfstudio/transforms.json'));names=sorted(Path(f['file_path']).name for f in m['frames'])
if mode=='split':
 for side in ['left','right']:
  d=p/'sam_frames'/side;d.mkdir(parents=True,exist_ok=True)
  for n in names:
   if not n.endswith('_'+side+'.png'):continue
   src=p/'nerfstudio/images_2'/n;assert src.exists(),src;dest=d/n
   if not dest.exists():dest.symlink_to(os.path.relpath(src,d))
 print('Prepared independent camera sequences',len(names))
elif mode=='combine':
 dst=p/'nerfstudio/masks_2_dyn';dst.mkdir(exist_ok=True);fracs=[];all_sizes=[]
 for side in ['left','right']:
  ns=[n for n in names if n.endswith('_'+side+'.png')]
  for idx,n in enumerate(ns):
   src=p/'sam_masks'/side/f'frame_{idx:05d}.png';assert src.exists(),src
   human=Image.open(src).convert('L');lens=Image.open(p/'nerfstudio/masks_2'/n).convert('L');human=human.resize(lens.size,Image.Resampling.NEAREST);a=np.array(human)>127;b=np.array(lens)>127;assert b.any();fracs.append(float(((~a)&b).sum()/b.sum()));all_sizes.append(lens.size);Image.fromarray(((a&b)*255).astype('uint8')).save(dst/n)
 r={'frames':len(fracs),'person_fraction_quantiles':np.quantile(fracs,[0,.5,.9,.99,1]).tolist(),'frames_with_people':int((np.array(fracs)>.001).sum()),'frames_masked_over_half':int((np.array(fracs)>.5).sum()),'original_lens_masks_preserved':True};(p/'mask-report.json').write_text(json.dumps(r,indent=2));print(r);assert r['frames_with_people']>0,'No detections in a capture known to contain a person'
elif mode=='adapter':
 import pycolmap
 rec=pycolmap.Reconstruction(str(p/'colmap/sparse_ba_aligned'));err=np.array([q.error for q in rec.points3D.values()]);stats=json.load(open(p/'colmap/align_stats.json'));gate={'median_reprojection_px':float(np.median(err)),'points':len(err),**stats};gate['accepted']=bool(np.isfinite(err).all() and len(err)>10000 and np.median(err)<1.0 and stats['residual_med_cm']<25 and stats['residual_p95_cm']<60);(p/'ba-acceptance.json').write_text(json.dumps(gate,indent=2));assert gate['accepted'],gate
 meta=json.load(open(p/'nerfstudio_ba/transforms.json'));d=p/'spirula_ba';d.mkdir(exist_ok=True)
 for f in meta['frames']:
  n=Path(f['file_path']).name;f['file_path']='images_2/'+n;f['mask_path']='masks_2_dyn/'+n
  with (p/'nerfstudio/images_2'/n).open('rb') as h:hdr=h.read(24)
  w,h=struct.unpack('>II',hdr[16:24]);sx=w/f['w'];sy=h/f['h'];assert abs(sx-sy)<1e-9
  for k in ['fl_x','cx']:f[k]*=sx
  for k in ['fl_y','cy']:f[k]*=sy
  f['w']=w;f['h']=h
 for sub in ['images_2','masks_2_dyn','points3d.ply']:
  link=d/sub
  if not link.exists():link.symlink_to('../nerfstudio/'+sub)
 (d/'transforms.json').write_text(json.dumps(meta,indent=2));print('Adapter frames',len(meta['frames']))
