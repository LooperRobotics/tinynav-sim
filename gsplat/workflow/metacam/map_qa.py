import numpy as np, open3d as o, json
from pathlib import Path
p=Path('/scene')
t=np.loadtxt(p/'data/new_map/traj_tum.txt'); d=np.diff(t[:,1:4],axis=0); dt=np.diff(t[:,0]); c=o.io.read_point_cloud(str(p/'data/new_map/global.pcd')); c=c.voxel_down_sample(.06); xyz=np.asarray(c.points)
# Inspect horizontal support candidates by elevation histogram; do not assume the lowest visible plane is ground.
h,edges=np.histogram(xyz[:,2],bins=np.arange(xyz[:,2].min(),xyz[:,2].max()+.05,.05)); ix=np.argsort(h)[-8:][::-1]
r={'poses':len(t),'duration_s':float(t[-1,0]-t[0,0]),'finite':bool(np.isfinite(t).all()),'max_step_m':float(np.linalg.norm(d,axis=1).max()),'path_length_m':float(np.linalg.norm(d,axis=1).sum()),'height_quantiles':np.quantile(t[:,3],[0,.1,.5,.9,1]).tolist(),'cloud_points_6cm':len(xyz),'height_modes':[{'z':float((edges[i]+edges[i+1])/2),'count':int(h[i])} for i in ix]}
np.savez_compressed(p/'map-qa.npz',xyz=xyz,trajectory=t)
(p/'map-qa.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
