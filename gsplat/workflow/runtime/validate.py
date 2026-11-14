from pathlib import Path
import sys,json,math
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent));from volume import CollisionVolume
p=Path('/scene');g=CollisionVolume(p/'geometry/collision-voxels.npz');t=np.load(p/'map-qa.npz')['trajectory'];count=0
for i in range(10,len(t)-10,10):
 x,y=t[i,1:3];v=t[i+5,1:3]-t[i-5,1:3];yaw=math.atan2(v[1],v[0]);z=g.floor(x,y)+.32
 if g.free(x,y,z,yaw):
  count+=1;old=(x,y,z,yaw);target=(x+20,y,z,yaw);accepted=g.constrain(old,target);assert all(np.isfinite(accepted))
im=np.array(Image.open(p/'sim-runtime/smoke-robot.png').convert('RGB'));d=np.array(Image.open(p/'sim-runtime/depth-robot.png'));r={'free_trajectory_samples':count,'tested_trajectory_samples':len(range(10,len(t)-10,10)),'render_mean':float(im.mean()),'render_std':float(im.std()),'lidar_depth_valid_fraction':float((d>0).mean()),'collision_dimension':'XYZ','eye_height_above_floor_m':.68,'status':'smoke_only_not_navigation_acceptance'};assert count>10 and im.std()>5 and (d>0).mean()>.1,r;(p/'simulation-smoke.json').write_text(json.dumps(r,indent=2));print(r)
