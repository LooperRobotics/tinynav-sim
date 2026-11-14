import os,time,json,math,threading,argparse,sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization
from gs_sensor_ring import SensorRingWriter
from volume import CollisionVolume
import open3d as o3d
SCENE=Path('/scene');CONFIG=Path(os.environ.get('METACAM_SCENE_ASSETS',str(SCENE/'sim-runtime/scene-assets.json')));ASSETS=json.loads(CONFIG.read_text()) if CONFIG.exists() else {};ROOT=Path(os.environ.get('METACAM_RUNTIME_DIR',str(SCENE/'sim-runtime')));ROOT.mkdir(exist_ok=True);W,H,F=544,480,272.
DEV='cuda';torch.set_grad_enabled(False)
INFRA_HZ=20.;SIM_HZ=12.
args=argparse.ArgumentParser();args.add_argument('--smoke',action='store_true');args.add_argument('--resume',action='store_true');args.add_argument('--max-frames',type=int,default=0);a=args.parse_args()
p=PlyData.read(ASSETS.get('splat',str(SCENE/'splat.ply')))['vertex'];names=p.data.dtype.names
arr=lambda ks:np.stack([p[k] for k in ks],-1).astype('float32')
def gpu(v):return torch.as_tensor(v,device=DEV).contiguous()
means=gpu(arr(['x','y','z']));quats=gpu(arr([f'rot_{i}' for i in range(4)]));quats=quats/torch.linalg.norm(quats,dim=-1,keepdim=True);scales=gpu(np.exp(arr([f'scale_{i}' for i in range(3)])));opacity=torch.sigmoid(gpu(np.array(p['opacity'],dtype='float32')))
dc=arr([f'f_dc_{i}' for i in range(3)])[:,None,:];restnames=sorted([n for n in names if n.startswith('f_rest_')],key=lambda s:int(s[7:]));rest=arr(restnames).reshape(-1,3,len(restnames)//3).transpose(0,2,1);colors=gpu(np.concatenate([dc,rest],1));degree=int(math.sqrt(colors.shape[1])-1)
print('Loaded',len(means),'splats SH',degree,flush=True)
grid=CollisionVolume(ASSETS.get('collision',str(SCENE/'geometry/collision-voxels.npz')));plane=grid.plane
mesh=o3d.io.read_triangle_mesh(ASSETS.get('mesh',str(SCENE/'geometry/lidar-surfaces.ply')));ray_scene=o3d.t.geometry.RaycastingScene();ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
yy,xx=np.mgrid[:H,:W];ray_dirs=np.stack([(xx-W/2)/F,(yy-H/2)/F,np.ones((H,W))],-1).astype('float32')
def floor(x,y):return float(-(plane[0]*x+plane[1]*y+plane[3])/plane[2])
trajectory=np.load(Path('/scene/map-qa.npz'))['trajectory'];spawn=None
for i in range(20,len(trajectory)-20):
 xy=trajectory[i,1:3];v=trajectory[i+10,1:3]-trajectory[i-10,1:3];yaw=math.atan2(v[1],v[0])
 if np.linalg.norm(v)>.3 and grid.free(*xy,floor(*xy)+.32,yaw) and grid.free(*(xy+np.array([math.cos(yaw),math.sin(yaw)])*.8),floor(*(xy+np.array([math.cos(yaw),math.sin(yaw)])*.8))+.32,yaw):spawn=[*map(float,xy),floor(*xy)+.32,yaw];break
assert spawn is not None,'No collision-free spawn'
state=spawn[:]
if a.resume and (ROOT/'state.json').exists():
 saved=json.loads((ROOT/'state.json').read_text());candidate=saved['position']+[math.radians(saved['yaw_deg'])]
 if grid.free(*candidate):state=candidate
applied_cmd=[0.,0.,0.];lock=threading.Lock();t0=time.monotonic();stop=False;blocked=False;frames=0;last_rgb=None
ring=SensorRingWriter(str(ROOT/'sensors.bin'),W,H);opt=np.array([[0,0,1],[-1,0,0],[0,-1,0]],dtype=np.float32);K=gpu(np.array([[[F,0,W/2],[0,F,H/2],[0,0,1]]]*1,dtype='float32'))
def camera_pose(s,right=False):
 x,y,z,yaw=s;c=math.cos(yaw);ss=math.sin(yaw);R=np.array([[c,-ss,0],[ss,c,0],[0,0,1]],dtype=np.float32);t=np.array([x,y,z])+R@np.array([.19,-.0765 if right else -.0255,.36]);C=R@opt;T=np.eye(4,dtype=np.float32);T[:3,:3]=C;T[:3,3]=t;return T

def motion():
 global state,blocked,applied_cmd
 last=time.monotonic()
 while not stop:
  now=time.monotonic();dt=min(.03,now-last);last=now;cmd=np.zeros(3)
  if (ROOT/"reset").exists():
   with lock:state=spawn[:]
   (ROOT/"reset").unlink(missing_ok=True)
  try:
   f=ROOT/'cmd.txt'
   if time.time()-f.stat().st_mtime<.3:cmd=np.array([float(v) for v in f.read_text().split()[:3]])
  except (OSError,ValueError):pass
  try:
   m=json.loads((ROOT/'manual.json').read_text())
   if time.time()-m['t']<.3:cmd=np.array(m['cmd'],dtype=float)
  except (OSError,ValueError,KeyError):pass
  cmd=np.clip(cmd,[-1,-.3,-1.2],[1,.3,1.2])
  with lock:
   x,y,z,yaw=state;c=math.cos(yaw);ss=math.sin(yaw);nx=x+(c*cmd[0]-ss*cmd[1])*dt;ny=y+(ss*cmd[0]+c*cmd[1])*dt;nyaw=yaw+cmd[2]*dt
   target=(nx,ny,floor(nx,ny)+.32,nyaw);accepted=grid.constrain(state,target);blocked=np.linalg.norm(np.array(accepted)[:2]-target[:2])>1e-6 or abs(math.atan2(math.sin(accepted[3]-target[3]),math.cos(accepted[3]-target[3])))>1e-6;applied_cmd=cmd.tolist();state=list(accepted);s=state[:]
  q=[0,0,math.sin(s[3]/2),math.cos(s[3]/2)];ring.publish_gt(now-t0,s[:3],q);ring.publish_imu(now-t0,[0,0,float(cmd[2])],[0,0,9.81],q);time.sleep(.01)

def render(s,with_depth=True):
 Ts=[camera_pose(s,r) for r in [False]];views=gpu(np.stack([np.linalg.inv(T) for T in Ts]));out,alpha,_=rasterization(means,quats,scales,opacity,colors,views,K,W,H,sh_degree=degree,render_mode='RGB',near_plane=.1,far_plane=40.,packed=True)
 rgb=(out[...,:3].clamp(0,1)*255).byte().cpu().numpy()
 if not with_depth:return rgb,np.zeros((H,W),dtype='uint16'),Ts[0]
 world_dirs=ray_dirs@Ts[0][:3,:3].T;rays=np.concatenate([np.broadcast_to(Ts[0][:3,3],world_dirs.shape),world_dirs],axis=-1).astype('float32');hit=ray_scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy();valid=np.isfinite(hit)&(hit>.15)&(hit<6);depth=np.zeros((H,W),dtype='uint16');depth[valid]=(hit[valid]*1000).astype('uint16');return rgb,depth,Ts[0]

def save_preview(rgb):
 tmp=ROOT/'front.tmp.jpg';Image.fromarray(rgb).save(tmp,quality=85);os.replace(tmp,ROOT/'front.jpg')
if a.smoke:
 for label,s in [('robot',state),('capture',[state[0],state[1],0.,state[3]])]:
  start=time.monotonic();rgb,depth,T=render(s);save_preview(rgb[0]);Image.fromarray(rgb[0]).save(ROOT/f'smoke-{label}.png');Image.fromarray(depth).save(ROOT/f'depth-{label}.png');print(label,'render_ms',(time.monotonic()-start)*1000,'depth_valid',float((depth>0).mean()),'pose',s,flush=True)
else:
 threading.Thread(target=motion,daemon=True).start()
 next_infra=time.monotonic();next_sync=next_infra;sim_frames=0
 while True:
  begin=time.monotonic()
  with lock:s=state[:]
  sync_frame=begin+1e-6>=next_sync
  if sync_frame:
   while next_sync<=begin+1e-6:next_sync+=1/SIM_HZ
   sim_frames+=1
  rgb,depth,T=render(s,with_depth=sync_frame);tm=time.monotonic()-t0
  # Optical-camera rotation -> quaternion, using the same convention as the existing emulator.
  from scipy.spatial.transform import Rotation
  ring.publish_gt_cam(tm,T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat());gray=(rgb.astype(np.float32)@np.array([.299,.587,.114])).astype('uint8');ring.publish_cameras(gray[0],np.zeros_like(gray[0]),depth,rgb[0][...,::-1].copy(),tm,sync_frame=sync_frame)
  if sync_frame:save_preview(rgb[0])
  frames+=1;status={'position':s[:3],'yaw_deg':math.degrees(s[3]),'frames':frames,'sim_frames':sim_frames,'infra1_hz_target':INFRA_HZ,'sim_hz_target':SIM_HZ,'render_ms':(time.monotonic()-begin)*1000,'blocked':bool(blocked),'cmd_vel':applied_cmd,'blocked_reason':'collision' if blocked else '', 'spawn':spawn,'camera':'front only','geometry':'6cm 3D LiDAR voxel surfaces','scene_variant':ASSETS.get('variant','original'),'depth':'LiDAR mesh raycast, no calibrated sensor noise'};tmp=ROOT/'state.tmp';tmp.write_text(json.dumps(status));os.replace(tmp,ROOT/'state.json')
  if frames%100==0:print(status,flush=True)
  if a.max_frames and frames>=a.max_frames:
   print('FINITE_BENCHMARK',frames,'frames',time.monotonic()-t0,'seconds',flush=True);break
  next_infra+=1/INFRA_HZ
  if next_infra<time.monotonic()-1/INFRA_HZ:next_infra=time.monotonic()
  time.sleep(max(0,next_infra-time.monotonic()))
