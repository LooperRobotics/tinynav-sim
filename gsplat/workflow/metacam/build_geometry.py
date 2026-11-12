import numpy as np,json,time
from pathlib import Path
from scipy.spatial import cKDTree
import open3d as o3d
p=Path('/scene');out=p/'geometry';out.mkdir(exist_ok=True)
a=np.load(p/'map-qa.npz');x=a['xyz'];traj=a['trajectory'];plane=np.array(json.load(open(p/'floor-qa.json'))['plane']);floor=-(x[:,0]*plane[0]+x[:,1]*plane[1]+plane[3])/plane[2]
lo=traj[:,1:3].min(0)-5;hi=traj[:,1:3].max(0)+5;keep=((x[:,:2]>=lo)&(x[:,:2]<=hi)).all(1)&(x[:,2]-floor>-.15)&(x[:,2]-floor<3.8);points=x[keep];counts=cKDTree(x).query_ball_point(points,.15,return_length=True,workers=8);points=points[counts>=5];res=.06;origin=np.floor(points.min(0)/res)*res;ijk=np.unique(np.floor((points-origin)/res).astype('int32'),axis=0);shape=ijk.max(0)+3
np.savez_compressed(out/'collision-voxels.npz',indices=ijk,origin=origin,resolution=res,shape=shape,floor_plane=plane)
# Surface geometry retains vertical gaps. Only exposed voxel faces become mesh triangles.
occ=np.zeros(shape,dtype=bool);occ[tuple(ijk.T)]=True;vertices=[];triangles=[];offset=0
for axis in range(3):
 others=[i for i in range(3) if i!=axis]
 for sign in [-1,1]:
  neighbor=ijk.copy();neighbor[:,axis]+=sign;valid=(neighbor>=0).all(1)&(neighbor<shape).all(1);filled=np.zeros(len(ijk),bool);filled[valid]=occ[tuple(neighbor[valid].T)];cells=ijk[~filled]
  corners=np.zeros((4,3),dtype='float64');corners[:,axis]=1 if sign>0 else 0;corners[:,others[0]]=[0,1,1,0];corners[:,others[1]]=[0,0,1,1]
  v=((cells[:,None,:]+corners)*res+origin).reshape(-1,3);base=np.arange(len(cells))*4+offset
  ts=np.stack([base,base+1,base+2,base,base+2,base+3],axis=1).reshape(-1,3);vertices.append(v);triangles.append(ts);offset+=len(v)
mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(vertices)),o3d.utility.Vector3iVector(np.concatenate(triangles).astype('int32')));mesh.remove_duplicated_vertices();mesh.compute_triangle_normals();o3d.io.write_triangle_mesh(str(out/'lidar-surfaces.ply'),mesh,write_ascii=False)
config={'source':'3D LiDAR voxel surface, spatial support filtered','resolution_m':res,'voxels':len(ijk),'triangles':len(mesh.triangles),'vertices':len(mesh.vertices),'floor_plane':plane.tolist(),'height_is_retained':True,'dynamic_people_removed_from_lidar':False,'collision_proxy':{'length_m':1.0,'width_m':.5,'bottom_above_floor_m':.12,'top_above_floor_m':.72,'status':'simulation proxy, not measured Go2 hull'},'validation':'pending camera-depth alignment and passage tests; do not treat missing surfaces as confirmed free'};(out/'geometry-report.json').write_text(json.dumps(config,indent=2));print(json.dumps(config,indent=2),flush=True)
