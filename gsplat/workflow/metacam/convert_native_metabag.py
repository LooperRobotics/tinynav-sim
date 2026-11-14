"""Read-only MetaBag V2 header adapter + native CustomMsg ROS2 conversion."""
import json,re,struct,time,collections,sys,heapq
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import rosbags.rosbag1.reader as reader_module
from rosbags.rosbag1 import Reader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores,get_typestore,get_types_from_msg
class MetaRegex:
 def __getattr__(self,name):return getattr(re,name)
 def match(self,pattern,string,*args,**kwargs):
  if string=='#METABAG V2.0\n' and pattern.startswith('#ROSBAG'):string='#ROSBAG V2.0\n'
  return re.match(pattern,string,*args,**kwargs)
reader_module.re=MetaRegex()
sources=sorted(Path('/capture/data').glob('data_*'));scene=Path('/scene');dst=scene/'ros2_bag';scene.mkdir(exist_ok=True)
if dst.exists():raise SystemExit('output exists; refusing overwrite')
repo=Path('/root/slam_ws/src/lightning-lm');sys.path.insert(0,str(repo/'scripts'));import convert_ros1_to_ros2_vec as ref
s1=get_typestore(Stores.ROS1_NOETIC);_,s2=ref.make_typestores();counts=collections.Counter();header_times=collections.defaultdict(list);imus=[];selftests=0
# ROS1 native CustomPoint is packed19B, ROS2 CDR array elements are20B except final padding.
dt1=np.dtype([('offset','<u4'),('x','<f4'),('y','<f4'),('z','<f4'),('reflectivity','u1'),('tag','u1'),('line','u1')]);dt2=np.dtype({'names':dt1.names,'formats':['<u4','<f4','<f4','<f4','u1','u1','u1'],'offsets':[0,4,8,12,16,17,18],'itemsize':20})
def fast_custom(raw):
 global selftests
 seq,sec,nsec,flen=struct.unpack_from('<IIII',raw,0);frame=bytes(raw[16:16+flen]).decode();o=16+flen;timebase,num=struct.unpack_from('<QI',raw,o);lidar_id=raw[o+12];rsvd=bytes(raw[o+13:o+16]);n=struct.unpack_from('<I',raw,o+16)[0];assert num==n and len(raw)==o+20+n*19
 arr=np.frombuffer(raw,dtype=dt1,count=n,offset=o+20);out=np.zeros(n,dtype=dt2)
 for key in dt1.names:out[key]=arr[key]
 frame_bytes=frame.encode()+b'\0';payload=bytearray(struct.pack('<iII',sec,nsec,len(frame_bytes))+frame_bytes)
 while len(payload)%8:payload+=b'\0'
 payload+=struct.pack('<QIB',timebase,num,lidar_id)+rsvd
 payload+=struct.pack('<I',n)+(out.tobytes()[:-1] if n else b'')
 encoded=b'\0\1\0\0'+payload
 if selftests<5:
  message=s1.deserialize_ros1(raw,'livox_ros_driver2/msg/CustomMsg');expected=s2.serialize_cdr(message,'livox_ros_driver2/msg/CustomMsg');assert bytes(expected)==bytes(encoded),(len(expected),len(encoded));selftests+=1
  if selftests==5:print('native lidar self-test:5 frames byte-identical',flush=True)
 return encoded,sec+nsec*1e-9,sorted(set(arr['line'].tolist())),int(arr['offset'].max())
with ExitStack() as stack:
 readers=[stack.enter_context(Reader(src)) for src in sources]
 for rr in readers:
  print("Source",rr.path,"range",rr.start_time,rr.end_time,"messages",rr.message_count,flush=True)
 connections=[c for rr in readers for c in rr.connections]
 for c in connections:
  if c.msgtype=='livox_ros_driver2/msg/CustomMsg':s1.register(get_types_from_msg(c.msgdef.data,c.msgtype))
 topics={c.topic:c for c in connections if c.topic in ('/livox/lidar','/livox/imu','/camera/left/jpeg','/camera/right/jpeg')}
 required = {'/livox/lidar', '/livox/imu', '/camera/left/jpeg', '/camera/right/jpeg'}
 if required - topics.keys():
  raise ValueError(f'Missing required topics: {sorted(required - topics.keys())}')
 with Writer(dst,version=8) as writer:
  outputs={t:writer.add_connection(t,c.msgtype,typestore=s2) for t,c in topics.items()};start=time.time();lines=set();maxoff=0
  streams=[rr.messages(connections=[c for c in rr.connections if c.topic in topics]) for rr in readers]
  for c,t,raw in heapq.merge(*streams,key=lambda x:x[1]):
   if c.topic=='/livox/lidar':encoded,stamp,ll,off=fast_custom(raw);lines.update(ll);maxoff=max(maxoff,off)
   else:
    m=s1.deserialize_ros1(raw,c.msgtype);stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
    if c.topic=='/livox/imu':
     imu=[stamp,m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z,m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z];imus.append(imu);m.linear_acceleration.x*=9.80665;m.linear_acceleration.y*=9.80665;m.linear_acceleration.z*=9.80665
    elif counts[c.topic]==0:(scene/('first-left.jpg' if '/left/' in c.topic else 'first-right.jpg')).write_bytes(bytes(m.data))
    encoded=s2.serialize_cdr(m,c.msgtype)
   writer.write(outputs[c.topic],t,encoded);counts[c.topic]+=1;header_times[c.topic].append(stamp)
   if sum(counts.values())%5000==0:print(dict(counts),'elapsed',round(time.time()-start,1),flush=True)
  assert all(counts[t]==sum(c.msgcount for c in connections if c.topic==t) for t in topics)
np.save(scene/'imu-raw.npy',np.array(imus));report={'input':[str(src) for src in sources],'output':str(dst),'reader_adapter':'Only recognizes MetaBag magic in reader; record offsets and original bytes unchanged. Full index read and message count verification.','native_lidar_not_pointcloud2':True,'lidar_selftest_frames':selftests,'counts':dict(counts),'lidar_line_values':sorted(lines),'max_point_offset_ns':maxoff,'imu_scale':9.80665,'interval_quantiles':{t:np.quantile(np.diff(v),[0,.5,.99,1]).tolist() for t,v in header_times.items()}};(scene/'conversion-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
