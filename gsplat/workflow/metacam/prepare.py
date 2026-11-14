import json,numpy as np,cv2
from pathlib import Path
p=Path('/scene');c=json.load(open(p/'calib/calibration-original.json'));report=[]
for cam in c['cameras']:
 im=cv2.imread(str(p/('first-'+cam['name']+'.jpg')));h,w=im.shape[:2];sx=w/cam['width'];sy=h/cam['height'];assert abs(sx-sy)<1e-9;report.append({'camera':cam['name'],'source':[cam['width'],cam['height']],'recorded':[w,h],'scale':sx});cam['width']=w;cam['height']=h
 for k in ['fl_x','cx']:cam['intrinsic'][k]*=sx
 for k in ['fl_y','cy']:cam['intrinsic'][k]*=sy
(p/'calib/calibration.json').write_text(json.dumps(c,indent=2));(p/'calibration-adaptation.json').write_text(json.dumps(report,indent=2))
a=np.load(p/'imu-raw.npy');s=a[a[:,0]<a[0,0]+2];r={'first_2s_samples':len(s),'gyro_rms_rad_s':float(np.sqrt(np.mean(s[:,4:7]**2))),'acc_norm_median_g':float(np.median(np.linalg.norm(s[:,1:4],axis=1))),'acc_norm_std_g':float(np.std(np.linalg.norm(s[:,1:4],axis=1)))};(p/'imu-quality.json').write_text(json.dumps(r,indent=2));print(json.dumps(report));print(json.dumps(r))
