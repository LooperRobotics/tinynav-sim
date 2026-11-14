import math
import numpy as np

class CollisionVolume:
    def __init__(self,path,front=.5,rear=.5,half_width=.25,bottom=-.20,top=.40):
        a=np.load(path); self.origin=a['origin'];self.r=float(a['resolution']);self.shape=a['shape'];self.plane=a['floor_plane'];self.front,self.rear,self.hw=front,rear,half_width;self.bottom,self.top=bottom,top;self.radius=math.hypot(max(front,rear),half_width)
        self.occ=np.zeros(self.shape,dtype=bool);self.occ[tuple(a['indices'].T)]=True
    def floor(self,x,y):
        a,b,c,d=self.plane;return float(-(a*x+b*y+d)/c)
    def free(self,x,y,z,yaw):
        low=np.floor((np.array([x-self.radius,y-self.radius,z+self.bottom])-self.origin)/self.r).astype(int);high=np.floor((np.array([x+self.radius,y+self.radius,z+self.top])-self.origin)/self.r).astype(int)
        if (low<0).any() or (high>=self.shape).any():return False
        cells=np.argwhere(self.occ[low[0]:high[0]+1,low[1]:high[1]+1,low[2]:high[2]+1])+low
        if not len(cells):return True
        xy=self.origin[:2]+cells[:,:2]*self.r;rel=xy+self.r/2-[x,y];c,s=math.cos(yaw),math.sin(yaw);f=rel@np.array([c,s]);l=rel@np.array([-s,c]);h=self.r/2;reach=h*(abs(c)+abs(s));sep_f=np.abs(f-(self.front-self.rear)/2)>(self.front+self.rear)/2+reach;sep_l=np.abs(l)>self.hw+reach;mid=np.array([x,y])+(self.front-self.rear)/2*np.array([c,s]);d=xy+h-mid;sep_x=np.abs(d[:,0])>(self.front+self.rear)/2*abs(c)+self.hw*abs(s)+h;sep_y=np.abs(d[:,1])>(self.front+self.rear)/2*abs(s)+self.hw*abs(c)+h
        return bool(np.all(sep_f|sep_l|sep_x|sep_y))
    def constrain(self,old,target):
        dyaw=math.atan2(math.sin(target[3]-old[3]),math.cos(target[3]-old[3]));n=max(1,math.ceil(max(math.dist(old[:3],target[:3]),abs(dyaw)*self.radius)/(self.r/3)))
        def path(end):
            return all(self.free(*(np.array(old[:3])+(np.array(end[:3])-old[:3])*i/n),old[3]+dyaw*i/n) for i in range(1,n+1))
        if path(target):return tuple(target)
        turn=(*old[:3],target[3])
        if path(turn):return turn
        return tuple(old)
