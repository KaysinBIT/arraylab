"""Full-visible-disk grid thinning and continuous position/voltage synthesis.

The search grid is deliberately independent of final peak/boundary validation.
No candidate is reported feasible from its coarse search score alone.
"""
from copy import deepcopy
import math
import numpy as np
from arraylab.framework.parallel import check_cancelled,ordered_map
from . import engine as E
from .linear_optimizer import rng

HALF=3.010299956639812


def defaults():
    return dict(mode='grid',count=32,gridNx=11,gridNy=11,dx=.5,dy=.5,minSpacing=.5,width=5,height=5,
        amplitudeMin=.2,optimizePositions=True,optimizeAmplitudes=True,initialization='generated',initialPositions=[],
        scans=[dict(x=0,y=0),dict(x=20,y=0),dict(x=-20,y=0),dict(x=0,y=20),dict(x=0,y=-20)],
        frequencies=[9,10,11],targetSll=-10,maxHpbwX=18,maxHpbwY=18,mainlobeRadiusU=.24,mainlobeRadiusV=.24,iterations=500,seed=42)

def continuous_defaults():return {**defaults(),'mode':'continuous','iterations':2000}


def envelope(o):return (o['width'],o['height']) if o['mode']=='continuous' else ((o['gridNx']-1)*o['dx'],(o['gridNy']-1)*o['dy'])
def electrical(base,o):return max(envelope(o))*max(o['frequencies'])/base['referenceGHz']
def ellipse(c,u,v):return ((u-c['expectedU'])/c['radiusU'])**2+((v-c['expectedV'])/c['radiusV'])**2


def cases(base,o):
    result=[]
    for f in o['frequencies']:
        ratio=f/base['referenceGHz'];factor=1/ratio if base['phaseMode']=='phase' else 1
        for scan in o['scans']:
            result.append(dict(index=len(result),scanX=scan['x'],scanY=scan['y'],frequencyGHz=f,
                expectedU=math.sin(scan['x']*E.RAD)*factor,expectedV=math.sin(scan['y']*E.RAD)*factor,
                radiusU=o['mainlobeRadiusU']/ratio,radiusV=o['mainlobeRadiusV']/ratio,ratio=ratio))
    return result


def geometry(points):
    xy=np.array([[p['x'],p['y']] for p in points]);a=np.array([p.get('amplitude',1) for p in points])
    distances=np.linalg.norm(xy[:,None]-xy[None,:],axis=2);np.fill_diagonal(distances,np.inf)
    gap=float(distances.min());amin=float(a.min());amax=float(a.max())
    return dict(minSpacing=gap if math.isfinite(gap) else None,width=float(np.ptp(xy[:,0])),height=float(np.ptp(xy[:,1])),count=len(points),
                taperEfficiency=float(a.sum()**2/(len(a)*(a@a))),amplitudeMin=amin,amplitudeMax=amax,amplitudeDynamicRangeDb=20*math.log10(amax/amin))


def verify_positions(points,o):
    if type(points) is not list or len(points)!=o['count']:raise ValueError('布局阵元数与count不一致。')
    if any(type(p) is not dict or not E.finite(p.get('x')) or not E.finite(p.get('y')) for p in points):raise ValueError('布局坐标须为有限数值。')
    tol=1e-8 if o['mode']=='continuous' else 1e-7;w,h=envelope(o)
    for p in points:
        if abs(p['x'])>w/2+tol or abs(p['y'])>h/2+tol or math.hypot(p['x'],p['y'])<tol:raise ValueError('布局包含越界或中心点。')
        a=p.get('amplitude',1)
        if o['mode']=='continuous':
            if not E.finite(a) or not o['amplitudeMin']-tol<=a<=1+tol or p.get('phaseDeg',0)!=0:raise ValueError('连续幅度须在正下限至1内，附加相位为零。')
        elif a!=1 or p.get('phaseDeg',0)!=0 or abs((p['x']+w/2)/o['dx']-E.js_round((p['x']+w/2)/o['dx']))>tol or abs((p['y']+h/2)/o['dy']-E.js_round((p['y']+h/2)/o['dy']))>tol:raise ValueError('格点布局须落在候选格点，单位幅度和零附加相位。')
        partner=next((q for q in points if q is not p and abs(q['x']+p['x'])<tol and abs(q['y']+p['y'])<tol),None)
        if partner is None or abs(partner.get('amplitude',1)-a)>tol:raise ValueError('布局需反演对称且同对等幅。')
    g=geometry(points)
    if g['minSpacing']<o['minSpacing']-tol:raise ValueError('布局违反真实欧氏最小间距或有重复位置。')
    if o['mode']=='continuous' and abs(g['amplitudeMax']-1)>tol:raise ValueError('初始幅度最大值须已归一化为1，不静默重标定。')


def validate(config=None,options=None):
    if options is not None and type(options) is not dict:return dict(ok=False,errors=['二维设置须为对象。'],options=None)
    o={**defaults(),**deepcopy(options or {})};base={**E.defaults(),**(config or {})};errors=E.validate(base)['errors']
    def number(key,low,high,integer=False):
        v=o[key]
        if not E.finite(v) or not low<=v<=high or integer and v!=int(v):errors.append(f'{key}须在{low}～{high}内。')
        elif integer:o[key]=int(v)
    for key,lo,hi,integer in [('count',4,96,True),('minSpacing',.05,6,False),('targetSll',-50,0,False),('maxHpbwX',1,120,False),('maxHpbwY',1,120,False),('mainlobeRadiusU',.03,.8,False),('mainlobeRadiusV',.03,.8,False),('iterations',0,5000,True),('seed',0,4294967295,True)]:number(key,lo,hi,integer)
    if E.finite(o['count']) and o['count']%2:errors.append('二维反演对称综合要求偶数阵元。')
    if o['mode']=='grid':
        for key,lo,hi,integer in [('gridNx',3,15,True),('gridNy',3,15,True),('dx',.05,3,False),('dy',.05,3,False)]:number(key,lo,hi,integer)
    elif o['mode']=='continuous':
        number('width',.1,120);number('height',.1,120);number('amplitudeMin',.05,1)
        if type(o['optimizePositions']) is not bool or type(o['optimizeAmplitudes']) is not bool:errors.append('坐标与幅度开关须为布尔值。')
        if not o['optimizePositions'] and not o['optimizeAmplitudes'] and o['iterations']!=0:errors.append('非零迭代至少启用一个优化变量。')
        if o['initialization'] not in ('generated','provided'):errors.append('初值来源应为generated/provided。')
    else:errors.append('二维模式须为grid/continuous。')
    if type(o['scans']) is not list or not 1<=len(o['scans'])<=25:errors.append('需1～25个扫描方向。')
    else:
        seen=set()
        for s in o['scans']:
            if type(s) is not dict or not E.finite(s.get('x')) or not E.finite(s.get('y')) or abs(s['x'])>85 or abs(s['y'])>85 or math.sin(s['x']*E.RAD)**2+math.sin(s['y']*E.RAD)**2>=1-1e-10:errors.append('扫描方向须严格位于可见域、各方向角在±85°内。');continue
            key=(s['x'],s['y'])
            if key in seen:errors.append('扫描方向不能重复。')
            seen.add(key)
    if type(o['frequencies']) is not list or not 1<=len(o['frequencies'])<=25:errors.append('需1～25个频率。')
    else:
        if any(not E.finite(f) or not .1<=f<=300 or E.finite(base['referenceGHz']) and f>10*base['referenceGHz'] for f in o['frequencies']):errors.append('频率须0.1～300GHz，且不超过参考频率10倍。')
        elif len(set(o['frequencies']))!=len(o['frequencies']):errors.append('频率不能重复。')
    if type(o['scans']) is list and type(o['frequencies']) is list and len(o['scans'])*len(o['frequencies'])>25:errors.append('频率×扫描方向组合最多25个。')
    if not errors:
        if o['mode']=='grid':
            sites=o['gridNx']*o['gridNy']-(1 if o['gridNx']%2 and o['gridNy']%2 else 0)
            if o['count']>sites:errors.append('选取数量超过去除中心后的候选格点数。')
        if electrical(base,o)>12+1e-10:errors.append('最高频率下单轴候选区域跨度不能超过12λ。')
        for c in cases(base,o):
            ts=np.arange(720)*2*math.pi/720
            if np.max(ellipse(c,np.cos(ts),np.sin(ts)))<=1.0001:errors.append('排除椭圆覆盖可见域，请减小排除半径。')
        if o['mode']=='continuous' and o['initialization']=='provided':
            try:verify_positions(o['initialPositions'],o)
            except ValueError as exc:errors.append(str(exc))
    return dict(ok=not errors,errors=errors,options=o)


def layout_config(base,o,points,c=None):
    return {**deepcopy(base),'name':f"二维稀疏阵 · {o['count']} 元",'geometry':'custom',
            'custom':[dict(x=p['x'],y=p['y'],amplitude=p.get('amplitude',1) if o['mode']=='continuous' else 1,phaseDeg=0) for p in points],
            'window':'uniform','frequencyGHz':c['frequencyGHz'] if c else base['referenceGHz'],
            'steerX':c['scanX'] if c else o['scans'][0]['x'],'steerY':c['scanY'] if c else o['scans'][0]['y']}


def build_grid(c,step,boundary_count):
    lu=math.ceil((-1-c['expectedU'])/step);hu=math.floor((1-c['expectedU'])/step)
    lv=math.ceil((-1-c['expectedV'])/step);hv=math.floor((1-c['expectedV'])/step);nx=hu-lu+1;ny=hv-lv+1
    ug,vg=np.meshgrid(c['expectedU']+np.arange(lu,hu+1)*step,c['expectedV']+np.arange(lv,hv+1)*step)
    valid=ug*ug+vg*vg<=1+1e-10;mapping=np.full((ny,nx),-1,int);mapping[valid]=np.arange(np.count_nonzero(valid))
    us=list(ug[valid]);vs=list(vg[valid]);kinds=['grid']*len(us)
    for i in range(boundary_count):
        t=i*2*math.pi/boundary_count
        for u,v,kind in [(math.cos(t),math.sin(t),'disk'),(c['expectedU']+c['radiusU']*math.cos(t),c['expectedV']+c['radiusV']*math.sin(t),'ellipse')]:
            if u*u+v*v<=1+1e-10:us.append(u);vs.append(v);kinds.append(kind)
    u=np.asarray(us);v=np.asarray(vs);q=ellipse(c,u,v)
    return dict(c=c,step=step,loU=lu,loV=lv,nx=nx,ny=ny,map=mapping,u=u,v=v,q=q,kinds=kinds,outside=np.flatnonzero(q>=1-1e-9),boundary=np.array([i for i,k in enumerate(kinds) if k=='disk'],int))


def local_peaks(grid,values,within):
    mapping=grid['map'];center=mapping[1:-1,1:-1];valid=center>=0;safe=np.maximum(center,0);value=values[safe]
    valid &= value>-299.999
    if within:valid &= grid['q'][safe]<1-1e-8
    strict=np.zeros(center.shape,bool)
    for j in range(3):
        for i in range(3):
            if i==j==1:continue
            neighbor=mapping[j:j+center.shape[0],i:i+center.shape[1]];nv=values[np.maximum(neighbor,0)]
            valid &= (neighbor>=0)&(nv<=value+1e-10);strict |= value>nv+1e-10
    ys,xs=np.nonzero(valid&strict);result=[]
    for j,i in zip(ys+1,xs+1):
        idx=int(mapping[j,i]);result.append(dict(idx=idx,gi=int(j*grid['nx']+i),value=float(values[idx]),u=float(grid['u'][idx]),v=float(grid['v'][idx])))
    c=grid['c']
    result.sort(key=(lambda p:((p['u']-c['expectedU'])**2+(p['v']-c['expectedV'])**2,-p['value'])) if within else lambda p:-p['value'])
    return result


def row_width(grid,values,peak,axis):
    j,i=divmod(peak['gi'],grid['nx']);row=grid['map'][j,:] if axis=='x' else grid['map'][:,i]
    indices=row[row>=0];where=np.flatnonzero(indices==peak['idx'])
    if not len(where):return None
    angles=np.arcsin(np.clip(grid['u' if axis=='x' else 'v'][indices],-1,1))/E.RAD;v=values[indices];p=int(where[0]);level=peak['value']-HALF
    left=right=None
    def cross(a,b):return float(angles[a]+(angles[b]-angles[a])*(level-v[a])/(v[b]-v[a]))
    for k in range(p-1,-1,-1):
        if v[k]<=level:left=cross(k,k+1);break
    for k in range(p+1,len(v)):
        if v[k]<=level:right=cross(k-1,k);break
    return None if left is None or right is None else right-left


def base_case(c):
    return {**{k:c[k] for k in ('index','scanX','scanY','frequencyGHz','expectedU','expectedV','radiusU','radiusV')},
            **dict.fromkeys(('peakU','peakV','peakDb','sll2D','hpbwX','hpbwY','sidelobeU','sidelobeV','pointingErrorDeg')),
            'resolved':False,'feasible':False,'issues':[]}


def finish_case(r,c,o,peak,side=None,hx=None,hy=None,clipped=False,connectivity_unresolved=False):
    if peak is None:r['issues'].append('未解析期望方向附近局部二维峰值。');return r
    uc=math.sin(c['scanX']*E.RAD);vc=math.sin(c['scanY']*E.RAD);wc=math.sqrt(max(0,1-uc*uc-vc*vc));wp=math.sqrt(max(0,1-peak['u']**2-peak['v']**2))
    error=math.acos(float(np.clip(uc*peak['u']+vc*peak['v']+wc*wp,-1,1)))/E.RAD
    r.update(peakU=peak['u'],peakV=peak['v'],peakDb=peak['value'],sll2D=side['value']-peak['value'] if side else None,hpbwX=hx,hpbwY=hy,
             sidelobeU=side['u'] if side else None,sidelobeV=side['v'] if side else None,pointingErrorDeg=error)
    if clipped:r['issues'].append('与预期主峰连通的半功率区域触及可见域边界，主瓣被边界截断；此检查独立于旁瓣排除椭圆。')
    if connectivity_unresolved:r['issues'].append('细网格未解析与预期主峰连通的半功率区域。')
    if hx is None or hy is None:r['issues'].append('X/Y切面未解析完整半功率波束。')
    if side is None:r['issues'].append('未解析排除椭圆以外旁瓣区域。')
    r['resolved']=bool(not clipped and not connectivity_unresolved and hx is not None and hy is not None and side is not None)
    if r['sll2D'] is not None and r['sll2D']>o['targetSll']+1e-8:r['issues'].append('区域旁瓣超限。')
    if hx is not None and hx>o['maxHpbwX']+1e-8:r['issues'].append('X切面HPBW超限。')
    if hy is not None and hy>o['maxHpbwY']+1e-8:r['issues'].append('Y切面HPBW超限。')
    r['feasible']=bool(r['resolved'] and r['sll2D']<=o['targetSll']+1e-8 and hx<=o['maxHpbwX']+1e-8 and hy<=o['maxHpbwY']+1e-8)
    return r


def connected_half_power_boundary(grid,values,peak,field=None):
    """Trace only the intended peak's component, independently of the SLL mask."""
    mapping=grid['map'];nx=grid['nx'];ny=grid['ny'];step=grid['step'];level=peak['value']-HALF
    def above(value):return value is not None and math.isfinite(value) and value>=level-1e-8
    def point(index):return (float(grid['u'][index]),float(grid['v'][index]))
    def segment(a,b):
        if field is None:return True
        count=max(2,math.ceil(math.hypot(a[0]-b[0],a[1]-b[1])/(step/3)))
        return all(above(field(a[0]+(b[0]-a[0])*i/count,a[1]+(b[1]-a[1])*i/count)) for i in range(1,count))
    mask=(mapping>=0)&np.isfinite(values[np.maximum(mapping,0)])&(values[np.maximum(mapping,0)]>=level-1e-8)
    candidates=np.flatnonzero(mask.ravel());flat=mapping.ravel();start=None;distance=math.inf
    for gi in candidates:
        idx=int(flat[gi]);p=point(idx);d=(p[0]-peak['u'])**2+(p[1]-peak['v'])**2
        if d<distance and d<=4*step*step and segment((peak['u'],peak['v']),p):start=int(gi);distance=d
    if start is None:return {'clipped':False,'resolved':False}
    if field is None:
        # Same 8-neighbor reachability as the reference queue, executed in C for
        # the many search iterations. Fine links still use independent samples.
        from scipy.ndimage import label
        labels,_=label(mask,structure=np.ones((3,3),int));connected=labels==labels.ravel()[start]
    else:
        connected=np.zeros((ny,nx),bool);connected.ravel()[start]=True;queue=[start]
        for gi in queue:
            row,col=divmod(gi,nx);p=point(flat[gi])
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dx==dy==0:continue
                    x=col+dx;y=row+dy
                    if not 0<=x<nx or not 0<=y<ny or connected[y,x] or not mask[y,x]:continue
                    if segment(p,point(mapping[y,x])):connected[y,x]=True;queue.append(y*nx+x)
    ids=mapping[connected]
    maximum=float(np.max(grid['u'][ids]**2+grid['v'][ids]**2))
    if math.sqrt(maximum)<1-2*step:return {'clipped':False,'resolved':True}
    def touches(p,value):
        if not above(value):return False
        col=E.js_round((p[0]-grid['c']['expectedU'])/step-grid['loU']);row=E.js_round((p[1]-grid['c']['expectedV'])/step-grid['loV'])
        for dy in range(-2,3):
            for dx in range(-2,3):
                x=col+dx;y=row+dy
                if not 0<=x<nx or not 0<=y<ny or not connected[y,x]:continue
                q=point(mapping[y,x])
                if math.hypot(q[0]-p[0],q[1]-p[1])<=2*step and segment(q,p):return True
        return False
    if field is not None:
        for i in range(720):
            angle=2*math.pi*i/720;p=(math.cos(angle),math.sin(angle))
            if touches(p,field(*p)):return {'clipped':True,'resolved':True}
    else:
        for idx in grid['boundary']:
            if touches(point(idx),values[idx]):return {'clipped':True,'resolved':True}
    return {'clipped':False,'resolved':True}


def coarse_case(grid,values,o):
    c=grid['c'];r=base_case(c)
    if c['expectedU']**2+c['expectedV']**2>=1-1e-10:r['issues'].append('期望主阶位于可见域外或边界。');return r
    peaks=local_peaks(grid,values,True)
    if not peaks:return finish_case(r,c,o,None)
    peak=peaks[0];outside=grid['outside'];side=None
    if len(outside):
        index=int(outside[np.argmax(values[outside])]);side=dict(u=float(grid['u'][index]),v=float(grid['v'][index]),value=float(values[index]))
    connectivity=connected_half_power_boundary(grid,values,peak)
    return finish_case(r,c,o,peak,side,row_width(grid,values,peak,'x'),row_width(grid,values,peak,'y'),connectivity['clipped'],not connectivity['resolved'])


def summary(points,rows):
    def worst(key):
        vals=[c[key] for c in rows];return max(vals) if all(E.finite(v) for v in vals) else None
    return {**geometry(points),'worstSll':worst('sll2D'),'maxHpbwX':worst('hpbwX'),'maxHpbwY':worst('hpbwY'),
            'caseCount':len(rows),'passCount':sum(c['feasible'] for c in rows),'feasible':all(c['feasible'] for c in rows),'cases':rows}


def objective(s,o):
    score=0 if all(c['feasible'] for c in s['cases']) else 1000;worst=-300
    for c in s['cases']:
        if not c['resolved'] or any(not E.finite(c[k]) for k in ('sll2D','hpbwX','hpbwY')):score+=10000;continue
        worst=max(worst,c['sll2D'])
        score+=max(0,c['sll2D']-o['targetSll'])**2+100*max(0,c['hpbwX']/o['maxHpbwX']-1)**2+100*max(0,c['hpbwY']/o['maxHpbwY']-1)**2
        score+=.005*(c['hpbwX']/o['maxHpbwX']+c['hpbwY']/o['maxHpbwY'])
    return score+.05*(worst+80)


def field_function(config,elements,cancelled=None):
    return E.scalar_field(config,elements,cancelled=cancelled)


def hill(field,start,step,allowed):
    best=dict(u=start['u'],v=start['v'],value=field(start['u'],start['v']));h=step
    for _ in range(70):
        if h<=.00001:break
        next_=best
        for j in (-1,0,1):
            for i in (-1,0,1):
                if i==j==0:continue
                u=best['u']+i*h;v=best['v']+j*h
                if u*u+v*v>1+1e-12 or not allowed(u,v):continue
                value=field(u,v)
                if value>next_['value']+1e-12:next_=dict(u=u,v=v,value=value)
        if next_ is best:h*=.5
        else:best=next_
    return best


def boundary_peak(field,c,kind,count,outside):
    def point(t):return (math.cos(t),math.sin(t)) if kind=='disk' else (c['expectedU']+c['radiusU']*math.cos(t),c['expectedV']+c['radiusV']*math.sin(t))
    samples=[]
    for i in range(count):
        t=i*2*math.pi/count;u,v=point(t);q=ellipse(c,u,v);ok=u*u+v*v<=1+1e-10 and (q>=1-1e-9 if outside else q<1)
        samples.append(dict(u=u,v=v,t=t,value=field(u,v) if ok else -math.inf))
    candidates=[p for i,p in enumerate(samples) if math.isfinite(p['value']) and p['value']>=samples[(i-1)%count]['value'] and p['value']>=samples[(i+1)%count]['value']]
    candidates.sort(key=lambda p:-p['value']);best=None
    for p in candidates[:16]:
        b=p;step=2*math.pi/count
        for _ in range(24):
            if step<=1e-6:break
            next_=b
            for sign in (-1,1):
                t=b['t']+sign*step;u,v=point(t);q=ellipse(c,u,v)
                if u*u+v*v>1+1e-10 or (q<1-1e-9 if outside else q>=1):continue
                value=field(u,v)
                if value>next_['value']:next_=dict(u=u,v=v,t=t,value=value)
            if next_ is b:step*=.5
            else:b=next_
        if best is None or b['value']>best['value']:best=b
    return best


def fine_width(field,peak,axis):
    fixed=peak['v' if axis=='x' else 'u'];center=math.asin(float(np.clip(peak['u' if axis=='x' else 'v'],-1,1)))/E.RAD
    limit=math.asin(math.sqrt(max(0,1-fixed*fixed)))/E.RAD;level=peak['value']-HALF;edges=[]
    for sign in (-1,1):
        a=center;va=peak['value'];cross=None
        for i in range(1,2000):
            b=max(-limit,min(limit,center+sign*i*.1));var=math.sin(b*E.RAD);vb=field(var,fixed) if axis=='x' else field(fixed,var)
            if vb<=level:cross=a+(b-a)*(level-va)/(vb-va);break
            if b==-limit or b==limit:break
            a=b;va=vb
        edges.append(cross)
    return None if None in edges else edges[1]-edges[0]


def fine_case(base,o,points,c,step,*,cancelled=None):
    check_cancelled(cancelled);r=base_case(c)
    if c['expectedU']**2+c['expectedV']**2>=1-1e-10:r['issues'].append('期望主阶位于可见域外或边界。');return r
    config=layout_config(base,o,points,c);elements=E.build_elements(config);field=field_function(config,elements,cancelled)
    grid=build_grid(c,step,360);values=E.field_values(config,elements,grid['u'],grid['v'],cancelled=cancelled)
    local=local_peaks(grid,values,True)
    if not local:return finish_case(r,c,o,None)
    peak=hill(field,local[0],step/2,lambda u,v:ellipse(c,u,v)<1-1e-8)
    if ellipse(c,peak['u'],peak['v'])>1-1e-5 or peak['u']**2+peak['v']**2>1-1e-6:r['issues'].append('期望峰在掩膜或可见域边界，局部主瓣未解析。');return r
    side=None
    def keep(p):
        nonlocal side
        if p and math.isfinite(p['value']) and (side is None or p['value']>side['value']):side=p
    for idx in grid['outside']:keep(dict(u=float(grid['u'][idx]),v=float(grid['v'][idx]),value=float(values[idx])))
    outside=[p for p in local_peaks(grid,values,False) if ellipse(c,p['u'],p['v'])>=1]
    for p in outside[:16]:keep(hill(field,p,step/2,lambda u,v:ellipse(c,u,v)>=1-1e-10))
    keep(boundary_peak(field,c,'disk',720,True));keep(boundary_peak(field,c,'ellipse',720,True))
    connectivity=connected_half_power_boundary(grid,values,peak,field)
    return finish_case(r,c,o,peak,side,fine_width(field,peak,'x'),fine_width(field,peak,'y'),connectivity['clipped'],not connectivity['resolved'])


def evaluate_layout(config,options,points,quality='fine',*,workers=None,cancelled=None,progress=None):
    valid=validate(config,options)
    if not valid['ok']:raise ValueError('；'.join(valid['errors']))
    if quality not in ('fine','coarse'):raise ValueError('quality须为fine/coarse。')
    base=E.normalized_config(config);o=valid['options'];verify_positions(points,o);step=min(.035,.25/electrical(base,o))
    rows,_=ordered_map(lambda c:fine_case(base,o,points,c,step,cancelled=cancelled),cases(base,o),workers=workers,
                       work_units=len(points)*len(cases(base,o))*int(4/step**2),progress=progress,cancelled=cancelled)
    return summary(points,rows)


def pairs(o):
    result=[]
    for j in range(o['gridNy']):
        for i in range(o['gridNx']):
            x=(i-(o['gridNx']-1)/2)*o['dx'];y=(j-(o['gridNy']-1)/2)*o['dy']
            if (x>1e-10 or abs(x)<1e-10 and y>1e-10) and 2*math.hypot(x,y)>=o['minSpacing']-1e-10:result.append(dict(x=x,y=y))
    return result


def pair_distance(a,b):return min(math.hypot(a['x']-b['x'],a['y']-b['y']),math.hypot(a['x']+b['x'],a['y']+b['y']))
def pair_positions(items):
    result=[]
    for p in items:
        for sign in (1,-1):result.append(dict(x=sign*p['x'],y=sign*p['y'],amplitude=p.get('amplitude',1),phaseDeg=0))
    return sorted(result,key=lambda p:(p['y'],p['x']))


def contributions(base,o,grid,pair):
    c=grid['c'];config=layout_config(base,o,pair_positions([{**pair,'amplitude':1}]),c);elements=E.build_elements(config)
    p=2*math.pi*c['ratio']*(grid['u'][:,None]*np.array([e['x'] for e in elements])+grid['v'][:,None]*np.array([e['y'] for e in elements]))+np.array([e['phaseDeg'] for e in elements])*E.RAD
    return np.cos(p).sum(1).astype(np.float32),np.sin(p).sum(1).astype(np.float32)


def precompute(base,o,pool,cancelled):
    step=min(.08,.6/electrical(base,o));records=[]
    for c in cases(base,o):
        check_cancelled(cancelled);grid=build_grid(c,step,180);n=len(grid['u'])
        re=np.empty((len(pool),n),np.float32);im=np.empty_like(re)
        for i,p in enumerate(pool):
            check_cancelled(cancelled);re[i],im[i]=contributions(base,o,grid,p)
        element=np.ones(n) if base['elementPower']==0 else np.maximum(0,1-grid['u']**2-grid['v']**2)**(base['elementPower']/2)
        records.append(dict(grid=grid,re=re,im=im,element=element,real=np.zeros(n),imag=np.zeros(n)))
    return dict(cases=records,step=step,sumAmplitude=o['count'])


def add_pair(pre,index,factor):
    for c in pre['cases']:c['real']+=factor*c['re'][index].astype(float);c['imag']+=factor*c['im'][index].astype(float)
def assess(pre,o,points):return summary(points,[coarse_case(c['grid'],E.db((c['real']**2+c['imag']**2)/pre['sumAmplitude']**2*c['element']),o) for c in pre['cases']])


def _initial_continuous(o,random,cancelled):
    if o['initialization']=='provided':
        used=set();result=[]
        for i,p in enumerate(o['initialPositions']):
            if i in used:continue
            j=next(j for j,q in enumerate(o['initialPositions']) if j!=i and j not in used and abs(q['x']+p['x'])<1e-8 and abs(q['y']+p['y'])<1e-8)
            used|={i,j};result.append(dict(x=p['x'],y=p['y'],amplitude=p.get('amplitude',1)))
        return result
    for _ in range(50):
        check_cancelled(cancelled);selected=[]
        for _ in range(5000):
            if len(selected)==o['count']//2:return selected
            p=dict(x=(random()-.5)*o['width'],y=(random()-.5)*o['height'],amplitude=1)
            if 2*math.hypot(p['x'],p['y'])>=o['minSpacing'] and all(pair_distance(p,q)>=o['minSpacing'] for q in selected):selected.append(p)
        if len(selected)==o['count']//2:return selected
    raise ValueError('有界连续装填未找到初值，请放宽间距或提供初值；不是不可行性证明。')


def optimize(config=None,options=None,*,workers=None,cancelled=None,progress=None):
    valid=validate(config,options)
    if not valid['ok']:raise ValueError('；'.join(valid['errors']))
    base=E.normalized_config(config);o=valid['options'];random=rng(o['seed']);continuous=o['mode']=='continuous'
    def notify(iteration,value,fraction,stage):
        check_cancelled(cancelled)
        if progress:progress(dict(iteration=iteration,total=o['iterations'],bestSll=value,progress=fraction,stage=stage))
    if continuous:
        current_pairs=_initial_continuous(o,random,cancelled);initial_pairs=deepcopy(current_pairs);initial_points=pair_positions(initial_pairs);verify_positions(initial_points,o)
        pool=current_pairs;selected=list(range(len(pool)))
    else:
        pool=pairs(o);n=o['count']//2
        if len(pool)<n:raise ValueError('满足对内间距的候选对不足。')
        compatible=np.array([[i!=j and pair_distance(a,b)>=o['minSpacing']-1e-10 for j,b in enumerate(pool)] for i,a in enumerate(pool)])
        selected=None
        for _ in range(120):
            order=list(range(len(pool)))
            for i in range(len(order)-1,0,-1):
                j=math.floor(random()*(i+1));order[i],order[j]=order[j],order[i]
            picked=[]
            for i in order:
                if all(compatible[i,j] for j in picked):picked.append(i)
                if len(picked)==n:selected=picked;break
            if selected is not None:break
        if selected is None:raise ValueError('有界初始化未找到间距可行布局，不代表几何不可行。')
        initial_points=pair_positions([pool[i] for i in selected]);initial_pairs=None
    notify(0,None,0,'search');pre=precompute(base,o,pool,cancelled)
    for i in selected:add_pair(pre,i,current_pairs[i]['amplitude'] if continuous else 1)
    if continuous:pre['sumAmplitude']=2*sum(p['amplitude'] for p in current_pairs)
    current=assess(pre,o,initial_points);current_score=objective(current,o);best_score=current_score;best_summary=current
    best=deepcopy(current_pairs) if continuous else selected[:];history=[dict(iteration=0,bestSll=current['worstSll'],score=current_score)]
    accepted_position=accepted_amplitude=0
    for iteration in range(1,o['iterations']+1):
        check_cancelled(cancelled);fraction=iteration/max(1,o['iterations']);candidate=None
        if not continuous:
            outgoing=incoming=-1
            for _ in range(80):
                at=math.floor(random()*len(selected));p=math.floor(random()*len(pool))
                if p not in selected and all(k==at or compatible[p,j] for k,j in enumerate(selected)):outgoing=at;incoming=p;break
            if outgoing>=0:
                removed=selected[outgoing];add_pair(pre,removed,-1);add_pair(pre,incoming,1)
                proposal=selected[:];proposal[outgoing]=incoming;candidate=assess(pre,o,pair_positions([pool[i] for i in proposal]));s=objective(candidate,o)
                if s<=current_score or random()<math.exp(min(700,(current_score-s)/(.8*.025**fraction))):selected=proposal;current=candidate;current_score=s
                else:add_pair(pre,incoming,-1);add_pair(pre,removed,1)
        else:
            at=math.floor(random()*len(current_pairs));old=current_pairs[at];mode=random()
            move_position=o['optimizePositions'] and (not o['optimizeAmplitudes'] or mode<.6)
            move_amplitude=o['optimizeAmplitudes'] and (not o['optimizePositions'] or mode>=.4)
            p=old.copy();valid_move=not move_position
            if move_position:
                scale=.22*.12**fraction
                for _ in range(35):
                    global_=random()<.04
                    proposed={**old,'x':(random()-.5)*o['width'] if global_ else old['x']+(random()-.5)*o['width']*scale,
                              'y':(random()-.5)*o['height'] if global_ else old['y']+(random()-.5)*o['height']*scale}
                    if abs(proposed['x'])<=o['width']/2 and abs(proposed['y'])<=o['height']/2 and 2*math.hypot(proposed['x'],proposed['y'])>=o['minSpacing'] and all(j==at or pair_distance(proposed,q)>=o['minSpacing'] for j,q in enumerate(current_pairs)):p=proposed;valid_move=True;break
                if not valid_move and move_amplitude:p=old.copy();move_position=False;valid_move=True
            if valid_move:
                if move_amplitude:
                    value=old['amplitude']+(random()-.5)*(1-o['amplitudeMin'])*(.7-.5*fraction);lo=o['amplitudeMin'];hi=1
                    if hi==lo:value=lo
                    while value<lo or value>hi:value=2*lo-value if value<lo else 2*hi-value
                    p['amplitude']=value
                scale=1/max(p['amplitude'],*(q['amplitude'] for i,q in enumerate(current_pairs) if i!=at)) if move_amplitude else 1
                proposal=[{**(p if i==at else q),'amplitude':(p if i==at else q)['amplitude']*scale} for i,q in enumerate(current_pairs)]
                replacements=[contributions(base,o,c['grid'],p) for c in pre['cases']] if move_position else None
                old_sum=pre['sumAmplitude']
                for ci,c in enumerate(pre['cases']):
                    nr,ni=replacements[ci] if replacements else (c['re'][at],c['im'][at])
                    c['real']=(c['real']-old['amplitude']*c['re'][at].astype(float)+p['amplitude']*nr.astype(float))*scale
                    c['imag']=(c['imag']-old['amplitude']*c['im'][at].astype(float)+p['amplitude']*ni.astype(float))*scale
                pre['sumAmplitude']=(old_sum-2*old['amplitude']+2*p['amplitude'])*scale
                candidate=assess(pre,o,pair_positions(proposal));s=objective(candidate,o)
                accept=s<=current_score or random()<math.exp(min(700,(current_score-s)/(.45*.02**fraction)))
                if accept:
                    if replacements:
                        for ci,c in enumerate(pre['cases']):c['re'][at],c['im'][at]=replacements[ci]
                    current_pairs=proposal;current=candidate;current_score=s;accepted_position+=move_position;accepted_amplitude+=move_amplitude
                else:
                    for ci,c in enumerate(pre['cases']):
                        nr,ni=replacements[ci] if replacements else (c['re'][at],c['im'][at])
                        c['real']=c['real']/scale-p['amplitude']*nr.astype(float)+old['amplitude']*c['re'][at].astype(float)
                        c['imag']=c['imag']/scale-p['amplitude']*ni.astype(float)+old['amplitude']*c['im'][at].astype(float)
                    pre['sumAmplitude']=old_sum
        if candidate is not None and s<best_score-1e-10:best=deepcopy(proposal);best_score=s;best_summary=candidate
        if iteration%max(1,o['iterations']//50)==0 or iteration==o['iterations']:
            history.append(dict(iteration=iteration,bestSll=best_summary['worstSll'],score=best_score));notify(iteration,best_summary['worstSll'],.85*fraction,'search')
    notify(o['iterations'],best_summary['worstSll'],.86,'verification')
    before=evaluate_layout(base,o,initial_points,workers=workers,cancelled=cancelled)
    notify(o['iterations'],before['worstSll'],.93,'verification')
    proposed_points=pair_positions(best if continuous else [pool[i] for i in best]);candidate=evaluate_layout(base,o,proposed_points,workers=workers,cancelled=cancelled)
    improved=not (before['feasible'] and not candidate['feasible']) and objective(candidate,o)<objective(before,o)-1e-8
    final=proposed_points if improved else initial_points;after=candidate if improved else before
    result=dict(kind='planar2d',method=o['mode'],config=layout_config(base,o,final),baselineConfig=layout_config(base,o,initial_points),options=deepcopy(o),before=before,after=after,history=history,
        iterations=o['iterations'],seed=o['seed'],improved=improved,feasible=after['feasible'],notes=[
        '反演对称二维布局；连续模式优化真实坐标与共享电压幅度，格点模式仅抽稀单位幅度位置。',
        '二维旁瓣为缩放排除椭圆外全可见圆盘峰值相对期望局部峰，不是第一零点旁瓣。',
        '一组坐标和共享幅度联合用于全部频率×扫描方向；HPBW仅约束穿过实际峰的X/Y切面。',
        '先保留全部约束可行性，再比较旁瓣和宽度。细验证追踪独立于排除椭圆的半功率连通区域，不将分离栅瓣误作主瓣截断。',
        '有限采样和有界搜索不证明全域或全局最优。',
        '细验证确认改善。' if improved else '细验证未确认改善，回退完整初始布局。',
        '离散联合约束全部达标。' if after['feasible'] else '至少一个工况未达标，仅可作为候选。'],
        verification=dict(coarseUvStep=pre['step'],fineUvStep=min(.035,.25/electrical(base,o)),boundarySamples=720,cutStepDeg=.1,peakRefinementUv=.00001,
            mask='ellipse centered at ideal intended direction; radii(reference)*f0/f',criterion='full visible disk outside explicit ellipse',mainlobeBoundaryCheck='sampled half-power connected component, independent of SLL mask; intermediate field checks on fine-grid links',numericalFloorDb=-300,fineBaselineScore=objective(before,o),fineCandidateScore=objective(candidate,o),baselineFallback=not improved,finiteSampling=True))
    if continuous:
        final_pairs=best if improved else initial_pairs;changes=[dict(pairIndex=i,before=a,after=b,coordinateMoved=math.hypot(a['x']-b['x'],a['y']-b['y'])>1e-9,amplitudeChanged=abs(a['amplitude']-b['amplitude'])>1e-9) for i,(a,b) in enumerate(zip(initial_pairs,final_pairs))]
        result['diagnostics']=dict(coordinateMovedCount=2*sum(p['coordinateMoved'] for p in changes),amplitudeChangedCount=2*sum(p['amplitudeChanged'] for p in changes),initialization=o['initialization'],sharedWeights=True,acceptedPositionMoves=accepted_position,acceptedAmplitudeMoves=accepted_amplitude,pairChanges=changes)
        result['verification'].update(sharedWeights=True,initialization=o['initialization'],amplitudeNormalization='relative voltage, maximum 1')
    notify(o['iterations'],after['worstSll'],1,'verification')
    return result


def field_at(config,positions,frequency_ghz,scan,u,v):
    if type(positions) is not list or any(type(p) is not dict or not E.finite(p.get('amplitude',1)) or p.get('amplitude',1)<=0 or p.get('phaseDeg',0)!=0 for p in positions):raise ValueError('field_at要求正幅度、零附加相位。')
    c=E.normalized_config({**(config or {}),'geometry':'custom','custom':[{**p,'amplitude':p.get('amplitude',1),'phaseDeg':0} for p in positions],'frequencyGHz':frequency_ghz,'steerX':scan['x'],'steerY':scan['y']})
    return E.pattern_at(c,E.build_elements(c),u,v)
