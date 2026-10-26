"""Reference-frequency phased-array analysis, ported from ArrayStudio 1.2.1.

Angles are direction angles: u=sin(X), v=sin(Y), not azimuth/elevation.
Power includes the selected scalar cos(theta)**p element envelope.
"""
from copy import deepcopy
import math
import numpy as np
from arraylab.framework.parallel import check_cancelled, ordered_map

C=299792458.0
RAD=math.pi/180
HALF=3.01029995664
SOURCE_VERSION='1.2.1'
SOURCE_ARCHIVE_SHA256='04708ed44ab2ea911318351061f4e5b4b88450caae78d265c3db317aa398329d'


def defaults():
    return dict(name='16 × 16 平面阵',geometry='rectangular',nx=16,ny=16,dx=.5,dy=.5,
        referenceGHz=10,frequencyGHz=10,steerX=0,steerY=0,window='uniform',sidelobeDb=-30,
        phaseBits=0,phaseMode='phase',elementPower=0,custom=[])


def finite(x):return type(x) in (int,float) and math.isfinite(x)
def js_round(x):return math.floor(x+.5)
def wrap(x):return (x+180)%360-180
def db(power):return np.maximum(-80.,10*np.log10(np.maximum(power,1e-8)))


def validate(config):
    if type(config) is not dict:return {'ok':False,'errors':['配置必须为对象。']}
    c={**defaults(),**config};errors=[]
    def number(key,low,high,integer=False):
        x=c[key]
        if not finite(x) or not low<=x<=high or integer and x!=int(x):errors.append(f'{key}须为{low}～{high}的'+('整数。' if integer else '有限数值。'))
    for key,choices in [('geometry',('linear','rectangular','custom')),('window',('uniform','hann','hamming','taylor')),('phaseMode',('phase','ttd'))]:
        if c[key] not in choices:errors.append(f'{key}选项无效。')
    for key,low,high,integer in [('referenceGHz',.1,300,False),('frequencyGHz',.1,300,False),('steerX',-89.9,89.9,False),('steerY',-89.9,89.9,False),('phaseBits',0,8,True),('elementPower',0,4,False),('sidelobeDb',-60,-15,False)]:number(key,low,high,integer)
    if finite(c['frequencyGHz']) and finite(c['referenceGHz']) and c['frequencyGHz']>10*c['referenceGHz']:errors.append('分析频率不能超过参考频率10倍。')
    if finite(c['steerX']) and finite(c['steerY']) and math.sin(c['steerX']*RAD)**2+math.sin(c['steerY']*RAD)**2>1+1e-12:errors.append('扫描方向超出可见域：sin²(X)+sin²(Y)≤1。')
    if c['geometry']!='custom':
        number('nx',2,64,True);number('dx',.1,3)
        if c['geometry']=='rectangular':
            number('ny',1,32,True);number('dy',.1,3)
            if finite(c['nx']) and finite(c['ny']) and c['nx']*c['ny']>1024:errors.append('总阵元数不能超过1024。')
        if c['window']=='hann' and (c['nx']==2 or c['geometry']=='rectangular' and c['ny']==2):errors.append('对称Hann窗长度2时全零，请增加阵元数或更换加权。')
    else:
        if type(c['custom']) is not list or not 2<=len(c['custom'])<=1024:errors.append('自定义阵列需要2～1024个阵元。')
        else:
            valid=True
            for i,e in enumerate(c['custom']):
                if type(e) is not dict or any(not finite(e.get(k)) for k in ('x','y','amplitude','phaseDeg')) or abs(e['x'])>200 or abs(e['y'])>200 or not 0<=e['amplitude']<=1000:
                    errors.append(f'自定义阵元{i+1}坐标/幅度/附加相位无效。');valid=False;break
            if valid and sum(e['amplitude'] for e in c['custom'])<=0:errors.append('自定义幅度不能全部为零。')
    if not errors:
        ratio=c['frequencyGHz']/c['referenceGHz']
        if c['geometry']=='custom':sx=np.ptp([e['x'] for e in c['custom']]);sy=np.ptp([e['y'] for e in c['custom']])
        else:sx=(c['nx']-1)*c['dx'];sy=(c['ny']-1)*c['dy'] if c['geometry']=='rectangular' else 0
        if max(sx,sy)*ratio>300:errors.append('分析频率下的单轴阵元中心跨度不能超过300λ。')
    return {'ok':not errors,'errors':errors}


def normalized_config(config=None):
    config={} if config is None else config
    check=validate(config)
    if not check['ok']:raise ValueError('；'.join(check['errors']))
    c={**defaults(),**deepcopy(config)}
    for key in ('nx','ny','phaseBits'):
        if finite(c[key]) and c[key]==int(c[key]):c[key]=int(c[key])
    c['custom']=[{key:e[key] for key in ('x','y','amplitude','phaseDeg')} for e in c['custom']]
    return c


def window_weights(n,kind,sidelobe_db=-30):
    n=int(n)
    if n==1:return [1.]
    if kind=='hann':w=np.hanning(n)
    elif kind=='hamming':w=np.hamming(n)
    elif kind=='taylor':
        from scipy.signal.windows import taylor
        w=taylor(n,nbar=min(4,n),sll=abs(sidelobe_db),norm=True,sym=True)
    else:w=np.ones(n)
    return np.maximum(w,0).tolist()


def build_elements(config):
    c=normalized_config(config);u0=math.sin(c['steerX']*RAD);v0=math.sin(c['steerY']*RAD)
    ratio=c['frequencyGHz']/c['referenceGHz'];factor=ratio if c['phaseMode']=='ttd' else 1
    step=360/2**c['phaseBits'] if c['phaseBits'] else 0
    if c['geometry']=='custom':raw=c['custom']
    else:
        ny=c['ny'] if c['geometry']=='rectangular' else 1
        wx=window_weights(c['nx'],c['window'],c['sidelobeDb']);wy=window_weights(ny,c['window'],c['sidelobeDb'])
        raw=[dict(x=(i-(c['nx']-1)/2)*c['dx'],y=(j-(ny-1)/2)*c['dy'],amplitude=wx[i]*wy[j],phaseDeg=0) for j in range(ny) for i in range(c['nx'])]
    elements=[];error=0.;total=0.
    for e in raw:
        ideal=wrap(e['phaseDeg']-360*factor*(e['x']*u0+e['y']*v0));actual=wrap(js_round(ideal/step)*step if step else ideal)
        error+=e['amplitude']**2*wrap(actual-ideal)**2;total+=e['amplitude']**2
        elements.append(dict(x=e['x'],y=e['y'],amplitude=e['amplitude'],phaseDeg=actual))
    return elements


def phase_rms(config,elements):
    c=normalized_config(config);factor=c['frequencyGHz']/c['referenceGHz'] if c['phaseMode']=='ttd' else 1
    raw=c['custom'] if c['geometry']=='custom' else [dict(phaseDeg=0) for _ in elements]
    errors=[e['amplitude']**2*wrap(e['phaseDeg']-wrap(r['phaseDeg']-360*factor*(e['x']*math.sin(c['steerX']*RAD)+e['y']*math.sin(c['steerY']*RAD))))**2 for e,r in zip(elements,raw)]
    return math.sqrt(sum(errors)/sum(e['amplitude']**2 for e in elements))


def field_values(config,elements,u,v,*,cancelled=None):
    """Vectorized block evaluation; out-of-visible-domain values are NaN."""
    u,v=np.broadcast_arrays(np.asarray(u,float),np.asarray(v,float));shape=u.shape
    uf,vf=u.ravel(),v.ravel();out=np.full(uf.shape,np.nan)
    valid=np.isfinite(uf)&np.isfinite(vf)&(uf*uf+vf*vf<=1+1e-12)
    indices=np.flatnonzero(valid)
    x=np.array([e['x'] for e in elements]);y=np.array([e['y'] for e in elements])
    a=np.array([e['amplitude'] for e in elements]);phase=np.array([e['phaseDeg'] for e in elements])*RAD
    k=2*math.pi*config['frequencyGHz']/config['referenceGHz'];den=float(a.sum())
    for start in range(0,len(indices),1024):
        check_cancelled(cancelled);at=indices[start:start+1024]
        p=k*(uf[at,None]*x+vf[at,None]*y)+phase
        re=np.cos(p)@a;im=np.sin(p)@a
        power=(re*re+im*im)/(den*den) if den>0 else np.zeros(len(at))
        if config['elementPower']:power*=np.maximum(0,1-uf[at]**2-vf[at]**2)**(config['elementPower']/2)
        out[at]=db(power)
    return out.reshape(shape)


def pattern_at(config,elements,u,v):
    if not finite(u) or not finite(v) or u*u+v*v>1+1e-12:return None
    return float(field_values(config,elements,u,v))


def cut(config,elements,axis,step_deg=.25,*,cancelled=None):
    if axis not in ('x','y') or not finite(step_deg) or not .01<=step_deg<=5:raise ValueError('切面轴须x/y，步长须0.01～5°。')
    fixed=math.sin(config['steerY' if axis=='x' else 'steerX']*RAD)
    limit=math.asin(math.sqrt(max(0,1-fixed*fixed)))/RAD;n=math.ceil(2*limit/step_deg)
    angles=np.linspace(-limit,limit,n+1) if n else np.array([0.])
    var=np.sin(angles*RAD);values=field_values(config,elements,var if axis=='x' else fixed,fixed if axis=='x' else var,cancelled=cancelled)
    return [{'angle':float(a),'db':float(v)} for a,v in zip(angles,values) if math.isfinite(v)]


def analyze_cut(samples,expected_angle=0,intended=False):
    empty=dict(hpbw=None,sll=None,peakAngle=None,peakDb=None,mainLeft=None,mainRight=None)
    if not samples or len(samples)<3:return empty
    expected_angle=expected_angle if finite(expected_angle) else 0
    if intended and not samples[0]['angle']-1e-10<=expected_angle<=samples[-1]['angle']+1e-10:return empty
    values=np.array([s['db'] for s in samples]);maximum=float(values.max())
    if np.ptp(values)<1e-7:return empty
    peaks=[i for i in range(len(samples)) if (i==0 or values[i]>=values[i-1]) and (i==len(values)-1 or values[i]>=values[i+1]) and ((i>0 and values[i]>values[i-1]+1e-12) or (i<len(values)-1 and values[i]>values[i+1]+1e-12))]
    if not peaks:return empty
    eligible=peaks if intended else [i for i in peaks if values[i]>=maximum-.05]
    # The tolerance only resolves essentially equidistant sampled peaks.
    best=eligible[0]
    for i in eligible[1:]:
        d=abs(samples[i]['angle']-expected_angle)-abs(samples[best]['angle']-expected_angle)
        if d<-1e-10 or abs(d)<=1e-10 and values[i]>values[best]:best=i
    p=best;peak=float(values[p]);left3=right3=None;l=r=None
    def crossing(a,b,level):return a['angle']+(b['angle']-a['angle'])*(level-a['db'])/(b['db']-a['db'])
    for i in range(p-1,-1,-1):
        if values[i]<=peak-HALF:left3=crossing(samples[i],samples[i+1],peak-HALF);break
    for i in range(p+1,len(values)):
        if values[i]<=peak-HALF:right3=crossing(samples[i-1],samples[i],peak-HALF);break
    for i in range(p-1,0,-1):
        if values[i]<=values[i-1] and values[i]<=values[i+1] and values[i]<peak-.01:l=i;break
    for i in range(p+1,len(values)-1):
        if values[i]<=values[i-1] and values[i]<=values[i+1] and values[i]<peak-.01:r=i;break
    side=[]
    if l is not None:side.extend(values[:l])
    if r is not None:side.extend(values[r+1:])
    return dict(hpbw=None if left3 is None or right3 is None else right3-left3,sll=float(max(side)-peak) if side else None,
                peakAngle=samples[p]['angle'],peakDb=peak,mainLeft=samples[l]['angle'] if l is not None else None,mainRight=samples[r]['angle'] if r is not None else None)


def grating_lobes(c):
    if c['geometry']=='custom':return []
    ratio=c['frequencyGHz']/c['referenceGHz'];factor=ratio if c['phaseMode']=='ttd' else 1
    uc=math.sin(c['steerX']*RAD)*factor/ratio;vc=math.sin(c['steerY']*RAD)*factor/ratio
    two=c['geometry']=='rectangular' and c['ny']>1
    ml=math.ceil((-1-uc)*c['dx']*ratio-1e-10);mh=math.floor((1-uc)*c['dx']*ratio+1e-10)
    nl=math.ceil((-1-vc)*c['dy']*ratio-1e-10) if two else 0;nh=math.floor((1-vc)*c['dy']*ratio+1e-10) if two else 0
    out=[]
    for m in range(ml,mh+1):
        for n in range(nl,nh+1):
            if m==n==0:continue
            u=uc+m/(c['dx']*ratio);v=vc+n/(c['dy']*ratio) if two else 0
            if u*u+v*v<=1+1e-10:out.append(dict(u=u,v=v,xDeg=math.asin(np.clip(u,-1,1))/RAD,yDeg=math.asin(np.clip(v,-1,1))/RAD))
    return out


def evaluate(config=None,*,cancelled=None):
    c=normalized_config(config);elements=build_elements(c);ratio=c['frequencyGHz']/c['referenceGHz']
    width=float(np.ptp([e['x'] for e in elements]));height=float(np.ptp([e['y'] for e in elements]))
    step=max(.01,min(.25,6/max(width*ratio,height*ratio,1)))
    cut_x=cut(c,elements,'x',step,cancelled=cancelled);cut_y=cut(c,elements,'y',step,cancelled=cancelled)
    expected=lambda angle:math.asin(float(np.clip(math.sin(angle*RAD)/(ratio if c['phaseMode']=='phase' else 1),-1,1)))/RAD
    mx=analyze_cut(cut_x,expected(c['steerX']));my=analyze_cut(cut_y,expected(c['steerY']))
    a=np.array([e['amplitude'] for e in elements]);eta=float(a.sum()**2/(len(a)*(a@a)))
    gl=grating_lobes(c);warnings=[]
    if gl:warnings.append(f'周期晶格有{len(gl)}个非零阶栅瓣方向进入可见域；量化和单元模型会改变实际峰值。')
    if c['geometry']=='custom':warnings.append('自定义阵列直接使用逐阵元激励；非周期阵列不使用周期晶格栅瓣判据。')
    if c['window']=='taylor' and c['geometry']!='custom':warnings.append('Taylor设计旁瓣是窗函数参数，实际有限阵列旁瓣以计算结果为准。')
    if c['phaseMode']=='phase' and abs(ratio-1)>1e-9:warnings.append('固定移相保持参考频率激励及物理位置，异频分析可能产生波束斜视。')
    if c['phaseMode']=='ttd' and c['phaseBits']>0:warnings.append('真时延按分析频率校相后施加相位量化，不是固定延时量化器硬件模型。')
    if height<1e-10 and abs(c['steerY'])>1e-8:warnings.append('该阵列没有Y向孔径，无法独立控制Y方向波束。')
    if gl and height<1e-10:warnings.append('直线阵栅瓣为锥面，列表取v=0代表方向。')
    if c['geometry']=='custom' and len({(e['x'],e['y']) for e in elements})<len(elements):warnings.append('坐标重合通道保留相干叠加，理想相干增益不等于实际辐射增益。')
    if c['elementPower']>0:warnings.append('含标量单元功率cos(theta)^p，theta从阵面法线起算。')
    if mx['hpbw'] is None or height>0 and my['hpbw'] is None:warnings.append('部分切面未解析完整半功率主瓣，宽度保留为空。')
    uc=math.sin(c['steerX']*RAD)/(ratio if c['phaseMode']=='phase' else 1);vc=math.sin(c['steerY']*RAD)/(ratio if c['phaseMode']=='phase' else 1)
    if uc*uc+vc*vc>1+1e-10:warnings.append('理想主阶已移出当前可见域，当前峰值可能属于其他瓣。')
    return dict(config=c,elements=elements,cutX=cut_x,cutY=cut_y,metrics=dict(count=len(elements),lambdaMm=C/(c['frequencyGHz']*1e6),
        widthMm=width*C/(c['referenceGHz']*1e6),heightMm=height*C/(c['referenceGHz']*1e6),hpbwX=mx['hpbw'],hpbwY=None if height<1e-10 else my['hpbw'],
        sllX=mx['sll'],sllY=None if height<1e-10 else my['sll'],peakAngleX=mx['peakAngle'],peakAngleY=None if height<1e-10 else my['peakAngle'],
        taperEfficiency=eta,coherentGainDb=10*math.log10(len(elements)*eta),pointingPowerDb=pattern_at(c,elements,math.sin(c['steerX']*RAD),math.sin(c['steerY']*RAD)),
        gratingLobes=gl,phaseRmsDeg=phase_rms(c,elements)),warnings=warnings)


def scan_study(config,options=None,*,workers=None,cancelled=None,progress=None):
    c=normalized_config(config);o={'maxAngle':60,'step':5,**(options or {})}
    if not finite(o['maxAngle']) or not 0<=o['maxAngle']<=85 or not finite(o['step']) or not 1<=o['step']<=30:raise ValueError('扫描半角0～85°，步长1～30°。')
    angles=[0];k=1
    while k*o['step']<o['maxAngle']-1e-10:angles.extend([-k*o['step'],k*o['step']]);k+=1
    if o['maxAngle']>0:angles.extend([-o['maxAngle'],o['maxAngle']])
    active=[e for e in build_elements(c) if e['amplitude']>0];origin=active[0];rank=0;axis=None
    for e in active:
        x=e['x']-origin['x'];y=e['y']-origin['y'];length=math.hypot(x,y)
        if axis is None and length>1e-10:axis=(x/length,y/length);rank=1
        elif axis is not None and abs(x*axis[1]-y*axis[0])>1e-9*max(1,length):rank=2;break
    visibility='planar' if rank==2 else 'linear_cone' if rank==1 else 'degenerate';no_x=rank==0 or rank==1 and abs(axis[0])<1e-10
    if rank==2:model_note='二维期望方向必须处于完整可见圆盘；切面固定在命令v。'
    elif rank==1 and no_x:model_note='共线阵没有X向孔径，不能用单元方向图宽度代替X向阵列波束宽度。'
    elif rank==1 and abs(axis[1])<1e-10:model_note='该阵列没有独立Y向孔径；线阵主瓣为同一u的锥面，本表保留固定命令v切面。'
    elif rank==1:model_note='倾斜共线阵仅约束阵轴方向余弦，本表跟踪锥面与固定命令v切面的交点。'
    else:model_note='有效位置无空间孔径，不能形成独立阵列扫描波束。'
    def one(angle):
        r=evaluate({**c,'steerX':angle},cancelled=cancelled);m=r['metrics']
        factor=c['frequencyGHz']/c['referenceGHz'] if c['phaseMode']=='phase' else 1
        eu=math.sin(angle*RAD)/factor;ev=math.sin(c['steerY']*RAD)/factor;fv=math.sin(c['steerY']*RAD)
        outside=(eu*eu+ev*ev>1+1e-12) if rank==2 else abs(eu*axis[0]+ev*axis[1])>1+1e-12 if rank else False
        cu=None if no_x else (eu*axis[0]+ev*axis[1]-axis[1]*fv)/axis[0] if rank==1 else eu
        expected=math.asin(cu)/RAD if cu is not None and abs(cu)<=1 else None
        outside_cut=not no_x and (expected is None or not r['cutX'] or not r['cutX'][0]['angle']-1e-10<=expected<=r['cutX'][-1]['angle']+1e-10)
        intended=dict(hpbw=None,sll=None,peakAngle=None,peakDb=None) if outside or outside_cut or no_x else analyze_cut(r['cutX'],expected,True)
        if intended['sll'] is not None and intended['peakDb']+intended['sll']<=-79.999:intended['sll']=None
        if outside:status,reason='outside_visible','理想期望主阶移出可见域；当前切面峰不能替代该主瓣。'
        elif no_x:status,reason='incomplete_beam','有效阵列没有X向孔径，单元方向图不能替代阵列扫描波束。'
        elif outside_cut:status,reason='outside_cut','期望方向不在当前固定命令v的可见切面内。'
        elif intended['peakAngle'] is None or intended['hpbw'] is None:status,reason='incomplete_beam','未解析期望方向附近完整局部半功率主瓣。'
        elif intended['sll'] is None:status,reason='unresolved_sidelobe','主瓣已解析，第一极小值以外旁瓣区域未解析。'
        elif intended['sll']>.05:status,reason='stronger_lobe','存在强于期望局部主瓣的其他方向峰值。'
        else:status,reason='resolved','当前切面期望局部主瓣和旁瓣已解析。'
        resolved=status in ('resolved','stronger_lobe')
        return dict(angle=angle,hpbw=intended['hpbw'],sll=intended['sll'],expectedAngle=expected,expectedU=eu,expectedV=ev,expectedCutU=cu,
                    peakAngle=intended['peakAngle'],dominantPeakAngle=m['peakAngleX'],intendedPeakDb=intended['peakDb'],pointingOffset=None if intended['peakAngle'] is None else intended['peakAngle']-angle,
                    pointingPowerDb=m['pointingPowerDb'],gratingCount=len(m['gratingLobes']),resolved=resolved,status=status,reason=reason,visibilityModel=visibility,modelNote=model_note,
                    warning=reason if not resolved or status=='stronger_lobe' else None)
    angles=[a for a in sorted(angles) if math.sin(a*RAD)**2+math.sin(c['steerY']*RAD)**2<=1+1e-12]
    rows,_=ordered_map(one,angles,workers=workers,work_units=len(active)*len(angles)*1440,progress=progress,cancelled=cancelled)
    return rows


def uv_map(config,elements=None,size=81,*,cancelled=None):
    if type(size) is not int or not 21<=size<=301:raise ValueError('u–v显示网格须为21～301。')
    c=normalized_config(config);elements=elements or build_elements(c);axis=np.linspace(-1,1,size)
    u,v=np.meshgrid(axis,axis);values=field_values(c,elements,u,v,cancelled=cancelled)
    return dict(u=axis.tolist(),v=axis.tolist(),values=[[None if not math.isfinite(p) else float(p) for p in row] for row in values])
