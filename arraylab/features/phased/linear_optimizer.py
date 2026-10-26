"""Seeded mirror-symmetric continuous line-position synthesis."""
from copy import deepcopy
import math
import numpy as np
from arraylab.framework.parallel import check_cancelled
from . import engine as E

DEFAULTS=dict(count=12,aperture=8,minSpacing=.45,scanMax=30,targetSll=-15,maxHpbw=12,iterations=500,seed=42)


def rng(seed):
    state=int(seed)&0xffffffff
    def random():
        nonlocal state
        state=(state+0x6D2B79F5)&0xffffffff;t=state
        t=((t^(t>>15))*(t|1))&0xffffffff
        t^=(t+(((t^(t>>7))*(t|61))&0xffffffff))&0xffffffff
        return ((t^(t>>14))&0xffffffff)/4294967296
    return random


def validate(options=None):
    if options is not None and type(options) is not dict:return dict(ok=False,errors=['一维优化设置须为对象。'],options=None)
    o={**DEFAULTS,**(options or {})};errors=[]
    for key,lo,hi,integer in [('count',6,24,True),('aperture',.5,100,False),('minSpacing',.01,3,False),('scanMax',0,70,False),('targetSll',-60,-1,False),('maxHpbw',.1,90,False),('iterations',20,10000,True),('seed',0,4294967295,True)]:
        v=o[key]
        if not E.finite(v) or not lo<=v<=hi or integer and v!=int(v):errors.append(f'{key}超出范围或类型错误。')
        elif integer:o[key]=int(v)
    if not errors:
        if o['count']%2:errors.append('镜像对称要求偶数阵元。')
        if o['aperture']+1e-10<(o['count']-1)*o['minSpacing']:errors.append('孔径不足以满足阵元数与最小间距。')
    return dict(ok=not errors,errors=errors,options=o)


def positions(positive):return [-x for x in reversed(positive)]+list(positive)
def configuration(base,positive):return {**deepcopy(base),'geometry':'custom','window':'uniform','ny':1,'steerY':0,'nx':len(positive)*2,'custom':[dict(x=x,y=0,amplitude=1,phaseDeg=0) for x in positions(positive)]}
def geometry_metrics(positive):
    xs=positions(positive);return dict(minSpacing=float(min(np.diff(xs))),aperture=xs[-1]-xs[0])


def analyze(samples,expected):
    empty=dict(hpbw=None,sll=None,peakAngle=None,peakDb=None)
    if not samples or len(samples)<3:return empty
    values=np.array([s['db'] for s in samples]);angles=np.array([s['angle'] for s in samples]);last=len(values)-1
    peaks=[i for i in range(len(values)) if values[i]>-299.999 and (i>0 and values[i]>values[i-1]+1e-12 or i<last and values[i]>values[i+1]+1e-12) and (i==0 or values[i]>=values[i-1]) and (i==last or values[i]>=values[i+1])]
    if not peaks:return empty
    peak=peaks[0]
    for i in peaks[1:]:
        d=abs(angles[i]-expected)-abs(angles[peak]-expected)
        if d<-1e-10 or abs(d)<1e-10 and values[i]>values[peak]:peak=i
    p=values[peak];lh=rh=peak
    while lh>0 and values[lh]>p-E.HALF:lh-=1
    while rh<last and values[rh]>p-E.HALF:rh+=1
    def cross(a,b):
        return (angles[a]+angles[b])/2 if abs(values[b]-values[a])<1e-15 else angles[a]+(angles[b]-angles[a])*(p-E.HALF-values[a])/(values[b]-values[a])
    width=cross(rh-1,rh)-cross(lh,lh+1) if lh<peak<rh and values[lh]<=p-E.HALF and values[rh]<=p-E.HALF else None
    left=right=peak
    while left>0 and values[left-1]<=values[left]+1e-12:left-=1
    while right<last and values[right+1]<=values[right]+1e-12:right+=1
    side=[]
    if left>0:side.extend(values[:left])
    if right<last:side.extend(values[right+1:])
    return dict(hpbw=None if width is None else float(width),sll=float(max(side)-p) if side and max(side)>-299.999 else None,peakAngle=float(angles[peak]),peakDb=float(p))


def is_feasible(m,o):
    return bool(m['resolved'] and m['worstSll'] is not None and m['hpbw'] is not None and m['worstSll']<=o['targetSll']+1e-8 and m['hpbw']<=o['maxHpbw']+1e-8 and m['minSpacing']>=o['minSpacing']-1e-8)


def score(m,o):
    return 10000 if not m['resolved'] or m['worstSll'] is None or m['hpbw'] is None else (0 if is_feasible(m,o) else 1000)+m['worstSll']+10*max(0,m['hpbw']-o['maxHpbw'])


def metrics(base,positive,angles,step,*,fine=False,cancelled=None):
    ratio=base['frequencyGHz']/base['referenceGHz'];factor=ratio if base['phaseMode']=='ttd' else 1
    xs=np.array(positions(positive));scans=[]
    if not fine:
        count=E.js_round(180/step);grid=np.linspace(-90,90,count+1);sine=np.sin(grid*E.RAD)
    for angle in angles:
        check_cancelled(cancelled)
        desired=math.sin(angle*E.RAD)*factor/ratio
        if abs(desired)>=1:
            scans.append(dict(angle=angle,expectedU=desired,expectedAngle=None,frequencyGHz=base['frequencyGHz'],phaseMode=base['phaseMode'],sll=None,hpbw=None,peakAngle=None,peakDb=None,resolved=False,status='outside_visible',reason='期望主阶波束位于可见域外或边界，不能以其他瓣替代完整主瓣。',_desired=desired));continue
        expected=math.asin(desired)/E.RAD;c={**configuration(base,positive),'steerX':angle};els=E.build_elements(c)
        if fine:
            samples=E.cut(c,els,'x',step,cancelled=cancelled)
        else:
            phases=np.array([e['phaseDeg'] for e in els])*E.RAD
            p=2*math.pi*ratio*sine[:,None]*xs+phases
            values=E.db((np.cos(p).sum(1)**2+np.sin(p).sum(1)**2)/len(xs)**2*np.maximum(0,1-sine*sine)**(base['elementPower']/2))
            samples=[dict(angle=float(a),db=float(v)) for a,v in zip(grid,values)]
        m=E.analyze_cut(samples,expected,True,E.cut_field(c,els,'x',cancelled=cancelled)) if fine else analyze(samples,expected)
        status='incomplete_beam' if m['hpbw'] is None else 'unresolved_sidelobe' if m['sll'] is None else 'stronger_lobe' if m['sll']>.05 else 'resolved'
        reasons={'incomplete_beam':'未解析出完整的期望局部半功率主瓣。','unresolved_sidelobe':'未解析主瓣外的旁瓣区域。','stronger_lobe':'其他方向存在更强峰值。','resolved':'期望局部主瓣与旁瓣已解析。'}
        scans.append(dict(angle=angle,**{k:m[k] for k in ('hpbw','sll','peakAngle','peakDb')},expectedAngle=expected,expectedU=desired,frequencyGHz=base['frequencyGHz'],phaseMode=base['phaseMode'],resolved=m['hpbw'] is not None and m['sll'] is not None,status=status,reason=reasons[status],_desired=desired))
    resolved=all(s['sll'] is not None and s['hpbw'] is not None and abs(s['_desired'])<1 for s in scans)
    worst=max(s['sll'] if s['sll'] is not None else 0 for s in scans);width=max(s['hpbw'] if s['hpbw'] is not None else 180 for s in scans)
    if not resolved:
        if any(s['sll'] is None for s in scans):worst=None
        if any(s['hpbw'] is None for s in scans):width=None
    for s in scans:s.pop('_desired')
    return dict(metrics={**geometry_metrics(positive),'worstSll':worst,'hpbw':width,'resolved':resolved},scans=scans)


def optimize(config=None,options=None,*,cancelled=None,progress=None,workers=None):
    checked=validate(options)
    if not checked['ok']:raise ValueError('；'.join(checked['errors']))
    o=checked['options'];base=E.normalized_config(config)
    if abs(base['steerY'])>1e-10:raise ValueError('一维综合要求Y方向角为0°。')
    electrical=o['aperture']*base['frequencyGHz']/base['referenceGHz']
    if electrical>100:raise ValueError('当前频率下电孔径不能超过100λ。')
    coarse=min(.5,6/electrical);fine=min(.1,1/electrical);random=rng(o['seed'])
    half=o['count']//2;pitch=o['aperture']/(o['count']-1);initial=[(i+.5)*pitch for i in range(half)]
    angles=[-o['scanMax'],0,o['scanMax']] if o['scanMax'] else [0]
    current=initial[:];cm=metrics(base,current,angles,coarse,cancelled=cancelled)['metrics'];cs=score(cm,o)
    best=current[:];bm=cm;bs=cs;history=[]
    def notify(i):
        check_cancelled(cancelled);history.append(dict(iteration=i,worstSll=bm['worstSll'],hpbw=bm['hpbw']))
        if progress:progress(dict(iteration=i,total=o['iterations'],bestSll=bm['worstSll'],progress=i/o['iterations']))
    notify(0)
    for i in range(1,o['iterations']+1):
        check_cancelled(cancelled);cycle=(i-1)%min(250,o['iterations'])
        if cycle==0 and i>1:current=best[:];cm=bm;cs=bs
        temperature=1.4*.035**(cycle/min(250,o['iterations']));candidate=current[:]
        index=math.floor(random()*(half-1));lo=o['minSpacing']/2 if index==0 else candidate[index-1]+o['minSpacing'];hi=candidate[index+1]-o['minSpacing']
        if hi>lo+1e-12:
            if random()<.2:candidate[index]=lo+random()*(hi-lo)
            else:
                scale=(hi-lo)*(.05+.3*temperature/1.4);candidate[index]=max(lo,min(hi,candidate[index]+(random()+random()-1)*scale))
        candidate_metrics=metrics(base,candidate,angles,coarse,cancelled=cancelled)['metrics'];candidate_score=score(candidate_metrics,o)
        if candidate_score<cs or random()<math.exp(min(700,(cs-candidate_score)/temperature)):current=candidate;cm=candidate_metrics;cs=candidate_score
        if candidate_score<bs-1e-10:best=candidate[:];bm=candidate_metrics;bs=candidate_score
        if i%max(1,o['iterations']//50)==0 or i==o['iterations']:notify(i)
    before=metrics(base,initial,angles,fine,fine=True,cancelled=cancelled);checked=metrics(base,best,angles,fine,fine=True,cancelled=cancelled)
    baseline_score=score(before['metrics'],o);candidate_score=score(checked['metrics'],o);fallback=False
    if is_feasible(before['metrics'],o) and not is_feasible(checked['metrics'],o) or candidate_score>baseline_score+1e-8:best=initial[:];checked=before;fallback=True
    after=checked['metrics'];improved=after['worstSll'] is not None and before['metrics']['worstSll'] is not None and after['worstSll']<before['metrics']['worstSll']-1e-6
    feasible=is_feasible(after,o)
    return dict(config=configuration(base,best),before=before['metrics'],after=after,history=history,iterations=o['iterations'],seed=o['seed'],improved=improved,feasible=feasible,options=deepcopy(o),coarseStep=coarse,fineStep=fine,
        baselineFallback=fallback,fineBaselineScore=baseline_score,fineCandidateScore=candidate_score,
        scanResults=checked['scans'],baselineScanResults=before['scans'],notes=[
        '固定端点、固定孔径、均匀幅度的一维镜像对称连续位置退火；不保证全局最优。',
        '基准为同阵元数和孔径的等间距阵，不是当前手工阵列。',
        '先保留全部约束可行性，再比较旁瓣；不以改善旁瓣交换已满足的宽度约束。',
        '细验证已回退同孔径等间距基准。' if fallback else '细验证保留搜索候选。',
        '最终以更细网格独立复核，粗网格改善不替代最终改善。',
        f'只验证X扫描角{angles}；不保证连续角域或宽带。坐标以参考波长λ0表示。',
        '主瓣跟踪期望方向最近局部峰，更强栅瓣计为旁瓣，SLL可为正。',
        '离散约束达标。' if feasible else '离散约束未全部达标，仅作为候选。'])
