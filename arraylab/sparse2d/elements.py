"""Identical scalar element envelopes; strict CSV import and frozen snapshots."""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

MODELS={'isotropic':'理想点源（各向同性）','dipole':'薄线偶极子（默认半波）',
        'patch':'典型贴片近似（+z 前向）','imported':'导入单元方向图'}
VALUE_COLUMNS=('response_db','amplitude_linear','power_linear')


def validate_element_config(c):
    if c.element_model not in MODELS: raise ValueError('未知单元模型。')
    if c.element_import_mode not in ('grid','axisymmetric'): raise ValueError('导入模式应为 grid 或 axisymmetric。')
    if c.element_dipole_axis not in ('x','y','z'): raise ValueError('偶极子轴向应为 x/y/z。')
    if not np.isfinite(c.element_dipole_length) or not .05<=c.element_dipole_length<=1:
        raise ValueError('偶极子总长度必须在 0.05–1.0 λ0 内。')
    if not np.isfinite(c.element_patch_hpbw) or not 30<=c.element_patch_hpbw<=150:
        raise ValueError('贴片近似单元的半功率全宽必须在 30–150° 内。')
    if not isinstance(c.element_file,str) or (c.element_model=='imported' and not c.element_file.strip()):
        raise ValueError('请先选择单元方向图 CSV。')


def _import_csv(path,mode):
    path=Path(path)
    if path.stat().st_size>10*1024*1024: raise ValueError('方向图 CSV 不能超过 10 MiB。')
    raw=path.read_bytes()
    try: text=raw.decode('utf-8-sig')
    except UnicodeError: raise ValueError('方向图 CSV 需为 UTF-8 或 UTF-8 BOM 编码。') from None
    lines=[line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    reader=csv.DictReader(io.StringIO('\n'.join(lines)))
    fields=reader.fieldnames or []
    if len(fields)!=len(set(fields)): raise ValueError('CSV 表头存在重复列。')
    value_columns=[name for name in VALUE_COLUMNS if name in fields]
    allowed={'theta_deg','phi_deg','frequency_ghz',*VALUE_COLUMNS}
    if 'theta_deg' not in fields or len(value_columns)!=1 or set(fields)-allowed:
        raise ValueError('CSV 需要 theta_deg，以及 response_db/amplitude_linear/power_linear 中恰好一列；可附 phi_deg、frequency_ghz。')
    if mode=='grid' and 'phi_deg' not in fields:
        raise ValueError('二维网格导入需要 phi_deg；单切面只能显式选择轴对称假设。')
    column=value_columns[0]; records={}; phi_values={}; rows=0; peak=0.
    for row in reader:
        rows+=1
        if rows>200000: raise ValueError('方向图最多 200000 行。')
        if None in row or any(v is None or not str(v).strip() for v in row.values()):
            raise ValueError(f'CSV 数据行 {rows} 有缺失或多余字段。')
        try:
            theta=float(row['theta_deg']); phi=float(row.get('phi_deg','0'))
            frequency=float(row['frequency_ghz']) if 'frequency_ghz' in fields else None
            value=float(row[column])
        except (ValueError,TypeError): raise ValueError(f'CSV 数据行 {rows} 含非数值字段。') from None
        if not np.isfinite(theta+phi) or not 0<=theta<=180 or not -360<=phi<=360:
            raise ValueError(f'CSV 数据行 {rows}：θ 应为 0–180°、φ 为 −360–360°，角度非弧度。')
        if frequency is not None and (not np.isfinite(frequency) or not .1<=frequency<=1000):
            raise ValueError(f'CSV 数据行 {rows} 的频率必须为 0.1–1000 GHz。')
        if column=='response_db':
            if np.isneginf(value): amplitude=0.
            elif not np.isfinite(value) or not -600<=value<=300:
                raise ValueError(f'CSV 数据行 {rows}：response_db 范围为 −600–300，−inf 表示零点。')
            else: amplitude=10**(value/20)
        else:
            if not np.isfinite(value) or value<0: raise ValueError(f'CSV 数据行 {rows} 的线性幅度/功率必须为非负有限数。')
            amplitude=np.sqrt(value) if column=='power_linear' else value
        phi=phi%360
        phi_values.setdefault(frequency,set()).add(phi)
        key=(frequency,theta,phi if mode=='grid' else 0.)
        if key in records and not np.isclose(records[key],amplitude,rtol=1e-6,atol=0.):
            raise ValueError('同一角度/频率有冲突数据，或 φ=0/360 周期接缝不一致。')
        records.setdefault(key,float(amplitude)); peak=max(peak,float(amplitude))
    if not records or peak<=0: raise ValueError('方向图数据为空或全部为零。')
    frequencies=sorted(phi_values,key=lambda f:-1 if f is None else f)
    planes=[]
    for frequency in frequencies:
        if mode=='axisymmetric' and len(phi_values[frequency])>1:
            raise ValueError('轴对称模式只能使用无 φ 列或单一 φ 的切面，不会丢弃多个方位。')
        items={(theta,phi):value/peak for (f,theta,phi),value in records.items() if f==frequency}
        theta=sorted({key[0] for key in items}); phi=sorted({key[1] for key in items})
        if len(theta)<2 or theta[0]>1e-8 or theta[-1]<90-1e-8:
            raise ValueError('每个频率的 θ 采样必须覆盖 0–90° 前半球，不进行角域外推。')
        if len(items)!=len(theta)*len(phi): raise ValueError('二维角网格缺少采样点；必须提供完整 θ×φ 矩形网格。')
        max_phi_gap=0.
        if mode=='grid':
            gaps=np.diff(np.r_[phi,phi[0]+360]); max_phi_gap=float(gaps.max())
            if len(phi)<4 or max_phi_gap>90+1e-8:
                raise ValueError('φ 需覆盖一整圈，至少 4 个不同方位，周期相邻间隔不大于 90°。')
        values=np.array([[items[(t,p)] for p in phi] for t in theta])
        for t in (0.,180.):
            if t in theta and not np.allclose(values[theta.index(t)],values[theta.index(t),0],rtol=1e-5,atol=1e-7):
                raise ValueError('极点 θ=0/180 的标量总幅度随 φ 不一致；请检查角定义或是否导入了极化分量。')
        planes.append({'frequency_ghz':frequency,'theta_deg':theta,'phi_deg':phi,
                       'amplitude':values.tolist(),'max_theta_step_deg':float(np.diff(theta).max()),
                       'max_phi_step_deg':max_phi_gap})
    return {'planes':planes,'source':{'filename':path.name,'path':str(path.resolve()),
        'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'rows':rows,'value_column':column},
        'normalization':'One global amplitude maximum across all imported angles and frequencies; interpolation is linear in field magnitude',
        'input_peak_db':float(20*np.log10(peak)),'frequency_independent':frequencies==[None]}


def make_snapshot(c):
    validate_element_config(c)
    snap={'schema_version':1,'model':c.element_model,'name':MODELS[c.element_model],
        'reference_ghz':c.reference_ghz,'coordinate':'theta from +z, phi from +x toward +y; front hemisphere only',
        'scalar_only':True,'parameters':{}}
    if c.element_model=='imported':
        snap.update(_import_csv(c.element_file,c.element_import_mode))
        snap['parameters']={'import_mode':c.element_import_mode}
    elif c.element_model=='dipole':
        snap['parameters']={'axis':c.element_dipole_axis,'length_lambda_ref':c.element_dipole_length}
        snap['normalization']='Thin-wire sinusoidal-current envelope; fixed physical length, normalized to sampled peak at each frequency; no impedance/efficiency model'
    elif c.element_model=='patch':
        hpbw=c.element_patch_hpbw
        snap['parameters']={'hpbw_deg':hpbw,'field_exponent':float(np.log(.5)/(2*np.log(np.cos(np.deg2rad(hpbw/2))))),'normal':'+z'}
        snap['normalization']='Front cos(theta)**q scalar envelope, frequency-independent; an illustrative patch approximation, not a cavity/full-wave patch solution'
    else: snap['normalization']='Unit scalar field in every direction and frequency; ideal point source'
    fingerprint={key:value for key,value in snap.items() if key!='planes'}
    snap['fingerprint']=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()
    return snap


class ElementPattern:
    def __init__(self,snapshot):
        self.snapshot=snapshot; self.model=snapshot['model']; self._peaks={}; self._planes=[]
        for plane in snapshot.get('planes',[]):
            theta=np.array(plane['theta_deg']); phi=np.array(plane['phi_deg']); values=np.array(plane['amplitude'])
            if snapshot['parameters']['import_mode']=='grid':
                interpolation=RegularGridInterpolator((theta,np.r_[phi,phi[0]+360]),np.column_stack((values,values[:,0])),bounds_error=True)
            else: interpolation=None
            self._planes.append((plane['frequency_ghz'],theta,phi,values,interpolation))

    def response(self,uv,frequency_ghz):
        uv=np.asarray(uv,dtype=float).reshape(-1,2)
        squared=np.sum(uv**2,axis=1); radial=np.sqrt(squared); z=np.sqrt(np.maximum(0.,1-squared))
        if np.any(radial>1+1e-8): raise ValueError('单元响应只支持 u²+v²≤1 的前半球。')
        if self.model=='isotropic': return np.ones(len(uv))
        if self.model=='patch': return z**self.snapshot['parameters']['field_exponent']
        if self.model=='dipole':
            p=self.snapshot['parameters']; axis=p['axis']; cosine={'x':uv[:,0],'y':uv[:,1],'z':z}[axis]
            alpha=np.pi*p['length_lambda_ref']*frequency_ghz/self.snapshot['reference_ghz']
            def envelope(mu):
                sine=np.sqrt(np.maximum(0.,1-mu**2))
                return np.abs(np.divide(np.cos(alpha*mu)-np.cos(alpha),sine,out=np.zeros_like(mu),where=sine>1e-8))
            if frequency_ghz not in self._peaks:
                self._peaks[frequency_ghz]=float(np.max(envelope(np.linspace(-1,1,8193))))
            peak=self._peaks[frequency_ghz]
            if peak<=1e-14: raise ValueError('偶极子模型在此频率无法形成有效响应。')
            return envelope(np.clip(cosine,-1,1))/peak
        theta=np.rad2deg(np.arcsin(np.clip(radial,0,1)))
        phi=np.rad2deg(np.arctan2(uv[:,1],uv[:,0]))%360
        def angular(plane):
            _,ta,pa,values,interpolation=plane
            if interpolation is None: return np.interp(theta,ta,values[:,0])
            query_phi=np.where(phi<pa[0],phi+360,phi)
            return interpolation(np.column_stack((np.clip(theta,ta[0],ta[-1]),query_phi)))
        if self.snapshot['frequency_independent']: return angular(self._planes[0])
        frequencies=np.array([p[0] for p in self._planes])
        if frequency_ghz<frequencies[0]-1e-9 or frequency_ghz>frequencies[-1]+1e-9:
            raise ValueError(f'导入方向图未覆盖 {frequency_ghz:g} GHz；禁止频率外推。')
        f=float(np.clip(frequency_ghz,frequencies[0],frequencies[-1]))
        high=int(np.searchsorted(frequencies,f))
        if high==0: return angular(self._planes[0])
        if high==len(frequencies): return angular(self._planes[-1])
        if abs(f-frequencies[high])<1e-10: return angular(self._planes[high])
        fraction=(f-frequencies[high-1])/(frequencies[high]-frequencies[high-1])
        return (1-fraction)*angular(self._planes[high-1])+fraction*angular(self._planes[high])

    def steering_response(self,case):
        t,p=np.deg2rad([case['theta'],case['phi']]); uv=np.array([[np.sin(t)*np.cos(p),np.sin(t)*np.sin(p)]])
        value=float(self.response(uv,case['frequency_ghz'])[0])
        if value<=1e-8:
            raise ValueError(f"工况 {case['label']} 位于单元零点或响应过低（≤−160 dB），不能按扫描点归一化。")
        return value

    def normalized_response(self,uv,case):
        return self.response(uv,case['frequency_ghz'])/self.steering_response(case)


def build_element(c,snapshot=None):
    return ElementPattern(make_snapshot(c) if snapshot is None else snapshot)


def write_example_csv(path,axisymmetric=False):
    """Write a new illustrative table; never overwrite user data."""
    path=Path(path)
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['theta_deg','response_db'] if axisymmetric else ['frequency_ghz','theta_deg','phi_deg','response_db'])
        for frequency in ([None] if axisymmetric else [9.,10.,11.]):
            rolloff={None:1.,9.:.9,10.:1.,11.:.8}[frequency]
            for theta in range(0,91,15):
                for phi in ([0] if axisymmetric else range(0,361,30)):
                    t,p=np.deg2rad([theta,phi])
                    amplitude=rolloff*max(0.,np.cos(t))**1.2*(1-.2*np.sin(t)**2*np.cos(p)**2)
                    value='-inf' if theta==90 else f'{20*np.log10(amplitude):.10g}'
                    writer.writerow([theta,value] if axisymmetric else [frequency,theta,phi,value])
    return path
