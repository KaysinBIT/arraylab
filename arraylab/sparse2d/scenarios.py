"""Discrete scan/frequency cases; coordinates are fixed in reference wavelengths."""
from dataclasses import replace
import numpy as np

LIGHT_MM_GHZ=299.792458


def scenarios(c):
    if not isinstance(c.scan_angles,str) or not isinstance(c.frequencies_ghz,str):
        raise ValueError('扫描角列表和频率列表必须为文本。')
    text=c.scan_angles.strip().replace('；',';').replace('，',',').replace('\n',';')
    angles=[]
    if text:
        for item in text.split(';'):
            if not item.strip(): continue
            values=item.split(',')
            if len(values)!=2: raise ValueError('扫描角格式：0,0;30,0;30,90（每项 θ,φ）。')
            try: t,p=map(float,values)
            except ValueError: raise ValueError('扫描角必须为数值 θ,φ。') from None
            if not np.isfinite(t+p) or not 0<=t<=60 or not -360<=p<=360:
                raise ValueError('联合扫描角 θ 范围 0–60°，φ 范围 −360–360°。')
            if (t,p) not in angles: angles.append((t,p))
    else: angles=[(float(c.theta),float(c.phi))]
    text=c.frequencies_ghz.strip().replace('，',',').replace(';',',').replace('；',',')
    try: frequencies=[float(x.strip()) for x in text.split(',') if x.strip()] if text else [float(c.reference_ghz)]
    except ValueError: raise ValueError('频率格式：9,10,11，单位 GHz。') from None
    frequencies=list(dict.fromkeys(frequencies))
    if not angles or not frequencies: raise ValueError('扫描角和频率列表不能为空项列表。')
    if any(not np.isfinite(f) or not .1<=f<=1000 for f in frequencies):
        raise ValueError('频率必须为 0.1–1000 GHz 内有限正数。')
    if any(not .25<=f/c.reference_ghz<=4 for f in frequencies):
        raise ValueError('Demo 要求频率/参考频率在 0.25–4 之间。')
    if len(angles)*len(frequencies)>25:
        raise ValueError('Demo 最多联合 25 个扫描角与频率的组合。')
    return [{'index':i,'theta':t,'phi':p,'frequency_ghz':f,'frequency_ratio':f/c.reference_ghz,
             'label':f'θ={t:g}°, φ={p:g}°, {f:g} GHz'}
            for i,(t,p,f) in enumerate((t,p,f) for t,p in angles for f in frequencies)]


def case_config(c,case):
    """Angular sampling only; do not use this scaled config to move physical layout."""
    return replace(c,theta=case['theta'],phi=case['phi'],spacing=c.spacing*case['frequency_ratio'])
