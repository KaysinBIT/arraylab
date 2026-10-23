"""Post-sparsification amplitude weighting on unchanged physical positions."""
from dataclasses import asdict,dataclass
import time
import warnings
import numpy as np
from scipy.signal import windows
from .core import (Config,admissible_pairs,amplitude_minimax,available_mask,
    evaluate_layouts,evaluation_domains,joint_matrix)
from ..parallel import requested_workers,check_cancelled
from .elements import build_element

METHODS={'minimax':'副瓣优化 LP','taylor':'Taylor','chebyshev':'Chebyshev',
         'gaussian':'Gaussian','uniform':'均匀'}


@dataclass(frozen=True)
class Weighting:
    method: str = 'minimax'
    dr_db: float = 20.0
    sll_db: float = 30.0
    nbar: int = 4
    sigma: float = .35

    def validate(self,c):
        if self.method not in METHODS:
            raise ValueError('Unknown weighting method')
        if not np.isfinite(self.dr_db) or not 0<=self.dr_db<=40:
            raise ValueError('加权幅度范围必须在 0–40 dB 内。')
        if self.method in ('taylor','chebyshev') and (not np.isfinite(self.sll_db) or not 20<=self.sll_db<=80):
            raise ValueError('窗函数参考副瓣抑制参数必须在 20–80 dB 内。')
        if self.method=='taylor' and (type(self.nbar) is not int or not 2<=self.nbar<=min(10,c.nx,c.ny)):
            raise ValueError('Taylor nbar 必须为 2 至 min(10,nx,ny) 的整数。')
        if self.method=='gaussian' and (not np.isfinite(self.sigma) or not .05<=self.sigma<=1):
            raise ValueError('Gaussian σ/网格跨度必须在 0.05–1.0 内。')
        # All metadata must remain valid strict JSON, even for inactive controls.
        if not all(np.isfinite(x) for x in (self.sll_db,self.sigma)):
            raise ValueError('加权参数必须为有限数。')
        return self


def window_weights(c,integer,settings):
    """Sample separable parent-grid windows at actual retained coordinates."""
    def axis(n):
        if settings.method=='uniform': return np.ones(n)
        if settings.method=='taylor':
            return windows.taylor(n,nbar=settings.nbar,sll=settings.sll_db,norm=True,sym=True)
        if settings.method=='chebyshev':
            # SciPy's low-attenuation warning concerns spectral ENBW monotonicity;
            # this is an antenna-amplitude reference, independently re-evaluated.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='This window is not suitable.*',category=UserWarning)
                return windows.chebwin(n,at=settings.sll_db,sym=True)
        return windows.gaussian(n,std=settings.sigma*(n-1),sym=True)
    wx,wy=axis(c.nx),axis(c.ny)
    values=wx[integer[:,0]]*wy[integer[:,1]]
    if not np.all(np.isfinite(values)) or np.any(values<=0):
        raise ValueError('当前参数生成了非正窗权重，请调整窗参数。')
    floor=float(values.max()/10**(settings.dr_db/20))
    clipped=bool(np.any(values<floor))
    values=np.maximum(values,floor)
    return values/values.sum(),{'floor_applied':clipped,
        'sampling':'separable nx-by-ny symmetric parent-grid window sampled at retained x/y indices',
        'gaussian_sigma_definition':'sigma_x=sigma*(nx-1), sigma_y=sigma*(ny-1) in candidate-index units',
        'reference_sll_is_not_sparse_psl_guarantee':True}


def add_weighting(bundle,source_index,settings=Weighting(),progress=None,*,workers=None,cancelled=None):
    """Return a new bundle; append one result and preserve every source result."""
    workers=requested_workers(workers);check_cancelled(cancelled)
    c=Config(**bundle['config']).validate(); settings.validate(c)
    element=build_element(c,bundle.get('element_pattern'))
    if type(source_index) is not int or not 0<=source_index<len(bundle['results']):
        raise ValueError('请选择一项已计算的布局。')
    tell=progress or (lambda message:None)
    source=bundle['results'][source_index]
    indices=np.asarray(source['indices'],dtype=int)
    integer=np.asarray(source['integer'],dtype=int)
    allowed=available_mask(c)
    if not np.all(allowed[indices]): raise ValueError('源布局包含中心禁布位置。')
    start=time.perf_counter()
    if settings.method=='minimax':
        tell('Fixed sparse-support amplitude optimization')
        pair_lookup={int(index):position for position,index in enumerate(indices)}
        pairs=admissible_pairs(c)
        selected=np.array([j for j,(a,b) in enumerate(pairs) if a in pair_lookup and b in pair_lookup])
        if len(selected)*2!=len(indices):
            raise ValueError('当前受限 LP 需要完整的中心对称位置对。')
        gw,info=amplitude_minimax(joint_matrix(c,element)[:,selected],settings.dr_db)
        weights=np.zeros(len(indices))
        for pair,g in zip(pairs[selected],gw):
            for index in pair: weights[pair_lookup[int(index)]]=g/2
    else:
        tell('Sample window on retained physical positions')
        weights,info=window_weights(c,integer,settings)
    if np.any(weights<=0): raise RuntimeError('加权必须保留全部源阵元。')
    weights=weights/weights.sum()
    layout_name=source.get('weighting',{}).get('layout_name',source['name'])
    number=len(bundle['results'])+1
    metadata={**asdict(settings),'source_index':source_index,'source_name':source['name'],
        'layout_name':layout_name,'support_preserved':True,
        'operation':'replace amplitudes on fixed support, not multiply previous weights',
        'details':info}
    result={'name':f'{layout_name} + {settings.method} #{number}',
        'indices':indices.copy(),'integer':integer.copy(),'xy':np.array(source['xy'],copy=True),
        'weights':weights,'notes':'Post-weighting: original positions/count preserved; original result retained. Window SLL is a reference, not a guarantee on sparse PSL.',
        'history':[],'weighting':metadata,'seconds':time.perf_counter()-start}
    tell('Validate weighted pattern on dense and shifted grids')
    evaluated,execution=evaluate_layouts([result],c,evaluation_domains(c),element,workers=workers,cancelled=cancelled)
    result=evaluated[0];execution['scope']='sparse_weighted_validation'
    return {**bundle,'results':[*bundle['results'],result],'last_weighted_index':len(bundle['results']),'execution':execution,
            'model':{**bundle['model'],'post_weighting':'Uses calculated config snapshot; replaces amplitudes on the selected support; positive weights with separate DR cap; phase steering and positions unchanged'}}
