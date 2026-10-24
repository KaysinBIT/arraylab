"""Independent narrowband planar-array experiment. Coordinates are in wavelengths.

Symmetric groups allow real-valued LP constraints; ArrayLab bounds validation parallelism.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import linprog
from .constraints import feasible_groups,pair_conflicts,spacing_stats
from .scenarios import LIGHT_MM_GHZ,case_config,scenarios
from .elements import build_element,validate_element_config
from .phase_quantization import validate_bits,evaluate_quantization,sampled_peak,quantized_af
from .numerics import bounded_call,minimax_solver,solve_lp
from .geometry import coarray
from .resources import ensure_resources
from ..parallel import ordered_map, requested_workers, check_cancelled


@dataclass(frozen=True)
class Config:
    nx: int = 12
    ny: int = 12
    k: int = 36
    spacing: float = 0.5
    theta: float = 0.0
    phi: float = 0.0
    seed: int = 42
    iterations: int = 1400
    opt_grid: int = 51
    eval_grid: int = 181
    hole_shape: str = 'none'
    hole_x: float = 2.0
    hole_y: float = 2.0
    amplitude_dr_db: float = 20.0
    reference_ghz: float = 10.0
    frequencies_ghz: str = ''
    scan_angles: str = ''
    min_spacing: float = 0.0
    element_model: str = 'isotropic'
    element_file: str = ''
    element_import_mode: str = 'grid'
    element_dipole_axis: str = 'x'
    element_dipole_length: float = 0.5
    element_patch_hpbw: float = 90.0
    phase_bits: int = 0
    lp_solver: str = 'auto'

    def validate(self):
        minimax_solver(self.lp_solver,0)
        for name in ('nx', 'ny', 'k', 'seed', 'iterations', 'opt_grid', 'eval_grid'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if min(self.nx,self.ny)<1:
            raise ValueError('nx 和 ny 必须为正整数。')
        if not 1 <= self.k <= self.nx * self.ny:
            raise ValueError('K 必须为 1 至 nx*ny 的整数。')
        if self.k%2 and (not self.nx%2 or not self.ny%2):
            raise ValueError('中心对称布局的奇数 K 需要 nx、ny 均为奇数，以保留中心阵元。')
        if not all(np.isfinite(x) for x in (self.spacing, self.theta, self.phi)):
            raise ValueError('parameters must be finite')
        if not 0.1 <= self.spacing <= 2 or not 0 <= self.theta <= 60 or not -360 <= self.phi <= 360:
            raise ValueError('spacing: 0.1..2 lambda; theta: 0..60 deg; phi: -360..360 deg')
        if not 0 <= self.seed < 2**32 or not 1 <= self.iterations <= 20000:
            raise ValueError('seed: 0..2**32-1; iterations: 1..20000')
        if not 25 <= self.opt_grid <= 101 or not 81 <= self.eval_grid <= 401:
            raise ValueError('optimization grid: 25..101; validation grid: 81..401')
        if self.hole_shape not in ('none','rectangle','circle'):
            raise ValueError('hole_shape must be none, rectangle, or circle')
        if not all(np.isfinite(x) for x in (self.hole_x,self.hole_y,self.amplitude_dr_db)):
            raise ValueError('Hole dimensions and amplitude dynamic range must be finite')
        if min(self.hole_x,self.hole_y)<=0 or not 0<=self.amplitude_dr_db<=40:
            raise ValueError('Hole dimensions must be positive; amplitude dynamic range: 0..40 dB')
        if self.k%2 and self.hole_shape!='none':
            raise ValueError('奇数 K 必须保留中心阵元，与中心禁布区域不兼容。')
        if not np.isfinite(self.reference_ghz) or not .1<=self.reference_ghz<=1000:
            raise ValueError('参考频率必须在 0.1–1000 GHz 内。')
        if not np.isfinite(self.min_spacing) or not 0<=self.min_spacing<=100:
            raise ValueError('最小中心距必须在 0–100 λ0 内，0 表示不追加约束。')
        scenarios(self)
        # A cheap configuration check must not allocate an arbitrary user grid.
        # Large geometries are checked after the run/restore resource preflight.
        if self.nx*self.ny<=32768:
            validate_geometry(self)
        validate_element_config(self)
        validate_bits(self.phase_bits)
        return self


def scan_uv(c):
    t, p = np.deg2rad([c.theta, c.phi])
    return np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p)])


def candidates(c):
    xx, yy = np.meshgrid(np.arange(c.nx), np.arange(c.ny))
    integer = np.column_stack((xx.ravel(), yy.ravel()))
    xy = (integer - [(c.nx-1)/2, (c.ny-1)/2]) * c.spacing
    # Flattened reversal pairs each position with its inversion through origin.
    first=np.arange(len(xy)//2)
    groups = np.column_stack((first,len(xy)-1-first))
    return integer, xy, groups


def required_center(c):
    """Exact-K symmetric layouts need the central singleton precisely for odd K."""
    return c.nx*c.ny//2 if c.k%2 else None


def validate_geometry(c):
    available=int(available_mask(c).sum())
    if c.k>available:
        raise ValueError(f'中心禁布后只剩 {available} 个可用位置，目标 K={c.k} 过大；请减小 K 或孔尺寸。')
    if 2*len(admissible_pairs(c))+int(required_center(c) is not None)<c.k:
        raise ValueError('中心对称及最小间距约束下可用位置不足 K；请减小 K 或间距。')


def available_mask(c):
    """False inside the central forbidden region, including its boundary."""
    xy=candidates(c)[1]
    if c.hole_shape=='none':
        return np.ones(len(xy),dtype=bool)
    if c.hole_shape=='rectangle':
        forbidden=(abs(xy[:,0])<=c.hole_x/2+1e-12)&(abs(xy[:,1])<=c.hole_y/2+1e-12)
    else:
        forbidden=np.sum(xy**2,axis=1)<=(c.hole_x/2)**2+1e-12
    return ~forbidden


def admissible_pairs(c):
    pairs=candidates(c)[2]
    allowed=available_mask(c)
    pairs=pairs[allowed[pairs[:,0]]&allowed[pairs[:,1]]]
    xy=candidates(c)[1]
    valid=np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1)>=c.min_spacing-1e-10
    if required_center(c) is not None:
        valid &= np.linalg.norm(xy[pairs[:,0]],axis=1)>=c.min_spacing-1e-10
    return pairs[valid]


def visible_grid(n, shifted=False):
    # Shifted validation points do not coincide with the optimization grid.
    axis = np.linspace(-1, 1, n)
    if shifted:
        axis = (axis[1:] + axis[:-1])/2
    uu, vv = np.meshgrid(axis, axis)
    uv = np.column_stack((uu.ravel(), vv.ravel()))
    return uv[np.sum(uv**2, axis=1) <= 1 + 1e-12]


def sample_domain(c, n, shifted=False):
    """Include exact exclusion edges and visible-circle boundary, not only grid."""
    center=scan_uv(c)
    rx,ry=1/(c.nx*c.spacing),1/(c.ny*c.spacing)
    axis=np.linspace(-1,1,n*2)
    edges=[]
    for sign in (-1,1):
        edges.append(np.column_stack((np.full(len(axis),center[0]+sign*rx),axis)))
        edges.append(np.column_stack((axis,np.full(len(axis),center[1]+sign*ry))))
    edges.append(np.array([center,center+[rx,0],center+[-rx,0],center+[0,ry],center+[0,-ry],
                           center+[rx,ry],center+[rx,-ry],center+[-rx,ry],center+[-rx,-ry]]))
    angle=np.linspace(0,2*np.pi,n*4,endpoint=False)
    edges.append(np.column_stack((np.cos(angle),np.sin(angle))))
    uv=np.vstack([visible_grid(n,shifted)]+edges)
    return np.unique(uv[np.sum(uv**2,axis=1)<=1+1e-12],axis=0)


def sidelobe_mask(uv, c):
    # Fixed reference first-null rectangle. Not a per-layout null search.
    delta = np.abs(uv - scan_uv(c))
    return (delta[:, 0] >= 1/(c.nx*c.spacing)-1e-12) | (delta[:, 1] >= 1/(c.ny*c.spacing)-1e-12)


def manifold(xy, uv, steering=(0, 0)):
    return np.exp(2j*np.pi*((np.asarray(uv)-steering) @ np.asarray(xy).T))


def field(xy, weights, uv, steering=(0, 0)):
    # Chunking bounds peak allocation for large validation grids.
    uv = np.asarray(uv)
    chunk=max(1,min(2048,(32*1024*1024)//max(1,48*len(xy))))
    return np.concatenate([manifold(xy, uv[i:i+chunk], steering) @ weights
                           for i in range(0, len(uv), chunk)])


def response_field(xy,weights,uv,c,case,element=None,view_mode='total',phase_mode='ideal'):
    if phase_mode not in ('ideal','quantized'): raise ValueError('Unknown phase mode')
    element=build_element(c) if element is None else element
    if view_mode=='element': return element.response(uv,case['frequency_ghz'])
    cc=case_config(c,case)
    af=(quantized_af(xy,weights,uv,case,c.phase_bits) if phase_mode=='quantized' and c.phase_bits else
        field(xy*case['frequency_ratio'],weights,uv,scan_uv(cc)))
    if view_mode=='array_factor' or element.model=='isotropic': return af
    return af*element.normalized_response(uv,case)


def db(a):
    return 20*np.log10(np.maximum(np.abs(a), 1e-12))


def group_matrix(c, uv,steering=None,frequency_ratio=1.):
    _, xy, _ = candidates(c)
    pairs=admissible_pairs(c)
    center=scan_uv(c) if steering is None else steering
    positions=xy[pairs[:,0]]
    if required_center(c) is not None:
        positions=np.vstack((positions,np.zeros((1,2))))
    return np.cos(2*np.pi*frequency_ratio*((uv-center) @ positions.T))


def joint_matrix(c,element=None,*,progress=None,cancelled=None):
    element=build_element(c) if element is None else element
    domains=[]
    cases=scenarios(c)
    for index,case in enumerate(cases,1):
        check_cancelled(cancelled)
        cc=case_config(c,case); uv=sample_domain(cc,c.opt_grid); side=uv[sidelobe_mask(uv,cc)]
        if not len(side): raise ValueError(f"工况 {case['label']}：Reference mainlobe mask covers the visible disk；请增大孔径。")
        element.steering_response(case)
        domains.append((case,cc,side))
    result=None; offset=0
    for index,(case,cc,side) in enumerate(domains,1):
        check_cancelled(cancelled)
        if progress: progress(f'构造优化矩阵：工况 {index}/{len(cases)}')
        block=group_matrix(c,side,scan_uv(cc),case['frequency_ratio'])
        if element.model!='isotropic': block*=element.normalized_response(side,case)[:,None]
        if len(domains)==1:return block
        if result is None:result=np.empty((sum(len(row[2]) for row in domains),block.shape[1]),dtype=block.dtype)
        result[offset:offset+len(block)]=block; offset+=len(block)
    return result


def evaluation_domains(c):
    domains=[]
    for case in scenarios(c):
        cc=case_config(c,case); uv=sample_domain(cc,c.opt_grid); side=uv[sidelobe_mask(uv,cc)]
        if not len(side): raise ValueError(f"工况 {case['label']}：Reference mainlobe mask covers the visible disk；请增大孔径。")
        domains.append((case,cc,(side,sample_domain(cc,c.eval_grid),sample_domain(cc,c.eval_grid+2,True))))
    return domains


def random_groups(c, rng):
    pairs=admissible_pairs(c)
    return feasible_groups(pair_conflicts(candidates(c)[1],pairs,c.min_spacing),c.k//2,rng)[0]


def regular_groups(c):
    # Separable, inversion-symmetric, approximately equally spaced subsets.
    integer, _, pairs = candidates(c)
    if c.nx%2==0 and c.ny%2==0:
        sizes = [(a,b) for a in range(2,c.nx+1,2) for b in range(2,c.ny+1,2) if a*b == c.k]
    else:
        sizes=[(a,c.k//a) for a in range(1,min(c.nx,c.k)+1)
               if c.k%a==0 and c.k//a<=c.ny and (a%2==0 or c.nx%2)
               and ((c.k//a)%2==0 or c.ny%2)]
    if not sizes:
        return None
    a,b = min(sizes, key=lambda ab: abs(ab[0]/ab[1] - c.nx/c.ny))
    xs = np.array([(c.nx-1)//2]) if a==1 else np.rint(np.linspace(0,c.nx-1,a)).astype(int)
    ys = np.array([(c.ny-1)//2]) if b==1 else np.rint(np.linspace(0,c.ny-1,b)).astype(int)
    chosen = np.isin(integer[:,0], xs) & np.isin(integer[:,1], ys)
    if not np.array_equal(chosen,chosen[::-1]) or int(chosen.sum())!=c.k:
        return None
    if np.any(chosen&~available_mask(c)):
        return None
    if spacing_stats(candidates(c)[1][chosen],c.min_spacing)['spacing_violations']:
        return None
    selected=np.flatnonzero(chosen[admissible_pairs(c)[:,0]])
    return selected if len(selected)*2+int(required_center(c) is not None)==c.k else None


def exchange(matrix, initial, passes=6,conflicts=None,*,progress=None,cancelled=None,
             fixed_field=None,normalization=None):
    chosen = np.array(initial, copy=True)
    total = matrix[:,chosen].sum(axis=1)
    if fixed_field is not None: total=total+fixed_field
    divisor=len(chosen) if normalization is None else normalization
    history = [float(np.max(np.abs(total))/divisor)]
    check_cancelled(cancelled)
    if not len(chosen): return chosen,history
    for iteration in range(passes):
        check_cancelled(cancelled)
        if progress: progress(f'贪心交换：第 {iteration+1}/{passes} 轮')
        outside = np.setdiff1d(np.arange(matrix.shape[1]),chosen)
        if not len(outside):
            break
        best = (history[-1], None, None)
        for slot, old in enumerate(chosen):
            check_cancelled(cancelled)
            valid=outside
            if conflicts is not None:
                remaining=np.delete(chosen,slot)
                valid=outside[~conflicts[np.ix_(outside,remaining)].any(axis=1)]
            if not len(valid): continue
            batch=max(1,524288//len(matrix))
            slot_best=(float('inf'),None)
            for start in range(0,len(valid),batch):
                check_cancelled(cancelled)
                choices=valid[start:start+batch]
                values = np.max(np.abs(total[:,None]-matrix[:,old,None]+matrix[:,choices]),axis=0)/divisor
                idx = int(np.argmin(values))
                if values[idx] < slot_best[0]:
                    slot_best=(float(values[idx]),int(choices[idx]))
            if slot_best[0] < best[0]-1e-12:
                best=(slot_best[0],slot,slot_best[1])
        if best[1] is None:
            break
        slot, new = best[1:]
        total += matrix[:,new]-matrix[:,chosen[slot]]
        chosen[slot] = new
        history.append(best[0])
    return np.sort(chosen), history


def anneal(matrix, initial, rng, iterations,conflicts=None,*,progress=None,cancelled=None,
           fixed_field=None,normalization=None):
    chosen = np.array(initial, copy=True)
    outside = np.setdiff1d(np.arange(matrix.shape[1]),chosen)
    total = matrix[:,chosen].sum(axis=1)
    if fixed_field is not None: total=total+fixed_field
    divisor=len(chosen) if normalization is None else normalization
    score = float(np.max(np.abs(total))/divisor)
    best, best_score, history = chosen.copy(), score, [score]
    check_cancelled(cancelled)
    if not len(outside) or not len(chosen):
        return np.sort(best), history
    for it in range(iterations):
        check_cancelled(cancelled)
        if progress and it % max(1,min(100,iterations//20)) == 0:
            progress(f'模拟退火：{it}/{iterations}')
        slot, other = int(rng.integers(len(chosen))), int(rng.integers(len(outside)))
        old, new = chosen[slot], outside[other]
        if conflicts is not None and conflicts[new,np.delete(chosen,slot)].any():
            history.append(best_score); continue
        trial = total + matrix[:,new] - matrix[:,old]
        trial_score = float(np.max(np.abs(trial))/divisor)
        temperature = 0.025*(0.0005/0.025)**(it/max(1,iterations-1))
        if trial_score < score or rng.random() < np.exp(min(0,(score-trial_score)/temperature)):
            total, score = trial, trial_score
            chosen[slot], outside[other] = new, old
            if score < best_score:
                best, best_score = chosen.copy(),score
        history.append(best_score)
    check_cancelled(cancelled)
    if progress: progress(f'模拟退火：{iterations}/{iterations}')
    return np.sort(best), history


def reweighted_lp(matrix, rounds=5,*,lp_solver='auto',progress=None,cancelled=None):
    """Positive real group weights, symmetry, fixed sampled sidelobe cap.

    Each group weight is the total weight of an inversion pair or singleton.
    This restricted real LP is not the general complex-valued SOCP.
    """
    n = matrix.shape[1]
    method=minimax_solver(lp_solver,n)
    check_cancelled(cancelled)
    if progress: progress(f'首次最小最大 LP：{method}，{n} 个对称组')
    constraints = np.vstack((matrix,-matrix))
    minimum,first_diagnostics = solve_lp(linprog,np.r_[np.zeros(n),1.0],
        A_ub=np.column_stack((constraints,-np.ones(len(constraints)))),
        b_ub=np.zeros(len(constraints)), A_eq=np.array([np.r_[np.ones(n),0]]), b_eq=[1],
        bounds=[(0,None)]*(n+1), method=method,cancelled=cancelled)
    if not minimum.success:
        raise RuntimeError(f'minimax LP failed: {minimum.message}')
    cap = max(0.08, float(minimum.x[-1])*1.35)
    weights, history = minimum.x[:-1], []
    diagnostics=[first_diagnostics]
    for iteration in range(rounds):
        check_cancelled(cancelled)
        if progress: progress(f'重加权 LP：{iteration+1}/{rounds}')
        penalty = 1/(weights+0.01)
        solution,details = solve_lp(linprog,penalty,A_ub=constraints,b_ub=np.full(len(constraints),cap),
                          A_eq=np.ones((1,n)),b_eq=[1],bounds=(0,None),method='highs',cancelled=cancelled)
        if not solution.success:
            raise RuntimeError(f'reweighted LP failed: {solution.message}')
        weights = solution.x
        diagnostics.append(details)
        history.append({'support_groups':int(np.count_nonzero(weights>1e-7)),
                        'sampled_peak':float(np.max(np.abs(matrix@weights)))})
    check_cancelled(cancelled)
    return weights, {'cap_linear':cap,'minimax_linear':float(minimum.x[-1]),'rounds':history,
                     'requested_solver':lp_solver,'minimax_solver':method,'reweighted_solver':'highs',
                     'solver_diagnostics':diagnostics,
                     'sampled_residual':float(np.max(np.abs(matrix@weights))-cap)}


def amplitude_minimax(matrix, dynamic_range_db,*,lp_solver='auto',progress=None,cancelled=None,
                      group_sizes=None):
    """Fixed support, positive weights, ratio(max/min) <= 10**(DR/20).

    Auxiliary q bounds each physical amplitude g_j/group_size_j; no element disappears.
    The optimum is global only for this finite-sample, restricted real LP.
    """
    n=matrix.shape[1]
    sizes=np.ones(n) if group_sizes is None else np.asarray(group_sizes,dtype=float)
    if sizes.shape!=(n,) or not np.all(np.isfinite(sizes)) or np.any(sizes<=0):
        raise ValueError('每个对称组的阵元数必须为正且与矩阵列数一致。')
    method=minimax_solver(lp_solver,n)
    check_cancelled(cancelled)
    if progress: progress(f'固定布局幅度 LP：{method}，{n} 个对称组')
    ratio=10**(dynamic_range_db/20)
    pattern=np.vstack((matrix,-matrix))
    pattern=np.column_stack((pattern,np.zeros(len(pattern)),-np.ones(len(pattern))))
    lower=np.column_stack((-np.eye(n),sizes,np.zeros(n)))
    upper=np.column_stack((np.eye(n),-ratio*sizes,np.zeros(n)))
    bounds=np.vstack((pattern,lower,upper))
    solution,diagnostics=solve_lp(linprog,np.r_[np.zeros(n+1),1.0],A_ub=bounds,b_ub=np.zeros(len(bounds)),
        A_eq=np.array([np.r_[np.ones(n),0,0]]),b_eq=[1],bounds=(0,None),method=method,cancelled=cancelled)
    if not solution.success:
        raise RuntimeError(f'Fixed-support amplitude LP failed: {solution.message}')
    weights=solution.x[:n]
    return weights,{'requested_dynamic_range_db':dynamic_range_db,
                    'requested_solver':lp_solver,'minimax_solver':method,'solver_diagnostics':diagnostics,
                    'actual_dynamic_range_db':float(db((weights/sizes).max()/(weights/sizes).min())),
                    'sampled_peak_linear':float(np.max(np.abs(matrix@weights))),
                    'lp_peak_linear':float(solution.x[-1]),
                    'preserves_support':True}


def structured_arrays():
    nested = np.unique(np.r_[np.arange(3),4*np.arange(1,4)-1])
    x,y = np.meshgrid(nested,nested)
    product = np.column_stack((x.ravel(),y.ravel()))
    a,b = 3,4
    x1,y1 = np.meshgrid(np.arange(a)*b,np.arange(a)*b)
    x2,y2 = np.meshgrid(np.arange(b)*a,np.arange(b)*a)
    coprime = np.unique(np.vstack((np.column_stack((x1.ravel(),y1.ravel())),
                                  np.column_stack((x2.ravel(),y2.ravel())))),axis=0)
    x,y=np.meshgrid(np.arange(12),np.arange(12))
    box=np.column_stack((x.ravel(),y.ravel()))
    box=box[(box[:,0]==0)|(box[:,0]==11)|(box[:,1]==0)]
    return {'Nested product (3+3)^2':product,'Coprime planar (3,4)':coprime,'Open box (12,12)':box}


def cut_width(xy, weights, c, along):
    center=scan_uv(c)
    transverse=center[1-along]
    extent=np.sqrt(max(0,1-transverse**2))
    axis=np.linspace(-extent,extent,2001)
    uv=np.tile(center,(len(axis),1)); uv[:,along]=axis
    amplitude=np.abs(field(xy,weights,uv,center))
    # Width of the connected above-half-power interval containing steering.
    inside=amplitude>=np.sqrt(0.5)
    peak=int(np.argmin(abs(axis-center[along])))
    if not inside[peak]:
        return None
    left=right=peak
    while left>0 and inside[left-1]: left-=1
    while right<len(axis)-1 and inside[right+1]: right+=1
    if left==0 or right==len(axis)-1:
        return None
    level=np.sqrt(0.5)
    lo=np.interp(level,amplitude[left-1:left+1],axis[left-1:left+1])
    hi=np.interp(level,amplitude[right:right+2][::-1],axis[right:right+2][::-1])
    return float(hi-lo)


def response_cut_width(result,c,case,element,along):
    cc=case_config(c,case); center=scan_uv(cc)
    if element.model=='isotropic':
        return cut_width(result['xy']*case['frequency_ratio'],result['weights'],cc,along)
    extent=np.sqrt(max(0.,1-center[1-along]**2)); axis=np.linspace(-extent,extent,2001)
    uv=np.tile(center,(len(axis),1)); uv[:,along]=axis
    amplitude=abs(response_field(result['xy'],result['weights'],uv,c,case,element))
    radius=1/((c.nx if along==0 else c.ny)*cc.spacing)
    main=np.flatnonzero(abs(axis-center[along])<=radius+1e-12)
    if not len(main): return None
    peak=int(main[np.argmax(amplitude[main])]); level=amplitude[peak]*np.sqrt(.5)
    if level<=0: return None
    inside=amplitude>=level; left=right=peak
    while left>0 and inside[left-1]: left-=1
    while right<len(axis)-1 and inside[right+1]: right+=1
    if left==0 or right==len(axis)-1: return None
    lo=np.interp(level,amplitude[left-1:left+1],axis[left-1:left+1])
    hi=np.interp(level,amplitude[right:right+2][::-1],axis[right:right+2][::-1])
    return float(hi-lo)


def evaluate_result(result,c,domains=None,element=None,*,cancelled=None):
    """Evaluate one fixed layout with the same protocol as the synthesis run."""
    if domains is None: domains=evaluation_domains(c)
    element=build_element(c) if element is None else element
    available=available_mask(c)
    per_case=[]
    for case,cc,grids in domains:
        check_cancelled(cancelled)
        amplitudes=[]; peaks=[]; af_peaks=[]; total_peaks=[]; main_peaks=[]; main_locations=[]
        response_at_scan=element.steering_response(case)
        for grid_index,uv in enumerate(grids):
            check_cancelled(cancelled)
            af=field(result['xy']*case['frequency_ratio'],result['weights'],uv,scan_uv(cc))
            total=af if element.model=='isotropic' else af*element.normalized_response(uv,case)
            mask=sidelobe_mask(uv,cc)
            values=abs(total[mask]); where=int(np.argmax(values))
            amplitudes.append(float(values[where])); peaks.append(uv[mask][where].tolist())
            af_peaks.append(float(np.max(abs(af[mask]))))
            if grid_index:
                where_total=int(np.argmax(abs(total)))
                total_peaks.append((float(abs(total[where_total])),uv[where_total].tolist()))
                main_peaks.append(float(np.max(abs(total[~mask]))))
                main_locations.append(sampled_peak(abs(total[~mask]),uv[~mask],scan_uv(cc)))
        total_peak=max(total_peaks,key=lambda item:item[0]); main_peak=max(main_peaks)
        per_case.append({**case,'optimization_psl_db':float(db(amplitudes[0])),
            'validation_psl_db':float(db(max(amplitudes[1:]))),
            'grid_gap_db':float(db(max(amplitudes[1:]))-db(amplitudes[0])),
            'validation_peak_uv':peaks[1] if amplitudes[1]>=amplitudes[2] else peaks[2],
            'hpbw_u':response_cut_width(result,c,case,element,0),'hpbw_v':response_cut_width(result,c,case,element,1),
            'array_factor_psl_db':float(db(max(af_peaks[1:]))),
            'element_at_steering_db':float(db(response_at_scan)),
            'total_peak_db':float(db(total_peak[0])),'total_peak_uv':total_peak[1],
            'mainlobe_peak_db':float(db(main_peak)),
            'mainlobe_peak_uv':min((item for item in main_locations if item[0]>=main_peak-max(1e-14,main_peak*1e-12)),
                key=lambda item:np.sum((np.asarray(item[1])-scan_uv(cc))**2))[1],
            'relative_to_main_peak_psl_db':float(db(max(amplitudes[1:])/main_peak)),
            'hpbw_definition':'peak of each steering cut within reference mainlobe box; not 3D global-peak beamwidth'})
    result['scenario_metrics']=per_case
    worst=max(per_case,key=lambda item:item['validation_psl_db'])
    optimization=max(item['optimization_psl_db'] for item in per_case)
    ca=coarray(result['integer'])
    distance=spacing_stats(result['xy'],c.min_spacing)
    for case in per_case:
        case['min_distance_lambda_frequency']=(distance['min_distance_lambda_ref']*case['frequency_ratio']
                                              if distance['min_distance_lambda_ref'] is not None else None)
    if distance['spacing_violations']:
        raise RuntimeError('Generated layout violates the requested minimum spacing')
    result['metrics']={'elements':len(result['xy']),'fill':len(result['xy'])/(c.nx*c.ny),
        'available_fill':len(result['xy'])/int(available.sum()),
        'forbidden_elements':int(np.count_nonzero(~available[result['indices']])),
        'amplitude_dynamic_range_db':float(db(result['weights'].max()/result['weights'].min())),
        'optimization_psl_db':optimization,'validation_psl_db':worst['validation_psl_db'],
        'grid_gap_db':worst['validation_psl_db']-optimization,
        'validation_peak_uv':worst['validation_peak_uv'],
        'hpbw_u':worst['hpbw_u'],'hpbw_v':worst['hpbw_v'],
        'element_at_steering_db':worst['element_at_steering_db'],
        'total_peak_db':worst['total_peak_db'],'mainlobe_peak_db':worst['mainlobe_peak_db'],
        'relative_to_main_peak_psl_db':worst['relative_to_main_peak_psl_db'],
        'array_factor_validation_psl_db':max(item['array_factor_psl_db'] for item in per_case),
        'case_count':len(per_case),'worst_case_index':worst['index'],'worst_case_label':worst['label'],
        **distance,'min_distance_mm':distance['min_distance_lambda_ref']*LIGHT_MM_GHZ/c.reference_ghz if distance['min_distance_lambda_ref'] is not None else None,
        'taper_efficiency':float(1/(len(result['weights'])*np.sum(result['weights']**2))),
        'aperture_x_lambda':float(np.ptp(result['xy'][:,0])),
        'aperture_y_lambda':float(np.ptp(result['xy'][:,1])),
        **{key:value for key,value in ca.items() if key not in ('lags','multiplicity')}}
    evaluate_quantization(result,c,domains,element,cancelled=cancelled)
    return result


def evaluate_layouts(results,c,domains,element,*,workers=None,progress=None,cancelled=None):
    # Prime the dipole normalization cache before sharing this read-only object.
    for case,_,_ in domains: element.steering_response(case)
    units=sum(len(row['xy']) for row in results)*sum(len(grid) for _,_,grids in domains for grid in grids)
    def evaluate(row):return evaluate_result(dict(row),c,domains,element,cancelled=cancelled)
    rows,execution=ordered_map(evaluate,results,workers=workers,work_units=units,progress=progress,cancelled=cancelled)
    execution['scope']='sparse_layout_validation'
    return rows,execution


def run(c=Config(), progress=None, *, workers=None, cancelled=None):
    workers=requested_workers(workers)
    c.validate()
    ensure_resources(c,workers=workers)
    validate_geometry(c)
    check_cancelled(cancelled)
    def tell(message):
        check_cancelled(cancelled)
        if progress:progress(message)
        check_cancelled(cancelled)
    integer,xy,_=candidates(c)
    pairs=admissible_pairs(c)
    center=required_center(c)
    has_center=center is not None
    groups=len(pairs)+int(has_center)
    available=available_mask(c)
    element=build_element(c)
    matrix=bounded_call(lambda:joint_matrix(c,element,progress=tell,cancelled=cancelled),cancelled=cancelled)
    rng=np.random.default_rng(c.seed)
    conflicts=pair_conflicts(xy,pairs,c.min_spacing)
    tell('检查间距可行性及公共布局')
    initial,pool,packing=feasible_groups(conflicts,c.k//2,rng,progress=tell,cancelled=cancelled)
    def columns(selected):
        return np.r_[selected,len(pairs)].astype(int) if has_center else selected
    def sizes(selected):
        return np.r_[np.full(len(selected),2),1] if has_center else None
    pool_columns=columns(pool)
    pair_matrix=matrix[:,:len(pairs)] if has_center else matrix
    uniform_options=({'fixed_field':matrix[:,-1]/2,'normalization':c.k/2} if has_center else {})
    raw=[]
    amplitude_info={}
    def add(name,selected=None,group_weights=None,notes='',history=None,seconds=0):
        if group_weights is None:
            group_weights=np.zeros(groups)
            if has_center:
                count=2*len(selected)+1
                group_weights[selected]=2/count
                group_weights[-1]=1/count
            else:
                group_weights[selected]=1/len(selected)
        w=np.zeros(len(xy)); w[pairs[:,0]]=group_weights[:len(pairs)]/2; w[pairs[:,1]]=group_weights[:len(pairs)]/2
        if has_center: w[center]=group_weights[-1]
        active=np.flatnonzero(w>1e-9)
        w=w[active]; w/=w.sum()
        raw.append({'name':name,'indices':active,'xy':xy[active],'integer':integer[active],
                    'weights':w,'notes':notes,'history':history or [],'seconds':seconds})
    hollow=c.hole_shape!='none'
    restricted=len(pool)<len(pairs)
    excludes_center=bool(c.nx%2 and c.ny%2 and not has_center and not hollow)
    reference_name=('Feasible packing reference' if restricted else ('Hollow uniform' if hollow else
        ('Paired reference (center excluded)' if excludes_center else 'Dense reference')))
    reference_notes=('Uniform feasible packing reference; not necessarily a maximum packing' if restricted else
        ('All admissible inversion pairs; central singleton excluded for even K; count differs from K' if excludes_center else
         'All admissible positions, uniform amplitude; count differs from K'))
    add(reference_name,pool,notes=reference_notes)
    joint=len(scenarios(c))>1
    if hollow or joint:
        tell('优化参考布局幅度')
        start=time.perf_counter(); ag,info=amplitude_minimax(matrix[:,pool_columns],c.amplitude_dr_db,
            lp_solver=c.lp_solver,progress=tell,cancelled=cancelled,group_sizes=sizes(pool))
        name='Packing amplitude LP' if restricted else ('Hollow amplitude LP' if hollow else 'Joint amplitude LP')
        amplitude_info[name]=info
        full=np.zeros(groups); full[pool_columns]=ag
        add(name,group_weights=full,seconds=time.perf_counter()-start,
            notes='Same support as the feasible reference; one shared positive amplitude vector optimized across all cases')
    regular=regular_groups(c)
    if regular is not None:
        add('Regular thinning',regular,notes='Symmetric separable rounded grid, not exact uniform spacing')
    add('Random symmetric',initial,notes='One seeded realization, not an ensemble average')
    tell('开始贪心交换')
    start=time.perf_counter(); selected,hist=bounded_call(lambda:exchange(pair_matrix,initial,
        conflicts=conflicts if conflicts.any() else None,progress=tell,cancelled=cancelled,
        **uniform_options),cancelled=cancelled)
    add('Greedy exchange',selected,history=hist,seconds=time.perf_counter()-start)
    tell('开始模拟退火')
    start=time.perf_counter(); selected,hist=bounded_call(lambda:anneal(pair_matrix,initial,rng,c.iterations,
        conflicts=conflicts if conflicts.any() else None,progress=tell,cancelled=cancelled,
        **uniform_options),cancelled=cancelled)
    add('Simulated annealing',selected,history=hist,seconds=time.perf_counter()-start)
    if hollow or joint:
        tell('优化所选布局幅度')
        start=time.perf_counter(); sg,info=amplitude_minimax(matrix[:,columns(selected)],c.amplitude_dr_db,
            lp_solver=c.lp_solver,progress=tell,cancelled=cancelled,group_sizes=sizes(selected))
        amplitude_info['SA amplitude LP']=info
        full=np.zeros(groups); full[columns(selected)]=sg
        add('SA amplitude LP',group_weights=full,seconds=time.perf_counter()-start,
            notes='Exactly the same K positions as Simulated annealing; only positive amplitudes change, with bounded dynamic range')
    tell('开始重加权 L1 线性规划')
    start=time.perf_counter(); gw,lp=reweighted_lp(matrix[:,pool_columns],lp_solver=c.lp_solver,progress=tell,cancelled=cancelled)
    lp_seconds=time.perf_counter()-start
    selected=pool[np.argsort(-gw[:len(pool)],kind='stable')[:c.k//2]]
    add('L1 top-K uniform',selected,notes='Rank LP support, select K, reset equal amplitude; cap no longer guaranteed',seconds=lp_seconds)
    full=np.zeros(groups); full[pool_columns]=gw
    add('L1 weighted solution',group_weights=full,notes='Variable support on the documented feasible pool; shared positive taper across cases; not equal-budget comparison',seconds=lp_seconds)
    tell('构造独立密网格及偏移网格')
    domains=evaluation_domains(c)
    raw,execution=evaluate_layouts(raw,c,domains,element,workers=workers,cancelled=cancelled,
        progress=lambda done,total:tell(f'验证布局：{done}/{total}'))
    return {'config':asdict(c),'results':raw,'lp':lp,'amplitude_lp':amplitude_info,'execution':execution,
            'element_pattern':element.snapshot,'scenarios':scenarios(c),'spacing':{**packing,'minimum_lambda_ref':c.min_spacing,
                'minimum_mm':c.min_spacing*LIGHT_MM_GHZ/c.reference_ghz,'lp_pool_pair_indices':pool.tolist(),
                'required_center_index':center},
            'candidate_counts':{'total':len(xy),'available':int(available.sum()),'forbidden':int((~available).sum()),
                'admissible_pair_elements':2*len(pairs),'admissible_center_elements':int(has_center)},
            'model':{'coordinates':'fixed physical xy in reference wavelengths lambda0; u=sin(theta)cos(phi), v=sin(theta)sin(phi)',
                     'normalization':'field at steering = 1, 20log10 amplitude',
                     'mask':'outside fixed reference first-null rectangle: |u-u0|>=1/(nx*d) OR |v-v0|>=1/(ny*d)',
                     'validation':'max over dense and shifted visible-disk grids; sampled, not continuous bound',
                     'assumptions':'per-case narrowband, far field, identical co-oriented scalar element envelope, no mutual coupling or polarization; only +z hemisphere evaluated',
                     'symmetry':'centrosymmetric support and positive real taper; odd K fixes the central singleton, even K excludes it; L1 weighted reference has variable support; steering phase applied analytically',
                     'central_hole':'rectangle: full X/Y widths; circle: diameter=hole_x, hole_y ignored; lambda units; boundary forbidden',
                     'dynamic_range':'amplitude DR applies only to fixed-support amplitude LP methods; reweighted L1 is a separate unconstrained reference',
                     'lp_solver':'auto: minimax uses highs-ipm for >=128 support groups (pairs or central singleton), otherwise highs; explicit highs/highs-ipm selects minimax only; five reweighting rounds always highs; all sampling and constraints retained',
                     'joint_synthesis':'One shared physical support and one shared amplitude vector; minimize worst sampled PSL across the Cartesian product of angles and frequencies',
                     'frequency_steering':'Ideal per-frequency phase correction (TTD-equivalent); phase=-2*pi*(f/f0)*(x*u0+y*v0); no fixed-phase beam-squint model',
                     'spacing':'Pairwise Euclidean center distance in lambda0; equality is allowed; checked on every generated result',
                     'aggregate_metrics':'PSL is worst across cases relative to total field at steering; HPBW is based on each steering-cut mainlobe peak, at the worst validation case',
                     'element_pattern':'Total=common scalar element envelope * array factor; imported data embedded in snapshot, no source-file reread for later weighting/plots; no absolute gain inference',
                     'phase_quantization':'phase_bits=0 disables; 1..12 bit nearest-code replay after ideal synthesis; fixed positions/amplitudes; each frequency reprogrammed; loss retained in plots, not quantization-aware optimization'}}
