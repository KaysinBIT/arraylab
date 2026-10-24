"""Exact pairwise center-distance checks and feasible symmetric-group selection."""
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_array
from ..parallel import check_cancelled
from .numerics import solve_milp
from .resources import FIELD_WORKSPACE_BYTES


def pair_conflicts(xy,pairs,minimum):
    n=len(pairs)
    conflicts=np.zeros((n,n),dtype=bool)
    if minimum<=0 or not n: return conflicts
    points=np.asarray(xy)[pairs]
    block_rows=max(1,min(n,FIELD_WORKSPACE_BYTES//(48*n)))
    for start in range(0,n,block_rows):
        block=conflicts[start:start+block_rows]
        for left in range(points.shape[1]):
            for right in range(points.shape[1]):
                distances=np.linalg.norm(points[start:start+block_rows,left,None,:]-points[None,:,right,:],axis=-1)
                block|=distances<minimum-1e-10
    np.fill_diagonal(conflicts,False)
    return conflicts


def spacing_stats(xy,minimum):
    xy=np.asarray(xy)
    if len(xy)<2:
        return {'min_distance_lambda_ref':None,'spacing_violations':0}
    closest=float('inf'); violations=0
    block_rows=max(1,min(len(xy),FIELD_WORKSPACE_BYTES//(48*len(xy))))
    for start in range(0,len(xy),block_rows):
        distances=np.linalg.norm(xy[start:start+block_rows,None,:]-xy[None,:,:],axis=-1)
        for local,row in enumerate(distances):
            values=row[start+local+1:]
            if len(values):
                closest=min(closest,float(values.min()))
                violations+=int(np.count_nonzero(values<minimum-1e-10))
    return {'min_distance_lambda_ref':closest,'spacing_violations':violations}


def feasible_groups(conflicts,target,rng,*,progress=None,cancelled=None):
    check_cancelled(cancelled)
    n=len(conflicts)
    if target>n: raise ValueError('对称配对及最小间距约束下可用位置不足，请减小 K 或间距。')
    if not conflicts.any():
        return np.sort(rng.choice(n,target,replace=False)),np.arange(n),{'initializer':'no inter-group conflicts','pool_is_global_optimum':False}
    best=[]
    for iteration in range(64):
        check_cancelled(cancelled)
        if progress and iteration%8==0:progress(f'间距可行布局搜索：{iteration}/64')
        chosen=[]
        for index in rng.permutation(n):
            check_cancelled(cancelled)
            if not chosen or not conflicts[index,chosen].any(): chosen.append(int(index))
        if len(chosen)>len(best): best=chosen
    solver_status=None
    if len(best)<target:
        check_cancelled(cancelled)
        if progress:progress('间距整数可行性检查：最多 10 秒')
        a,b=np.nonzero(np.triu(conflicts,1)); rows=np.repeat(np.arange(len(a)),2)
        cols=np.column_stack((a,b)).ravel()
        adjacency=coo_array((np.ones(len(cols)),(rows,cols)),shape=(len(a),n)).tocsc()
        constraints=[LinearConstraint(adjacency,-np.inf,np.ones(len(a))),
                     LinearConstraint(np.ones((1,n)),target,np.inf)]
        solution=solve_milp(milp,np.zeros(n),integrality=np.ones(n),bounds=Bounds(0,1),
                      constraints=constraints,cancelled=cancelled)
        solver_status=int(solution.status)
        if solution.x is None:
            if solution.status==2:
                raise ValueError('当前网格、中心禁布、中心对称和最小间距约束下，目标 K 不可行（整数可行性检查）。')
            raise ValueError('间距可行性检查未找到布局或达到时间上限；这不证明不可行，请降低 K/间距或缩小网格重试。')
        best=np.flatnonzero(solution.x>.5).tolist()
        if len(best)<target or conflicts[np.ix_(best,best)].any():
            raise RuntimeError('Spacing feasibility solver returned an invalid support')
        for index in rng.permutation(n):
            check_cancelled(cancelled)
            if index not in best and not conflicts[index,best].any(): best.append(int(index))
    pool=np.sort(best)
    initial=np.sort(rng.choice(pool,target,replace=False))
    check_cancelled(cancelled)
    if progress:progress('间距可行布局搜索完成')
    return initial,pool,{'initializer':'64 randomized maximal packings; MILP fallback if needed',
        'milp_status':solver_status,'milp_native_threads':1,'milp_time_limit_seconds':10.,
        'pool_groups':len(pool),'pool_is_global_optimum':False,
        'lp_search_scope':'reweighted L1 uses this feasible pool; greedy/SA search all admissible pairs with hard conflict checks'}
