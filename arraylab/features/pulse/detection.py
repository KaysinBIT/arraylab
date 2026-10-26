"""Power-domain CFAR and finite-sample detection statistics.

CA/OS scale factors below are for a single complex Gaussian cell (exponential
power) and independent identically distributed training cells. Correlated
range/Doppler cells are deliberately not advertised as achieving nominal Pfa.
"""
from __future__ import annotations

from functools import lru_cache
import math
import numpy as np
from scipy import ndimage, signal
from scipy.optimize import brentq
from scipy.stats import beta

# Bound the temporary training-cell matrix independently of range-grid width.
OS_WORK_VALUES = 262144


@lru_cache(maxsize=256)
def os_scale(training_count: int, order: int, pfa: float) -> float:
    """Scale of the order-th smallest exponential training power (1-based)."""
    n, k = int(training_count), int(order)
    if not 1 <= k <= n or not 0 < pfa < 1:
        raise ValueError("OS-CFAR requires 1 <= order <= training count and 0 < Pfa < 1")
    factors = np.arange(n-k+1, n+1, dtype=float)
    target = math.log(pfa)
    def equation(alpha):
        return float(np.sum(np.log(factors)-np.log(factors+alpha))-target)
    upper = 1.0
    while equation(upper) > 0:
        upper *= 2
    return float(brentq(equation, 0, upper, xtol=1e-12))


def cfar_2d(power, settings, cancel=None):
    """Return threshold, boolean mask and estimated noise; axes are [v, r]."""
    power = np.asarray(power, dtype=float)
    if power.ndim != 2 or np.any(power < 0):
        raise ValueError("CFAR power must be a nonnegative two-dimensional matrix")
    tr, tv = map(int, settings.get("training", [8, 4]))
    gr, gv = map(int, settings.get("guard", [2, 1]))
    if min(tr,tv,gr,gv) < 0 or tr+tv == 0:
        raise ValueError("Invalid CFAR training/guard dimensions")
    hr, hv = tr+gr, tv+gv
    shape = (2*hv+1, 2*hr+1)
    footprint = np.ones(shape, dtype=bool)
    footprint[hv-gv:hv+gv+1, hr-gr:hr+gr+1] = False
    count = int(footprint.sum())
    pfa = float(settings.get("pfa", 1e-4))
    if not 0 < pfa < 1 or count == 0:
        raise ValueError("Invalid nominal Pfa or empty training set")
    threshold = np.full(power.shape, np.nan)
    noise = np.full(power.shape, np.nan)
    if power.shape[0] < shape[0] or power.shape[1] < shape[1]:
        return threshold, np.zeros(power.shape,dtype=bool), noise
    valid = np.isfinite(power)
    interior = np.zeros(power.shape,dtype=bool)
    interior[hv:power.shape[0]-hv if hv else None,
             hr:power.shape[1]-hr if hr else None] = True
    # A rectangular training ring is two disjoint separable, positive sums.
    # Unlike outer-box minus guard-box this avoids catastrophic subtraction
    # of a strong CUT; unlike direct 2-D convolution it is linear in widths.
    vertical_edges=np.ones(2*hv+1);vertical_edges[tv:tv+2*gv+1]=0
    horizontal_edges=np.ones(2*hr+1);horizontal_edges[tr:tr+2*gr+1]=0
    def ring_sum(values):
        top_bottom=ndimage.convolve1d(values,vertical_edges,axis=0,mode='constant',cval=0)
        top_bottom=ndimage.convolve1d(top_bottom,np.ones(2*hr+1),axis=1,mode='constant',cval=0)
        middle=ndimage.convolve1d(values,np.ones(2*gv+1),axis=0,mode='constant',cval=0)
        sides=ndimage.convolve1d(middle,horizontal_edges,axis=1,mode='constant',cval=0)
        return top_bottom+sides
    valid_training = ring_sum(valid.astype(float)) >= count-0.5
    accepted = interior & valid_training & valid
    clean = np.where(valid,power,0)
    method = settings.get("method","ca").lower()
    if method == "ca":
        estimate = ring_sum(clean)/count
        scale = count * math.expm1(-math.log(pfa)/count)
        noise[accepted] = estimate[accepted]
        threshold[accepted] = estimate[accepted]*scale
    elif method == "os":
        order = min(count,max(1,int(math.ceil(count*float(settings.get("rank_fraction",0.75))))))
        scale = os_scale(count,order,pfa)
        windows = np.lib.stride_tricks.sliding_window_view(clean, shape)
        # Partition both Doppler rows and range columns. A single long row
        # with a large guard/training footprint can otherwise require GiB.
        mean_order = float(np.sum(1/np.arange(count-order+1,count+1)))
        block=max(1,min(256,OS_WORK_VALUES//count))
        for i in range(windows.shape[0]):
            if cancel and cancel(): raise InterruptedError("CFAR cancelled")
            j = i+hv
            for first in range(0,windows.shape[1],block):
                if cancel and cancel():raise InterruptedError('CFAR cancelled')
                last=min(first+block,windows.shape[1])
                row=windows[i,first:last][:,footprint]
                ordered=np.partition(row,order-1,axis=-1)[:,order-1]
                sl=slice(hr+first,hr+last)
                ok=accepted[j,sl]
                noise[j,sl]=np.where(ok,ordered/mean_order,np.nan)
                threshold[j,sl]=np.where(ok,ordered*scale,np.nan)
    else:
        raise ValueError("Unknown CFAR method")
    return threshold, accepted & (power>threshold), noise


def peak_cells(power, mask, max_count=64, min_separation=(1,2)):
    """Data-only local maxima with deterministic nonmaximum suppression."""
    power = np.asarray(power,float)
    finite=np.isfinite(power)
    safe=np.where(finite,power,-np.inf)
    local = ndimage.maximum_filter(safe, size=(3,3),mode="constant",cval=-np.inf)
    candidates=np.flatnonzero(np.asarray(mask,bool)&finite&(power>0)&(safe>=local))
    # Keep O(number of peaks) numeric arrays, not millions of Python objects.
    order=candidates[np.lexsort((candidates,-power.ravel()[candidates]))]
    selected = []
    for flat in order:
        row,col=divmod(int(flat),power.shape[1])
        if any(abs(row-r)<=min_separation[0] and abs(col-c)<=min_separation[1] for r,c in selected):
            continue
        selected.append((row,col))
        if len(selected)>=max_count: break
    return selected


def binomial_interval(successes: int, trials: int, confidence=0.95):
    """Exact Clopper-Pearson interval; zero successes has a nonzero upper bound."""
    k,n = int(successes),int(trials)
    if n<=0 or not 0<=k<=n: raise ValueError("Invalid binomial counts")
    alpha = 1-float(confidence)
    lo = 0.0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1))
    hi = 1.0 if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return {"count":k,"trials":n,"estimate":k/n,"confidence":confidence,
            "lower":lo,"upper":hi,
            "zero_event_one_sided_upper":1-(1-confidence)**(1/n) if k==0 else None}
