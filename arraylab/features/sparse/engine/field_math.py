"""Pure sparse-array field and mask operations, shared within this feature."""
import numpy as np


def scan_uv(c):
    t, p = np.deg2rad([c.theta, c.phi])
    return np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p)])

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

def db(a):
    return 20*np.log10(np.maximum(np.abs(a), 1e-12))
