"""Full-sphere isotropic planar array factors; explicit numerical peak estimates."""
import math
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from arraylab.shared.numerics.array_fields import field_uv, integrated_power






def peak_search(coordinates, weights, preferred_uv):
    centered = coordinates - np.mean(coordinates, axis=0)
    active = abs(weights) > 0
    for candidate in (np.asarray(preferred_uv), np.zeros(2)):
        values = weights[active] * np.exp(2j*np.pi*(centered[active] @ candidate))
        if np.all(abs(np.angle(values * np.conj(values[0]))) < 1e-10):
            return candidate, float(np.sum(abs(weights))**2), "coherent_bound", 0.0
    diameter = float(np.max(np.linalg.norm(centered[:,None,:]-centered[None,:,:],axis=2)))
    size = max(81, math.ceil(16 * diameter) + 1)
    axis = np.linspace(-1, 1, size)
    u,v = np.meshgrid(axis,axis,indexing="xy")
    valid = u*u+v*v <= 1+1e-14
    power = np.full(u.shape,-np.inf)
    power[valid] = abs(field_uv(np.column_stack([u[valid],v[valid]]),centered,weights))**2
    maxima = np.flatnonzero(valid & (power == maximum_filter(power,size=3,mode="constant",cval=-np.inf)))
    indices = maxima[np.argsort(power.ravel()[maxima])[-32:]]
    seeds = [np.array([u.ravel()[i],v.ravel()[i]]) for i in indices]
    seeds.extend([np.asarray(preferred_uv),np.zeros(2)])
    objective_scale = max(float(np.max(power[valid])), 1e-300)
    best_value, best_uv = -1.0, np.zeros(2)
    def value_gradient(point):
        terms = weights*np.exp(2j*np.pi*(centered @ point))
        field = np.sum(terms)
        derivative = 2j*np.pi*np.sum(terms[:,None]*centered,axis=0)
        return float(abs(field)**2), 2*np.real(np.conj(field)*derivative)
    for seed in seeds:
        solution = minimize(lambda point: -value_gradient(point)[0] / objective_scale,seed,
                            jac=lambda point: -value_gradient(point)[1] / objective_scale,method="SLSQP",
                            bounds=[(-1,1),(-1,1)],constraints={"type":"ineq","fun":lambda point:1-np.dot(point,point),
                                                             "jac":lambda point:-2*point},
                            options={"ftol":1e-12,"maxiter":160})
        for point in (seed,solution.x):
            if not np.isfinite(point).all():continue
            norm = np.linalg.norm(point)
            if norm > 1: point = point / norm
            value = value_gradient(point)[0]
            if value > best_value:
                best_uv,best_value = point.copy(),value
    return best_uv,best_value,"disk_grid_multistart_estimate",2/(size-1)


def planar_pattern_analysis(parameters, coordinates, weights, reference, preferred_uv):
    p = parameters
    weights = np.asarray(weights,dtype=complex)
    angles = np.arange(-90,90,p.step_deg)
    angles = np.append(angles[angles<90-1e-10],90.0)
    phi_cut = math.radians(p.cut_phi_deg)
    directions = np.column_stack([np.sin(np.deg2rad(angles))*math.cos(phi_cut),np.sin(np.deg2rad(angles))*math.sin(phi_cut)])
    theta,phi = np.linspace(0,180,61),np.linspace(-180,180,121)
    tt,pp = np.meshgrid(np.deg2rad(theta),np.deg2rad(phi),indexing="ij")
    uv = np.stack([np.sin(tt)*np.cos(pp),np.sin(tt)*np.sin(pp)],axis=-1)
    def db(field): return (20*np.log10(np.maximum(abs(field)/reference,1e-15))).tolist()
    report = {"cut_angles_deg":angles.tolist(),"cut_db":db(field_uv(directions,coordinates,weights)),
              "sphere_theta_deg":theta.tolist(),"sphere_phi_deg":phi.tolist(),"sphere_db":db(field_uv(uv,coordinates,weights)),
              "metrics":{"directivity_dbi":None,"representative_peak_theta_deg":None,"representative_peak_phi_deg":None,
                         "peak_level_db":None,"peak_search_method":"undefined","search_grid_uv_step":None,
                         "peak_is_not_claimed_unique":True}}
    scale = float(np.max(abs(weights)))
    if scale == 0: return report
    w = weights/scale
    best,peak,method,grid_step = peak_search(coordinates,w,preferred_uv)
    integral = integrated_power(coordinates,w)
    metrics = report["metrics"]
    metrics.update(directivity_dbi=10*math.log10(peak/integral),peak_level_db=20*math.log10(math.sqrt(peak)*scale/reference),
                   peak_search_method=method,search_grid_uv_step=grid_step)
    if np.count_nonzero(abs(weights)>0)>1:
        radius = min(1.0,float(np.linalg.norm(best)))
        metrics["representative_peak_theta_deg"] = math.degrees(math.asin(radius))
        metrics["representative_peak_phi_deg"] = None if radius < 1e-10 else math.degrees(math.atan2(best[1],best[0]))
    return report
