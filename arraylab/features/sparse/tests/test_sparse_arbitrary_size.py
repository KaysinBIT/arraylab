"""Integer-grid and singleton checks against explicit complex array sums."""
from dataclasses import asdict,replace

import numpy as np
import pytest

import arraylab.features.sparse.engine.core as core
from arraylab.features.sparse.engine.core import Config, run, required_center, admissible_pairs, candidates
from arraylab.features.sparse.engine.headless import jsonable, restore_bundle
from arraylab.features.sparse.engine.weighting import Weighting, add_weighting, window_weights
from arraylab.features.sparse.engine.elements import build_element


def small(**changes):
    return replace(Config(nx=5,ny=7,k=7,iterations=3,opt_grid=25,eval_grid=81),**changes)


@pytest.mark.parametrize('nx,ny,k',[(5,7,7),(5,6,8),(6,5,8),(21,23,11),(24,26,12),(1,9,5),(9,1,5),(21,25,501)])
def test_custom_size_exact_counts_and_inversion(nx,ny,k):
    config=small(nx=nx,ny=ny,k=k)
    bundle=run(config,workers=1)
    methods={'Random symmetric','Greedy exchange','Simulated annealing','L1 top-K uniform','Regular thinning'}
    for row in bundle['results']:
        lookup=dict(zip(row['indices'],row['weights']))
        assert all(lookup[nx*ny-1-index]==weight for index,weight in lookup.items())
        assert len(row['indices'])==len(set(row['indices']))
        assert np.isclose(row['weights'].sum(),1)
        if row['name'] in methods:
            assert len(row['indices'])==k
            assert np.allclose(row['weights'],1/k)
            if k%2:assert nx*ny//2 in lookup
    restored,restored_config,_=restore_bundle(jsonable(bundle))
    assert restored_config==config
    for original,loaded in zip(bundle['results'],restored['results']):
        np.testing.assert_array_equal(original['indices'],loaded['indices'])
        np.testing.assert_allclose(original['weights'],loaded['weights'],rtol=0,atol=0)


@pytest.mark.parametrize('changes',[{'nx':0},{'ny':-1},{'nx':True},{'ny':2.5},{'k':0},
    {'nx':6,'ny':7,'k':7},{'nx':7,'ny':6,'k':7},{'k':37},
    {'hole_shape':'circle'},{'hole_shape':'rectangle'}, {'min_spacing':10}])
def test_invalid_counts_and_center_constraints(changes):
    with pytest.raises(ValueError):small(**changes).validate()


def test_center_distance_filters_pairs_before_search():
    config=small(min_spacing=.6)
    pairs=admissible_pairs(config)
    xy=candidates(config)[1]
    assert np.all(np.linalg.norm(xy[pairs[:,0]],axis=1)>=.6-1e-10)
    assert not any(np.isclose(np.linalg.norm(xy[pairs[:,0]],axis=1),.5))
    bundle=run(config,workers=1)
    assert all(row['metrics']['spacing_violations']==0 for row in bundle['results'])


def test_even_k_on_odd_grid_excludes_center_and_labels_reference():
    bundle=run(small(k=8),workers=1)
    assert bundle['results'][0]['name']=='Paired reference (center excluded)'
    assert 'central singleton excluded' in bundle['results'][0]['notes']
    assert all(17 not in row['indices'] for row in bundle['results'])


def test_odd_uniform_objective_uses_element_envelope():
    config=small(theta=20,phi=35,element_model='patch')
    bundle=run(config,workers=1)
    element=build_element(config)
    matrix=core.joint_matrix(config,element)
    pairs=admissible_pairs(config)
    for row in bundle['results']:
        if row['name'] not in ('Greedy exchange','Simulated annealing'):continue
        selected=[index for index,pair in enumerate(pairs) if pair[0] in row['indices']]
        expected=np.max(abs((2*matrix[:,selected].sum(axis=1)+matrix[:,-1])/config.k))
        assert row['history'][-1]==pytest.approx(expected,abs=1e-12)
        for case,cc,(side,*_) in core.evaluation_domains(config):
            explicit=core.response_field(row['xy'],row['weights'],side,config,case,element)
            assert np.max(abs(explicit))==pytest.approx(expected,abs=1e-12)


def test_center_group_dynamic_range_is_per_element():
    config=small()
    bundle=run(config,workers=1)
    source=next(i for i,row in enumerate(bundle['results']) if row['name']=='Simulated annealing')
    uniform=add_weighting(bundle,source,Weighting(method='minimax',dr_db=0),workers=1)
    weights=uniform['results'][-1]['weights']
    np.testing.assert_allclose(weights,np.full(config.k,1/config.k),atol=1e-12)
    tapered=add_weighting(bundle,source,Weighting(method='minimax',dr_db=6),workers=1)
    row=tapered['results'][-1]
    assert 20*np.log10(row['weights'].max()/row['weights'].min())<=6+1e-8
    assert row['weighting']['details']['actual_dynamic_range_db']==pytest.approx(row['metrics']['amplitude_dynamic_range_db'])
    np.testing.assert_array_equal(row['indices'],bundle['results'][source]['indices'])
    restore_bundle(jsonable(tapered))


def test_joint_case_amplitude_lp_keeps_singleton_and_uniform_zero_dr():
    bundle=run(small(scan_angles='0,0;20,30',amplitude_dr_db=0),workers=1)
    rows=[row for row in bundle['results'] if row['name'] in ('Joint amplitude LP','SA amplitude LP')]
    assert len(rows)==2
    for row in rows:
        assert 17 in row['indices']
        np.testing.assert_allclose(row['weights'],np.full(len(row['indices']),1/len(row['indices'])),atol=1e-12)


def test_singleton_search_reweight_and_restore():
    config=small(nx=1,ny=9,k=1)
    bundle=run(config,workers=1)
    index=next(i for i,row in enumerate(bundle['results']) if row['name']=='Simulated annealing')
    row=bundle['results'][index]
    np.testing.assert_array_equal(row['indices'],[4])
    np.testing.assert_array_equal(row['weights'],[1.])
    result=add_weighting(bundle,index,Weighting(method='minimax',dr_db=0),workers=1)
    np.testing.assert_array_equal(result['results'][-1]['weights'],[1.])
    restore_bundle(jsonable(result))
    for method in ('uniform','gaussian','taylor','chebyshev'):
        setting=Weighting(method=method,nbar=2).validate(config)
        weights,_=window_weights(config,np.array([[0,4]]),setting)
        np.testing.assert_array_equal(weights,[1.])


def test_tiny_aperture_reports_undefined_sidelobe_domain():
    config=small(nx=1,ny=1,k=1)
    config.validate()
    with pytest.raises(ValueError,match='covers the visible disk'):
        run(config,workers=1)


def test_configuration_large_integer_validation_has_no_grid_allocation(monkeypatch):
    def unexpected(*args,**kwargs):raise AssertionError('configuration must not allocate this grid')
    monkeypatch.setattr(core,'candidates',unexpected)
    config=small(nx=100000,ny=100001,k=8)
    assert config.validate() is config
    def denied(*args,**kwargs):raise ValueError('resource preflight reached')
    monkeypatch.setattr(core,'ensure_resources',denied)
    with pytest.raises(ValueError,match='resource preflight reached'):run(config,workers=1)


def test_restore_checks_resources_before_large_candidate_allocation(monkeypatch):
    import arraylab.features.sparse.engine.headless as headless
    def denied(*args,**kwargs):
        assert kwargs['operation']=='restore'
        raise ValueError('restore resource preflight reached')
    monkeypatch.setattr(headless,'ensure_resources',denied)
    data={'config':asdict(small(nx=100000,ny=100001,k=8))}
    with pytest.raises(ValueError,match='restore resource preflight reached'):restore_bundle(data)


def test_even_pair_sequence_and_legacy_uniform_objective():
    config=small(nx=6,ny=8,k=16)
    integer,xy,pairs=candidates(config)
    np.testing.assert_array_equal(pairs,np.column_stack((np.arange(24),np.arange(47,23,-1))))
    rng=np.random.default_rng(config.seed)
    initial=core.random_groups(config,rng)
    np.testing.assert_array_equal(initial,np.sort(np.random.default_rng(config.seed).choice(24,8,replace=False)))
    uv=core.sample_domain(config,25)
    side=uv[core.sidelobe_mask(uv,config)]
    matrix=core.group_matrix(config,side)
    chosen,hist=core.exchange(matrix,initial)
    selected=pairs[chosen].ravel()
    direct=core.field(xy[selected],np.full(len(selected),1/len(selected)),side,core.scan_uv(config))
    assert hist[-1]==pytest.approx(np.max(abs(direct)),abs=2e-14)
    assert required_center(config) is None


@pytest.mark.parametrize('nx,ny,k', [(7,7,7),(7,7,1),(1,9,1),(9,1,1)])
def test_regular_single_point_axis_uses_the_grid_center(nx,ny,k):
    config=small(nx=nx,ny=ny,k=k)
    pairs=core.admissible_pairs(config)
    selected=core.regular_groups(config)
    assert selected is not None
    ids=np.r_[pairs[selected].ravel(),required_center(config)].astype(int)
    assert len(ids)==k
    points=core.candidates(config)[1][ids]
    assert np.any(np.all(points==0,axis=1))
    if k==1: np.testing.assert_array_equal(points,[[0.,0.]])


def test_reevaluate_preflights_new_grid_and_workers(monkeypatch):
    import arraylab.features.sparse.engine.headless as headless
    config=small(nx=3,ny=3,k=3)
    bundle=run(config,workers=1)
    original=headless.ensure_resources
    def check(c,**kwargs):
        if kwargs['operation']=='evaluate':
            assert c.eval_grid==201 and kwargs['workers']==4
            raise ValueError('new evaluation budget checked')
        return original(c,**kwargs)
    monkeypatch.setattr(headless,'ensure_resources',check)
    with pytest.raises(ValueError,match='new evaluation budget checked'):
        headless.reevaluate(jsonable(bundle),eval_grid=201,workers=4)
