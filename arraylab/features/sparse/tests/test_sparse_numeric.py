"""Independent analytic and numerical checks; run without ArrayLab installed."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from arraylab.features.sparse.engine.core import Config, anneal, candidates, coarray, cut_width, exchange, field, group_matrix, manifold, random_groups, regular_groups, reweighted_lp, run, sample_domain, scan_uv, sidelobe_mask, structured_arrays, available_mask, admissible_pairs, amplitude_minimax
from arraylab.features.sparse.engine.core import joint_matrix, evaluation_domains
from arraylab.features.sparse.engine.constraints import feasible_groups, pair_conflicts, spacing_stats
from arraylab.features.sparse.engine.scenarios import LIGHT_MM_GHZ, case_config, scenarios
from arraylab.features.sparse.engine.weighting import Weighting, add_weighting, window_weights


class MathematicalChecks(unittest.TestCase):
    def test_full_planar_array_against_geometric_sum(self):
        c=Config(nx=6,ny=8,k=16,theta=25,phi=37)
        xy=candidates(c)[1]; uv=np.array([[0.,0.],[.25,-.36],scan_uv(c),[-.7,.1]])
        du,dv=(uv-scan_uv(c)).T
        def analytic(n,d): return np.sinc(n*c.spacing*d)/np.sinc(c.spacing*d)
        expected=analytic(c.nx,du)*analytic(c.ny,dv)
        np.testing.assert_allclose(field(xy,np.ones(len(xy))/len(xy),uv,scan_uv(c)),expected,atol=2e-14)

    def test_complex_sum_against_scalar_steering_phase(self):
        c=Config(theta=32,phi=-70)
        xy=np.array([[0,0],[.2,.7],[1.1,-.6]])
        weights=np.array([.1,.6,.3]); uv=np.array([[.1,.2],scan_uv(c)])
        import cmath
        expected=[sum(w*cmath.exp(2j*np.pi*float(np.dot(p,u-scan_uv(c))))
                      for p,w in zip(xy,weights)) for u in uv]
        np.testing.assert_allclose(field(xy,weights,uv,scan_uv(c)),expected,atol=2e-15)

    def test_group_reduction_matches_full_complex_manifold(self):
        c=Config(theta=40,phi=60); _,xy,pairs=candidates(c)
        uv=np.array([[.2,.1],[-.6,.3],scan_uv(c)])
        expected=(manifold(xy,uv,scan_uv(c))[:,pairs[:,0]]+manifold(xy,uv,scan_uv(c))[:,pairs[:,1]])/2
        np.testing.assert_allclose(group_matrix(c,uv),expected,atol=1e-14)

    def test_boundary_sampling_contains_exact_guard_and_horizon(self):
        c=Config(); uv=sample_domain(c,51)
        self.assertTrue(np.any(np.isclose(uv[:,0],1/6,atol=1e-14,rtol=0)))
        self.assertTrue(np.any(np.all(np.isclose(uv,[1,0]),axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(uv,[1/6,0]),axis=1)))
        self.assertGreater(np.count_nonzero(sidelobe_mask(uv,c)),100)

    def test_coarray_by_independent_brute_force(self):
        coords=np.array([[0,0],[1,0],[0,2]])
        expected={(int(a[0]-b[0]),int(a[1]-b[1])) for a in coords for b in coords}
        ca=coarray(coords)
        self.assertEqual(set(map(tuple,ca['lags'])),expected)
        self.assertEqual(sum(ca['multiplicity']),len(coords)**2)
        self.assertEqual(ca['unique_lags'],7)
        self.assertEqual(ca['bounding_holes'],8)

    def test_nested_product_and_open_box_have_full_difference_rectangle(self):
        structures=structured_arrays()
        for name in ('Nested product (3+3)^2','Open box (12,12)'):
            ca=coarray(structures[name])
            self.assertEqual(ca['unique_lags'],529)
            self.assertEqual(ca['central_rectangle_lags'],529)
            self.assertEqual(ca['bounding_holes'],0)
        self.assertEqual(len(structures['Coprime planar (3,4)']),24)

    def test_cut_width_against_continuous_root(self):
        from scipy.optimize import brentq
        c=Config(); xy=candidates(c)[1]
        root=brentq(lambda u: np.sinc(c.nx*c.spacing*u)/np.sinc(c.spacing*u)-np.sqrt(.5),.001,1/(c.nx*c.spacing))
        self.assertAlmostEqual(cut_width(xy,np.ones(len(xy))/len(xy),c,0),2*root,places=5)

    def test_selection_budgets_and_reproducibility(self):
        c=Config(); rng=np.random.default_rng(c.seed)
        a=random_groups(c,rng)
        np.testing.assert_array_equal(a,random_groups(c,np.random.default_rng(c.seed)))
        self.assertEqual(len(regular_groups(c)),c.k//2)
        self.assertEqual(len(a),c.k//2); self.assertEqual(len(set(a)),len(a))

    def test_greedy_and_annealing_keep_best_and_budget(self):
        c=Config(nx=6,ny=6,k=12,iterations=150)
        uv=sample_domain(c,31); matrix=group_matrix(c,uv[sidelobe_mask(uv,c)])
        initial=random_groups(c,np.random.default_rng(3))
        for selected,hist in (exchange(matrix,initial),anneal(matrix,initial,np.random.default_rng(4),150)):
            self.assertEqual(len(selected),6); self.assertEqual(len(set(selected)),6)
            self.assertTrue(np.all(np.diff(hist)<=1e-12))
            self.assertAlmostEqual(max(abs(matrix[:,selected].sum(axis=1)))/6,hist[-1],places=12)
        result1=anneal(matrix,initial,np.random.default_rng(4),150)
        result2=anneal(matrix,initial,np.random.default_rng(4),150)
        np.testing.assert_array_equal(result1[0],result2[0])

    def test_lp_actual_sampled_constraint(self):
        c=Config(nx=6,ny=6,k=12); uv=sample_domain(c,31)
        matrix=group_matrix(c,uv[sidelobe_mask(uv,c)])
        weights,info=reweighted_lp(matrix)
        self.assertGreaterEqual(weights.min(),-1e-10)
        self.assertAlmostEqual(weights.sum(),1,places=10)
        self.assertLessEqual(max(abs(matrix@weights)),info['cap_linear']+1e-7)

    def test_invalid_parameters(self):
        for kwargs in ({'nx':0},{'k':15},{'k':200},{'spacing':float('nan')},{'theta':61},{'seed':-1},{'iterations':0},{'nx':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): Config(**kwargs).validate()
        with self.assertRaisesRegex(ValueError,'covers the visible disk'):
            run(Config(nx=4,ny=4,k=8,spacing=.1))

    def test_hole_masks_and_pair_mapping(self):
        for shape,count in [('rectangle',128),('circle',132)]:
            c=Config(hole_shape=shape)
            c.validate(); allowed=available_mask(c)
            self.assertEqual(int(allowed.sum()),count)
            xy=candidates(c)[1]
            expected=[]
            for x,y in xy:
                inside=(abs(x)<=1 and abs(y)<=1) if shape=='rectangle' else x*x+y*y<=1
                expected.append(not inside)
            np.testing.assert_array_equal(allowed,expected)
            pairs=admissible_pairs(c)
            self.assertEqual(len(pairs)*2,count)
            self.assertTrue(np.all(allowed[pairs]))
            np.testing.assert_allclose(xy[pairs[:,0]],-xy[pairs[:,1]],atol=0)
        np.testing.assert_array_equal(available_mask(Config(hole_shape='circle',hole_y=1)),
                                      available_mask(Config(hole_shape='circle',hole_y=9)))
        self.assertIsNone(regular_groups(Config(hole_shape='rectangle')))

    def test_hollow_array_against_full_minus_inner_analytic_pattern(self):
        c=Config(hole_shape='rectangle',theta=30,phi=35)
        _,xy,_=candidates(c); xy=xy[available_mask(c)]
        uv=np.array([[.1,.2],[-.4,.3],scan_uv(c),[.7,-.4]])
        du,dv=(uv-scan_uv(c)).T
        def axis(n,value): return np.sinc(n*c.spacing*value)/np.sinc(c.spacing*value)
        expected=(144*axis(12,du)*axis(12,dv)-16*axis(4,du)*axis(4,dv))/128
        np.testing.assert_allclose(field(xy,np.ones(128)/128,uv,scan_uv(c)),expected,atol=2e-14)

    def test_hole_group_matrix_matches_complex_field(self):
        c=Config(hole_shape='circle',theta=35,phi=20)
        xy=candidates(c)[1]; pairs=admissible_pairs(c)
        uv=np.array([[.1,.2],[-.4,.3],scan_uv(c)])
        a=manifold(xy,uv,scan_uv(c))
        np.testing.assert_allclose(group_matrix(c,uv),(a[:,pairs[:,0]]+a[:,pairs[:,1]])/2,atol=2e-14)

    def test_fixed_support_lp_dynamic_range_and_uniform_feasibility(self):
        c=Config(nx=6,ny=6,k=12,hole_shape='rectangle',hole_x=1,hole_y=1)
        uv=sample_domain(c,31); matrix=group_matrix(c,uv[sidelobe_mask(uv,c)])
        uniform=np.ones(matrix.shape[1])/matrix.shape[1]
        for limit in (0.,10.,20.):
            weights,info=amplitude_minimax(matrix,limit)
            self.assertGreater(weights.min(),0)
            self.assertAlmostEqual(weights.sum(),1,places=10)
            self.assertLessEqual(20*np.log10(weights.max()/weights.min()),limit+1e-6)
            self.assertLessEqual(max(abs(matrix@weights)),max(abs(matrix@uniform))+1e-8)
            self.assertAlmostEqual(max(abs(matrix@weights)),info['lp_peak_linear'],places=8)
            if limit==0: np.testing.assert_allclose(weights,uniform,atol=1e-10)

    def test_hole_budget_validation(self):
        for kwargs in ({'hole_shape':'unknown'},{'hole_x':0},{'hole_y':float('nan')},
                       {'amplitude_dr_db':-1},{'amplitude_dr_db':41},
                       {'hole_shape':'rectangle','k':144},{'hole_shape':'circle','hole_x':100}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): Config(**kwargs).validate()

    def test_hollow_run_support_preserved_and_exported(self):
        from arraylab.features.sparse.engine.demo import export_bundle
        c=Config(nx=6,ny=6,k=12,hole_shape='rectangle',hole_x=1,hole_y=1,
                 amplitude_dr_db=10,iterations=80,opt_grid=25,eval_grid=81)
        bundle=run(c); results={r['name']:r for r in bundle['results']}
        allowed=available_mask(c)
        for result in results.values():
            self.assertTrue(np.all(allowed[result['indices']]))
            self.assertEqual(result['metrics']['forbidden_elements'],0)
        for before,after in [('Hollow uniform','Hollow amplitude LP'),('Simulated annealing','SA amplitude LP')]:
            np.testing.assert_array_equal(results[before]['indices'],results[after]['indices'])
            self.assertLessEqual(results[after]['metrics']['amplitude_dynamic_range_db'],10+1e-6)
            self.assertLessEqual(results[after]['metrics']['optimization_psl_db'],results[before]['metrics']['optimization_psl_db']+1e-7)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            target=Path(temp)/'掏洞结果'; export_bundle(bundle,target)
            data=json.loads((target/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(data['config']['hole_shape'],'rectangle')
            self.assertEqual(data['candidate_counts']['forbidden'],4)
            self.assertIn('SA amplitude LP',data['amplitude_lp'])
            self.assertIn('实际幅度范围',(target/'report.html').read_text(encoding='utf-8'))

    def test_batch_model_and_safe_export(self):
        from arraylab.features.sparse.engine.demo import export_bundle
        c=Config(nx=4,ny=4,k=8,iterations=80,opt_grid=25,eval_grid=81)
        bundle=run(c)
        for r in bundle['results']:
            self.assertAlmostEqual(float(r['weights'].sum()),1,places=12)
            self.assertAlmostEqual(abs(field(r['xy'],r['weights'],np.array([scan_uv(c)]),scan_uv(c))[0]),1,places=12)
            self.assertLessEqual(r['metrics']['validation_psl_db'],1e-9)
            if r['name'] not in ('Dense reference','L1 weighted solution'):
                self.assertEqual(r['metrics']['elements'],c.k)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            target=Path(temp)/'中文结果'
            export_bundle(bundle,target)
            parsed=json.loads((target/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(parsed['config']['k'],8)
            self.assertTrue((target/'report.html').exists())
            with (target/'layout_1.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(len(list(csv.DictReader(f))),16)
            before=(target/'summary.json').read_bytes()
            with self.assertRaises(FileExistsError): export_bundle(bundle,target)
            self.assertEqual(before,(target/'summary.json').read_bytes())

    def test_gaussian_weights_follow_coordinates_not_sparse_order(self):
        c=Config(nx=6,ny=8,k=12)
        integer=np.array([[0,1],[4,6],[2,3],[5,0],[1,7]])
        settings=Weighting(method='gaussian',dr_db=40,sigma=.35)
        weights,info=window_weights(c,integer,settings)
        centered=integer-[(c.nx-1)/2,(c.ny-1)/2]
        expected=np.exp(-.5*((centered[:,0]/(.35*(c.nx-1)))**2+(centered[:,1]/(.35*(c.ny-1)))**2))
        expected/=expected.sum()
        np.testing.assert_allclose(weights,expected,atol=2e-15)
        perm=np.array([3,0,4,1,2]); shuffled,_=window_weights(c,integer[perm],settings)
        np.testing.assert_allclose(shuffled,weights[perm],atol=2e-15)
        self.assertFalse(info['floor_applied'])

    def test_taylor_and_chebyshev_dense_pattern_against_axis_product(self):
        from scipy.signal import windows
        import warnings
        c=Config(nx=8,ny=6,k=12,theta=30,phi=35)
        integer,xy,_=candidates(c)
        uv=np.array([[0.,0.],[.2,-.3],scan_uv(c),[-.6,.2]])
        for method in ('taylor','chebyshev'):
            setting=Weighting(method=method,dr_db=40,sll_db=50,nbar=4)
            weights,_=window_weights(c,integer,setting)
            maker=(lambda n:windows.taylor(n,nbar=4,sll=50,norm=True,sym=True)) if method=='taylor' else (lambda n:windows.chebwin(n,at=50,sym=True))
            wx,wy=maker(c.nx),maker(c.ny)
            self.assertLess(20*np.log10((wx.max()*wy.max())/(wx.min()*wy.min())),40)
            x=(np.arange(c.nx)-(c.nx-1)/2)*c.spacing
            y=(np.arange(c.ny)-(c.ny-1)/2)*c.spacing
            expected=np.array([sum(a*np.exp(2j*np.pi*p*(u[0]-scan_uv(c)[0])) for a,p in zip(wx,x))/wx.sum()*
                               sum(a*np.exp(2j*np.pi*p*(u[1]-scan_uv(c)[1])) for a,p in zip(wy,y))/wy.sum() for u in uv])
            np.testing.assert_allclose(field(xy,weights,uv,scan_uv(c)),expected,atol=2e-14)

    def test_post_weighting_all_methods_preserve_sparse_and_hollow_support(self):
        for shape in ('none','rectangle','circle'):
            c=Config(nx=6,ny=6,k=12,hole_shape=shape,hole_x=1,hole_y=1,iterations=60,opt_grid=25,eval_grid=81)
            original=run(c); source_index=next(i for i,r in enumerate(original['results']) if r['name']=='Simulated annealing')
            source=original['results'][source_index]; before_weights=source['weights'].copy()
            for method in ('uniform','taylor','chebyshev','gaussian','minimax'):
                changed=add_weighting(original,source_index,Weighting(method=method,dr_db=10))
                r=changed['results'][-1]
                self.assertEqual(len(changed['results']),len(original['results'])+1)
                np.testing.assert_array_equal(r['indices'],source['indices'])
                np.testing.assert_array_equal(r['xy'],source['xy'])
                np.testing.assert_array_equal(source['weights'],before_weights)
                self.assertGreater(r['weights'].min(),0)
                self.assertAlmostEqual(r['weights'].sum(),1,places=12)
                self.assertEqual(r['metrics']['elements'],12)
                self.assertEqual(r['metrics']['forbidden_elements'],0)
                self.assertEqual(r['metrics']['unique_lags'],source['metrics']['unique_lags'])
                self.assertLessEqual(r['metrics']['amplitude_dynamic_range_db'],10+1e-7)
                self.assertAlmostEqual(abs(field(r['xy'],r['weights'],np.array([scan_uv(c)]),scan_uv(c))[0]),1,places=12)
                if method=='minimax': self.assertLessEqual(r['metrics']['optimization_psl_db'],source['metrics']['optimization_psl_db']+1e-7)

    def test_zero_dynamic_range_and_clipped_gaussian_keep_all_elements(self):
        c=Config(nx=6,ny=6,k=12,iterations=40,opt_grid=25,eval_grid=81)
        original=run(c); index=next(i for i,r in enumerate(original['results']) if r['name']=='Random symmetric')
        for method in ('uniform','taylor','chebyshev','gaussian','minimax'):
            result=add_weighting(original,index,Weighting(method=method,dr_db=0))['results'][-1]
            np.testing.assert_allclose(result['weights'],np.ones(12)/12,atol=1e-10)
        result=add_weighting(original,index,Weighting(method='gaussian',dr_db=6,sigma=.05))['results'][-1]
        self.assertTrue(result['weighting']['details']['floor_applied'])
        self.assertLessEqual(result['metrics']['amplitude_dynamic_range_db'],6+1e-8)

    def test_invalid_weighting_and_reweighting_replaces_amplitudes(self):
        c=Config(nx=4,ny=4,k=8,iterations=20,opt_grid=25,eval_grid=81)
        for kwargs in ({'method':'bad'},{'dr_db':float('nan')},{'dr_db':41},
                       {'method':'taylor','sll_db':10},{'method':'taylor','nbar':True},
                       {'method':'taylor','nbar':5},{'method':'gaussian','sigma':0}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): Weighting(**kwargs).validate(c)
        bundle=run(c); idx=next(i for i,r in enumerate(bundle['results']) if r['name']=='Greedy exchange')
        one=add_weighting(bundle,idx,Weighting(method='taylor'))
        two=add_weighting(one,len(one['results'])-1,Weighting(method='uniform'))
        np.testing.assert_allclose(two['results'][-1]['weights'],np.ones(8)/8,atol=1e-14)
        self.assertEqual(two['results'][-1]['weighting']['layout_name'],'Greedy exchange')
        for idx_bad in (-1,100,True):
            with self.assertRaises(ValueError): add_weighting(bundle,idx_bad)

    def test_weighting_export_keeps_source_and_comparison(self):
        from arraylab.features.sparse.engine.demo import export_bundle
        c=Config(nx=4,ny=4,k=8,iterations=20,opt_grid=25,eval_grid=81)
        bundle=run(c); idx=next(i for i,r in enumerate(bundle['results']) if r['name']=='L1 top-K uniform')
        result=add_weighting(bundle,idx,Weighting(method='minimax',dr_db=10))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            target=Path(temp)/'稀疏后加权'; export_bundle(result,target)
            data=json.loads((target/'summary.json').read_text(encoding='utf-8'))
            new=data['results'][-1]; old=data['results'][new['weighting']['source_index']]
            self.assertEqual(new['indices'],old['indices'])
            self.assertTrue((target/f'weighting_{len(data["results"])}.png').exists())
            with (target/'weighting_comparison.csv').open(encoding='utf-8-sig') as f:
                comparisons=list(csv.DictReader(f))
            self.assertEqual(comparisons[0]['support_preserved'],'True')
            self.assertEqual(comparisons[0]['weight_method'],'minimax')

    def test_joint_scenario_cartesian_product_and_validation(self):
        c=Config(scan_angles='0,0；30,0;30,90',frequencies_ghz='9，10,11,10')
        c.validate(); cases=scenarios(c)
        self.assertEqual(len(cases),9)
        self.assertEqual([(r['theta'],r['phi'],r['frequency_ghz']) for r in cases],
                         [(t,p,f) for t,p in [(0,0),(30,0),(30,90)] for f in [9,10,11]])
        self.assertEqual(scenarios(Config())[0]['frequency_ratio'],1)
        for kwargs in ({'scan_angles':'30'},{'scan_angles':'70,0'},{'frequencies_ghz':'nan'},
                       {'reference_ghz':0},{'frequencies_ghz':'1'},{'min_spacing':-1},
                       {'scan_angles':';'},{'scan_angles':'0,0;10,0;20,0;30,0;40,0;50,0','frequencies_ghz':'8,9,10,11,12'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): Config(**kwargs).validate()

    def test_joint_matrix_against_independent_complex_pair_sum(self):
        c=Config(nx=6,ny=6,k=8,scan_angles='0,0;30,35',frequencies_ghz='9,11',opt_grid=25,eval_grid=81)
        xy=candidates(c)[1]; pairs=admissible_pairs(c); expected=[]
        for case in scenarios(c):
            cc=case_config(c,case); uv=sample_domain(cc,c.opt_grid); side=uv[sidelobe_mask(uv,cc)]
            a=np.exp(2j*np.pi*case['frequency_ratio']*((side-scan_uv(cc))@xy.T))
            expected.append((a[:,pairs[:,0]]+a[:,pairs[:,1]])/2)
        np.testing.assert_allclose(joint_matrix(c),np.vstack(expected),atol=3e-14)

    def test_frequency_response_against_physical_millimeter_sum(self):
        c=Config(nx=6,ny=6,k=8,reference_ghz=10,scan_angles='30,35',frequencies_ghz='9,11')
        xy=candidates(c)[1][[0,5,12,23,30,35]]; weights=np.array([1,2,3,3,2,1],float)/12
        mm=xy*LIGHT_MM_GHZ/c.reference_ghz; uv=np.array([[.1,.3],[-.4,.2]])
        for case in scenarios(c):
            center=scan_uv(case_config(c,case))
            expected=np.array([sum(w*np.exp(2j*np.pi*case['frequency_ghz']/LIGHT_MM_GHZ*np.dot(p,d-center)) for w,p in zip(weights,mm)) for d in uv])
            np.testing.assert_allclose(field(xy*case['frequency_ratio'],weights,uv,center),expected,atol=3e-14)
        np.testing.assert_array_equal(mm,xy*LIGHT_MM_GHZ/10)

    def test_pair_distance_conflicts_and_feasible_initialization(self):
        c=Config(nx=8,ny=8,k=12,min_spacing=.6)
        xy=candidates(c)[1]; pairs=admissible_pairs(c); conflict=pair_conflicts(xy,pairs,c.min_spacing)
        for i in range(len(pairs)):
            for j in range(i):
                closest=min(float(np.linalg.norm(xy[a]-xy[b])) for a in pairs[i] for b in pairs[j])
                self.assertEqual(bool(conflict[i,j]),closest<c.min_spacing-1e-10)
        a,pool,_=feasible_groups(conflict,6,np.random.default_rng(7))
        b,pool2,_=feasible_groups(conflict,6,np.random.default_rng(7))
        np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(pool,pool2)
        self.assertEqual(len(a),6)
        self.assertEqual(spacing_stats(xy[pairs[a].ravel()],.6)['spacing_violations'],0)
        self.assertFalse(conflict[np.ix_(pool,pool)].any())

    def test_spacing_infeasibility_is_not_silently_relaxed(self):
        conflict=~np.eye(4,dtype=bool)
        with self.assertRaisesRegex(ValueError,'不可行'):
            feasible_groups(conflict,2,np.random.default_rng(1))
        c=Config(nx=4,ny=4,k=4,min_spacing=.8)
        xy=candidates(c)[1]; pairs=admissible_pairs(c)
        self.assertTrue(np.all(np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1)>=.8-1e-10))
        with self.assertRaises(ValueError): run(Config(nx=4,ny=4,k=16,min_spacing=1.))

    def test_greedy_and_anneal_reject_distance_conflicts(self):
        c=Config(nx=8,ny=8,k=12,min_spacing=.6,opt_grid=25,eval_grid=81)
        xy=candidates(c)[1]; pairs=admissible_pairs(c); conflict=pair_conflicts(xy,pairs,.6)
        initial,_,_=feasible_groups(conflict,6,np.random.default_rng(2)); matrix=joint_matrix(c)
        for chosen,history in [exchange(matrix,initial,conflicts=conflict),anneal(matrix,initial,np.random.default_rng(3),300,conflicts=conflict)]:
            self.assertEqual(len(chosen),6)
            self.assertEqual(spacing_stats(xy[pairs[chosen].ravel()],.6)['spacing_violations'],0)
            self.assertTrue(np.all(np.diff(history)<=1e-12))

    def test_joint_run_aggregates_cases_and_preserves_physical_spacing(self):
        c=Config(nx=6,ny=6,k=8,min_spacing=.6,scan_angles='0,0;30,0',frequencies_ghz='9,11',iterations=60,opt_grid=25,eval_grid=81)
        bundle=run(c)
        self.assertEqual(len(bundle['scenarios']),4)
        for result in bundle['results']:
            self.assertEqual(len(result['scenario_metrics']),4)
            metrics=result['metrics']; cases=result['scenario_metrics']
            worst=max(cases,key=lambda row:row['validation_psl_db'])
            self.assertEqual(metrics['validation_psl_db'],worst['validation_psl_db'])
            self.assertEqual(metrics['worst_case_index'],worst['index'])
            self.assertEqual(metrics['spacing_violations'],0)
            self.assertGreaterEqual(metrics['min_distance_lambda_ref'],.6-1e-10)
            self.assertAlmostEqual(metrics['min_distance_mm'],metrics['min_distance_lambda_ref']*LIGHT_MM_GHZ/10,places=10)
            for case in cases:
                self.assertAlmostEqual(case['min_distance_lambda_frequency'],metrics['min_distance_lambda_ref']*case['frequency_ratio'],places=12)
                cc=case_config(c,case)
                self.assertAlmostEqual(abs(field(result['xy']*case['frequency_ratio'],result['weights'],np.array([scan_uv(cc)]),scan_uv(cc))[0]),1,places=12)

    def test_joint_post_weighting_uses_all_cases_with_one_vector(self):
        c=Config(nx=6,ny=6,k=8,min_spacing=.6,scan_angles='0,0;30,0',frequencies_ghz='9,11',iterations=50,opt_grid=25,eval_grid=81)
        bundle=run(c); index=next(i for i,r in enumerate(bundle['results']) if r['name']=='Simulated annealing')
        changed=add_weighting(bundle,index,Weighting(method='minimax',dr_db=10))
        result=changed['results'][-1]; source=bundle['results'][index]
        np.testing.assert_array_equal(result['indices'],source['indices'])
        self.assertEqual(len(result['weights']),8)
        self.assertEqual(len(result['scenario_metrics']),4)
        self.assertLessEqual(result['metrics']['optimization_psl_db'],source['metrics']['optimization_psl_db']+1e-7)
        pairmap={int(i):pos for pos,i in enumerate(result['indices'])}; pairs=admissible_pairs(c)
        columns=[j for j,(a,b) in enumerate(pairs) if int(a) in pairmap and int(b) in pairmap]
        gw=np.array([2*result['weights'][pairmap[int(pairs[j,0])]] for j in columns])
        actual=float(np.max(abs(joint_matrix(c)[:,columns]@gw)))
        self.assertAlmostEqual(actual,result['weighting']['details']['sampled_peak_linear'],places=10)

    def test_joint_export_has_shared_coordinates_amplitudes_and_case_phases(self):
        from arraylab.features.sparse.engine.demo import export_bundle
        c=Config(nx=4,ny=4,k=4,min_spacing=.6,scan_angles='0,0;30,0',frequencies_ghz='9,11',iterations=20,opt_grid=25,eval_grid=81)
        bundle=run(c)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            target=Path(temp)/'联合结果'; export_bundle(bundle,target)
            self.assertTrue((target/'joint_cases.png').is_file())
            with (target/'scenario_metrics.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),len(bundle['results'])*4)
            with (target/'excitation_cases.csv').open(encoding='utf-8-sig') as f: excitation=list(csv.DictReader(f))
            positions={}
            for row in excitation:
                key=(row['method'],row['candidate_index'])
                physical=(float(row['x_mm']),float(row['y_mm']),float(row['shared_amplitude']))
                if key in positions: self.assertEqual(positions[key],physical)
                positions[key]=physical
                theta,phi=np.deg2rad([float(row['theta_deg']),float(row['phi_deg'])])
                center=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi)])
                expected=-360*float(row['frequency_ghz'])/LIGHT_MM_GHZ*np.dot(physical[:2],center)
                self.assertAlmostEqual(float(row['steering_phase_deg']),expected,places=10)

    def test_joint_holes_spacing_and_equality_constraints_together(self):
        self.assertEqual(spacing_stats(np.array([[0.,0.],[.5,0.]]),.5)['spacing_violations'],0)
        for shape in ('rectangle','circle'):
            c=Config(nx=6,ny=6,k=8,hole_shape=shape,hole_x=1,hole_y=1,min_spacing=.6,
                     scan_angles='0,0;30,0',frequencies_ghz='9,11',iterations=40,opt_grid=25,eval_grid=81)
            bundle=run(c)
            for result in bundle['results']:
                self.assertEqual(len(result['scenario_metrics']),4)
                self.assertTrue(np.all(available_mask(c)[result['indices']]))
                self.assertEqual(result['metrics']['forbidden_elements'],0)
                self.assertEqual(result['metrics']['spacing_violations'],0)
                self.assertGreaterEqual(result['metrics']['min_distance_lambda_ref'],.6-1e-10)


if __name__=='__main__': unittest.main()
