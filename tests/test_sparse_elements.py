"""Element physics, CSV semantics, immutable snapshots, and combined synthesis."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from arraylab.sparse2d.core import (Config,admissible_pairs,field,joint_matrix,group_matrix,run,
                  response_field,response_cut_width,sample_domain,scan_uv,sidelobe_mask)
from arraylab.sparse2d.scenarios import case_config,scenarios
from arraylab.sparse2d.elements import ElementPattern,build_element,make_snapshot,write_example_csv
from arraylab.sparse2d.weighting import Weighting,add_weighting


def table_rows(frequencies=(9.,10.,11.),with_phi=True,column='response_db'):
    rows=[]
    for frequency in frequencies:
        factor={None:1.,9.:.8,10.:1.,11.:.6}[frequency]
        for theta in (0.,30.,60.,90.):
            for phi in ((0.,90.,180.,270.,360.) if with_phi else (0.,)):
                t,p=np.deg2rad([theta,phi])
                amplitude=factor*(.2+.8*np.cos(t))*(1-.2*np.sin(t)**2*np.cos(p)**2)
                value=20*np.log10(amplitude) if column=='response_db' else amplitude**2 if column=='power_linear' else amplitude
                row={'theta_deg':theta,column:value}
                if frequency is not None: row['frequency_ghz']=frequency
                if with_phi: row['phi_deg']=phi
                rows.append(row)
    return rows


def save_table(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def uv(theta,phi=0):
    t,p=np.deg2rad([theta,phi]); return np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p)])


class ElementChecks(unittest.TestCase):
    def test_default_point_source_is_exactly_unity(self):
        c=Config(); self.assertEqual(c.element_model,'isotropic')
        element=build_element(c); directions=np.array([[0.,0.],[.3,.4],[1.,0.]])
        for frequency in (9.,10.,11.): np.testing.assert_array_equal(element.response(directions,frequency),np.ones(3))
        xy=np.array([[.1,.2],[-.1,-.2]]); weights=np.ones(2)/2; case=scenarios(c)[0]
        np.testing.assert_array_equal(response_field(xy,weights,directions,c,case,element),field(xy,weights,directions))

    def test_half_wave_dipole_axes_and_analytic_value(self):
        x=build_element(Config(element_model='dipole',element_dipole_axis='x'))
        values=x.response(np.array([[0.,0.],[.5,0.],[1.,0.]]),10.)
        np.testing.assert_allclose(values,[1.,np.sqrt(2/3),0.],atol=2e-14)
        y=build_element(Config(element_model='dipole',element_dipole_axis='y'))
        np.testing.assert_allclose(y.response(np.array([[0.,0.],[0.,.5],[0.,1.]]),10.),values,atol=2e-14)
        z=build_element(Config(element_model='dipole',element_dipole_axis='z'))
        self.assertEqual(z.response(np.array([[0.,0.]]),10.)[0],0)
        self.assertNotAlmostEqual(x.response(np.array([[.5,0.]]),9.)[0],x.response(np.array([[.5,0.]]),11.)[0],places=6)

    def test_patch_hpbw_and_frequency_independent_approximation(self):
        element=build_element(Config(element_model='patch',element_patch_hpbw=90))
        directions=np.array([uv(0),uv(45),uv(60),uv(90)])
        np.testing.assert_allclose(element.response(directions,10),[1,np.sqrt(.5),.5,0],atol=2e-14)
        np.testing.assert_array_equal(element.response(directions,9),element.response(directions,11))
        c=Config(nx=4,ny=4,k=4,element_model='patch')
        result={'xy':np.array([[0.,0.]]),'weights':np.ones(1)}
        width=response_cut_width(result,c,scenarios(c)[0],element,0)
        self.assertAlmostEqual(width,np.sqrt(2),places=5)

    def test_imported_grid_units_and_exact_samples(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            for column in ('response_db','amplitude_linear','power_linear'):
                path=Path(directory)/(column+'.csv'); save_table(path,table_rows(column=column))
                element=build_element(Config(element_model='imported',element_file=str(path)))
                np.testing.assert_allclose(element.response(np.array([uv(0),uv(60,0),uv(60,90)]),10),[1,.51,.6],atol=2e-14)
                np.testing.assert_allclose(element.response(np.array([uv(0)]),9),[.8],atol=2e-14)
                self.assertFalse(element.snapshot['frequency_independent'])

    def test_periodic_angular_and_frequency_interpolation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'grid.csv'; save_table(path,table_rows())
            element=build_element(Config(element_model='imported',element_file=str(path)))
            probe=np.array([uv(45,45),uv(30,-45),uv(30,315)])
            corners=np.array([(.2+.8*np.cos(np.deg2rad(t)))*(1-.2*np.sin(np.deg2rad(t))**2*np.cos(np.deg2rad(p))**2) for t in (30,60) for p in (0,90)])
            self.assertAlmostEqual(element.response(probe,10)[0],corners.mean(),places=12)
            self.assertAlmostEqual(element.response(probe,10)[1],element.response(probe,10)[2],places=12)
            self.assertAlmostEqual(element.response(np.array([uv(0)]),9.5)[0],.9,places=12)
            self.assertAlmostEqual(element.response(np.array([uv(0)]),10.5)[0],.8,places=12)
            for frequency in (8.,12.):
                with self.assertRaisesRegex(ValueError,'外推'): element.response(probe,frequency)

    def test_axisymmetric_import_requires_explicit_assumption(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'cut.csv'; save_table(path,table_rows(frequencies=(None,),with_phi=False))
            with self.assertRaisesRegex(ValueError,'轴对称'):
                build_element(Config(element_model='imported',element_file=str(path)))
            element=build_element(Config(element_model='imported',element_file=str(path),element_import_mode='axisymmetric'))
            values=element.response(np.array([uv(30,0),uv(30,80),uv(30,230)]),123.)
            np.testing.assert_allclose(values,values[0],atol=2e-14)
            self.assertTrue(element.snapshot['frequency_independent'])

    def test_import_rejects_holes_seams_poles_and_invalid_values(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base=table_rows(frequencies=(10.,))
            missing=[row for row in base if not (row['theta_deg']==30 and row['phi_deg']==90)]
            seam=[dict(row) for row in base]; seam[4]['response_db']=-20
            pole=[dict(row) for row in base]; pole[1]['response_db']=-20
            sparse_phi=[row for row in base if row['phi_deg'] in (0,90)]
            no_pole=[row for row in base if row['theta_deg']!=0]
            invalid=[dict(row) for row in base]; invalid[5]['response_db']='nan'
            for i,rows in enumerate((missing,seam,pole,sparse_phi,no_pole,invalid)):
                path=Path(directory)/f'bad_{i}.csv'; save_table(path,rows)
                with self.subTest(i=i),self.assertRaises(ValueError):
                    build_element(Config(element_model='imported',element_file=str(path)))

    def test_template_roundtrip_zero_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'示例.csv'; write_example_csv(path)
            element=build_element(Config(element_model='imported',element_file=str(path)))
            self.assertEqual(element.response(np.array([uv(90)]),10)[0],0)
            with self.assertRaises(FileExistsError): write_example_csv(path)
            rebuilt=ElementPattern(json.loads(json.dumps(element.snapshot,allow_nan=False)))
            np.testing.assert_array_equal(rebuilt.response(np.array([uv(30,35)]),10),element.response(np.array([uv(30,35)]),10))

    def test_element_factor_is_inside_joint_optimization_matrix(self):
        c=Config(nx=4,ny=4,k=4,opt_grid=25,eval_grid=81,element_model='patch',scan_angles='0,0;30,35',frequencies_ghz='9,11')
        expected=[]
        for case in scenarios(c):
            cc=case_config(c,case); points=sample_domain(cc,c.opt_grid); points=points[sidelobe_mask(points,cc)]
            factor=np.sqrt(np.maximum(0,1-np.sum(points**2,axis=1)))/np.cos(np.deg2rad(case['theta']))
            expected.append(group_matrix(c,points,scan_uv(cc),case['frequency_ratio'])*factor[:,None])
        np.testing.assert_allclose(joint_matrix(c),np.vstack(expected),atol=3e-14)

    def test_total_field_views_and_steering_null_rejection(self):
        c=Config(element_model='patch',theta=30,phi=35)
        case=scenarios(c)[0]; element=build_element(c); xy=np.array([[.3,.2],[-.3,-.2]])
        weights=np.ones(2)/2; points=np.array([uv(0),uv(45,20),scan_uv(c)])
        af=field(xy,weights,points,scan_uv(c)); expected=af*np.sqrt(np.maximum(0,1-np.sum(points**2,axis=1)))/np.cos(np.deg2rad(30))
        np.testing.assert_allclose(response_field(xy,weights,points,c,case,element),expected,atol=2e-14)
        np.testing.assert_array_equal(response_field(xy,weights,points,c,case,element,'array_factor'),af)
        self.assertAlmostEqual(abs(response_field(xy,weights,points,c,case,element)[-1]),1,places=13)
        with self.assertRaisesRegex(ValueError,'零点'):
            run(Config(nx=4,ny=4,k=4,element_model='dipole',element_dipole_axis='z',iterations=20,opt_grid=25,eval_grid=81))

    def test_imported_snapshot_survives_source_change_and_removal(self):
        from arraylab.sparse2d.demo import export_bundle
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'源方向图.csv'; save_table(path,table_rows())
            c=Config(nx=4,ny=4,k=4,iterations=20,opt_grid=25,eval_grid=81,element_model='imported',element_file=str(path),scan_angles='0,0;30,0',frequencies_ghz='9,11')
            bundle=run(c); original_hash=bundle['element_pattern']['source']['sha256']
            save_table(path,table_rows(column='power_linear'))
            self.assertNotEqual(make_snapshot(c)['source']['sha256'],original_hash)
            path.unlink()
            source=next(i for i,r in enumerate(bundle['results']) if r['name']=='Simulated annealing')
            weighted=add_weighting(bundle,source,Weighting(method='minimax',dr_db=10))
            self.assertEqual(weighted['element_pattern']['source']['sha256'],original_hash)
            np.testing.assert_array_equal(weighted['results'][-1]['indices'],bundle['results'][source]['indices'])
            out=Path(directory)/'snapshot_export'; export_bundle(weighted,out)
            snapshot=json.loads((out/'element_pattern.json').read_text(encoding='utf-8'))
            self.assertEqual(snapshot['source']['sha256'],original_hash)
            self.assertTrue((out/'element_samples.csv').exists())
            self.assertTrue((out/'element_preview.png').exists())

    def test_element_models_combine_with_joint_hole_and_spacing(self):
        for model in ('dipole','patch'):
            c=Config(nx=6,ny=6,k=8,hole_shape='rectangle',hole_x=1,hole_y=1,min_spacing=.6,
                     scan_angles='0,0;30,35',frequencies_ghz='9,11',element_model=model,
                     iterations=30,opt_grid=25,eval_grid=81)
            bundle=run(c)
            self.assertEqual(bundle['element_pattern']['model'],model)
            for r in bundle['results']:
                self.assertEqual(r['metrics']['spacing_violations'],0)
                self.assertEqual(r['metrics']['forbidden_elements'],0)
                self.assertEqual(len(r['scenario_metrics']),4)
                for row in r['scenario_metrics']:
                    self.assertTrue(np.isfinite(row['element_at_steering_db']))
                    self.assertGreaterEqual(row['total_peak_db'],-1e-8)


if __name__=='__main__': unittest.main()
