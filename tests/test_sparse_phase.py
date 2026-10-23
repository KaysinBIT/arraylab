"""Phase codes, independent complex replay, normalization, snapshots and real Qt."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy
import csv
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from arraylab.sparse2d.core import Config,field,run,response_field,scan_uv
from arraylab.sparse2d.demo import export_bundle
from arraylab.sparse2d.elements import write_example_csv
from arraylab.sparse2d.headless import dumps,jsonable,reevaluate,reweight,save_bundle,load_bundle
from arraylab.sparse2d.phase_quantization import quantize_degrees,excitation_phases,quantized_af,angular_distance
from arraylab.sparse2d.weighting import Weighting,add_weighting

ROOT=Path(__file__).resolve().parent


class PhaseQuantizationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=Config(nx=4,ny=4,k=8,theta=30,phi=35,iterations=10,opt_grid=25,eval_grid=81)
        cls.ideal=run(cls.c); cls.quant=run(replace(cls.c,phase_bits=3))


    def test_codes_wrap_and_half_step_rule(self):
        command,code,error=quantize_degrees([0,45,-45,135,-135,180,-180,315,359,-359],2)
        np.testing.assert_array_equal(command,[0,90,270,180,180,180,180,270,0,0])
        np.testing.assert_array_equal(code,[0,1,3,2,2,2,2,3,0,0])
        np.testing.assert_allclose(error,[0,45,-45,45,-45,0,0,-45,1,-1],atol=1e-12)
        np.testing.assert_array_equal(quantize_degrees([90,-90],1)[0],[180,180])

    def test_all_bit_widths_bound_periodicity_and_conjugacy(self):
        angles=np.r_[np.linspace(-720,720,10001),[-90,-45,45,90]]
        for bits in range(1,13):
            command,code,error=quantize_degrees(angles,bits); step=360/2**bits
            self.assertLessEqual(float(abs(error).max()),step/2+1e-9)
            self.assertTrue(np.all((code>=0)&(code<2**bits)))
            np.testing.assert_array_equal(command,code*step)
            np.testing.assert_allclose(np.exp(1j*np.deg2rad(quantize_degrees(-angles,bits)[0])),
                np.conj(np.exp(1j*np.deg2rad(command))),atol=2e-12)
            np.testing.assert_allclose(quantize_degrees(angles+360*7,bits)[0],command,atol=1e-10)

    def test_disabled_quantizer_and_field_are_exactly_ideal(self):
        angles=np.array([-720.,-.25,0.,180.,725.])
        command,code,error=quantize_degrees(angles,0)
        np.testing.assert_array_equal(command,angles); np.testing.assert_array_equal(error,np.zeros(5))
        np.testing.assert_array_equal(code,np.full(5,-1))
        case=self.ideal['scenarios'][0]; row=self.ideal['results'][0]; uv=np.array([[0,0],[.2,.3]])
        np.testing.assert_array_equal(quantized_af(row['xy'],row['weights'],uv,case,0),
            field(row['xy'],row['weights'],uv,scan_uv(self.c)))

    def test_illegal_bits_and_nonfinite_angles_rejected(self):
        for value in (True,False,-1,13,2.5,'3'):
            with self.subTest(value=value),self.assertRaises(ValueError): replace(self.c,phase_bits=value).validate()
        for values in ([np.nan],[np.inf]):
            with self.assertRaises(ValueError): quantize_degrees(values,3)

    def test_replay_matches_known_hardware_commands_not_ideal(self):
        xy=np.array([[0.,0.],[.5,0.],[0.,.5]]); weights=np.array([.2,.3,.5])
        case={'theta':30.,'phi':45.,'frequency_ratio':1.,'frequency_ghz':10.}
        uv=np.array([[0.,0.],[.2,-.1],[.6,.4]])
        commands=np.deg2rad([0.,270.,270.])
        expected=np.array([sum(w*np.exp(1j*(2*np.pi*np.dot(p,d)+phase)) for p,w,phase in zip(xy,weights,commands)) for d in uv])
        np.testing.assert_allclose(quantized_af(xy,weights,uv,case,2),expected,atol=2e-15)
        self.assertGreater(np.max(abs(expected-field(xy,weights,uv,scan_uv(Config(theta=30,phi=45))))),.1)

    def test_ideal_synthesis_is_preserved_and_gain_is_not_normalized_away(self):
        for ideal,actual in zip(self.ideal['results'],self.quant['results'],strict=True):
            np.testing.assert_array_equal(actual['indices'],ideal['indices'])
            np.testing.assert_array_equal(actual['weights'],ideal['weights'])
            self.assertEqual(actual['metrics'],ideal['metrics'])
            q=actual['phase_quantization']['scenarios'][0]
            case=self.quant['scenarios'][0]; phases=excitation_phases(actual['xy'],case,3)
            gain=abs(sum(w*np.exp(1j*np.deg2rad(e)) for w,e in zip(actual['weights'],phases['error_deg'])))
            self.assertAlmostEqual(q['steering_field_linear'],gain,places=13)
            self.assertAlmostEqual(q['validation_psl_db'],q['validation_psl_ideal_reference_db']-20*np.log10(gain),places=11)
            self.assertLessEqual(q['steering_loss_db'],-20*np.log10(np.cos(np.pi/8))+1e-10)
            field_at_scan=abs(response_field(actual['xy'],actual['weights'],np.array([scan_uv(self.c)]),replace(self.c,phase_bits=3),case,phase_mode='quantized')[0])
            self.assertAlmostEqual(field_at_scan,gain,places=12)
            self.assertLess(gain,1.)

    def test_broadside_is_zero_error_and_high_bits_converge(self):
        row=self.ideal['results'][0]; case={**self.ideal['scenarios'][0],'theta':0.,'phi':0.}
        np.testing.assert_array_equal(excitation_phases(row['xy'],case,1)['error_deg'],np.zeros(len(row['xy'])))
        uv=np.array([[0.,0.],[.2,.3],[-.7,.3]])
        case=self.ideal['scenarios'][0]
        ideal=quantized_af(row['xy'],row['weights'],uv,case,0)
        actual=quantized_af(row['xy'],row['weights'],uv,case,12)
        self.assertLessEqual(np.max(abs(actual-ideal)),np.deg2rad(180/2**12)+1e-12)

    def test_one_bit_scan_null_is_explicit_finite_json(self):
        c=replace(self.c,spacing=1.,phi=0.,phase_bits=1)
        bundle=run(c)
        for row in bundle['results']:
            q=row['phase_quantization']['scenarios'][0]
            self.assertFalse(q['steering_response_defined']); self.assertIsNone(q['validation_psl_db'])
            self.assertEqual(row['phase_quantization']['metrics']['invalid_steering_cases'],1)
            broadside=response_field(row['xy'],row['weights'],np.array([[0.,0.]]),c,bundle['scenarios'][0],phase_mode='quantized')
            self.assertAlmostEqual(abs(broadside[0]),1.,places=12)
        json.loads(dumps(bundle))

    def test_joint_cases_and_postweighting_replay_same_support(self):
        c=replace(self.c,nx=6,ny=6,scan_angles='0,0;30,35',frequencies_ghz='9,11',min_spacing=.6,hole_shape='circle',hole_x=.6,phase_bits=3)
        bundle=run(c); index=next(i for i,row in enumerate(bundle['results']) if row['name']=='Simulated annealing')
        previous=copy.deepcopy(bundle['results'][index]['phase_quantization'])
        weighted=add_weighting(bundle,index,Weighting(method='gaussian'))
        new=weighted['results'][-1]
        np.testing.assert_array_equal(new['indices'],bundle['results'][index]['indices'])
        self.assertEqual(bundle['results'][index]['phase_quantization'],previous)
        self.assertEqual(len(new['phase_quantization']['scenarios']),4)
        commands=[excitation_phases(new['xy'],case,3)['phase_code'] for case in bundle['scenarios']]
        self.assertTrue(np.any(commands[2]!=commands[3]))
        for case,q in zip(bundle['scenarios'],new['phase_quantization']['scenarios']):
            err=excitation_phases(new['xy'],case,3)['error_deg']
            expected=abs(np.sum(new['weights']*np.exp(1j*np.deg2rad(err))))
            self.assertAlmostEqual(q['steering_field_linear'],expected,places=13)

    def test_export_commands_reconstruct_field_and_metrics(self):
        row=self.quant['results'][0]; bundle={**self.quant,'results':[row]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target=export_bundle(bundle,Path(directory)/'report')
            for name in ('phase_quantization.csv','phase_quantization_1.png','layout_1_quantized.png'):
                self.assertTrue((target/name).is_file(),name)
            with (target/'excitation_cases.csv').open(encoding='utf-8-sig',newline='') as stream: records=list(csv.DictReader(stream))
            wave=299.792458/self.c.reference_ghz; uv=np.array([.2,-.15])
            direct=sum(float(r['shared_amplitude'])*np.exp(1j*(2*np.pi*(float(r['x_mm'])/wave*uv[0]+float(r['y_mm'])/wave*uv[1])+np.deg2rad(float(r['steering_phase_deg'])))) for r in records)
            expected=response_field(row['xy'],row['weights'],np.array([uv]),replace(self.c,phase_bits=3),bundle['scenarios'][0],phase_mode='quantized')[0]
            self.assertAlmostEqual(abs(direct-expected),0.,places=12)
            for record in records:
                self.assertAlmostEqual(float(record['steering_phase_deg']),int(record['phase_code'])*45,places=12)
                self.assertLessEqual(abs(float(record['phase_error_deg'])),22.5+1e-10)
            saved=json.loads((target/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['results'][0]['phase_quantization'],row['phase_quantization'])
            self.assertIn('移相器量化复核',(target/'report.html').read_text(encoding='utf-8'))

    def test_saved_replay_and_removed_element_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory); source=path/'element.csv'; write_example_csv(source)
            bundle=run(replace(self.c,element_model='imported',element_file=str(source)))
            save_bundle(bundle,path/'ideal'); source.unlink()
            restored=load_bundle(path/'ideal'); quant=reevaluate(restored,phase_bits=4)
            self.assertEqual(quant['config']['phase_bits'],4)
            np.testing.assert_array_equal(quant['results'][0]['indices'],restored['results'][0]['indices'])
            weighted=reweight(quant,'annealing',{'method':'uniform'})
            self.assertTrue(weighted['results'][-1]['phase_quantization']['enabled'])
            self.assertEqual(weighted['element_pattern'],bundle['element_pattern'])
            self.assertEqual(restored['config']['phase_bits'],0)


    def test_angular_error_is_geodesic_not_theta_phi_subtraction(self):
        self.assertAlmostEqual(angular_distance([0,0],[.5,0]),30.,places=12)
        self.assertAlmostEqual(angular_distance([.5,0],[0,.5]),np.rad2deg(np.arccos(.75)),places=12)




if __name__=='__main__': unittest.main()
