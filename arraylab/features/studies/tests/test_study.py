from dataclasses import replace
import json
import math

import numpy as np
import pytest

from arraylab.features.linear.model import ArrayParameters, calculate
from arraylab.features.linear.dataio import read_pattern
from arraylab.features.feed.model import FeedSettings, design_feed
from arraylab.features.planar.model import PlanarParameters, design_planar
from arraylab.features.studies.model import StudySettings, run_study, StudyCancelled, export_study_directory, save_study_configuration, load_study_configuration
from test_cli import run_cli
from test_feed import independent_circuit
from test_planar import quadrature_power


@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
def test_linear_scan_closed_form_and_same_seed_realization(topology):
    p = ArrayParameters(elements=8, step_deg=1)
    study = StudySettings(count=5, start_deg=-60, stop_deg=60)
    report = run_study(p, FeedSettings(topology=topology), study)
    for row in report["records"]:
        reference = calculate(replace(p, scan_deg=row["scan_theta_deg"]))
        assert row["metrics"]["target_direction_level_db"] == pytest.approx(0, abs=1e-12)
        assert row["metrics"]["target_direction_directivity_dbi"] == pytest.approx(reference.directivity_dbi, abs=1e-9)
        np.testing.assert_allclose(10**(np.array(row["pattern_db"])/20), 10**(reference.pattern_db/20), atol=1e-12)
    settings = FeedSettings(topology=topology, phase_bits=3, amplitude_error_db=.4, phase_error_deg=2, failed_ports=(2,), seed=91)
    report = run_study(p, settings, study)
    for row in report["records"]:
        single = design_feed(replace(p, scan_deg=row["scan_theta_deg"]), settings)
        assert row["ports"] == single["ports"]
        for port, first in zip(row["ports"], report["records"][0]["ports"]):
            assert port["amplitude_error_db"] == first["amplitude_error_db"]
            assert port["phase_error_deg"] == first["phase_error_deg"]
        np.testing.assert_allclose(row["pattern_db"], single["electrical"]["patterns"]["output"]["pattern_db"], atol=1e-8)
    assert report == run_study(p, settings, study)
    assert "statistics" not in report


def test_abcd_phases_and_independent_directional_power_integral():
    p = ArrayParameters(elements=6, spacing=.4, step_deg=2)
    s = FeedSettings(topology="symmetric_series", input_ohm=75, load_ohm=100, phase_bits=4,
                     weighting="custom", custom_amplitudes=(1,2,3,3,2,1), custom_phases_deg=(15,-20,0,10,30,5))
    report = run_study(p,s,StudySettings(count=3,start_deg=-25,stop_deg=35))
    _,_,voltages = independent_circuit(report["network"])
    indices = (np.arange(6)-2.5)*.4
    u,q = np.polynomial.legendre.leggauss(80)
    for row in report["records"]:
        weights = np.array([voltages[f"P{port['port']}"]/10*np.exp(-1j*np.deg2rad(port["applied_delay_deg"])) for port in row["ports"]])
        reported = np.array([complex(port["output_real"],port["output_imag"]) for port in row["ports"]])
        np.testing.assert_allclose(weights,reported,atol=1e-12)
        field = sum(w*np.exp(2j*np.pi*x*u) for w,x in zip(weights,indices))
        integral = np.sum(q*abs(field)**2)/2
        command = math.sin(math.radians(row["scan_theta_deg"]))
        target_power = abs(sum(w*np.exp(2j*np.pi*x*command) for w,x in zip(weights,indices)))**2
        assert row["metrics"]["target_direction_directivity_linear"] == pytest.approx(target_power/integral,rel=1e-11)
        # This metric is evaluated at the command, even when custom offsets shift the beam.
        assert row["metrics"]["target_direction_power_ratio"] == pytest.approx(target_power/np.sum(abs(weights))**2,abs=1e-12)


@pytest.mark.parametrize("axis", ["theta","phi"])
@pytest.mark.parametrize("geometry", ["rectangular","arbitrary"])
def test_planar_scan_matches_single_design_and_fixed_cut(axis,geometry):
    kwargs = {"nx":2,"ny":2} if geometry=="rectangular" else {"coordinates":((-.3,-.2),(.4,-.5),(.2,.4),(-.5,.3))}
    p = PlanarParameters(geometry=geometry,cut_phi_deg=19,scan_theta_deg=30,scan_phi_deg=25,step_deg=3,**kwargs)
    s = FeedSettings(topology="series",phase_bits=3,phase_error_deg=1,seed=17)
    settings = StudySettings(count=3,scan_axis=axis,start_deg=5 if axis=="theta" else -30,stop_deg=45)
    report = run_study(p,s,settings)
    assert report["cut_phi_deg"]==19 and "array_parameters" not in report["network"]
    for row in report["records"]:
        single = design_planar(replace(p,scan_theta_deg=row["scan_theta_deg"],scan_phi_deg=row["scan_phi_deg"]),s)
        np.testing.assert_allclose(row["pattern_db"],single["electrical"]["patterns"]["output"]["cut_db"],atol=1e-8)
        for port,other in zip(row["ports"],single["network"]["ports"]):
            assert port["desired_delay_deg"]==pytest.approx(other["desired_delay_deg"],abs=1e-10)


def test_single_element_all_failed_and_tiny_opposed_aperture():
    one=run_study(ArrayParameters(elements=1),FeedSettings(),StudySettings(count=2))
    assert all(row["metrics"]["target_direction_directivity_dbi"]==pytest.approx(0) for row in one["records"])
    failed=run_study(ArrayParameters(elements=2),FeedSettings(failed_ports=(1,2)),StudySettings(count=2))
    for row in failed["records"]:
        assert row["metrics"]["target_direction_directivity_linear"] is None
        assert row["metrics"]["target_direction_directivity_dbi"] is None
        assert row["pattern_db"]==[-300]*len(failed["angles_deg"])
        assert all(port["output_phase_deg"] is None for port in row["ports"])
    tiny=run_study(ArrayParameters(elements=2,spacing=1e-6),FeedSettings(weighting="custom",custom_amplitudes=(1,1),custom_phases_deg=(0,180)),StudySettings(count=2,start_deg=0,stop_deg=30))
    assert all(math.isfinite(row["metrics"]["target_direction_directivity_linear"]) for row in tiny["records"])
    json.dumps(failed,allow_nan=False)


def test_planar_target_direction_against_independent_sphere_quadrature():
    p=PlanarParameters(geometry="arbitrary",coordinates=((-1,-.2),(.3,-.7),(.6,.4),(-.5,.8)),scan_theta_deg=37,step_deg=2)
    s=FeedSettings(phase_bits=3,failed_ports=(2,),weighting="custom",custom_amplitudes=(1,2,3,2),custom_phases_deg=(10,-20,40,15))
    report=run_study(p,s,StudySettings(count=3,scan_axis="phi",start_deg=-70,stop_deg=55))
    points=p.positions()
    for row in report["records"]:
        weights=np.array([complex(port["output_real"],port["output_imag"]) for port in row["ports"]])
        theta,phi=np.deg2rad([row["scan_theta_deg"],row["scan_phi_deg"]])
        u,v=np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi)
        power=abs(sum(w*np.exp(2j*np.pi*(x*u+y*v)) for w,(x,y) in zip(weights,points)))**2
        expected=power/quadrature_power(points,weights)
        assert row["metrics"]["target_direction_directivity_linear"]==pytest.approx(expected,rel=1e-10)


@pytest.mark.parametrize("changes",[{"count":1},{"count":201},{"count":True},{"scan_axis":"phi"},{"scan_axis":"x"},{"start_deg":70,"stop_deg":60},{"start_deg":-91},{"stop_deg":91},{"stop_deg":float("nan")}])
def test_invalid_scan_rejected(changes):
    with pytest.raises(ValueError):run_study(ArrayParameters(),FeedSettings(),StudySettings(**changes))


def test_reject_wrong_phase_and_excessive_sampling():
    with pytest.raises(ValueError):run_study(ArrayParameters(),FeedSettings(phase_mode="equalize"),StudySettings())
    with pytest.raises(ValueError):run_study(ArrayParameters(step_deg=.01),FeedSettings(),StudySettings(count=200))
    with pytest.raises(ValueError):run_study(PlanarParameters(),FeedSettings(),StudySettings(start_deg=-1))


def test_supported_scale_and_mm_geometry():
    p=PlanarParameters(nx=20,ny=10,step_deg=2)
    a=run_study(p,FeedSettings(),StudySettings(count=2))
    mm=replace(p,coordinate_unit="mm",spacing_x=p.spacing_x*p.wavelength_mm,spacing_y=p.spacing_y*p.wavelength_mm)
    b=run_study(mm,FeedSettings(),StudySettings(count=2))
    for left,right in zip(a["records"],b["records"]):
        assert len(left["ports"])==200
        assert left["metrics"]["target_direction_level_db"]==pytest.approx(0,abs=1e-10)
        np.testing.assert_allclose(10**(np.array(left["pattern_db"])/20),10**(np.array(right["pattern_db"])/20),atol=1e-12)
    full=run_study(ArrayParameters(elements=1,step_deg=5),FeedSettings(),StudySettings(count=200,start_deg=-90,stop_deg=90))
    assert len(full["records"])==200
    assert full["records"][0]["scan_theta_deg"]==-90 and full["records"][-1]["scan_theta_deg"]==90


def test_cancel_before_start_between_points_and_before_publication():
    calls=[]
    with pytest.raises(StudyCancelled):run_study(ArrayParameters(),FeedSettings(),StudySettings(),cancelled=lambda:True,progress=lambda a,b:calls.append(a))
    assert not calls
    with pytest.raises(StudyCancelled):run_study(ArrayParameters(),FeedSettings(),StudySettings(count=3),cancelled=lambda:len(calls)==1,progress=lambda a,b:calls.append(a))
    assert calls==[1]
    calls=[]
    with pytest.raises(StudyCancelled):run_study(ArrayParameters(),FeedSettings(),StudySettings(count=2),cancelled=lambda:len(calls)==2,progress=lambda a,b:calls.append(a))
    assert calls==[1,2]


@pytest.mark.parametrize("planar",[False,True])
def test_configuration_headless_export_replay_and_failure_protection(planar,tmp_path):
    p = PlanarParameters(nx=2,ny=2,step_deg=2) if planar else ArrayParameters(elements=4,step_deg=2)
    s = FeedSettings(phase_bits=4,seed=18)
    study = StudySettings(count=3,start_deg=5,stop_deg=40)
    config=tmp_path/'中文 扫描.json';save_study_configuration(config,p,s,study)
    assert load_study_configuration(config)==(p,s,study)
    out=tmp_path/'导出';result=run_cli('--study-config',config,'--output-dir',out,'--plot','all',guard=True)
    assert result.returncode==0,result.stderr
    doc=json.loads(result.stdout)['study'];assert doc==run_study(p,s,study)
    assert json.loads((out/'study-report.json').read_text(encoding='utf-8'))==doc
    for row in doc['records']:
        angles,db,phase=read_pattern(out/f"study-{row['index']:03d}.dat")
        np.testing.assert_allclose(angles,doc['angles_deg'],rtol=0,atol=1e-9)
        np.testing.assert_allclose(db,row['pattern_db'],rtol=0,atol=1e-8);assert phase is None
    assert (out/'study.png').read_bytes().startswith(b'\x89PNG')
    again=run_cli('--study-config',out/'study-config.json',guard=True)
    assert json.loads(again.stdout)['study']==doc
    keep=(out/'study-report.json').read_bytes()
    assert run_cli('--study-config',config,'--output-dir',out,guard=True).returncode==3
    assert (out/'study-report.json').read_bytes()==keep
    assert run_cli('--study-config',config,'--elements',8,guard=True).returncode==2
    assert run_cli('--study','monte_carlo','--study-config',config,guard=True).returncode==2


@pytest.mark.parametrize("change",['version','unknown','missing','bool','kind','report','duplicate'])
def test_strict_config_rejects_invalid_document(change,tmp_path):
    path=tmp_path/'scan.json';save_study_configuration(path,ArrayParameters(),FeedSettings(),StudySettings())
    doc=json.loads(path.read_text(encoding='utf-8'))
    if change=='version':doc['schema_version']=999
    elif change=='unknown':doc['study_settings']['unknown']=0
    elif change=='missing':del doc['feed_settings']['seed']
    elif change=='bool':doc['study_settings']['count']=True
    elif change=='kind':doc['array_kind']='spherical'
    elif change=='report':doc=run_study(ArrayParameters(),FeedSettings(),StudySettings(count=2))
    raw=json.dumps(doc)
    if change=='duplicate':
        version=doc['schema_version'];raw=raw.replace(f'"schema_version": {version}',f'"schema_version": {version}, "schema_version": {version}')
    path.write_text(raw,encoding='utf-8')
    with pytest.raises(ValueError):load_study_configuration(path)


def test_failed_export_is_atomic(tmp_path,monkeypatch):
    report=run_study(ArrayParameters(),FeedSettings(),StudySettings(count=2))
    def fail(*args):raise OSError('simulated disk error')
    monkeypatch.setattr('arraylab.study.write_pattern',fail)
    with pytest.raises(OSError):export_study_directory(tmp_path/'result',report)
    assert not (tmp_path/'result').exists() and not list(tmp_path.glob('.arraylab-study-*'))
