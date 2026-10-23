from dataclasses import asdict,replace
import json
import math

import numpy as np
import pytest

from arraylab.core import ArrayParameters,calculate
from arraylab.feed import FeedSettings
from arraylab.planar import (PlanarParameters,design_planar,planar_amplitudes,read_coordinates_csv,
                            save_planar_configuration,load_planar_configuration,export_planar_directory)
from arraylab.planar_pattern import field_uv,integrated_power
from test_feed import independent_circuit
from test_cli import run_cli


@pytest.mark.parametrize("scan",[-30,0,30,70,90])
def test_planar_line_limit(scan):
    p=PlanarParameters(nx=8,ny=1,scan_theta_deg=abs(scan),scan_phi_deg=180 if scan<0 else 0,step_deg=.5)
    report=design_planar(p,FeedSettings(phase_mode="scan"))
    output=report["electrical"]["patterns"]["output"]
    baseline=calculate(ArrayParameters(elements=8,scan_deg=scan,step_deg=.5))
    assert output["metrics"]["directivity_dbi"]==pytest.approx(baseline.directivity_dbi,abs=1e-9)
    np.testing.assert_allclose(10**(np.array(output["cut_db"])/20),10**(baseline.pattern_db/20),atol=1e-12)


def test_two_by_two_broadside_closed_form():
    report=design_planar(PlanarParameters(nx=2,ny=2,scan_theta_deg=0),FeedSettings())
    expected=10*math.log10(4/(1+np.sinc(math.sqrt(2))))
    assert report["electrical"]["patterns"]["output"]["metrics"]["directivity_dbi"]==pytest.approx(expected,abs=1e-10)
    assert report["electrical"]["patterns"]["output"]["metrics"]["representative_peak_phi_deg"] is None


def quadrature_power(points,weights):
    # Independent spherical quadrature: integrate cos(theta) and azimuth.
    z,q=np.polynomial.legendre.leggauss(64)
    phi=np.arange(160)*2*np.pi/160
    total=0
    for cosine,weight in zip(z,q):
        radius=math.sqrt(1-cosine*cosine)
        field=np.zeros(len(phi),dtype=complex)
        for point,w in zip(points,weights):
            field+=w*np.exp(2j*np.pi*radius*(point[0]*np.cos(phi)+point[1]*np.sin(phi)))
        total+=weight*np.mean(abs(field)**2)/2
    return total


def test_arbitrary_integral_peak_search_and_front_back_symmetry():
    p=PlanarParameters(geometry="arbitrary",coordinates=((-1,-.3),(.2,-.4),(.5,.6),(-.5,.7),(0,0)),scan_theta_deg=33,scan_phi_deg=47,cut_phi_deg=47)
    s=FeedSettings(weighting="custom",custom_amplitudes=(1,2,3,2,1),custom_phases_deg=(20,-35,10,50,-15),phase_error_deg=3,seed=18)
    report=design_planar(p,s)
    ports=report["network"]["ports"]
    weights=np.array([complex(port["output_real"],port["output_imag"]) for port in ports])
    integral=quadrature_power(p.positions(),weights)
    assert integrated_power(p.positions(),weights)==pytest.approx(integral,rel=1e-11)
    output=report["electrical"]["patterns"]["output"]
    metric=output["metrics"]
    assert metric["peak_search_method"]=="disk_grid_multistart_estimate"
    theta,phi=np.deg2rad([metric["representative_peak_theta_deg"],metric["representative_peak_phi_deg"]])
    direction=np.sin(theta)*np.array([np.cos(phi),np.sin(phi)])
    peak=abs(sum(w*np.exp(2j*np.pi*np.dot(point,direction)) for w,point in zip(weights,p.positions())))**2
    assert metric["directivity_dbi"]==pytest.approx(10*math.log10(peak/integral),abs=1e-8)
    # Dense independently generated angular grid should not exceed refined maximum.
    tt,pp=np.meshgrid(np.linspace(0,np.pi/2,101),np.linspace(-np.pi,np.pi,241),indexing="ij")
    field=np.zeros_like(tt,dtype=complex)
    for w,point in zip(weights,p.positions()):field+=w*np.exp(2j*np.pi*np.sin(tt)*(point[0]*np.cos(pp)+point[1]*np.sin(pp)))
    assert peak>=float(np.max(abs(field)**2))*(1-1e-7)
    np.testing.assert_allclose(output["sphere_db"],np.array(output["sphere_db"])[::-1],atol=1e-9)


def test_translation_rotation_and_mm_units():
    points=np.array([[-.8,-.2],[.3,-.7],[.8,.5],[-.4,.6]])
    p=PlanarParameters(geometry="arbitrary",coordinates=tuple(map(tuple,points)),scan_theta_deg=33,scan_phi_deg=21,cut_phi_deg=21)
    s=FeedSettings(topology="symmetric_series")
    reference=design_planar(p,s)["electrical"]["patterns"]["output"]
    angle=np.deg2rad(25);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    moved=points@rotation.T+np.array([3,-2])
    rotated=design_planar(replace(p,coordinates=tuple(map(tuple,moved)),scan_phi_deg=46,cut_phi_deg=46),s)["electrical"]["patterns"]["output"]
    assert rotated["metrics"]["directivity_dbi"]==pytest.approx(reference["metrics"]["directivity_dbi"],abs=1e-10)
    np.testing.assert_allclose(rotated["cut_db"],reference["cut_db"],atol=1e-8)
    mm=design_planar(replace(p,coordinate_unit="mm",coordinates=tuple(map(tuple,points*p.wavelength_mm))),s)
    np.testing.assert_allclose(mm["electrical"]["patterns"]["output"]["cut_db"],reference["cut_db"],atol=1e-8)


def test_rectangular_separable_taylor_and_feed_powers():
    p=PlanarParameters(nx=4,ny=5)
    s=FeedSettings(weighting="taylor",sidelobe_db=30,taylor_nbar=3,topology="symmetric_series")
    a=planar_amplitudes(p,s).reshape(5,4)
    # Separable: every row is a scalar multiple of the central row.
    np.testing.assert_allclose(a/a[:,1,None],np.tile(a[2]/a[2,1],(5,1)),atol=1e-13)
    report=design_planar(p,s);zin,powers,_=independent_circuit(report["network"])
    expected=(a*a/np.sum(a*a)).ravel()
    assert zin==pytest.approx(50)
    for i,power in enumerate(expected,1):assert powers[f"P{i}"]==pytest.approx(power,abs=1e-12)


@pytest.mark.parametrize("changes",[
    {"nx":201},{"nx":True},{"nx":20,"ny":20},{"spacing_x":0},{"spacing_y":100},
    {"scan_theta_deg":-1},{"scan_theta_deg":91},{"scan_phi_deg":181},{"cut_phi_deg":float("nan")},
    {"geometry":"arbitrary","coordinates":[]},{"geometry":"arbitrary","coordinates":[[0,0],[0,0]]},
    {"geometry":"arbitrary","coordinates":[[0,0,0]]},{"geometry":"arbitrary","coordinates":[[True,0]]},
    {"coordinates":[[0,0]]},{"geometry":"unknown"},
])
def test_invalid_geometry(changes):
    with pytest.raises(ValueError):PlanarParameters(**changes).validate()


def test_planar_single_and_all_failed():
    single=design_planar(PlanarParameters(nx=1,ny=1),FeedSettings())
    metric=single["electrical"]["patterns"]["output"]["metrics"]
    assert metric["directivity_dbi"]==pytest.approx(0)
    assert metric["representative_peak_theta_deg"] is None
    failed=design_planar(PlanarParameters(nx=2,ny=2),FeedSettings(failed_ports=(1,2,3,4)))
    out=failed["electrical"]["patterns"]["output"]
    assert out["metrics"]["directivity_dbi"] is None
    assert np.all(np.array(out["sphere_db"])==-300)
    json.dumps(failed,allow_nan=False)


def test_small_aperture_opposed_phase_peak_search_scale():
    angle=math.radians(47)
    p=PlanarParameters(geometry="arbitrary",coordinates=((0,0),(2e-6*math.cos(angle),2e-6*math.sin(angle))))
    s=FeedSettings(weighting="custom",custom_amplitudes=(1,1),custom_phases_deg=(0,180),phase_mode="equalize")
    report=design_planar(p,s)
    assert report["electrical"]["patterns"]["output"]["metrics"]["directivity_dbi"]==pytest.approx(10*math.log10(3),abs=1e-5)


def test_200_port_geometry_and_coherent_peak():
    report=design_planar(PlanarParameters(nx=20,ny=10,scan_theta_deg=17,scan_phi_deg=-42),FeedSettings())
    metric=report["electrical"]["patterns"]["output"]["metrics"]
    assert metric["peak_search_method"]=="coherent_bound"
    assert metric["representative_peak_theta_deg"]==pytest.approx(17)
    assert metric["representative_peak_phi_deg"]==pytest.approx(-42)
    assert len(report["network"]["ports"])==200


def test_spherical_grid_and_signed_cut_agree():
    report=design_planar(PlanarParameters(nx=3,ny=4,scan_theta_deg=27,scan_phi_deg=33,cut_phi_deg=0,step_deg=1),FeedSettings())
    for pattern in report["electrical"]["patterns"].values():
        sphere=10**(np.array(pattern["sphere_db"])/20)
        cut=10**(np.array(pattern["cut_db"])/20)
        for index in range(31):
            assert cut[90+3*index]==pytest.approx(sphere[index,60],abs=1e-12)
            assert cut[90-3*index]==pytest.approx(sphere[index,120],abs=1e-12)


def test_planar_config_csv_and_cli_no_gui(tmp_path):
    p=PlanarParameters(nx=2,ny=3,scan_phi_deg=35,cut_phi_deg=35)
    s=FeedSettings(phase_bits=4,phase_error_deg=2,seed=29)
    config=tmp_path/"中文 平面阵.json"
    save_planar_configuration(config,p,s)
    assert load_planar_configuration(config)==(p,s)
    output=tmp_path/"输出"
    process=run_cli("--planar-config",config,"--output-dir",output,"--plot","all",guard=True)
    assert process.returncode==0,process.stderr
    doc=json.loads(process.stdout)["planar"]
    assert json.loads((output/"planar-report.json").read_text(encoding="utf-8"))==doc
    coords=read_coordinates_csv(output/"planar-coordinates.csv")
    np.testing.assert_allclose(coords,p.positions())
    for name in ("planar-cut.png","planar-map.png","planar-3d.png"):assert (output/name).read_bytes().startswith(b"\x89PNG")
    process=run_cli("--planar","--coordinates",output/"planar-coordinates.csv","--scan-phi",35,"--cut-phi",35,guard=True)
    assert process.returncode==0,process.stderr
    assert json.loads(process.stdout)["planar"]["geometry"]["geometry"]=="arbitrary"
    old=(output/"planar-report.json").read_bytes()
    again=run_cli("--planar-config",config,"--output-dir",output,guard=True)
    assert again.returncode==3 and (output/"planar-report.json").read_bytes()==old


@pytest.mark.parametrize("text",["element,x,y\n1,0,0\n1,1,1\n","element,x,y\n1,nan,0\n","x,y\n0,0\n","element,x,y\n1,0\n"])
def test_invalid_coordinate_file(text,tmp_path):
    path=tmp_path/"bad.csv";path.write_text(text)
    with pytest.raises(ValueError):read_coordinates_csv(path)


def test_planar_config_failure_preserves_file_and_output(tmp_path):
    path=tmp_path/"keep.json";path.write_text("keep")
    with pytest.raises(ValueError):save_planar_configuration(path,PlanarParameters(nx=200,ny=200),FeedSettings())
    assert path.read_text()=="keep"
    process=run_cli("--planar","--nx",200,"--ny",200,"--output-dir",tmp_path/"result",guard=True)
    assert process.returncode==2 and not (tmp_path/"result").exists()
    for args in [("--planar","--elements",8),("--planar","--config","examples/ula.json")]:
        assert run_cli(*args,guard=True).returncode==2
