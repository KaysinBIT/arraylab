from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from arraylab.features.pulse import service,io,presentation,metadata
from arraylab.features.pulse.config import C,project_hash
from arraylab.shared.simulation_io import read_npz

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())


def small_project():
    p=service.default_configuration()
    p['radar'].update(noise_enabled=True,receiver_bandwidth_hz=12e6)
    p['array'].update(rx_count=2,tx_count=1,beamwidth_deg=[180.,180.])
    p['timing'].update(pulses=16)
    p['processing'].update(doppler_bins=33,velocity_search_mps=[-32.,32.],doppler_window='rect')
    p['processing']['doa']['enabled']=False
    p['scene']['platform'].update(position_m=[0.,0.,0.])
    p['scene']['targets']=[{'id':'fixture','position_m':[C/24e6*240,0.,0.],
        'velocity_mps':[0.,0.,0.],'rcs_m2':1.,'fluctuation':'none','trajectory':[],'height_mode':'absolute'}]
    p['statistics'].update(trials=2,cut_range_m=3000.,cut_velocity_mps=0.)
    return p


@pytest.fixture(scope='module')
def simulated():
    return service.execute('simulate',small_project())


def test_budget_presets_and_source_contract():
    p=service.default_configuration();before=deepcopy(p)
    result,arrays=service.execute('budget',p)
    assert result['valid'] and not arrays and p==before
    assert result['source']['version']=='1.0.1'
    assert result['configuration']['scene']['platform']['height_mode']=='absolute'
    assert result['config_hash']==project_hash(result['configuration'])
    presets=service.presets();assert len(presets)==10
    assert all(set(row)=={'id','name','description','configuration'} for row in presets)
    assert all(row['configuration']['scene']['platform']['height_mode']=='absolute' for row in presets)
    from arraylab.features import pulse
    assert pulse.ACTIONS==metadata.ACTIONS and pulse.metadata is metadata
    assert dict(metadata.ACTIONS).keys()==set(service.ACTIONS)
    json.dumps(result,allow_nan=False)


def test_budget_and_saved_inputs_do_not_run_signal_engine(monkeypatch):
    from arraylab.features.pulse import engine
    monkeypatch.setattr(engine,'simulate',lambda *a,**kw:pytest.fail('Unexpected simulation'))
    result,_=service.execute('budget',small_project())
    assert result['valid']


def test_real_simulation_preserves_arrays_and_inputs(simulated):
    result,arrays=simulated
    assert result['kind']=='simulate' and result['valid']
    assert arrays['raw'].dtype==np.complex128 and arrays['rd_power'].ndim==2
    assert arrays['rd_power'].shape==(len(arrays['velocities_mps']),len(arrays['ranges_m']))
    assert result['array_units']['raw']=='sqrt(W)'
    assert result['array_manifest']['raw']['shape']==list(arrays['raw'].shape)
    assert result['acquisition']['times_s'][0]==0


def test_invalid_action_invalid_design_and_cancel():
    with pytest.raises(ValueError):service.execute('unexpected',{})
    with pytest.raises(InterruptedError):service.execute('simulate',{},cancel=lambda:True)
    project=small_project();project['resources']['memory_mib']=1
    result,_=service.execute('budget',project);assert not result['valid']
    with pytest.raises(ValueError):service.execute('simulate',project)
    stop=False
    def progress(f,message):
        nonlocal stop
        if f>=.1:stop=True
    with pytest.raises(InterruptedError):service.execute('simulate',small_project(),progress=progress,cancel=lambda:stop)


@pytest.mark.parametrize('suffix',['.json','.prlab'])
def test_project_roundtrip_is_source_compatible(tmp_path,suffix):
    project=small_project();path=tmp_path/('完整工程'+suffix)
    io.export_project(path,project)
    assert io.import_project(path)==project
    if suffix=='.prlab':
        with zipfile.ZipFile(path) as archive:
            assert set(archive.namelist())=={'project.json'}
            assert json.loads(archive.read('project.json'))['app']=='PulseLab'


def test_dem_resource_is_embedded_and_input_not_mutated(tmp_path):
    project=small_project();project['scene']['terrain'].update(kind='dem',enabled=True,
        x_m=[0.,1.,2.],y_m=[0.,1.,2.],heights_m=[[0.,0.,0.],[0.,None,0.],[0.,0.,0.]],
        valid=[[True,True,True],[True,False,True],[True,True,True]],metadata={'vertical_datum':'synthetic'})
    original=deepcopy(project);path=tmp_path/'地形.prlab';io.export_project(path,project)
    assert project==original and io.import_project(path)==project
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist())=={'project.json','terrain.json'}
        assert json.loads(archive.read('project.json'))['scene']['terrain']=={'resource':'terrain.json'}


@pytest.mark.parametrize('bad',[b'{"app":"PulseLab","app":"other"}',b'{"seed":NaN}',b'[]',b'{"app":"EquationLab"}',
                              b'{"format":"arraylab.simulation","feature":"equation","configuration":{}}'])
def test_bad_project_documents_rejected(bad):
    with pytest.raises(ValueError):io.import_project(bad)


@pytest.mark.parametrize('scene',[[],{'targets':None},{'targets':[None]}])
def test_malformed_scene_normalization_is_input_error(scene):
    with pytest.raises(ValueError):service.normalize_configuration({'scene':scene})


def test_archive_unknown_resource_is_not_followed():
    data=BytesIO()
    with zipfile.ZipFile(data,'w') as archive:
        archive.writestr('project.json',json.dumps({'scene':{'terrain':{'resource':'../outside.json'}}}))
        archive.writestr('../outside.json','{}')
    with pytest.raises(ValueError):io.import_project(data.getvalue())


def test_complete_exports_and_existing_directory_guard(simulated,tmp_path):
    result,arrays=simulated;path=io.export_result(tmp_path/'results',result,arrays)
    assert {p.name for p in path.iterdir()}=={'summary.json','project.json','project.prlab','arrays.npz','report.html','results.csv'}
    restored,loaded=read_npz(path/'arrays.npz')
    assert restored==result and io.import_project(path/'project.prlab')==result['configuration']
    for key,value in arrays.items():np.testing.assert_array_equal(loaded[key],value)
    assert 'ArrayLab' in (path/'report.html').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):io.export_result(path,result,arrays)


def test_signal_independent_statistics_and_inverse_are_explicit():
    project=small_project();result,arrays=service.execute('statistics',project)
    assert result['trials']==2 and arrays['trial_results'].shape==(2,4)
    assert result['pfa']['ci95'][1]>0
    p=small_project();p['requirements'].update(range_m=3000.,range_resolution_m=100.,velocity_resolution_m=100.,max_velocity_mps=5.,angle_resolution_deg=180.,rcs_m2=1.)
    p['design'].update(bandwidth_hz=[4e6],pulse_width_s=[2e-6],pri_s=[80e-6],pulses=[16],peak_power_w=[200.],rx_count=[2],max_candidates=1,verify_top=1,verification_trials=2)
    result,arrays=service.execute('inverse',p)
    assert result['kind']=='inverse' and result['screened_count']==1
    assert result['candidates'] and not result['candidates'][0]['passed']
    assert project_hash(service.array_configuration(result))==result['array_source']['config_hash']
    snapshot=deepcopy(result);candidate=service.apply_inverse_candidate(result,0)
    candidate['scene']['targets'][0]['rcs_m2']=999
    assert result==snapshot
    with pytest.raises(ValueError):service.apply_inverse_candidate(result,-1)


@pytest.mark.parametrize('view',[key for key,_ in metadata.VIEWS])
def test_plot_views_use_saved_result_without_recalculating(simulated,view,monkeypatch):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from arraylab.features.pulse import engine
    monkeypatch.setattr(engine,'simulate',lambda *a,**k:pytest.fail('View recalculated'))
    result,arrays=simulated;original=json.dumps(result,sort_keys=True)
    with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans']}):
        figure=Figure(figsize=(9,6));FigureCanvasAgg(figure)
        presentation.draw(figure,result,arrays,view)
        figure.canvas.draw()
        heads,rows,note=presentation.table(result,arrays,view)
    assert heads and isinstance(rows,list) and isinstance(note,str)
    assert json.dumps(result,sort_keys=True)==original


def test_scene_coordinates_and_metadata_templates():
    p=small_project();p['scene']['platform']['velocity_mps']=[10.,0.,0.]
    a=presentation.scene_preview(p,0);b=presentation.scene_preview(p,2)
    assert b['platform']['position_m'][0]-a['platform']['position_m'][0]==20
    assert 'scene.targets' in metadata.LIST_DEFAULTS and 'trajectory' in metadata.LIST_DEFAULTS


def test_public_numerical_api_is_qt_and_server_free():
    code="from arraylab.features.pulse import execute; import sys; r,a=execute('budget',{}); assert r['valid']; assert not any(k.startswith(('PySide6','fastapi','uvicorn')) for k in sys.modules)"
    process=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,timeout=30)
    assert process.returncode==0,process.stderr.decode('utf-8','replace')


def test_real_geotiff_import_and_portable_resource(tmp_path):
    pytest.importorskip('rasterio');pytest.importorskip('pyproj')
    from arraylab.features.pulse.scene import import_dem,terrain_grid
    terrain=import_dem(ROOT/'examples/pulse-demo-hill-utm.tif',
                       {'max_size':17,'height_unit':'m','vertical_datum':'synthetic'})
    assert terrain['metadata']['source_epsg']==32650
    assert terrain['metadata']['vertical_datum']=='synthetic'
    assert max(terrain['metadata']['analysis_shape'])<=17
    project=small_project();project['scene']['terrain']=terrain
    path=tmp_path/'DEM.prlab';io.export_project(path,project);restored=io.import_project(path)
    grid=terrain_grid(restored)
    assert grid['z_m'].shape==tuple(terrain['metadata']['analysis_shape'])
    assert np.any(grid['valid']) and np.isfinite(grid['z_m'][grid['valid']]).all()


def test_scene_table_tracks_display_time_without_signal_recalculation(monkeypatch):
    from arraylab.features.pulse import engine
    monkeypatch.setattr(engine,'simulate',lambda *a,**k:pytest.fail('Scene table recalculated signal'))
    project=small_project();project['scene']['platform']['velocity_mps']=[10.,0.,0.]
    result={'kind':'simulate','configuration':project};original=deepcopy(result)
    _,start,start_note=presentation.table(result,{},'scene',selection=0)
    _,later,note=presentation.table(result,{},'scene',selection=2.5)
    assert float(later[0][1])-float(start[0][1])==25.
    assert '输入场景' in note and 't=2.5 s' in note and '不重算信号' in note
    assert 't=0 s' in start_note and result==original


def test_inverse_scene_table_and_plot_share_array_candidate_and_time(monkeypatch):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from arraylab.features.pulse import engine
    monkeypatch.setattr(engine,'simulate',lambda *a,**k:pytest.fail('Scene view recalculated signal'))
    original_project=small_project();candidate=deepcopy(original_project)
    candidate['scene']['platform'].update(position_m=[100.,20.,50.],velocity_mps=[3.,2.,0.])
    candidate['scene']['targets'][0].update(position_m=[400.,10.,60.],velocity_mps=[4.,0.,-1.])
    candidate['statistics']['seed']=11
    source={'candidate_rank':2,'statistics_seed':11,'config_hash':project_hash(candidate)}
    stored=deepcopy(candidate);stored['statistics']['seed']=99
    result={'kind':'inverse','configuration':original_project,'array_source':source,
            'candidates':[{'rank':1,'project':deepcopy(original_project)},{'rank':2,'project':stored}]}
    before=deepcopy(result)
    _,rows,note=presentation.table(result,{},'scene',selection=3)
    assert [float(value) for value in rows[0][1:]]==[109.,26.,50.]
    assert [float(value) for value in rows[1][1:]]==[412.,10.,57.]
    assert '候选 #2' in note and 't=3 s' in note
    figure=Figure(figsize=(7,5));FigureCanvasAgg(figure)
    presentation.draw(figure,result,{},'scene',selection=3)
    ax=figure.axes[0]
    assert '候选 #2' in ax.get_title() and 't=3 s' in ax.get_title()
    np.testing.assert_allclose(np.asarray(ax.collections[0]._offsets3d).ravel(),[109.,26.,50.])
    assert result==before
