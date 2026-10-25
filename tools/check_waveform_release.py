"""Only radar waveform/antenna checks on the actual frozen desktop and CLI."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import uuid


def same(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(same(a,b) for a,b in zip(left,right))
    if type(left) in (int,float) and type(right) in (int,float):
        return math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-8)
    return left == right


def main():
    p=argparse.ArgumentParser();p.add_argument('release',type=Path);p.add_argument('report',type=Path)
    args=p.parse_args();release=args.release.resolve();args.report=args.report.resolve()
    work=args.report.parent/('波形 独立验收-'+uuid.uuid4().hex[:8]);work.mkdir(parents=True)
    env=os.environ.copy();windows=Path(env.get('SystemRoot','C:/Windows'))
    env.update(PATH=str(windows/'System32'),PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
               TEMP=str(work),TMP=str(work),MPLCONFIGDIR=str(work/'mpl'),
               ARRAYLAB_HISTORY_DIR=str(work/'linear-isolated'),ARRAYLAB_SPARSE_HISTORY_DIR=str(work/'sparse-isolated'))
    env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
    gui_env={**env,'QT_QPA_PLATFORM':'windows'};gui_env.pop('MPLBACKEND',None)
    gui=work/'gui';run=subprocess.run([str(release/'ArrayLab.exe'),'--smoke-test',str(gui),'--smoke-suite','waveform'],cwd=work,env=gui_env,timeout=180)
    desktop=json.loads((gui/'smoke.json').read_text(encoding='utf-8'))
    if run.returncode or not desktop['success']:raise RuntimeError(desktop.get('error','Waveform GUI checks failed'))
    assert desktop['suite']=='waveform' and desktop['full_desktop_suite'] is False
    cli_env={**env,'QT_QPA_PLATFORM':'invalid-test-platform','DISPLAY':'','MPLBACKEND':'QtAgg'}
    checks=[]
    def command(*flags,code=0,gui_entry=False):
        exe=release/('ArrayLab.exe' if gui_entry else 'ArrayLab-cli.exe')
        process=subprocess.run([str(exe),*map(str,flags)],cwd=work,env=cli_env,capture_output=True,timeout=90)
        out=process.stdout.decode('utf-8');error=process.stderr.decode('utf-8')
        assert process.returncode==code,(process.returncode,out[-1000:],error[-1000:])
        return out,error
    for name in ('single','ab'):
        destination=work/('cli-'+name)
        out,_=command('--waveform-config',gui/f'waveform-{name}-config.json','--output-dir',destination,'--plot','all','--workers','1')
        actual=json.loads(out)['waveform'];reference=desktop['waveform_results'][name]
        for key in ('configuration','analysis','spectra','range_recovery','pri_recovery'):
            assert same(actual[key],reference[key]),(name,key)
        checks.append(name+'_GUI_and_CLI_match_all_waveform_and_recovery_fields')
        for group in actual['spectra']:
            for view in ('overview','phase','spectra','rd','range_recovery','pri_recovery'):
                assert (destination/f'waveform-{group}-{view}.png').is_file()
            assert (destination/f'{group}-range.csv').is_file() and (destination/f'{group}-doppler.csv').is_file()
        checks.append(name+'_all_profiles_PNG_and_CSV_without_Qt_display')
        assert json.loads((destination/'waveform-record.json').read_text(encoding='utf-8'))['report']==actual
        checks.append(name+'_exported_record_preserves_result')
    actual=json.loads(command('waveform','--config',gui/'waveform-import-v2.json','--workers','1')[0])['waveform']
    assert actual['configuration']['abOptions']['enabled']
    checks.append('source_WaveLab_v2_import_works_in_frozen_CLI')
    out=work/'windowed-headless'
    command('--headless','--waveform-config',gui/'waveform-single-config.json','--output-dir',out,gui_entry=True)
    assert same(json.loads((out/'summary.json').read_text(encoding='utf-8'))['analysis'],desktop['waveform_results']['single']['analysis'])
    checks.append('windowed_EXE_waveform_headless_dispatch')
    command('--waveform-config',gui/'waveform-single-config.json','--output-dir',out,code=3)
    checks.append('existing_waveform_outputs_are_not_overwritten')
    bad=work/'bad-waveform.json';bad.write_text('{"format":"arraylab.waveform","schema_version":99,"configuration":{}}',encoding='utf-8')
    _,error=command('--waveform-config',bad,code=2);assert json.loads(error)['error']['type']=='input'
    checks.append('invalid_waveform_configuration_returns_input_error')
    # New 1.4 algorithms must be present in the actual Qt-free executable.
    radar_document=json.loads((release/'examples'/'radar-dml-coherent.json').read_text(encoding='utf-8'))
    radar_document['antennaDesign']['sparseOptions'].update(iterations=100,trials=2,snapshots=32,scanMaxDeg=10)
    radar_input=work/'radar-1.4.json';radar_input.write_text(json.dumps(radar_document),encoding='utf-8')
    for action in ('sparse','evaluate'):
        destination=work/('radar-'+action)
        out,_=command('waveform','--config',radar_input,'--radar-action',action,'--workers','2','--output-dir',destination,'--plot','sparse_design')
        actual_radar=json.loads(out)['waveform'];sparse=actual_radar['sparse_design']
        assert actual_radar['source']['version']=='1.4' and sparse['valid'] and sparse['action']==action
        assert sparse['superresolution']['method']=='dml' and sparse['superresolution']['valid']
        assert json.loads((destination/'waveform-record.json').read_text(encoding='utf-8'))['report']==actual_radar
        assert (destination/'sparse-design.json').is_file() and (destination/'waveform-A-sparse_design.png').is_file()
        checks.append('radar_'+action+'_DML_and_complete_snapshot_without_Qt')
    result={'success':True,'scope':'Only waveform/antenna/gain and directly affected entrypoints; no full suite',
            'full_desktop_suite':False,'gui_checks':len(desktop['checks']),'gui_seconds':desktop['elapsed_seconds'],
            'headless_checks':checks,'app_version':actual['app_version'],'work_directory':str(work),
            'host':platform.platform(),'environment':'Existing Windows host; system-only PATH and isolated history/cache; not a clean OS image'}
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'success':True,'gui_checks':result['gui_checks'],'headless_checks':len(checks),'work_directory':str(work)},ensure_ascii=False))


if __name__=='__main__':main()
