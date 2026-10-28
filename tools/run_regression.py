"""Isolated per-file regression run: no shared Qt application or user history."""
from pathlib import Path
import argparse,os,sys,subprocess,time,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arraylab.framework.process import WindowsJob,resume_child
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_name',nargs='?',default=time.strftime('%Y%m%d-%H%M%S'))
parser.add_argument('test_files',nargs='*')
parser.add_argument('--timeout',type=float,default=180)
args=parser.parse_args()
if not 0<args.timeout<float('inf'):parser.error('--timeout must be positive and finite')
stage=args.run_name;out=(ROOT/'tmp/regression'/stage).resolve()
if not out.is_relative_to((ROOT/'tmp/regression').resolve()):raise ValueError('Run name must stay inside tmp/regression')
if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new run name to preserve previous evidence')
out.mkdir(parents=True,exist_ok=True)
rows=[]
files=sorted((ROOT/'tests').glob('test_*.py')) if not args.test_files else [(ROOT/x).resolve() for x in args.test_files]
if len({p.stem for p in files})!=len(files):raise ValueError('Test file names must be unique in one run')
if not all(p.is_file() and p.suffix=='.py' and p.is_relative_to(ROOT) for p in files):raise ValueError('Tests must be Python files inside the project')
for file in files:
    folder=out/file.stem;folder.mkdir(exist_ok=True);temp=folder/'temp';temp.mkdir(exist_ok=True)
    env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','QT_QPA_PLATFORM':'offscreen',
         'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLCONFIGDIR':str(folder/'mpl'),
         'ARRAYLAB_HISTORY_DIR':str(folder/'history'),'ARRAYLAB_SPARSE_HISTORY_DIR':str(folder/'sparse-history'),
         'TEMP':str(temp),'TMP':str(temp)}
    started=time.perf_counter();job=WindowsJob();timed_out=False
    with (folder/'pytest.log').open('wb') as log:
        p=subprocess.Popen([sys.executable,'-m','pytest',str(file),'-q','-p','no:cacheprovider',
             '--basetemp',str(folder/'pytest-temp'),'--junitxml',str(folder/'junit.xml')],cwd=ROOT,env=env,
             stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000004 if os.name=='nt' else 0)
        try:
            job.attach(p);resume_child(p)
            try:p.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:timed_out=True
        finally:
            job.close()
            if p.poll() is None:p.kill()
        p.wait(timeout=10)
    row={'file':file.name,'exit_code':p.returncode,'timeout':timed_out,'seconds':round(time.perf_counter()-started,3)}
    if (folder/'junit.xml').exists():
        suites=ET.parse(folder/'junit.xml').getroot().findall('testsuite')
        row.update({k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')})
        row['failed_cases']=[c.get('name') for s in suites for c in s.findall('testcase') if c.find('failure') is not None or c.find('error') is not None]
    rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)
print(json.dumps({'files':len(rows),'failed_files':[r['file'] for r in rows if r['exit_code']!=0],
                 'tests':sum(r.get('tests',0) for r in rows),'seconds':sum(r['seconds'] for r in rows)}),flush=True)

sys.exit(int(any(r['exit_code']!=0 for r in rows)))
