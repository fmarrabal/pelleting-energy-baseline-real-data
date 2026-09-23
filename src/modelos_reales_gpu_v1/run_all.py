"""Execute one GPU job at a time and retain each real child-process exit code."""
from pathlib import Path
import sys,subprocess,json,hashlib,time
ROOT=Path(__file__).resolve().parent

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

plan=json.loads((ROOT/'GPU_RUN_PLAN.json').read_text(encoding='utf8'))
assert sha(ROOT/'GPU_RUN_PLAN.json')==json.loads((ROOT/'GPU_RUN_LOCK.json').read_text())['sha256']
assert all(sha(ROOT/name)==h for name,h in plan['code_before_execution'].items())
results=[]
for script in ['run_supervised.py','run_foundation.py']:
    tick=time.perf_counter();print('START',script,flush=True)
    with (ROOT/(Path(script).stem+'.stdout.log')).open('w',encoding='utf8') as out, (ROOT/(Path(script).stem+'.stderr.log')).open('w',encoding='utf8') as err:
        completed=subprocess.run([sys.executable,'-X','utf8',str(ROOT/script)],cwd=ROOT.parent,stdout=out,stderr=err)
    results.append({'script':script,'exit_code':completed.returncode,'seconds':time.perf_counter()-tick})
    (ROOT/'PROCESS_COMPLETION.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print('FINISH',results[-1],flush=True)
    if completed.returncode:raise SystemExit(completed.returncode)
print('ALL GPU JOBS COMPLETE',flush=True)
