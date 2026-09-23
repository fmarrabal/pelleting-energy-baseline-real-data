"""Package GPU evidence and trained small-model checkpoints after verification."""
from common import *
import zipfile

def main():
    audit=json.loads((ROOT/'GPU_AUDIT.json').read_text(encoding='utf8'))
    numerical=json.loads((ROOT/'NUMERICAL_CHECKS.json').read_text(encoding='utf8'))
    plan=json.loads((ROOT/'GPU_RUN_PLAN.json').read_text(encoding='utf8'))
    assert audit['status']=='GPU_RETRAINING_COMPLETE' and numerical['status']=='PASS'
    assert all(sha(BASE/p)==h for p,h in plan['cpu_artifact_sha256'].items())
    assert all(sha(BASE/p)==h for p,h in plan['original_protected_sha256'].items())
    assert all(sha(ROOT/p)==h for p,h in plan['code_before_execution'].items())
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.suffix not in ['.py','.md','.json','.csv','.pt','.joblib','.svg','.png']:continue
        if p.name in ['PACKAGE_MANIFEST.json','DELIVERY.json']:continue
        files.append(p)
    checkpoint_files=[p for p in files if p.suffix=='.pt']
    assert len(checkpoint_files)==24
    manifest={'status':'VERIFIED_GPU_REPLICATION','cpu_and_originals_unchanged':True,'cuda_checkpoints':24,
      'files':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)],
      'external_requirements':['Original project data and archived CPU evidence for audit and reproduction','Pinned foundation checkpoints in existing local Hugging Face cache','Recorded Python/CUDA environment and JAX dependencies'],
      'scientific_scope':'Same development observations and fixed protocol; exploratory replication, no new final test.'}
    save_json(ROOT/'PACKAGE_MANIFEST.json',manifest)
    archive=ROOT/'Reentrenamiento_GPU_Datos_Reales_Resultados.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[ROOT/'PACKAGE_MANIFEST.json']:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    save_json(ROOT/'DELIVERY.json',{'status':'COMPLETE','report':str(ROOT/'INFORME_GPU.md'),'archive':str(archive),
      'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'archive_files':len(files)+1,
      'device':'NVIDIA RTX PRO 5000 Blackwell / CUDA','neural_checkpoint_count':24,
      'numerical_checks':len(numerical['checks']),'roundtrip_max_difference_kWh':numerical['largest_roundtrip_difference_kWh']})
    print('PACKAGED',archive,'bytes',archive.stat().st_size,flush=True)

if __name__=='__main__':main()
