"""Prepare an isolated GPU replication of the frozen real-record benchmark."""
from pathlib import Path
import hashlib,json,shutil,datetime
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
CPU=BASE/'modelos_reales_temporales_v1'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    if (ROOT/'GPU_RUN_PLAN.json').exists():raise RuntimeError('GPU replication is already initialized.')
    protocol=json.loads((CPU/'PROTOCOL.json').read_text(encoding='utf8'))
    assert sha(CPU/'PROTOCOL.json')==json.loads((CPU/'PROTOCOL_LOCK.json').read_text())['sha256']
    assert all(sha(BASE/p)==h for p,h in protocol['protected_sha256'].items())
    snapshot={}
    for folder in [CPU,CPU/'supervised',CPU/'foundation',CPU/'figures',CPU/'report']:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ['.py','.md','.json','.csv','.pt','.joblib','.png','.svg','.pdf','.zip','.tex']:
                snapshot[str(p.relative_to(BASE))]=sha(p)
    pdf=BASE/'output/pdf/Anexo_Modelos_Temporales_Datos_Reales.pdf'
    snapshot[str(pdf.relative_to(BASE))]=sha(pdf)
    for name in ['PROTOCOL.json','PROTOCOL_LOCK.json','donor_audit.csv','sample_membership.csv','warmup_exclusions.csv']:
        shutil.copyfile(CPU/name,ROOT/name)
    common=(CPU/'common.py').read_text(encoding='utf8')
    common=common.replace("sys.path.insert(0, str(ROOT / 'dependencies'))","sys.path.insert(0, str(BASE / 'modelos_reales_temporales_v1/dependencies'))")
    old="if device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']='-1'"
    assert old in common
    common=common.replace(old,"if device!='cuda': raise RuntimeError('GPU replication requires CUDA and at least 4096 MiB free; no CPU fallback.')")
    (ROOT/'common.py').write_text(common,encoding='utf8')
    supervised=(CPU/'run_supervised.py').read_text(encoding='utf8')
    supervised=supervised.replace("torch.set_num_threads(4)","torch.set_num_threads(4)\nassert DEVICE=='cuda' and torch.cuda.is_available()\nRUN_START=time.perf_counter()")
    supervised=supervised.replace('return m.to(DEVICE)',"m=m.to(DEVICE)\n    assert next(m.parameters()).is_cuda\n    return m")
    supervised=supervised.replace('tick=time.perf_counter()','torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()')
    supervised=supervised.replace("'parameters':sum(p.numel() for p in m.parameters()),'refit_curve':refitcurve", "'parameters':sum(p.numel() for p in m.parameters()),'refit_curve':refitcurve,\n                 'peak_cuda_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'peak_cuda_reserved_MiB':torch.cuda.max_memory_reserved()/2**20")
    supervised=supervised.replace("print('SUPERVISED COMPLETE',flush=True)","save_json(out/'completion.json',{'status':'COMPLETE','device':DEVICE,'device_name':torch.cuda.get_device_name(0),'new_seed_refits':len(runtime),'wall_seconds_including_preparation':time.perf_counter()-RUN_START})\n    print('SUPERVISED GPU COMPLETE',flush=True)")
    (ROOT/'run_supervised.py').write_text(supervised,encoding='utf8')
    foundation=(CPU/'run_foundation.py').read_text(encoding='utf8')
    foundation=foundation.replace('torch.set_num_threads(4)',"torch.set_num_threads(4)\nassert DEVICE=='cuda' and torch.cuda.is_available()\nRUN_START=time.perf_counter()")
    foundation=foundation.replace("tick=time.perf_counter();pred=[]","torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();pred=[]")
    foundation=foundation.replace("'revision':spec['revision']}","'revision':spec['revision'],'peak_cuda_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'peak_cuda_reserved_MiB':torch.cuda.max_memory_reserved()/2**20}")
    foundation=foundation.replace("print('FOUNDATION COMPLETE',flush=True)","save_json(out/'completion.json',{'status':'COMPLETE','device':DEVICE,'device_name':torch.cuda.get_device_name(0),'new_fold_variants':len(timings),'weights_finetuned':False,'xreg_solver_device':'cpu','wall_seconds_including_preparation':time.perf_counter()-RUN_START})\n    print('FOUNDATION GPU COMPLETE',flush=True)")
    (ROOT/'run_foundation.py').write_text(foundation,encoding='utf8')
    analysis=(CPU/'analyze.py').read_text(encoding='utf8')
    # Keep metric/figure computation identical. CPU-specific narrative is replaced
    # by compare_cpu_gpu.py, using the actual new measurements.
    cut=analysis.index("    effects_tft=contrasts.loc['TFT'")
    analysis=analysis[:cut]+"    print(scores.to_string())\n\nif __name__=='__main__':main()\n"
    (ROOT/'analyze.py').write_text(analysis,encoding='utf8')
    plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'purpose':'GPU replication explicitly requested by user after freeing GPU memory',
      'parent_protocol_sha256':sha(CPU/'PROTOCOL.json'),'protocol_changes':[],
      'execution_override':'CUDA required for TFT/NHITS training and pretrained-model forward passes; no CPU fallback. Ridge and official JAX XReg remain CPU as before.',
      'frozen_design':'Same four outer folds, inner selection procedure, context=16, horizon=1, 584 validation records, seeds [11,29,47], architecture, optimizer, budgets and checkpoint revisions.',
      'no_new_hyperparameter_search':True,'supervised':'Retrain from scratch, including inner epoch selection and outer refit.',
      'foundation':'Inference on CUDA with frozen weights; no fine-tuning.',
      'scientific_scope':'Exploratory device replication, not a new independent validation. No selection between CPU/GPU realizations by outer error.',
      'precision':'float32, CUDA matmul TF32=False and cuDNN TF32=False, no mixed precision; seeded stochastic trajectories can differ by device.',
      'timing_scope':'Operational timings, not a controlled hardware speed benchmark.',
      'cpu_artifact_sha256':snapshot,'original_protected_sha256':protocol['protected_sha256'],
      'code_before_execution':{p.name:sha(p) for p in ROOT.glob('*.py')}}
    (ROOT/'GPU_RUN_PLAN.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf8')
    (ROOT/'GPU_RUN_LOCK.json').write_text(json.dumps({'sha256':sha(ROOT/'GPU_RUN_PLAN.json')},indent=2),encoding='utf8')
    print('GPU replication prepared; protected CPU files:',len(snapshot),flush=True)

if __name__=='__main__':main()
