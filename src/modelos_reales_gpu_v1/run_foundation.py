from common import *
DEVICE,GPU_STATUS=device_policy()
import time,gc,torch
torch.set_num_threads(4)
assert DEVICE=='cuda' and torch.cuda.is_available()
RUN_START=time.perf_counter()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
    import timesfm
    from chronos import Chronos2Pipeline
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text(encoding='utf8'))
    assert sha(ROOT/'PROTOCOL.json')==json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())['sha256']
    out=ROOT/'foundation';out.mkdir(exist_ok=True)
    save_json(out/'hardware.json',{'device':DEVICE,'gpu_at_start':GPU_STATUS,'torch':torch.__version__,'threads':4})
    d=data();timings=[]
    all_windows={start:fold_windows(d,start,end)['val'] for start,end in FOLDS}
    for family in ['TimesFM_2.5','Chronos_2']:
        spec=protocol['foundations'][family];tick=time.perf_counter()
        if family=='TimesFM_2.5':
            model=timesfm.TimesFM_2p5_200M_torch.from_pretrained(spec['repo'],revision=spec['revision'],local_files_only=True,torch_compile=False)
            model.compile(timesfm.ForecastConfig(max_context=32,max_horizon=32,normalize_inputs=True,per_core_batch_size=1,
              use_continuous_quantile_head=False,force_flip_invariance=True,infer_is_positive=False,return_backcast=True))
        else:
            model=Chronos2Pipeline.from_pretrained(spec['repo'],revision=spec['revision'],local_files_only=True,device_map=DEVICE,dtype=torch.float32)
        assert str(next(model.model.parameters()).device).startswith(DEVICE)
        print(f'{family} loaded on {DEVICE} in {time.perf_counter()-tick:.1f}s',flush=True)
        for start,end in FOLDS:
            w=all_windows[start];hist=d.loc[w['donors'].ravel(),TARGET].to_numpy('float32').reshape(-1,L)
            for variant in ['univariate','covariates']:
                name=family+'_'+variant;dest=out/f'{name}_{start}.csv'
                if dest.exists():continue
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();pred=[]
                if family=='TimesFM_2.5':
                    # Each XReg fit is restricted to ONE query context. Batching contexts in
                    # native XReg pools their labels and would leak future observations.
                    for i,target in enumerate(w['ids']):
                        if variant=='univariate':
                            p,_=model.forecast(horizon=1,inputs=[hist[i]])
                            p=float(p[0,-1])
                        else:
                            rows=d.loc[np.r_[w['donors'][i],target]]
                            nc={k:[rows[k].to_numpy(float)] for k in NUM}
                            cc={k:[rows[k].astype(str).to_numpy()] for k in CAT}
                            pp,_=model.forecast_with_covariates(inputs=[hist[i]],dynamic_numerical_covariates=nc,
                              dynamic_categorical_covariates=cc,xreg_mode='xreg + timesfm',ridge=10,
                              normalize_xreg_target_per_input=True,force_on_cpu=True)
                            p=float(pp[0][0])
                        pred.append(p)
                        if i%40==0:print(name,start,i+1,'/',len(hist),flush=True)
                else:
                    for first in range(0,len(hist),16):
                        if variant=='univariate':inputs=hist[first:first+16,None,:]
                        else:
                            inputs=[]
                            for i in range(first,min(first+16,len(hist))):
                                rows=d.loc[np.r_[w['donors'][i],w['ids'][i]]]
                                vals={k:rows[k].to_numpy('float32') for k in NUM}
                                vals.update({k:rows[k].astype(str).to_numpy() for k in CAT})
                                inputs.append({'target':hist[i],'past_covariates':{k:v[:-1] for k,v in vals.items()},
                                 'future_covariates':{k:v[-1:] for k,v in vals.items()}})
                        q,_=model.predict_quantiles(inputs,prediction_length=1,quantile_levels=[.5],batch_size=64,context_length=L,cross_learning=False)
                        pred.extend(float(v[0,0,0].cpu()) for v in q)
                        if first%64==0:print(name,start,first+len(q),'/',len(hist),flush=True)
                prediction_frame(d,w['ids'],pred,name,start).to_csv(dest,index=False)
                timings.append({'model':name,'fold':start,'device':DEVICE,'seconds':time.perf_counter()-tick,'n':len(hist),'revision':spec['revision'],'peak_cuda_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'peak_cuda_reserved_MiB':torch.cuda.max_memory_reserved()/2**20})
                save_json(out/'runtime.json',timings)
                print(name,start,'COMPLETE',flush=True)
        del model;gc.collect()
        if DEVICE=='cuda':torch.cuda.empty_cache()
    save_json(out/'completion.json',{'status':'COMPLETE','device':DEVICE,'device_name':torch.cuda.get_device_name(0),'new_fold_variants':len(timings),'weights_finetuned':False,'xreg_solver_device':'cpu','wall_seconds_including_preparation':time.perf_counter()-RUN_START})
    print('FOUNDATION GPU COMPLETE',flush=True)

if __name__=='__main__':main()
