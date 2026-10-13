"""Post-run diagnostic only; does not alter pretrained code or predictions."""
from common import *
import torch
from chronos.chronos_bolt import InstanceNorm
from chronos.chronos2.preprocess import from_list_of_dicts
torch.set_num_threads(4)

def main():
    cfg=json.loads((BASE/'modelos_avanzados_v1/hf_cache/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c/config.json').read_text())
    use_arcsinh=cfg['chronos_config'].get('use_arcsinh',False)
    d=data();context=[];future=[];meta=[]
    for start,end in FOLDS:
        w=fold_windows(d,start,end)['val']
        for first in range(0,len(w['ids']),16):
            inputs=[];ids=w['ids'][first:first+16]
            for target,donors in zip(ids,w['donors'][first:first+16]):
                rows=d.loc[np.r_[donors,target]]
                vals={k:rows[k].to_numpy('float32') for k in NUM}
                vals.update({k:rows[k].astype(str).to_numpy() for k in CAT})
                inputs.append({'target':rows[TARGET].to_numpy('float32')[:-1],
                  'past_covariates':{k:v[:-1] for k,v in vals.items()},'future_covariates':{k:v[-1:] for k,v in vals.items()}})
            prepared=from_list_of_dicts(inputs,prediction_length=1)
            for target,p in zip(ids,prepared):
                context.append(p['context'].numpy());future.append(p['future_covariates'].numpy())
                meta.extend({'fold':start,'fila_fuente':int(target),'covariate':k} for k in [TARGET]+sorted(NUM+CAT))
    h=np.concatenate(context);f=np.concatenate(future)
    norm=InstanceNorm(use_arcsinh=use_arcsinh)
    hc=torch.tensor(h);fc=torch.tensor(f)
    ac,statsc=norm(hc);bc,_=norm(fc,statsc)
    ag,statsg=norm(hc.to('cuda'));bg,_=norm(fc.to('cuda'),statsg)
    diff=abs(ac.numpy()-ag.cpu().numpy()).max(axis=1)
    fdiff=abs(bc.numpy()-bg.cpu().numpy()).reshape(-1)
    constant=np.ptp(h,axis=1)==0
    q=pd.DataFrame(meta);q['constant_history_float32']=constant
    q['normalized_context_max_abs_difference']=diff;q['normalized_future_abs_difference']=fdiff
    q['CPU_scale']=statsc[1].numpy().reshape(-1);q['GPU_scale']=statsg[1].cpu().numpy().reshape(-1)
    q.sort_values('normalized_future_abs_difference',ascending=False).to_csv(ROOT/'chronos_normalization_diagnostic.csv',index=False)
    result={'status':'POST_HOC_DIAGNOSTIC_NO_PRIMARY_CHANGES','use_arcsinh':use_arcsinh,'native_prepared_channels_checked':len(h),
      'categorical_encoding':'Official per-item target encoding, based only on completed context labels; same 16-query preprocessing batches as primary run.',
      'constant_channels':int(constant.sum()),'constant_channels_context_difference_over_0_1':int((constant&(diff>.1)).sum()),
      'nonconstant_channels_context_difference_over_0_1':int((~constant&(diff>.1)).sum()),
      'max_normalized_context_difference':float(diff.max()),'max_normalized_future_difference':float(np.nanmax(fdiff)),
      'source_rule':'Native InstanceNorm substitutes eps only when scale equals exactly zero. Near-zero residual variance can amplify different CPU/GPU reductions.',
      'interpretation':'This reproduces a normalization discrepancy on actual covariates. It is a mechanism consistent with the frozen-model prediction differences, not a completed end-to-end correction or a causal attribution of every residual.',
      'primary_predictions_modified':False}
    save_json(ROOT/'CHRONOS_NORMALIZATION_DIAGNOSTIC.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
