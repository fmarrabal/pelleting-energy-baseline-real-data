"""Round-trip all newly trained GPU checkpoints; check frozen-model batching."""
from common import *
import gc,torch,joblib
from neuralforecast.models import TFT,NHITS
from neuralforecast.losses.pytorch import MSE
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
    assert torch.cuda.is_available()
    assert json.loads((ROOT/'supervised/completion.json').read_text())['status']=='COMPLETE'
    assert json.loads((ROOT/'foundation/completion.json').read_text())['status']=='COMPLETE'
    checks=[];d=data();cached={}
    for start,end in FOLDS:
        w=fold_windows(d,start,end)['val'];cached[start]=w
        e=joblib.load(ROOT/'supervised'/f'encoder_{start}.joblib')
        ids=w['ids'][:8];don=w['donors'][:8];rows=np.column_stack([don,ids])
        x=e['encoder'].transform(d.loc[rows.ravel()]).reshape(8,L+1,-1).astype('float32')
        h=((d.loc[don.ravel(),TARGET].to_numpy().reshape(8,L)-e['target_mean'])/e['target_std']).astype('float32')
        inputs={'insample_y':torch.tensor(h[:,:,None],device='cuda'),
          'insample_mask':torch.ones((8,L,1),device='cuda'),'futr_exog':torch.tensor(x,device='cuda'),'hist_exog':None,'stat_exog':None}
        for family in ['TFT','NHITS']:
            for seed in SEEDS:
                stem=f'{family}_{start}_seed{seed}'
                state=torch.load(ROOT/'supervised'/f'{stem}.pt',map_location='cpu',weights_only=True)
                cfg=dict(h=1,input_size=L,futr_exog_list=[f'x{i}' for i in range(x.shape[-1])],loss=MSE(),scaler_type='identity',random_seed=seed)
                if family=='TFT':m=TFT(**cfg,hidden_size=32,n_head=4,dropout=.1)
                else:m=NHITS(**cfg,mlp_units=[[64,64]]*3,n_blocks=[1]*3,n_pool_kernel_size=[2,2,1],n_freq_downsample=[1]*3)
                m.load_state_dict(state['state_dict'],strict=True);m=m.to('cuda').eval()
                assert next(m.parameters()).is_cuda
                with torch.no_grad():pred=m(inputs).reshape(-1).cpu().numpy()*e['target_std']+e['target_mean']
                old=pd.read_csv(ROOT/'supervised'/f'{stem}.csv').set_index('fila_fuente').loc[ids,'raw_prediction_kWh'].to_numpy()
                delta=float(abs(pred-old).max());assert delta<.005,(stem,delta)
                checks.append({'model':family,'fold':start,'seed':seed,'check':'CUDA_checkpoint_roundtrip_batch8_vs64','n':8,'max_abs_difference_kWh':delta})
                del m;gc.collect();torch.cuda.empty_cache()
    from chronos import Chronos2Pipeline
    spec=json.loads((ROOT/'PROTOCOL.json').read_text(encoding='utf8'))['foundations']['Chronos_2']
    m=Chronos2Pipeline.from_pretrained(spec['repo'],revision=spec['revision'],local_files_only=True,device_map='cuda',dtype=torch.float32)
    assert next(m.model.parameters()).is_cuda
    for start,end in FOLDS:
        w=cached[start];inputs=[]
        for i in range(8):
            rows=d.loc[np.r_[w['donors'][i],w['ids'][i]]]
            vals={k:rows[k].to_numpy('float32') for k in NUM};vals.update({k:rows[k].astype(str).to_numpy() for k in CAT})
            inputs.append({'target':d.loc[w['donors'][i],TARGET].to_numpy('float32'),
              'past_covariates':{k:v[:-1] for k,v in vals.items()},'future_covariates':{k:v[-1:] for k,v in vals.items()}})
        q,_=m.predict_quantiles(inputs,prediction_length=1,quantile_levels=[.5],batch_size=32,context_length=L,cross_learning=False)
        pred=np.array([float(v[0,0,0].cpu()) for v in q])
        old=pd.read_csv(ROOT/'foundation'/f'Chronos_2_covariates_{start}.csv').set_index('fila_fuente').loc[w['ids'][:8],'raw_prediction_kWh'].to_numpy()
        delta=float(abs(pred-old).max());assert delta<.01,(start,delta)
        checks.append({'model':'Chronos_2_covariates','fold':start,'check':'CUDA_native_batch32_vs64','n':8,'max_abs_difference_kWh':delta})
    save_json(ROOT/'NUMERICAL_CHECKS.json',{'status':'PASS','device':'cuda','checks':checks,
      'largest_roundtrip_difference_kWh':max(x['max_abs_difference_kWh'] for x in checks)})
    print('NUMERICAL CHECKS PASS',len(checks),'largest difference',max(x['max_abs_difference_kWh'] for x in checks),flush=True)

if __name__=='__main__':main()
