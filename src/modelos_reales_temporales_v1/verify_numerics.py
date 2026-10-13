"""Checks of serialization and native foundation batching, using saved results."""
from common import *
# Verification runs on the recorded device, not whichever device is now free.
hw=json.loads((ROOT/'foundation/hardware.json').read_text())
if hw['device']=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch,joblib
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main():
    import run_supervised as runner
    from chronos import Chronos2Pipeline
    from neuralforecast.models import TFT,NHITS
    from neuralforecast.losses.pytorch import MSE
    d=data();start,end=FOLDS[0];w=fold_windows(d,start,end)['val']
    checks=[]
    for family in ['TFT','NHITS']:
        seed=11;stem=f'{family}_{start}_seed{seed}'
        state=torch.load(ROOT/'supervised'/f'{stem}.pt',map_location='cpu',weights_only=True)
        e=joblib.load(ROOT/'supervised'/f'encoder_{start}.joblib')
        ids=w['ids'][:8];don=w['donors'][:8];rows=np.column_stack([don,ids])
        x=e['encoder'].transform(d.loc[rows.ravel()]).reshape(8,L+1,-1).astype('float32')
        hist=(d.loc[don.ravel(),TARGET].to_numpy().reshape(8,L)-e['target_mean'])/e['target_std']
        cfg=dict(h=1,input_size=L,futr_exog_list=[f'x{i}' for i in range(x.shape[-1])],loss=MSE(),scaler_type='identity',random_seed=seed)
        if family=='TFT':m=TFT(**cfg,hidden_size=32,n_head=4,dropout=.1)
        else:m=NHITS(**cfg,mlp_units=[[64,64]]*3,n_blocks=[1]*3,n_pool_kernel_size=[2,2,1],n_freq_downsample=[1]*3)
        m.load_state_dict(state['state_dict'],strict=True);m.eval()
        h=torch.tensor(hist[:,:,None],dtype=torch.float32)
        with torch.no_grad():pred=m({'insample_y':h,'insample_mask':torch.ones_like(h),'futr_exog':torch.tensor(x),
          'hist_exog':None,'stat_exog':None}).reshape(-1).numpy()*e['target_std']+e['target_mean']
        old=pd.read_csv(ROOT/'supervised'/f'{stem}.csv').set_index('fila_fuente').loc[ids,'raw_prediction_kWh'].to_numpy()
        delta=float(np.max(np.abs(pred-old)));assert delta<.005,(family,delta)
        checks.append({'check':'checkpoint_roundtrip_different_batch_size','model':family,'n':8,'max_abs_difference_kWh':delta})
    spec=json.loads((ROOT/'PROTOCOL.json').read_text(encoding='utf8'))['foundations']['Chronos_2']
    m=Chronos2Pipeline.from_pretrained(spec['repo'],revision=spec['revision'],local_files_only=True,device_map=hw['device'],dtype=torch.float32)
    inputs=[]
    for i in range(8):
        rows=d.loc[np.r_[w['donors'][i],w['ids'][i]]]
        vals={k:rows[k].to_numpy('float32') for k in NUM};vals.update({k:rows[k].astype(str).to_numpy() for k in CAT})
        inputs.append({'target':d.loc[w['donors'][i],TARGET].to_numpy('float32'),'past_covariates':{k:v[:-1] for k,v in vals.items()},'future_covariates':{k:v[-1:] for k,v in vals.items()}})
    qq,_=m.predict_quantiles(inputs,prediction_length=1,quantile_levels=[.5],batch_size=32,context_length=L,cross_learning=False)
    pred=np.array([float(v[0,0,0].cpu()) for v in qq])
    old=pd.read_csv(ROOT/'foundation'/f'Chronos_2_covariates_{start}.csv').set_index('fila_fuente').loc[w['ids'][:8],'raw_prediction_kWh'].to_numpy()
    delta=float(np.max(np.abs(pred-old)));assert delta<.01,delta
    checks.append({'check':'native_Chronos_covariates_batch32_vs64','n':8,'max_abs_difference_kWh':delta})
    # A target label is not a value in either past or future covariates.
    assert TARGET not in inputs[0]['past_covariates'] and TARGET not in inputs[0]['future_covariates']
    save_json(ROOT/'NUMERICAL_CHECKS.json',{'status':'PASS','checks':checks})
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
