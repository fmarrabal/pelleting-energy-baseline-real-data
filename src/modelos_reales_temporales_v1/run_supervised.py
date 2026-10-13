from common import *
DEVICE,GPU_STATUS=device_policy()
import copy,time,random,gc,torch,joblib
from neuralforecast.models import TFT,NHITS
from neuralforecast.losses.pytorch import MSE
from sklearn.linear_model import Ridge

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def init_model(family,nf,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    cfg=dict(h=1,input_size=L,futr_exog_list=[f'x{i}' for i in range(nf)],loss=MSE(),scaler_type='identity',random_seed=seed)
    if family=='TFT':m=TFT(**cfg,hidden_size=32,n_head=4,dropout=.1)
    else:m=NHITS(**cfg,mlp_units=[[64,64]]*3,n_blocks=[1]*3,n_pool_kernel_size=[2,2,1],n_freq_downsample=[1]*3)
    return m.to(DEVICE)

def tensors(a):
    return {k:torch.tensor(a[k],device=DEVICE,dtype=torch.float32) for k in ['x','hist','y']}

def forward(m,a,ids):
    h=a['hist'][ids,:,None]
    return m({'insample_y':h,'insample_mask':torch.ones_like(h),'futr_exog':a['x'][ids],
              'hist_exog':None,'stat_exog':None}).reshape(-1)

def infer(m,a):
    m.eval()
    with torch.no_grad():return torch.cat([forward(m,a,slice(i,i+64)) for i in range(0,len(a['y']),64)]).cpu().numpy()

def fit(family,tr,va,seed,epochs=40,selection=True):
    m=init_model(family,tr['x'].shape[-1],seed);a=tensors(tr);v=tensors(va) if va is not None else None
    opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed);best=float('inf');best_epoch=1;wait=0;curve=[]
    for epoch in range(1,epochs+1):
        m.train();order=rng.permutation(len(a['y']));loss_sum=0
        for i in range(0,len(order),64):
            ix=torch.tensor(order[i:i+64],device=DEVICE);opt.zero_grad(set_to_none=True)
            pred=forward(m,a,ix);loss=((pred-a['y'][ix])**2).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
            loss_sum+=loss.item()*len(ix)
        row={'epoch':epoch,'train_MSE':loss_sum/len(order)}
        if selection:
            score=float(np.mean(np.abs(infer(m,v)-va['y'])))
            row['inner_MAE_standardized']=score
            if score<best:best=score;best_epoch=epoch;wait=0
            else:wait+=1
        curve.append(row)
        if selection and epoch>=10 and wait>=8:break
    return m,best_epoch,curve

def main():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text(encoding='utf8'))
    assert sha(ROOT/'PROTOCOL.json')==json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())['sha256']
    out=ROOT/'supervised';out.mkdir(exist_ok=True)
    save_json(out/'hardware.json',{'device':DEVICE,'gpu_at_start':GPU_STATUS,'torch':torch.__version__,'threads':4})
    d=data();runtime=[]
    for start,end in FOLDS:
        ws=fold_windows(d,start,end)
        enc,mu,sd,a=encode(d,ws['train'],{'train':ws['train'],'val':ws['val']})
        ienc,imu,isd,inner=encode(d,ws['inner_train'],{'train':ws['inner_train'],'val':ws['inner_val']})
        joblib.dump({'encoder':enc,'target_mean':mu,'target_std':sd},out/f'encoder_{start}.joblib')
        for name,temporal in [('Ridge_actual',False),('Ridge_ventana',True)]:
            dest=out/f'{name}_{start}.csv'
            if dest.exists():continue
            def xx(v):return np.c_[v['x'].reshape(len(v['x']),-1),v['hist']] if temporal else v['x'][:,-1,:]
            m=Ridge(alpha=10).fit(xx(a['train']),a['train']['truth'])
            pred=m.predict(xx(a['val']));prediction_frame(d,ws['val']['ids'],pred,name,start).to_csv(dest,index=False)
            joblib.dump(m,out/f'{name}_{start}.joblib')
        for family in ['TFT','NHITS']:
            for seed in SEEDS:
                stem=f'{family}_{start}_seed{seed}';dest=out/f'{stem}.csv'
                if dest.exists():continue
                tick=time.perf_counter()
                m,best_epoch,curve=fit(family,inner['train'],inner['val'],seed)
                # Freeze epoch choice before exposing outer validation targets to metrics.
                save_json(out/f'{stem}_selection.json',{'seed':seed,'selected_epoch':best_epoch,'curve':curve,'inner_scale':isd,'outer_labels_used_for_selection':False})
                del m;gc.collect()
                m,_,refitcurve=fit(family,a['train'],None,seed,epochs=best_epoch,selection=False)
                pred=infer(m,tensors(a['val']))*sd+mu
                prediction_frame(d,ws['val']['ids'],pred,family,start,seed=seed).to_csv(dest,index=False)
                torch.save({'state_dict':m.cpu().state_dict(),'family':family,'seed':seed,'n_features':a['train']['x'].shape[-1],
                  'context':L,'selected_epoch':best_epoch,'target_mean':mu,'target_std':sd},out/f'{stem}.pt')
                row={'fold':start,'model':family,'seed':seed,'selected_epoch':best_epoch,'trained_epochs_inner':len(curve),
                 'n_train':len(a['train']['ids']),'n_val':len(a['val']['ids']),'seconds':time.perf_counter()-tick,'device':DEVICE,
                 'parameters':sum(p.numel() for p in m.parameters()),'refit_curve':refitcurve}
                runtime.append(row);save_json(out/'runtime.json',runtime)
                print(f'{stem} complete epoch={best_epoch} seconds={row["seconds"]:.1f}',flush=True)
                del m;gc.collect()
                if DEVICE=='cuda':torch.cuda.empty_cache()
    print('SUPERVISED COMPLETE',flush=True)

if __name__=='__main__':main()
