"""Post-result numerical diagnostic, never used to replace primary predictions."""
from common import *
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch,timesfm
from timesfm.utils import xreg_lib
torch.set_num_threads(4)

def main():
    d=data();w=fold_windows(d,*FOLDS[0])['val']
    pred=pd.read_csv(ROOT/'foundation/TimesFM_2.5_covariates_2025-01-25.csv').set_index('fila_fuente')
    ids=pred[pred.prediccion_kWh.gt(2000)].index.to_list()
    spec=json.loads((ROOT/'PROTOCOL.json').read_text(encoding='utf8'))['foundations']['TimesFM_2.5']
    m=timesfm.TimesFM_2p5_200M_torch.from_pretrained(spec['repo'],revision=spec['revision'],local_files_only=True,torch_compile=False)
    m.compile(timesfm.ForecastConfig(max_context=32,max_horizon=32,normalize_inputs=True,per_core_batch_size=1,
      use_continuous_quantile_head=False,force_flip_invariance=True,infer_is_positive=False,return_backcast=True))
    out=[]
    for target in ids:
        i=np.where(w['ids']==target)[0][0];rows=d.loc[np.r_[w['donors'][i],target]]
        yy=[rows[TARGET].to_numpy('float32')[:-1]];yn,stats=xreg_lib.normalize(yy)
        nt={k:[rows[k].to_numpy(float)[:-1]] for k in NUM};nv={k:[rows[k].to_numpy(float)[-1:]] for k in NUM}
        ct={k:[rows[k].astype(str).to_numpy()[:-1]] for k in CAT};cv={k:[rows[k].astype(str).to_numpy()[-1:]] for k in CAT}
        reg=xreg_lib.BatchedInContextXRegLinear(targets=yn,train_lens=[L],test_lens=[1],train_dynamic_numerical_covariates=nt,
          test_dynamic_numerical_covariates=nv,train_dynamic_categorical_covariates=ct,test_dynamic_categorical_covariates=cv)
        xpred,xctx,_,xx,xt=reg.fit(ridge=10,one_hot_encoder_drop=None,max_rows_per_col=0,force_on_cpu=True,
          debug_info=True,assert_covariates=True,assert_covariate_shapes=True)
        rr=yn[0]-xctx[0];pp,_=m.forecast(horizon=32,inputs=[rr]);resid=float(pp[0,-32])
        mu,sd=stats[0];total=(float(xpred[0][0])+resid)*float(sd)+float(mu)
        standardized={k:float((nv[k][0][0]-nt[k][0].mean())/(nt[k][0].std() if nt[k][0].std()>1e-6 else 1)) for k in NUM}
        # Independent float64 normal-equation solve of the same XReg system.
        # This is a diagnostic of numerical stability, not a new forecasting specification.
        rawy=np.asarray(reg.create_covariate_matrix(one_hot_encoder_drop=None,use_intercept=True,assert_covariates=True,assert_covariate_shapes=True)[0],dtype=float)
        X=np.asarray(xx,dtype=float);V=np.asarray(xt,dtype=float)
        Y=np.pad(rawy,(0,X.shape[0]-len(rawy)))
        beta=np.linalg.solve(X.T@X+10*np.eye(X.shape[1]),X.T@Y)
        double=float((V@beta)[0]);native=float(xpred[0][0])
        result={'source_row':target,'observed_kWh':float(rows.iloc[-1][TARGET]),'recorded_prediction_kWh':float(pred.loc[target,'prediccion_kWh']),
          'reconstructed_prediction_kWh':total,'reconstruction_difference_kWh':abs(total-float(pred.loc[target,'prediccion_kWh'])),
          'XReg_component_normalized':native,'TimesFM_residual_normalized':resid,'context_target_mean':float(mu),'context_target_sd':float(sd),
          'XReg_float64_vs_native_difference_normalized':abs(double-native),'current_numeric_distance_context_sd':standardized,
          'history_source_rows':w['donors'][i].tolist()}
        assert result['reconstruction_difference_kWh']<.02
        out.append(result)
    save_json(ROOT/'TIMESFM_EXTREME_DIAGNOSTIC.json',{'status':'POST_HOC_DIAGNOSTIC_NO_PRIMARY_CHANGES','cases':out})
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
