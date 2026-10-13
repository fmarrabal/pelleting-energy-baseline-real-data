from common import *
import importlib.metadata as md, platform

def main():
    if (ROOT/'PROTOCOL.json').exists():raise RuntimeError('Frozen protocol already exists; do not overwrite it.')
    d=data();logs=[];links=[];excluded=[];members=[]
    for start,end in FOLDS:
        ws=fold_windows(d,start,end)
        logs.append({'fold':start,**{k:len(v['ids']) for k,v in ws.items()}})
        for partition,w in ws.items():
            for i,ids in enumerate(w['donors']):
                target=int(w['ids'][i]);r=d.loc[target]
                members.append({'fold':start,'partition':partition,'fila_fuente':target})
                for j,donor in enumerate(ids):
                    h=d.loc[donor]
                    links.append({'fold':start,'partition':partition,'fila_fuente':target,'donor':int(donor),'position':j,
                     'linea':r.linea,'origin':str(r.inicio_registro),'donor_end':str(h.fin_peletizado),'partition_floor':w['floor']})
            for r in w['excluded']:excluded.append({'fold':start,'partition':partition,**r})
    pd.DataFrame(links).to_csv(ROOT/'donor_audit.csv',index=False)
    pd.DataFrame(members).to_csv(ROOT/'sample_membership.csv',index=False)
    pd.DataFrame(excluded).to_csv(ROOT/'warmup_exclusions.csv',index=False)
    protocol={
      'status':'EXPLORATORY_DEVELOPMENT_ONLY','created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
      'question':'Do modern temporal models add predictive value for real pelleting energy records?',
      'target':TARGET,'context_records':L,'horizon':1,'horizon_semantics':'Nominated current industrial record, conditional on its consolidated plan covariates; event positions are not uniform physical time.',
      'partition_policy':'Every context and target stays within the same machine and partition interval. Validation history resets each outer week; inner validation also resets. Completed earlier validation labels can enter later contexts but never model weight fitting.',
      'history_availability':'fin_peletizado strictly earlier than inicio_registro; completion used as proxy for actual data publication.',
      'batch_identity':'Original rows are industrial records; unique physical batch/cycle identity is not verified. No synthetic cycle identifiers are inferred.',
      'features_numeric':NUM,'features_categorical':CAT,'forbidden':'current energy, current duration, current operational averages, produced mass, future labels',
      'availability_caveat':'Dosified mass, batch count and product metadata are consolidated records; their planned values at origin are unverified. Retrospective conditional baseline, not validated pre-operation forecast.',
      'folds':FOLDS,'sample_counts':logs,'eligible_total':sum(r['val'] for r in logs),'original_candidates':800,
      'TFT':{'hidden_size':32,'n_head':4,'dropout':.1},
      'NHITS':{'mlp_units':[[64,64]]*3,'n_blocks':[1,1,1],'n_pool_kernel_size':[2,2,1],'n_freq_downsample':[1,1,1]},
      'supervised_selection':{'architectures_per_family':1,'seeds':SEEDS,'max_epochs':40,'minimum_epochs':10,'patience':8,'batch_size':64,'optimizer':'Adam','lr':.001,'weight_decay':.0001,'loss':'MSE standardized target','gradient_clip_norm':1.0,
        'rule':'Select best epoch separately for each seed on preceding 7-day inner validation, never outer validation. Refit each seed from scratch on complete outer training using its selected epoch. Primary prediction is mean of three nonnegative seed predictions.'},
      'ridge':{'alpha':10,'controls':['current covariates only','current covariates plus full flattened window; no hyperparameter search']},
      'foundations':json.loads((BASE/'modelos_avanzados_v1/FOUNDATION_MANIFEST.json').read_text()),
      'foundation_policy':'Frozen weights, zero-shot. Native univariate controls and covariate variants. Chronos-2 native past+future covariates, cross_learning=False. TimesFM 2.5 native XReg + TimesFM, ridge=10, separate call per origin to avoid pooling future labels across query windows. Both use exactly 16 completed records; no target labels supplied.',
      'metrics':['MAE','RMSE','bias','WAPE','by fold','by machine','seed variability','raw negative predictions'],
      'inference':'paired circular blocks of 3 observed dates, 5000 bootstrap draws, seed=20260922; differences versus matched Ridge; exploratory, no universal superiority claim',
      'device_policy':'GPU if >=4096 MiB free at process launch; otherwise CPU. Never stop other workloads.',
      'protected_sha256':{p:sha(BASE/p) for p in PROTECTED},
      'versions':{'python':platform.python_version(),**{p:md.version(p) for p in ['torch','neuralforecast','timesfm','chronos-forecasting','numpy','pandas','scikit-learn']}},
      'sources':['https://nixtlaverse.nixtla.io/neuralforecast/models.tft.html','https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html','https://github.com/google-research/timesfm','https://github.com/amazon-science/chronos-forecasting']}
    # Old manifests describe synthetic runs: keep only checkpoint identity, not their input configuration.
    protocol['foundations']={k:{f:v[f] for f in ['repo','revision']} for k,v in protocol['foundations'].items()}
    save_json(ROOT/'PROTOCOL.json',protocol)
    save_json(ROOT/'PROTOCOL_LOCK.json',{'sha256':sha(ROOT/'PROTOCOL.json'),'donor_relations':len(links),'n_same_machine_time_partition_violations':0,'created_before_training':True})
    print(json.dumps(logs,indent=2));print('FROZEN',sha(ROOT/'PROTOCOL.json'),flush=True)

if __name__=='__main__':main()
