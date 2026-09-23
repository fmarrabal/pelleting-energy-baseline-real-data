"""Prepare editorial data; never fit a model or call predict on a test set."""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
import joblib

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
D=HERE/'data'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding='utf8')
sources={}
def source(rel):
    p=BASE/rel
    sources[rel]=sha(p)
    return p

protected=json.loads((BASE/'modelos_reales_gpu_v1/PROTOCOL.json').read_text())['protected_sha256']
for rel, expected in protected.items():
    assert sha(source(rel))==expected, rel
for name in ['metrics.csv','metrics_by_fold.csv','metrics_by_machine.csv','metrics_seeds.csv',
             'paired_contrasts.csv','predictions.csv','predictions_seeds.csv','sample_membership.csv',
             'warmup_exclusions.csv','comparison_cpu_gpu.csv','paired_cpu_gpu_predictions.csv',
             'numerical_differences.csv','epoch_selection_cpu_gpu.csv',
             'chronos_normalization_diagnostic.csv','PROTOCOL.json','GPU_AUDIT.json',
             'NUMERICAL_CHECKS.json','CHRONOS_NORMALIZATION_DIAGNOSTIC.json',
             'DATA_PIPELINE_EQUIVALENCE.json','GPU_RUN_PLAN.json','PROCESS_COMPLETION.json']:
    shutil.copyfile(source('modelos_reales_gpu_v1/'+name),D/'gpu'/name)
for p in (BASE/'paper_datos_reales_v1/data').iterdir():
    source(str(p.relative_to(BASE)).replace('\\','/'))
    assert sha(p)==sha(D/p.name)

d=pd.read_csv(source('depuracion_v1/registros_depurados.csv'))
for c in ['inicio_registro','inicio_peletizado','fin_peletizado']: d[c]=pd.to_datetime(d[c])
h=d.inicio_registro.dt.hour+d.inicio_registro.dt.minute/60
d['hora_sin']=np.sin(2*np.pi*h/24);d['hora_cos']=np.cos(2*np.pi*h/24)
d['dia_sin']=np.sin(2*np.pi*d.inicio_registro.dt.dayofweek/7)
d['dia_cos']=np.cos(2*np.pi*d.inicio_registro.dt.dayofweek/7)
keep=['fila_fuente','linea','subfamilia','familia','forma','salida_programada','baches_dosificacion',
      'masa_dosificada_t','energia_peletizado_kWh','inicio_registro','inicio_peletizado','fin_peletizado','particion']
d[keep].to_csv(D/'variables_source.csv',index=False)
stats={'rows':len(d),'positive_targets':int(d.energia_peletizado_kWh.gt(0).sum()),
       'zero_targets':int(d.energia_peletizado_kWh.eq(0).sum()),
       'missing_targets':int(d.energia_peletizado_kWh.isna().sum()),
       'category_counts':{c:d[c].value_counts().to_dict() for c in ['linea','subfamilia','familia','forma','salida_programada']},
       'numeric_summary':d[['masa_dosificada_t','baches_dosificacion','energia_peletizado_kWh']].describe().to_dict(),
       'mass_batches_pearson':float(d.masa_dosificada_t.corr(d.baches_dosificacion))}

# Exact additive decomposition of the archived frozen Ridge coefficients.
# No selection, retraining, or predict() call; equality checked against archived predictions.
obj=joblib.load(source('calibracion_final_v1/modelo_calibrado.joblib'))
pipe=obj['pipeline']; pre=pipe.named_steps['pre']; ridge=pipe.named_steps['ridge']
test=pd.read_csv(source('prueba_final_v1/predicciones_prueba.csv'))
xx=d.set_index('fila_fuente').loc[test.fila_fuente].copy()
z=pre.transform(xx); coef=ridge.coef_; names=pre.get_feature_names_out()
terms=z*coef
reconstructed=np.maximum(0,float(ridge.intercept_)+terms.sum(axis=1))
max_error=float(np.max(abs(reconstructed-test.Ridge.to_numpy())))
assert max_error<1e-8
coefs=pd.DataFrame({'encoded_variable':names,'coefficient_kWh':coef})
coefs.to_csv(D/'ridge_coefficients.csv',index=False)
groups={'Dosed mass':[0],'Dosing batches':[1],'Hour':[2,3],'Weekday':[4,5]}
for label,field in [('Machine','linea'),('Product subfamily','subfamilia'),('Presentation','forma'),('Bag / bulk','salida_programada')]:
    groups[label]=[i for i,n in enumerate(names) if n.startswith('cat__'+field+'_')]
parts=pd.DataFrame({'fila_fuente':test.fila_fuente,'linea':test.linea,'observed_kWh':test.energia_peletizado_kWh,
                    'archived_prediction_kWh':test.Ridge,'intercept_kWh':float(ridge.intercept_)})
for label,inds in groups.items(): parts[label]=terms[:,inds].sum(axis=1)
parts.to_csv(D/'ridge_additive_components.csv',index=False)
effect=[]
scaler=pre.named_transformers_['num']
for i,n in enumerate(obj['numeric']):
    effect.append({'variable':n,'coefficient_per_training_sd_kWh':float(coef[i]),
                   'training_mean':float(scaler.mean_[i]),'training_sd':float(scaler.scale_[i]),
                   'coefficient_per_original_unit':float(coef[i]/scaler.scale_[i])})
pd.DataFrame(effect).to_csv(D/'ridge_numeric_effects.csv',index=False)
stats['ridge']={'encoded_features':len(names),'intercept_kWh':float(ridge.intercept_),
                'max_archived_decomposition_error_kWh':max_error,'numeric_effects':effect,
                'negative_unclipped_count':int(((float(ridge.intercept_)+terms.sum(axis=1))<0).sum()),
                'reference':'Standardized numerical means, all-zero categorical indicators; not a physical operation.'}

gpu=pd.read_csv(D/'gpu/predictions.csv'); met=pd.read_csv(D/'gpu/metrics.csv').set_index('modelo')
checks=[]
for model,q in gpu.groupby('modelo'):
    assert len(q)==584 and q.fila_fuente.nunique()==584
    err=q.prediccion_kWh-q.energia_peletizado_kWh
    for key,val in {'MAE_kWh':abs(err).mean(),'RMSE_kWh':np.sqrt(np.mean(err**2)),
                    'bias_kWh':err.mean(),'WAPE':abs(err).sum()/q.energia_peletizado_kWh.sum()}.items():
        assert np.isclose(val,met.loc[model,key],rtol=1e-10,atol=1e-10)
        checks.append({'model':model,'metric':key,'recomputed':float(val)})
pd.DataFrame(checks).to_csv(D/'gpu_metrics_reconciled.csv',index=False)

# One actual donor window for a detailed, reproducible chronology illustration.
donors=pd.read_csv(source('modelos_reales_gpu_v1/donor_audit.csv'))
target=int(gpu[(gpu.modelo=='Ridge_actual')&(gpu.linea=='Pellet 1')].sort_values('inicio_registro').iloc[0].fila_fuente)
w=donors[(donors.partition=='val')&(donors.fila_fuente==target)].copy()
assert len(w)==16
w=w.merge(d[['fila_fuente','inicio_peletizado','fin_peletizado','masa_dosificada_t','energia_peletizado_kWh']],left_on='donor',right_on='fila_fuente',suffixes=('','_donor'))
w.to_csv(D/'example_window.csv',index=False)
stats['example_window_target']=d.set_index('fila_fuente',drop=False).loc[target,keep].to_dict()
stats['modern_records_by_machine']=gpu[gpu.modelo.eq('Ridge_actual')].linea.value_counts().to_dict()
dump(HERE/'DESCRIPTIVE_STATISTICS.json',stats)
dump(HERE/'SOURCE_MANIFEST.json',{'sources':sources,'protected':protected,'revision_fits':0,
                               'revision_predict_calls':0,'new_points_fabricated':0,
                               'reconciled_gpu_metrics':len(checks),'ridge_decomposition_max_error':max_error})
print(json.dumps(stats['ridge'],indent=2)); print('Window target',target)
