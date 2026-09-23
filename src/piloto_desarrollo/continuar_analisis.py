"""Analisis y comparacion exploratoria. Calibracion/prueba excluidas del modelado."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json,hashlib,time,warnings,platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analisis_desarrollo_v2';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'depuracion_v1/registros_depurados.csv'
raw=pd.read_csv(SOURCE)
for c in raw:
    if c.startswith(('inicio_','fin_')):raw[c]=pd.to_datetime(raw[c])
cutoff=pd.Timestamp('2025-02-22')
dev=raw[raw.particion.eq('DESARROLLO')].copy()
# Evidence on grinding is an audit, not a use of reserved pelleting labels for modeling.
power=60*raw.energia_molienda_kWh/raw.duracion_molienda_min.where(raw.duracion_molienda_min.gt(0))
raw['potencia_molienda_equivalente_kW']=power
valid=power.notna()&raw.corriente_molienda_A.gt(0)
milling={
 'n':len(raw),'n_con_potencia_equivalente':int(power.notna().sum()),'potencia_equivalente_cuantiles':power.quantile([0,.25,.5,.75,.95,1]).to_dict(),
 'supera_592_1':int(power.gt(592.1).sum()),
 'energia_vs_duracion_spearman':float(raw[['energia_molienda_kWh','duracion_molienda_min']].corr(method='spearman').iloc[0,1]),
 'energia_vs_masa_spearman':float(raw[['energia_molienda_kWh','masa_dosificada_t']].corr(method='spearman').iloc[0,1]),
 'registros_con_intervalos_molienda_repetidos':int(raw.dropna(subset=['inicio_molienda','fin_molienda']).duplicated(['inicio_molienda','fin_molienda'],keep=False).sum()),
 'nota':'No se corrigen escalas. Potencia nominal no equivale a cota rigida de potencia electrica.'}
milling['por_linea']={}
for name,g in raw.groupby('linea'):
    pairs=g.sort_values('inicio_molienda').energia_molienda_kWh.diff().dropna()
    milling['por_linea'][name]={'n':len(g),'Pmediana_kW':float(g.potencia_molienda_equivalente_kW.median()),'supera_592_1':int(g.potencia_molienda_equivalente_kW.gt(592.1).sum()),'fraccion_de_descensos_energia_secuencial':float(pairs.lt(0).mean())}
(OUT/'auditoria_molienda.json').write_text(json.dumps(milling,ensure_ascii=False,indent=2),encoding='utf8')
raw.sort_values('potencia_molienda_equivalente_kW',ascending=False)[['fila_fuente','linea','inicio_molienda','fin_molienda','duracion_molienda_min','energia_molienda_kWh','corriente_molienda_A','potencia_molienda_equivalente_kW']].head(50).to_csv(OUT/'molienda_50_casos_revision.csv',index=False)

categories=['linea','subfamilia','forma','salida_programada']
numeric=['masa_dosificada_t','baches_dosificacion','hora_sin','hora_cos','dia_sin','dia_cos']
hour=dev.inicio_registro.dt.hour+dev.inicio_registro.dt.minute/60
dev['hora_sin']=np.sin(2*np.pi*hour/24);dev['hora_cos']=np.cos(2*np.pi*hour/24)
dev['dia_sin']=np.sin(2*np.pi*dev.inicio_registro.dt.dayofweek/7);dev['dia_cos']=np.cos(2*np.pi*dev.inicio_registro.dt.dayofweek/7)
good=dev.mascara_objetivo_peletizado.eq(1)&dev.apto_temporal_peletizado.eq(1)&dev.masa_dosificada_t.gt(0)&dev[numeric+categories].notna().all(axis=1)
folds=[('2025-01-25','2025-02-01'),('2025-02-01','2025-02-08'),('2025-02-08','2025-02-15'),('2025-02-15','2025-02-22')]
config={'numeric':numeric,'categories':categories,'objective':'energia_peletizado_kWh','folds':folds,
'alpha_ridge':10,'ExtraTrees':{'n_estimators':250,'min_samples_leaf':5,'max_features':1.0,'random_state':2026},
'HistGradientBoosting':{'max_iter':200,'max_leaf_nodes':7,'min_samples_leaf':20,'learning_rate':.05,'l2_regularization':10,'early_stopping':False,'random_state':2026},
'MLP':{'hidden_layer_sizes':[32,16],'solver':'adam','alpha':.1,'max_iter':800,'early_stopping':False,'seeds':[11,29,47]},
'temporal_policy':'Train labels end before fold start. Validation labels end before fold end. No calibration/test outcomes used.',
'historical_baseline':'Last five same-machine completed development records at each origin; sequential updates allowed, no refit.',
'availability':'Candidate plan variables assumed known at origin; not verified plant forecasting.',
'input_sha256':hashlib.file_digest(SOURCE.open('rb'),'sha256').hexdigest(),
'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
(OUT/'configuracion.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
predictions=[]; foldlog=[]; convergence=[]
history=dev[good].sort_values('fin_peletizado')
for start,end in folds:
    a,b=pd.Timestamp(start),pd.Timestamp(end)
    tr=dev[good&dev.inicio_registro.lt(a)&dev.fin_peletizado.lt(a)].copy()
    va=dev[good&dev.inicio_registro.ge(a)&dev.inicio_registro.lt(b)&dev.fin_peletizado.lt(b)].copy()
    assert tr.fin_peletizado.max()<a and va.fin_peletizado.max()<b<=cutoff
    enc=ColumnTransformer([('num',StandardScaler(),numeric),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categories)])
    X=enc.fit_transform(tr);V=enc.transform(va)
    y=tr.energia_peletizado_kWh.to_numpy();yt=va.energia_peletizado_kWh.to_numpy()
    median=tr.groupby('linea').energia_peletizado_kWh.median()
    secs=tr.groupby('linea').apply(lambda g:g.energia_peletizado_kWh.sum()/g.masa_dosificada_t.sum(),include_groups=False)
    estimates={'Mediana_maquina':va.linea.map(median).to_numpy(),'SEC_maquina':va.linea.map(secs).to_numpy()*va.masa_dosificada_t.to_numpy()}
    last=[]
    for _,r in va.iterrows():
        h=history[(history.linea==r.linea)&history.fin_peletizado.lt(r.inicio_registro)].tail(5)
        assert len(h) and h.fin_peletizado.max()<r.inicio_registro
        last.append(float((h.energia_peletizado_kWh/h.masa_dosificada_t).median()*r.masa_dosificada_t))
    estimates['SEC_ultimos5']=np.array(last)
    models={'Ridge':Ridge(alpha=10),
      'ExtraTrees':ExtraTreesRegressor(**config['ExtraTrees'],n_jobs=2),
      'HistGradientBoosting':HistGradientBoostingRegressor(**config['HistGradientBoosting'])}
    times={}
    for name,model in models.items():
        t=time.perf_counter();model.fit(X,y);estimates[name]=np.maximum(model.predict(V),0);times[name]=time.perf_counter()-t
    nn=[]
    for seed in config['MLP']['seeds']:
        model=MLPRegressor(hidden_layer_sizes=(32,16),solver='adam',alpha=.1,max_iter=800,early_stopping=False,random_state=seed)
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as ww:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(X,(y-y.mean())/y.std())
        forecast=np.maximum(0,model.predict(V)*y.std()+y.mean());nn.append(forecast)
        convergence.append({'fold':start,'seed':seed,'iterations':model.n_iter_,'warnings':[str(w.message) for w in ww],'MAE':float(abs(forecast-yt).mean()),'seconds':time.perf_counter()-t})
    estimates['MLP_3_semillas']=np.mean(nn,axis=0)
    n_candidates=int((dev.inicio_registro.ge(a)&dev.inicio_registro.lt(b)).sum())
    foldlog.append({'inicio':start,'fin_exclusivo':end,'train':len(tr),'validation':len(va),'candidatos':n_candidates,'excluidos':n_candidates-len(va),'max_fin_train':str(tr.fin_peletizado.max()),'max_fin_val':str(va.fin_peletizado.max()),'tiempos_s':times})
    for name,pred in estimates.items():
        q=va[['fila_fuente','linea','familia','subfamilia','inicio_registro','duracion_peletizado_min','energia_peletizado_kWh']].copy()
        q['fold']=start;q['modelo']=name;q['prediccion_kWh']=pred
        q['error']=pred-yt;q['error_abs']=abs(q.error);predictions.append(q)
    print('Bloque terminado',start,'train',len(tr),'val',len(va),flush=True)

allp=pd.concat(predictions,ignore_index=True)
allp.to_csv(OUT/'predicciones_desarrollo.csv',index=False)
def metrics(g):
    return pd.Series({'n':len(g),'MAE_kWh':g.error_abs.mean(),'RMSE_kWh':np.sqrt((g.error**2).mean()),'sesgo_kWh':g.error.mean(),'WAPE':g.error_abs.sum()/g.energia_peletizado_kWh.sum()})
global_metrics=allp.groupby('modelo').apply(metrics,include_groups=False).sort_values('MAE_kWh')
global_metrics.to_csv(OUT/'metricas_globales.csv')
for keys,label in [(['modelo','fold'],'por_bloque'),(['modelo','linea'],'por_maquina'),(['modelo','familia'],'por_familia')]:
    allp.groupby(keys).apply(metrics,include_groups=False).to_csv(OUT/f'metricas_{label}.csv')
allp[allp.duracion_peletizado_min.le(1440)].groupby('modelo').apply(metrics,include_groups=False).to_csv(OUT/'sensibilidad_sin_duraciones_mayores_24h.csv')

# Paired circular moving-block bootstrap of observed dates (all machines together).
err=allp.pivot(index='fila_fuente',columns='modelo',values='error_abs')
dates=allp.drop_duplicates('fila_fuente').set_index('fila_fuente').inicio_registro.dt.normalize()
err['fecha']=dates
daily=err.groupby('fecha').sum(numeric_only=True);counts=err.groupby('fecha').size().to_numpy()
rng=np.random.default_rng(20260920);B=3000;nd=len(daily);L=3
idx=((rng.integers(0,nd,size=(B,int(np.ceil(nd/L)),1))+np.arange(L))%nd).reshape(B,-1)[:,:nd]
ci=[]
for baseline in ['SEC_maquina','Ridge']:
    for name in global_metrics.index:
        if name==baseline:continue
        difference=(daily[baseline]-daily[name]).to_numpy()
        boots=difference[idx].sum(axis=1)/counts[idx].sum(axis=1)
        ci.append({'modelo':name,'referencia':baseline,'reduccion_MAE_kWh':float(difference.sum()/counts.sum()),'IC95_inferior':float(np.quantile(boots,.025)),'IC95_superior':float(np.quantile(boots,.975))})
pd.DataFrame(ci).to_csv(OUT/'bootstrap_diferencias.csv',index=False)

eda=dev[good&dev.fin_peletizado.lt(cutoff)].copy()
eda.groupby('linea').agg(n=('fila_fuente','size'),masa_mediana_t=('masa_dosificada_t','median'),energia_mediana_kWh=('energia_peletizado_kWh','median'),duracion_mediana_min=('duracion_peletizado_min','median')).to_csv(OUT/'descripcion_desarrollo.csv')
pd.crosstab(eda.linea,eda.familia).to_csv(OUT/'maquina_producto_desarrollo.csv')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.5));global_metrics.MAE_kWh.sort_values().plot.barh(ax=ax,color='#315a83');ax.invert_yaxis();ax.set_xlabel('MAE de desarrollo (kWh por registro)');ax.set_ylabel('');ax.set_title('Comparación temporal exploratoria — cuatro bloques');fig.tight_layout();fig.savefig(OUT/'comparacion_modelos.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for machine,g in eda.groupby('linea'):axes[0].scatter(g.masa_dosificada_t,g.energia_peletizado_kWh,s=9,alpha=.35,label=machine)
axes[0].set(xlabel='Masa dosificada (t)',ylabel='Energía de peletizado (kWh)',title='Desarrollo: masa y energía');axes[0].legend()
tab=pd.crosstab(eda.linea,eda.familia,normalize='index')*100;tab.plot.bar(stacked=True,ax=axes[1],colormap='tab20');axes[1].set(ylabel='% de registros',xlabel='',title='Mezcla de productos por máquina');axes[1].tick_params(axis='x',rotation=0);axes[1].legend(fontsize=6,loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig(OUT/'exploracion_desarrollo.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5));ax.hist(power.dropna().clip(upper=2000),bins=45,color='#63839f');ax.axvline(592.1,color='#a53e32',label='592,1 kW instalados documentados');ax.set(xlabel='Energía/duración de molienda (kW equivalentes)',ylabel='Registros',title='Auditoría de molienda — sin corregir factores');ax.legend();fig.tight_layout();fig.savefig(OUT/'auditoria_molienda.png',dpi=160);plt.close(fig)
manifest={'config':config,'bloques':foldlog,'MLP':convergence,'n_predicciones_por_modelo':int(global_metrics.n.iloc[0]),'dias_validacion':nd,'bootstrap':{'repeticiones':B,'longitud_bloque_dias_observados':L,'interpretacion':'IC exploratorios no ajustados por seleccion/comparaciones multiples'},'candidato_min_MAE_desarrollo':global_metrics.index[0],'calibracion_y_prueba_evaluadas':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
assert allp.inicio_registro.max()<cutoff
assert all(allp.groupby('modelo').fila_fuente.nunique()==len(err))
assert np.isfinite(allp.prediccion_kWh).all()
assert hashlib.file_digest(SOURCE.open('rb'),'sha256').hexdigest()==config['input_sha256']
print(global_metrics.to_string(),flush=True)
print('Molienda:',json.dumps(milling,ensure_ascii=False),flush=True)
