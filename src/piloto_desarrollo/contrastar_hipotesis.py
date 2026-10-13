from pathlib import Path
import hashlib,json,math,datetime
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'hipotesis_exploratorias_v1';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=ROOT/'depuracion_v1/registros_depurados.csv';source_hash=sha(src)
cfg={'fecha_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'H1':'Interaccion masa x maquina','H2':'Interaccion masa x subfamilia','H3':'Correccion media de residuos Ridge de ultimos20 finalizados de misma maquina dentro del bloque; minimo5, contraccion n/(n+10), reinicio0 cada bloque','alpha':10,'bootstrap':{'B':5000,'L_dias_observados':3,'seed':20260921,'IC95_y_IC98_333_Bonferroni_3':True},'seleccion':'MAE principal; RMSE, sesgo y estabilidad complementarios; no busqueda parametros','alcance':'Solo desarrollo; hipotesis formuladas tras conocer resultados previos, sin validacion independiente','source_sha256':source_hash}
(OUT/'PROTOCOLO.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf8')
d=pd.read_csv(src);d=d[d.particion.eq('DESARROLLO')].copy()
for c in ['inicio_registro','fin_peletizado']:d[c]=pd.to_datetime(d[c])
h=d.inicio_registro.dt.hour+d.inicio_registro.dt.minute/60
for label,x,period in [('hora',h,24),('dia',d.inicio_registro.dt.dayofweek,7)]:
    d[label+'_sin']=np.sin(2*np.pi*x/period);d[label+'_cos']=np.cos(2*np.pi*x/period)
num=['masa_dosificada_t','baches_dosificacion','hora_sin','hora_cos','dia_sin','dia_cos'];cat=['linea','subfamilia','forma','salida_programada']
d=d[d.mascara_objetivo_peletizado.eq(1)&d.apto_temporal_peletizado.eq(1)&d.masa_dosificada_t.gt(0)&d[num+cat].notna().all(axis=1)]
rows=[];logs=[];donors=[]
folds=[('2025-01-25','2025-02-01'),('2025-02-01','2025-02-08'),('2025-02-08','2025-02-15'),('2025-02-15','2025-02-22')]
for start,end in folds:
    a,b=pd.Timestamp(start),pd.Timestamp(end)
    tr=d[d.inicio_registro.lt(a)&d.fin_peletizado.lt(a)]
    va=d[d.inicio_registro.ge(a)&d.inicio_registro.lt(b)&d.fin_peletizado.lt(b)].sort_values(['inicio_registro','fila_fuente']).copy()
    assert tr.fin_peletizado.max()<a
    enc=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
    X=enc.fit_transform(tr);V=enc.transform(va);y=tr.energia_peletizado_kWh
    preds={'Ridge':np.maximum(0,Ridge(alpha=10).fit(X,y).predict(V))}
    for name,col in [('H1_masa_maquina','linea'),('H2_masa_subfamilia','subfamilia')]:
        oh=OneHotEncoder(handle_unknown='ignore',sparse_output=False);T=oh.fit_transform(tr[[col]]);W=oh.transform(va[[col]])
        # Numeric column0 is training-standardized mass; slope interactions only.
        xt=T*X[:,0,None];vt=W*V[:,0,None]
        pred=Ridge(alpha=10).fit(np.column_stack([X,xt]),y).predict(np.column_stack([V,vt]))
        preds[name]=np.maximum(0,pred)
    va['base']=preds['Ridge'];va['residuo']=va.energia_peletizado_kWh-va.base
    history=va.sort_values(['fin_peletizado','fila_fuente']);adjusted=[]
    for _,r in va.iterrows():
        prior=history[history.linea.eq(r.linea)&history.fin_peletizado.lt(r.inicio_registro)].tail(20)
        assert r.fila_fuente not in set(prior.fila_fuente)
        shift=float(prior.residuo.mean()*len(prior)/(len(prior)+10)) if len(prior)>=5 else 0.
        adjusted.append(max(0,r.base+shift));donors.append({'fila_fuente':r.fila_fuente,'fold':start,'n_donantes':len(prior),'ajuste_kWh':shift,'max_fin_donante':str(prior.fin_peletizado.max()),'filas_donantes':','.join(map(str,prior.fila_fuente))})
    preds['H3_sesgo_reciente']=np.array(adjusted)
    logs.append({'inicio':start,'train':len(tr),'val':len(va)})
    for name,pred in preds.items():
        z=va[['fila_fuente','linea','familia','inicio_registro','energia_peletizado_kWh']].copy();z['fold']=start;z['modelo']=name;z['pred']=pred;z['error']=pred-z.energia_peletizado_kWh;z['abs']=abs(z.error);rows.append(z)
p=pd.concat(rows,ignore_index=True)
def metrics(g):return pd.Series({'n':len(g),'MAE':g['abs'].mean(),'RMSE':np.sqrt((g.error**2).mean()),'sesgo':g.error.mean(),'WAPE':g['abs'].sum()/g.energia_peletizado_kWh.sum()})
m=p.groupby('modelo').apply(metrics,include_groups=False);m.to_csv(OUT/'metricas.csv')
for c in ['linea','fold','familia']:p.groupby(['modelo',c]).apply(metrics,include_groups=False).to_csv(OUT/f'metricas_por_{c}.csv')
p.to_csv(OUT/'predicciones.csv',index=False);pd.DataFrame(donors).to_csv(OUT/'donantes_sesgo.csv',index=False)
wide=p.pivot(index='fila_fuente',columns='modelo',values='abs');dates=p.drop_duplicates('fila_fuente').set_index('fila_fuente').inicio_registro.dt.normalize();wide['fecha']=dates
daily=wide.groupby('fecha').sum(numeric_only=True);counts=wide.groupby('fecha').size().to_numpy();nd=len(daily)
rng=np.random.default_rng(20260921);idx=((rng.integers(0,nd,size=(5000,math.ceil(nd/3),1))+np.arange(3))%nd).reshape(5000,-1)[:,:nd];den=counts[idx].sum(axis=1)
ci=[]
for name in ['H1_masa_maquina','H2_masa_subfamilia','H3_sesgo_reciente']:
    differences=(daily.Ridge-daily[name]).to_numpy();boot=differences[idx].sum(axis=1)/den
    ci.append({'hipotesis':name,'reduccion_MAE':float(differences.sum()/counts.sum()),'reduccion_pct':float(100*(m.loc['Ridge','MAE']-m.loc[name,'MAE'])/m.loc['Ridge','MAE']),'IC95_inf':np.quantile(boot,.025),'IC95_sup':np.quantile(boot,.975),'IC98_333_inf':np.quantile(boot,.05/6),'IC98_333_sup':np.quantile(boot,1-.05/6)})
pd.DataFrame(ci).to_csv(OUT/'contrastes_pareados.csv',index=False)
old=pd.read_csv(ROOT/'analisis_desarrollo_v2/predicciones_desarrollo.csv');old=old[old.modelo.eq('Ridge')]
check=p[p.modelo.eq('Ridge')].merge(old[['fila_fuente','prediccion_kWh']],on='fila_fuente',validate='one_to_one')
assert len(check)==800 and np.allclose(check.pred,check.prediccion_kWh,rtol=0,atol=1e-8)
assert sha(src)==source_hash and np.isfinite(p.pred).all() and not p.duplicated(['modelo','fila_fuente']).any()
(OUT/'verificacion.json').write_text(json.dumps({'n':800,'dias':nd,'folds':logs,'referencia_reproducida':True,'fuente_intacta':True,'prueba_reutilizada':False,'script_sha256':sha(Path(__file__))},indent=2),encoding='utf8')
print(m.to_string());print(pd.DataFrame(ci).to_string(index=False))
