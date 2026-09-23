"""Ajuste final de desarrollo y calibracion reservada; no evalua prueba."""
from pathlib import Path
import csv,json,hashlib,math,platform
import numpy as np
import pandas as pd
import sklearn,joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'calibracion_final_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'depuracion_v1/registros_depurados.csv'
source_hash=sha(source);protocol_hash=sha(OUT/'PROTOCOLO_FIJADO.md')
# Filter raw CSV records by partition before parsing numeric targets. No test labels used.
with source.open(encoding='utf-8-sig',newline='') as f:
    records=[r for r in csv.DictReader(f) if r['particion'] in ['DESARROLLO','CALIBRACION']]
d=pd.DataFrame(records)
num=['masa_dosificada_t','baches_dosificacion','hora_sin','hora_cos','dia_sin','dia_cos']
cat=['linea','subfamilia','forma','salida_programada']
for c in ['masa_dosificada_t','baches_dosificacion','energia_peletizado_kWh','duracion_peletizado_min']:d[c]=pd.to_numeric(d[c],errors='coerce')
for c in ['inicio_registro','fin_peletizado']:d[c]=pd.to_datetime(d[c],errors='coerce')
d.fila_fuente=d.fila_fuente.astype(int)
h=d.inicio_registro.dt.hour+d.inicio_registro.dt.minute/60
for label,x,period in [('hora',h,24),('dia',d.inicio_registro.dt.dayofweek,7)]:
    d[label+'_sin']=np.sin(2*np.pi*x/period);d[label+'_cos']=np.cos(2*np.pi*x/period)
valid=np.isfinite(d[num+['energia_peletizado_kWh','duracion_peletizado_min']]).all(axis=1)&d.energia_peletizado_kWh.gt(0)&d.duracion_peletizado_min.gt(0)&d.masa_dosificada_t.gt(0)&d.baches_dosificacion.gt(0)&d[cat].notna().all(axis=1)&d[cat].ne('').all(axis=1)
excluded=[];sets={}
for part,end in [('DESARROLLO','2025-02-22'),('CALIBRACION','2025-03-06')]:
    mask=d.particion.eq(part);before=d.fin_peletizado.lt(pd.Timestamp(end))
    sets[part]=d[mask&valid&before].copy()
    for _,r in d[mask&~(valid&before)].iterrows():
        excluded.append({'fila_fuente':r.fila_fuente,'particion':part,'entradas_objetivo_validos':bool(valid.loc[r.name]),'final_antes_del_corte':bool(before.loc[r.name])})
tr=sets['DESARROLLO'];ca=sets['CALIBRACION']
assert tr.fin_peletizado.max()<pd.Timestamp('2025-02-22')<=ca.inicio_registro.min()
assert ca.fin_peletizado.max()<pd.Timestamp('2025-03-06') and not set(tr.fila_fuente)&set(ca.fila_fuente)
pipe=Pipeline([('pre',ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])),('ridge',Ridge(alpha=10))])
pipe.fit(tr[num+cat],tr.energia_peletizado_kWh)
pred=np.maximum(0,pipe.predict(ca[num+cat]));scores=abs(ca.energia_peletizado_kWh.to_numpy()-pred)
quantiles={}
for alpha in [.1,.05]:
    rank=math.ceil((len(scores)+1)*(1-alpha));assert rank<=len(scores)
    quantiles[str(round(1-alpha,2))]={'n':len(scores),'rango':rank,'semiamplitud_kWh':float(np.sort(scores)[rank-1])}
sec=tr.groupby('linea').apply(lambda g:g.energia_peletizado_kWh.sum()/g.masa_dosificada_t.sum(),include_groups=False).to_dict()
bundle={'pipeline':pipe,'numeric':num,'categorical':cat,'quantiles':quantiles,'baseline_sec':sec,'clip_prediction_at_zero':True,'protocol_sha256':protocol_hash,'scope':'RETROSPECTIVO_CONDICIONAL_NO_VALIDADO_EN_PRUEBA'}
joblib.dump(bundle,OUT/'modelo_calibrado.joblib')
loaded=joblib.load(OUT/'modelo_calibrado.joblib')
assert np.allclose(pred,np.maximum(0,loaded['pipeline'].predict(ca[num+cat])),rtol=0,atol=1e-12)
pd.DataFrame(excluded).to_csv(OUT/'registros_excluidos.csv',index=False)
q=ca[['fila_fuente','linea','inicio_registro','fin_peletizado','energia_peletizado_kWh']].copy();q['prediccion']=pred;q['error_absoluto_calibracion']=scores;q.to_csv(OUT/'residuos_calibracion.csv',index=False)
tr[['fila_fuente','inicio_registro','fin_peletizado']].to_csv(OUT/'registros_entrenamiento.csv',index=False)
unknown={c:int((~ca[c].isin(tr[c].unique())).sum()) for c in cat}
manifest={'n_entrenamiento':len(tr),'n_calibracion':len(ca),'n_excluidos':len(excluded),'excluidos_por_particion':pd.DataFrame(excluded).particion.value_counts().to_dict(),'calibracion_por_maquina':ca.linea.value_counts().to_dict(),'categorias_desconocidas_calibracion':unknown,'quantiles':quantiles,'baseline_sec':sec,'source_sha256':source_hash,'protocol_sha256':protocol_hash,'model_sha256':sha(OUT/'modelo_calibrado.joblib'),'script_sha256':sha(Path(__file__)),'versiones':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},'verificacion':{'fechas_y_particiones':True,'modelo_serializado_reproduce_predicciones':True,'fuente_intacta':sha(source)==source_hash,'prueba_evaluada':False},'estado':'CALIBRADO_RETROSPECTIVO_CONDICIONAL'}
assert manifest['verificacion']['fuente_intacta']
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(manifest,indent=2,ensure_ascii=False))
