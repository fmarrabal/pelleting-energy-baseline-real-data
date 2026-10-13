"""Experimento exploratorio posterior a prueba: solo datos de desarrollo."""
from pathlib import Path
import json,hashlib,math,datetime
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'adaptacion_desarrollo_v1';OUT.mkdir(exist_ok=True)
source=ROOT/'depuracion_v1/registros_depurados.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
initial=sha(source)
cfg={'ventana_residuos':200,'metodos':['fijo','recientes_200'],'niveles':[.9,.95],'folds':['2025-01-25','2025-02-01','2025-02-08','2025-02-15'],'alpha_ridge':10,'regla':'Residuo disponible solo si fin_peletizado < inicio_registro predicho. Modelo fijo dentro del bloque. Orden por finalizacion y fila para empates.','alcance':'Exploratorio de desarrollo posterior a conocer limitaciones de prueba. No es nueva validacion independiente.','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/'configuracion.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf8')
d=pd.read_csv(source);d=d[d.particion.eq('DESARROLLO')].copy()
for c in ['inicio_registro','fin_peletizado']:d[c]=pd.to_datetime(d[c])
h=d.inicio_registro.dt.hour+d.inicio_registro.dt.minute/60
for label,x,period in [('hora',h,24),('dia',d.inicio_registro.dt.dayofweek,7)]:
    d[label+'_sin']=np.sin(2*np.pi*x/period);d[label+'_cos']=np.cos(2*np.pi*x/period)
num=['masa_dosificada_t','baches_dosificacion','hora_sin','hora_cos','dia_sin','dia_cos'];cat=['linea','subfamilia','forma','salida_programada']
d=d[d.mascara_objetivo_peletizado.eq(1)&d.apto_temporal_peletizado.eq(1)&d.masa_dosificada_t.gt(0)&d[num+cat].notna().all(axis=1)]
out=[];lineage=[]
for start in cfg['folds']:
    a=pd.Timestamp(start);c=a-pd.Timedelta('7 days');b=a+pd.Timedelta('7 days')
    tr=d[d.inicio_registro.lt(c)&d.fin_peletizado.lt(c)]
    ca=d[d.inicio_registro.ge(c)&d.inicio_registro.lt(a)&d.fin_peletizado.lt(a)]
    va=d[d.inicio_registro.ge(a)&d.inicio_registro.lt(b)&d.fin_peletizado.lt(b)].sort_values(['inicio_registro','fila_fuente'])
    enc=ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
    model=Ridge(alpha=10).fit(enc.fit_transform(tr),tr.energia_peletizado_kWh)
    ca=ca.copy();ca['pred']=np.maximum(0,model.predict(enc.transform(ca)))
    va=va.copy();va['pred']=np.maximum(0,model.predict(enc.transform(va)))
    history=pd.concat([ca,va]).sort_values(['fin_peletizado','fila_fuente'])
    history['score']=abs(history.energia_peletizado_kWh-history.pred)
    fixed=abs(ca.energia_peletizado_kWh-ca.pred).to_numpy()
    for _,r in va.iterrows():
        recent=history[history.fin_peletizado.lt(r.inicio_registro)].tail(200)
        assert r.fila_fuente not in set(recent.fila_fuente) and recent.fin_peletizado.max()<r.inicio_registro
        lineage.append({'fila_fuente':r.fila_fuente,'fold':start,'max_fin_residuo':str(recent.fin_peletizado.max()),'n_residuos':len(recent),'filas_residuos':','.join(map(str,recent.fila_fuente))})
        for method,s in [('fijo',fixed),('recientes_200',recent.score.to_numpy())]:
            for level in cfg['niveles']:
                rank=math.ceil((len(s)+1)*level);assert rank<=len(s)
                q=np.sort(s)[rank-1];lo=max(0,r.pred-q);hi=r.pred+q;y=r.energia_peletizado_kWh
                out.append({'fila_fuente':r.fila_fuente,'linea':r.linea,'inicio':r.inicio_registro,'fold':start,'metodo':method,'nominal':level,'pred':r.pred,'inferior':lo,'superior':hi,'cubierto':lo<=y<=hi,'ancho':hi-lo,'score':hi-lo+2/(1-level)*(max(lo-y,0)+max(y-hi,0))})
p=pd.DataFrame(out)
def metrics(g):return pd.Series({'n':len(g),'cobertura':g.cubierto.mean(),'ancho':g.ancho.mean(),'interval_score':g.score.mean()})
m=p.groupby(['metodo','nominal']).apply(metrics,include_groups=False);m.to_csv(OUT/'metricas.csv')
for group in ['linea','fold']:p.groupby(['metodo','nominal',group]).apply(metrics,include_groups=False).to_csv(OUT/f'metricas_por_{group}.csv')
p.to_csv(OUT/'predicciones.csv',index=False);pd.DataFrame(lineage).to_csv(OUT/'trazabilidad_residuos.csv',index=False)
old=pd.read_csv(ROOT/'incertidumbre_desarrollo_v1/intervalos_y_alertas.csv')
old=old[old.escenario.eq('plan_provisional')&old.metodo.eq('absoluto_global')]
check=p[p.metodo.eq('fijo')].merge(old,on=['fila_fuente','nominal'],suffixes=('_new','_old'),validate='one_to_one')
assert len(check)==1600
for c in ['pred','inferior','superior']:assert np.allclose(check[c+'_new'],check[c+'_old'],rtol=0,atol=1e-9)
assert sha(source)==initial
(OUT/'verificacion.json').write_text(json.dumps({'n_validacion':800,'n_predicciones':len(p),'referencia_fija_reproducida':True,'control_disponibilidad_residuos':True,'fuente_intacta':True,'prueba_reutilizada_para_metricas':False,'source_sha256':initial,'script_sha256':sha(Path(__file__))},indent=2),encoding='utf8')
print(m.to_string())
