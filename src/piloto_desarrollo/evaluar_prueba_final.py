"""Evaluacion final del modelo congelado, sin ajustes ni recalibracion."""
from pathlib import Path
import json,hashlib,math,datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'prueba_final_v1';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=ROOT/'calibracion_final_v1';man=json.loads((base/'manifest.json').read_text(encoding='utf8'))
source=ROOT/'depuracion_v1/registros_depurados.csv'
assert sha(source)==man['source_sha256']
assert sha(base/'modelo_calibrado.joblib')==man['model_sha256']
assert sha(base/'PROTOCOLO_FIJADO.md')==man['protocol_sha256']
if (OUT/'manifest.json').exists():raise RuntimeError('La prueba ya fue evaluada; consultar los resultados existentes.')
bundle=joblib.load(base/'modelo_calibrado.joblib')
d=pd.read_csv(source);d=d[d.particion.eq('PRUEBA')].copy();nc=len(d)
for c in ['inicio_registro','fin_peletizado']:d[c]=pd.to_datetime(d[c])
h=d.inicio_registro.dt.hour+d.inicio_registro.dt.minute/60
for label,x,period in [('hora',h,24),('dia',d.inicio_registro.dt.dayofweek,7)]:
    d[label+'_sin']=np.sin(2*np.pi*x/period);d[label+'_cos']=np.cos(2*np.pi*x/period)
num=bundle['numeric'];cat=bundle['categorical']
rules={'objetivo_invalido':~(np.isfinite(d.energia_peletizado_kWh)&d.energia_peletizado_kWh.gt(0)),
 'tiempo_invalido':~(d.duracion_peletizado_min.gt(0)&d.fin_peletizado.notna()&d.inicio_registro.ge(pd.Timestamp('2025-03-06'))),
 'entradas_invalidas':~(np.isfinite(d[num]).all(axis=1)&d[cat].notna().all(axis=1)&d[cat].ne('').all(axis=1)&d.masa_dosificada_t.gt(0)&d.baches_dosificacion.gt(0))}
bad=pd.DataFrame(rules,index=d.index);ex=d.loc[bad.any(axis=1),['fila_fuente']].join(bad);ex.to_csv(OUT/'exclusiones.csv',index=False)
d=d[~bad.any(axis=1)].copy()
assert d.linea.isin(bundle['baseline_sec']).all()
enc=bundle['pipeline'].named_steps['pre'].named_transformers_['cat']
for c,known in zip(cat,enc.categories_):d['desconocida_'+c]=~d[c].isin(known)
d['Ridge']=np.maximum(0,bundle['pipeline'].predict(d[num+cat]))
d['SEC']=d.linea.map(bundle['baseline_sec'])*d.masa_dosificada_t
d['semana']=d.inicio_registro.dt.to_period('W-SUN').dt.start_time.astype(str)
d['fecha']=d.inicio_registro.dt.normalize()
for model in ['Ridge','SEC']:
    d['error_'+model]=d[model]-d.energia_peletizado_kWh;d['abs_'+model]=abs(d['error_'+model])
for level,v in bundle['quantiles'].items():
    tag=str(round(float(level)*100));alpha=1-float(level);q=v['semiamplitud_kWh']
    d['L'+tag]=np.maximum(0,d.Ridge-q);d['U'+tag]=d.Ridge+q
    d['cubierto'+tag]=d.energia_peletizado_kWh.between(d['L'+tag],d['U'+tag])
    d['ancho'+tag]=d['U'+tag]-d['L'+tag]
    d['score'+tag]=d['ancho'+tag]+2/alpha*((d['L'+tag]-d.energia_peletizado_kWh).clip(lower=0)+(d.energia_peletizado_kWh-d['U'+tag]).clip(lower=0))
d['alerta_alta']=d.energia_peletizado_kWh.gt(d.U95);d['alerta_baja']=d.energia_peletizado_kWh.lt(d.L95)
def point(g):
    return [{'modelo':m,'n':len(g),'MAE':g['abs_'+m].mean(),'RMSE':np.sqrt((g['error_'+m]**2).mean()),'sesgo':g['error_'+m].mean(),'WAPE':g['abs_'+m].sum()/g.energia_peletizado_kWh.sum()} for m in ['Ridge','SEC']]
def interval(g):return [{'nominal':k,'n':len(g),'cobertura':g['cubierto'+k].mean(),'ancho_medio':g['ancho'+k].mean(),'interval_score':g['score'+k].mean()} for k in ['90','95']]
pg=pd.DataFrame(point(d));ig=pd.DataFrame(interval(d));pg.to_csv(OUT/'metricas_puntuales.csv',index=False);ig.to_csv(OUT/'metricas_intervalos.csv',index=False)
for group in ['linea','semana','familia']:
    pr=[];ir=[]
    for key,g in d.groupby(group):
        pr.extend([{group:key,**r} for r in point(g)]);ir.extend([{group:key,**r} for r in interval(g)])
    pd.DataFrame(pr).to_csv(OUT/f'puntuales_por_{group}.csv',index=False);pd.DataFrame(ir).to_csv(OUT/f'intervalos_por_{group}.csv',index=False)
pd.DataFrame(point(d[d.duracion_peletizado_min.le(1440)])).to_csv(OUT/'sensibilidad_duracion_24h.csv',index=False)
daily=d.groupby('fecha').agg(n=('fila_fuente','size'),ridge=('abs_Ridge','sum'),sec=('abs_SEC','sum'),cover90=('cubierto90','sum'),cover95=('cubierto95','sum'))
rng=np.random.default_rng(20260920);nd=len(daily);idx=((rng.integers(0,nd,size=(3000,math.ceil(nd/3),1))+np.arange(3))%nd).reshape(3000,-1)[:,:nd]
den=daily.n.to_numpy()[idx].sum(axis=1);delta=(daily.sec-daily.ridge).to_numpy()[idx].sum(axis=1)/den
ci={'reduccion_MAE_kWh':float(d.abs_SEC.mean()-d.abs_Ridge.mean()),'IC95_reduccion':np.quantile(delta,[.025,.975]).tolist(),'reduccion_relativa_pct':float(100*(1-d.abs_Ridge.mean()/d.abs_SEC.mean()))}
for k in ['90','95']:ci['IC95_cobertura_'+k]=np.quantile(daily['cover'+k].to_numpy()[idx].sum(axis=1)/den,[.025,.975]).tolist()
keep=['fila_fuente','linea','familia','inicio_registro','fin_peletizado','duracion_peletizado_min','energia_peletizado_kWh','Ridge','SEC','L90','U90','L95','U95','alerta_alta','alerta_baja']+['desconocida_'+c for c in cat]
d[keep].to_csv(OUT/'predicciones_prueba.csv',index=False)
d.loc[d.alerta_alta|d.alerta_baja,keep].to_csv(OUT/'alertas_prueba.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(10,4.5));pg.plot.bar(x='modelo',y='MAE',ax=axes[0],legend=False,color=['#315a83','#9c8a70']);axes[0].set(ylabel='MAE (kWh/registro)',xlabel='');axes[0].tick_params(axis='x',rotation=0)
weeks=pd.read_csv(OUT/'intervalos_por_semana.csv')
for nominal,g in weeks.groupby('nominal'):axes[1].plot(g.semana,100*g.cobertura,'o-',label=f'{nominal}% nominal')
axes[1].axhline(95,color='gray',ls='--');axes[1].axhline(90,color='gray',ls=':');axes[1].set(ylabel='Cobertura observada (%)',xlabel='Inicio de semana');axes[1].tick_params(axis='x',rotation=25);axes[1].legend();fig.suptitle('Prueba final: modelo e intervalos congelados');fig.tight_layout();fig.savefig(OUT/'resultado_prueba.png',dpi=170);plt.close(fig)
assert np.isfinite(d[['Ridge','SEC','L90','U90','L95','U95']]).all().all()
assert d.fila_fuente.is_unique and (d.L95<=d.L90).all() and (d.U95>=d.U90).all()
assert sha(source)==man['source_sha256'] and sha(base/'modelo_calibrado.joblib')==man['model_sha256']
summary={'fecha_ejecucion_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidatos':nc,'evaluados':len(d),'excluidos':len(ex),'inicio_min':str(d.inicio_registro.min()),'inicio_max':str(d.inicio_registro.max()),'fin_max':str(d.fin_peletizado.max()),'dias_observados':nd,'metricas_puntuales':point(d),'metricas_intervalos':interval(d),'bootstrap':ci,'alertas_altas':int(d.alerta_alta.sum()),'alertas_bajas':int(d.alerta_baja.sum()),'duracion_mayor_24h':int(d.duracion_peletizado_min.gt(1440).sum()),'categorias_desconocidas':{c:int(d['desconocida_'+c].sum()) for c in cat},'modelo_sha256':man['model_sha256'],'protocolo_sha256':man['protocol_sha256'],'fuente_sha256':man['source_sha256'],'script_sha256':sha(Path(__file__)),'reajustado':False,'recalibrado':False,'estado':'PRUEBA_EVALUADA_RETROSPECTIVA_CONDICIONAL'}
(OUT/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False,indent=2))
