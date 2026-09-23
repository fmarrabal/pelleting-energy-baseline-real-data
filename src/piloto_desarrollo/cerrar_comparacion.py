from pathlib import Path
import json,hashlib,math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'cierre_estudio';OUT.mkdir(exist_ok=True)
paths=[ROOT/'prueba_final_v1/predicciones_prueba.csv',ROOT/'calibracion_final_v1/residuos_calibracion.csv',ROOT/'calibracion_final_v1/modelo_calibrado.joblib']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
config={'ventana':200,'niveles':[90,95],'actualizacion':'Solo residuos con fin estrictamente anterior al inicio de la prediccion; orden fin y fila','modelo':'Ridge congelado de calibracion_final_v1','alcance':'Comparacion retrospectiva posterior a conocer la prueba original, no validacion independiente','fuentes_sha256':hashes}
(OUT/'configuracion.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
d=pd.read_csv(paths[0]);ca=pd.read_csv(paths[1])
for g in [d,ca]:
    for c in ['inicio_registro','fin_peletizado']:g[c]=pd.to_datetime(g[c])
assert len(d)==798 and len(ca)==380 and ca.fin_peletizado.max()<d.inicio_registro.min()
ca['score']=ca.error_absoluto_calibracion;d['score']=abs(d.energia_peletizado_kWh-d.Ridge)
history=pd.concat([ca[['fila_fuente','fin_peletizado','score']],d[['fila_fuente','fin_peletizado','score']]]).sort_values(['fin_peletizado','fila_fuente'])
rows=[];audit=[]
for _,r in d.iterrows():
    prior=history[history.fin_peletizado.lt(r.inicio_registro)].tail(200)
    assert len(prior)==200 and r.fila_fuente not in set(prior.fila_fuente)
    audit.append({'fila_fuente':r.fila_fuente,'max_fin_residuo':str(prior.fin_peletizado.max()),'filas_residuos':','.join(map(str,prior.fila_fuente))})
    for level in [90,95]:
        q=np.sort(prior.score.to_numpy())[math.ceil(201*level/100)-1]
        for method,lo,hi in [('fijo',r[f'L{level}'],r[f'U{level}']),('actualizado_200',max(0,r.Ridge-q),r.Ridge+q)]:
            y=r.energia_peletizado_kWh;alpha=1-level/100
            rows.append({'fila_fuente':r.fila_fuente,'linea':r.linea,'inicio':r.inicio_registro,'fin':r.fin_peletizado,'observado':y,'prediccion':r.Ridge,'metodo':method,'nominal':level,'inferior':lo,'superior':hi,'cubierto':lo<=y<=hi,'ancho':hi-lo,'interval_score':hi-lo+2/alpha*(max(lo-y,0)+max(y-hi,0)),'alerta_alta':y>hi,'alerta_baja':y<lo})
p=pd.DataFrame(rows);p['semana']=p.inicio.dt.to_period('W-SUN').dt.start_time.astype(str)
def metric(g):return pd.Series({'n':len(g),'cobertura':g.cubierto.mean(),'ancho_medio_kWh':g.ancho.mean(),'interval_score':g.interval_score.mean(),'alertas_altas':g.alerta_alta.sum(),'alertas_bajas':g.alerta_baja.sum()})
m=p.groupby(['metodo','nominal']).apply(metric,include_groups=False);m.to_csv(OUT/'comparacion_final.csv')
for c in ['linea','semana']:p.groupby(['metodo','nominal',c]).apply(metric,include_groups=False).to_csv(OUT/f'comparacion_por_{c}.csv')
p.to_csv(OUT/'intervalos_finales.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'trazabilidad_actualizacion.csv',index=False)
p[p.metodo.eq('actualizado_200')&p.nominal.eq(95)&(p.alerta_alta|p.alerta_baja)].to_csv(OUT/'alertas_actualizadas_95.csv',index=False)
assert all(sha(path)==hashes[str(path.relative_to(ROOT))] for path in paths)
assert np.isfinite(p[['inferior','superior']]).all().all()
summary={'n':len(d),'metricas':m.reset_index().to_dict('records'),'modelo_reajustado':False,'validacion_independiente_adaptacion':False,'control_etiquetas_futuras':True,'fuentes_intactas':True,'script_sha256':sha(Path(__file__))}
(OUT/'verificacion.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8');print(m.to_string())
