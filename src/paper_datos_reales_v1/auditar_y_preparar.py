"""Read-only audit of industrial sources and archived results. No model training.

All outputs are confined to this new manuscript directory. Original files,
calibration, test predictions and serialized models are never overwritten.
"""
from pathlib import Path
import ast, hashlib, json, shutil, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / 'data'
DATA.mkdir(exist_ok=True)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

sources = [
 'Datos Planta Peletizado.xlsx', 'depuracion_v1/registros_depurados.csv',
 'depuracion_v1/resumen.json', 'calibracion_final_v1/modelo_calibrado.joblib',
 'calibracion_final_v1/registros_entrenamiento.csv',
 'calibracion_final_v1/registros_excluidos.csv',
 'calibracion_final_v1/residuos_calibracion.csv',
 'prueba_final_v1/predicciones_prueba.csv', 'prueba_final_v1/manifest.json',
 'analisis_desarrollo_v2/predicciones_desarrollo.csv',
 'analisis_desarrollo_v2/metricas_globales.csv',
 'analisis_desarrollo_v2/configuracion.json',
 'hipotesis_exploratorias_v1/predicciones.csv',
 'hipotesis_exploratorias_v1/metricas.csv',
 'hipotesis_exploratorias_v1/contrastes_pareados.csv',
 'hipotesis_exploratorias_v1/donantes_sesgo.csv',
 'hipotesis_exploratorias_v1/PROTOCOLO.json',
 'cierre_estudio/intervalos_finales.csv',
 'cierre_estudio/comparacion_final.csv',
 'cierre_estudio/comparacion_por_linea.csv',
 'cierre_estudio/comparacion_por_semana.csv',
 'cierre_estudio/trazabilidad_actualizacion.csv',
 'paper_latex_v3/references.bib',
]
before = {p: sha(ROOT / p) for p in sources}
assert before[sources[0]] == 'd5968030da3e704e7505e9512151824d9384ae5e68259b71d01e1e9ee7647395'
assert before['calibracion_final_v1/modelo_calibrado.joblib'] == 'b355bf285ddda6aa6c06d8d55136fbddcc598358105fa0155d1d22a9948117e2'
w = openpyxl.load_workbook(ROOT / sources[0], read_only=True, data_only=True)
rows = iter(w.active.values)
headers = next(rows)
raw = pd.DataFrame(rows, columns=[v or f'blank_{i}' for i, v in enumerate(headers)])
w.close()
d = pd.read_csv(ROOT / sources[1], encoding='utf-8-sig')
for c in d:
    if c.startswith(('inicio_', 'fin_')):
        d[c] = pd.to_datetime(d[c], format='mixed')
tree = ast.parse((ROOT / 'depuracion_v1/depurar.py').read_text(encoding='utf-8-sig'))
mapping = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'mapping' for t in n.targets))
assert len(raw) == len(d) == 2745 and d.fila_fuente.tolist() == list(range(2, 2747))
comparisons = []
for original, clean in mapping.items():
    a, b = raw[original], d[clean]
    if clean.startswith(('inicio_', 'fin_')):
        a = pd.to_datetime(a, format='mixed')
        ok = (a.isna() & b.isna()) | a.eq(b)
    elif clean in ['salida_programada','linea','familia','subfamilia','forma','empaque_final','unidad_inventario','grupo_inventario']:
        a = a.map(lambda v: v.strip() if isinstance(v, str) else v)
        ok = (a.isna() & b.isna()) | a.eq(b)
    else:
        ok = np.isclose(pd.to_numeric(a), pd.to_numeric(b), rtol=1e-12, atol=1e-9, equal_nan=True)
    assert np.all(ok), f'Original mismatch: {original}'
    comparisons.append({'original':original,'columna_analitica':clean,'n':len(d),'coinciden':int(np.sum(ok))})
pd.DataFrame(comparisons).to_csv(DATA/'fidelidad_32_variables.csv', index=False, encoding='utf-8-sig')

# An explicit dictionary distinguishes recorded values from origin availability.
features = {'masa_dosificada_kg','baches_dosificacion','linea','subfamilia','forma','salida_programada'}
cats = {'linea','subfamilia','forma','salida_programada','familia','empaque_final','unidad_inventario','grupo_inventario'}
availability=[]
for original, clean in mapping.items():
    if clean == 'energia_peletizado_kWh':
        role, status, rule = 'objetivo','posterior','No imputar. Disponibilidad >= fin peletizado; falta sello de publicacion.'
    elif clean in features:
        role, status, rule = 'entrada_modelo_original','no_acreditada_antes_del_origen','Admitida en linea base retrospectiva. Para pronostico se requiere version del plan y sello previo.'
    elif clean.startswith('inicio_'):
        role, status, rule = 'cronologia','observacion_del_evento','Origen historico = minimo de inicios disponibles. No usar inicio futuro de otra etapa como predictor.'
    elif clean.startswith('fin_'):
        role, status, rule = 'cronologia_etiquetas','posterior','Solo para purga y llegada de etiquetas. Fin no prueba disponibilidad inmediata.'
    elif clean in cats:
        role, status, rule = 'contexto_no_usado','no_acreditada_antes_del_origen','No confundir clasificacion final con plan conocido.'
    else:
        role, status, rule = 'resumen_operacional_no_usado','posterior_o_no_documentada','Excluir de prediccion previa; no hay trayectoria ni sello de captura.'
    availability.append({'campo_excel':original,'campo_analitico':clean,'rol':role,
                         'disponibilidad':status,'regla':rule,'ausentes_original':int(raw[original].isna().sum())})
pd.DataFrame(availability).to_csv(DATA/'disponibilidad_variables.csv',index=False,encoding='utf-8-sig')

tr = pd.read_csv(ROOT/'calibracion_final_v1/registros_entrenamiento.csv')
ca = pd.read_csv(ROOT/'calibracion_final_v1/residuos_calibracion.csv')
te = pd.read_csv(ROOT/'prueba_final_v1/predicciones_prueba.csv')
ex = pd.read_csv(ROOT/'calibracion_final_v1/registros_excluidos.csv')
assert [len(tr),len(ca),len(te),len(ex)] == [1542,380,798,25]
sets = [set(x.fila_fuente) for x in [tr,ca,te,ex]]
assert len(set.union(*sets)) == 2745 and sum(map(len,sets)) == 2745
idx=d.set_index('fila_fuente')
for part, rows_, end in [('DESARROLLO',tr,'2025-02-22'),('CALIBRACION',ca,'2025-03-06')]:
    subset=idx.loc[rows_.fila_fuente]
    assert subset.particion.eq(part).all() and subset.fin_peletizado.lt(pd.Timestamp(end)).all()
assert idx.loc[te.fila_fuente].inicio_registro.ge(pd.Timestamp('2025-03-06')).all()
assert np.allclose(idx.loc[te.fila_fuente].energia_peletizado_kWh,te.energia_peletizado_kWh,rtol=0,atol=1e-10)
flow=[]
for part, rows_ in [('DESARROLLO',tr),('CALIBRACION',ca),('PRUEBA',te)]:
    q=d[d.particion.eq(part)]
    flow.append({'particion':part,'originales':len(q),'incluidos':len(rows_),'excluidos':len(q)-len(rows_),
                 'min_origen':str(q.inicio_registro.min()),'max_origen':str(q.inicio_registro.max())})
pd.DataFrame(flow).to_csv(DATA/'flujo_muestra.csv',index=False)

# Temporal diagnostics: overlap is not automatically a sensor error or duplicate.
overlaps=[]
for line,g in d.groupby('linea'):
    g=g.sort_values(['inicio_peletizado','fila_fuente'])
    prior_end=g.fin_peletizado.cummax().shift(1)
    valid=g.fin_peletizado.gt(g.inicio_peletizado)
    overlap=valid & g.inicio_peletizado.lt(prior_end)
    gaps=g.inicio_peletizado.diff().dt.total_seconds()/60
    overlaps.append({'linea':line,'n':len(g),'n_objetivos_positivos':int(g.energia_peletizado_kWh.gt(0).sum()),
      'n_solapa_intervalo_anterior':int(overlap.sum()),'n_duracion_mayor24h':int(g.duracion_peletizado_min.gt(1440).sum()),
      'mediana_min_entre_inicios':float(gaps.median()),'duracion_mediana_min':float(g.duracion_peletizado_min.median())})
pd.DataFrame(overlaps).to_csv(DATA/'estructura_temporal.csv',index=False)

devp=pd.read_csv(ROOT/'analisis_desarrollo_v2/predicciones_desarrollo.csv')
hp=pd.read_csv(ROOT/'hipotesis_exploratorias_v1/predicciones.csv')
devm=pd.read_csv(ROOT/'analisis_desarrollo_v2/metricas_globales.csv').set_index('modelo')
hm=pd.read_csv(ROOT/'hipotesis_exploratorias_v1/metricas.csv').set_index('modelo')
checks=[]
def metrics(y,p):
    e=p-y
    return dict(MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e**2))),
                sesgo=float(e.mean()),WAPE=float(np.abs(e).sum()/y.sum()))
for name,g in devp.groupby('modelo'):
    assert g.fila_fuente.is_unique and len(g)==800
    for k,v in metrics(g.energia_peletizado_kWh,g.prediccion_kWh).items():
        stored=devm.loc[name,{'MAE':'MAE_kWh','RMSE':'RMSE_kWh','sesgo':'sesgo_kWh','WAPE':'WAPE'}[k]]
        assert np.isclose(v,stored,rtol=0,atol=1e-10)
        checks.append({'panel':'desarrollo','modelo':name,'metrica':k,'valor':v,'diferencia':v-stored})
for name,g in hp.groupby('modelo'):
    assert g.fila_fuente.is_unique and len(g)==800
    for k,v in metrics(g.energia_peletizado_kWh,g.pred).items():
        assert np.isclose(v,hm.loc[name,k],rtol=0,atol=1e-10)
        checks.append({'panel':'hipotesis','modelo':name,'metrica':k,'valor':v,'diferencia':v-hm.loc[name,k]})
manifest=json.loads((ROOT/'prueba_final_v1/manifest.json').read_text(encoding='utf8'))
for row in manifest['metricas_puntuales']:
    for k,v in metrics(te.energia_peletizado_kWh,te[row['modelo']]).items():
        assert np.isclose(v,row[k],rtol=0,atol=1e-10)
        checks.append({'panel':'prueba_original','modelo':row['modelo'],'metrica':k,'valor':v,'diferencia':v-row[k]})
pa=devp[devp.modelo.eq('Ridge')].set_index('fila_fuente').prediccion_kWh.sort_index()
pb=hp[hp.modelo.eq('Ridge')].set_index('fila_fuente').pred.sort_index()
assert np.allclose(pa,pb,rtol=0,atol=1e-8)
assert max(pd.to_datetime(devp.inicio_registro,format='mixed'))<pd.Timestamp('2025-02-22')
pd.DataFrame(checks).to_csv(DATA/'metricas_reconciliadas.csv',index=False)

# Independently verify EVERY recorded donor, not only the logged maximum.
nd_h=nd_u=0
for r in pd.read_csv(ROOT/'hipotesis_exploratorias_v1/donantes_sesgo.csv').itertuples():
    if pd.isna(r.filas_donantes):
        assert r.n_donantes==0
        continue
    ids=[int(x) for x in str(r.filas_donantes).split(',')]
    q=idx.loc[ids]; target=idx.loc[r.fila_fuente]
    assert len(ids)==r.n_donantes and len(ids)==len(set(ids)) and r.fila_fuente not in ids
    assert q.fin_peletizado.lt(target.inicio_registro).all() and q.linea.eq(target.linea).all()
    assert q.inicio_registro.ge(pd.Timestamp(r.fold)).all()
    assert q.particion.eq('DESARROLLO').all()
    nd_h+=len(ids)
for r in pd.read_csv(ROOT/'cierre_estudio/trazabilidad_actualizacion.csv').itertuples():
    ids=[int(x) for x in r.filas_residuos.split(',')]
    assert len(ids)==200 and len(ids)==len(set(ids)) and r.fila_fuente not in ids
    q=idx.loc[ids]
    assert q.fin_peletizado.lt(idx.loc[r.fila_fuente,'inicio_registro']).all()
    assert set(ids).issubset(sets[1]|sets[2])
    nd_u+=len(ids)

iv=pd.read_csv(ROOT/'cierre_estudio/intervalos_finales.csv')
im=pd.read_csv(ROOT/'cierre_estudio/comparacion_final.csv').set_index(['metodo','nominal'])
for key,g in iv.groupby(['metodo','nominal']):
    assert len(g)==798 and g.fila_fuente.is_unique
    alpha=1-key[1]/100
    inside=(g.observado>=g.inferior)&(g.observado<=g.superior)
    score=g.superior-g.inferior+2/alpha*np.maximum(g.inferior-g.observado,0)+2/alpha*np.maximum(g.observado-g.superior,0)
    assert np.allclose(g.prediccion,te.set_index('fila_fuente').loc[g.fila_fuente,'Ridge'],atol=1e-10)
    for k,v in [('cobertura',inside.mean()),('ancho_medio_kWh',(g.superior-g.inferior).mean()),('interval_score',score.mean())]:
        assert np.isclose(v,im.loc[key,k],rtol=0,atol=1e-9)

# Portable plotting inputs, with no invented observations.
copies={
 'analisis_desarrollo_v2/metricas_globales.csv':'modelos_desarrollo.csv',
 'analisis_desarrollo_v2/predicciones_desarrollo.csv':'predicciones_desarrollo.csv',
 'hipotesis_exploratorias_v1/metricas.csv':'hipotesis_metricas.csv',
 'hipotesis_exploratorias_v1/contrastes_pareados.csv':'hipotesis_contrastes.csv',
 'hipotesis_exploratorias_v1/predicciones.csv':'hipotesis_predicciones.csv',
 'prueba_final_v1/predicciones_prueba.csv':'prueba_original.csv',
 'prueba_final_v1/manifest.json':'prueba_original_manifest.json',
 'cierre_estudio/comparacion_final.csv':'intervalos_global.csv',
 'cierre_estudio/comparacion_por_linea.csv':'intervalos_linea.csv',
 'cierre_estudio/comparacion_por_semana.csv':'intervalos_semana.csv',
 'cierre_estudio/intervalos_finales.csv':'intervalos_registro.csv',
}
for source,dest in copies.items(): shutil.copyfile(ROOT/source,DATA/dest)
d[['fila_fuente','linea','familia','subfamilia','masa_dosificada_t','energia_peletizado_kWh',
   'inicio_registro','inicio_peletizado','fin_peletizado','duracion_peletizado_min','particion']].to_csv(DATA/'registros_reales_figuras.csv',index=False)
audit={
 'fecha_utc':datetime.now(timezone.utc).isoformat(),'fuente':'Datos Planta Peletizado.xlsx',
 'registros':2745,'variables_originales':32,'celdas_originales_cotejadas':2745*32,
 'unidad':'registro operacional; correspondencia con orden/lote no confirmada',
 'origen_historico':'min(inicio_molienda, inicio_mezclado, inicio_peletizado)',
 'origen_min':str(d.inicio_registro.min()),'origen_max':str(d.inicio_registro.max()),
 'ultima_finalizacion_peletizado':str(d.fin_peletizado.max()),
 'n_energia_no_positiva_o_ausente':int((~d.energia_peletizado_kWh.gt(0)).sum()),
 'n_fin_molienda_tras_inicio_peletizado':int((d.fin_molienda>d.inicio_peletizado).sum()),
 'n_fin_mezclado_tras_inicio_peletizado':int((d.fin_mezclado>d.inicio_peletizado).sum()),
 'flujo':flow,'estructura_temporal':overlaps,'metricas_puntuales_verificadas':len(checks),
 'relaciones_donante_H3_verificadas':nd_h,'relaciones_donante_intervalos_verificadas':nd_u,
 'entrenamientos_nuevos':0,'reevaluaciones_modelo_prueba':0,
 'comparacion_nuevos_modelos_sobre_prueba':False,
 'lectura_resultados_ya_archivados':True,'disponibilidad_predictiva_confirmada':False,
 'origen_sintetico_de_datos_industriales':False,
 'sha256_antes':before,'fuentes_preservadas':all(sha(ROOT/p)==s for p,s in before.items()),
 'versiones':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
}
assert audit['fuentes_preservadas']
(HERE/'AUDITORIA.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({k:v for k,v in audit.items() if k not in ['sha256_antes']},indent=2,ensure_ascii=False))
