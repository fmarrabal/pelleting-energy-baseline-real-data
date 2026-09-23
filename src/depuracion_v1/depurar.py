from pathlib import Path
import pandas as pd
import numpy as np
import openpyxl, json, hashlib

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
src=ROOT/'Datos Planta Peletizado.xlsx'
initial_hash=hashlib.file_digest(src.open('rb'),'sha256').hexdigest()
w=openpyxl.load_workbook(src,read_only=True,data_only=True); rows=iter(w.active.values); headers=next(rows)
raw=pd.DataFrame(rows,columns=[x if x else f'separador_{i}' for i,x in enumerate(headers)])
mapping={
'DES_PRODUCTO_TIPO':'salida_programada','N_BACHES_DOS':'baches_dosificacion','PESO DOS (kg)':'masa_dosificada_kg','PELLT':'linea',
'CANTIDAD_BACHES_PEL':'baches_peletizado','CANT PELETIZADA':'produccion_t_asumida','KWH PELET':'energia_peletizado_kWh',
'PRESION VAPOR (psi)':'presion_vapor_psig','VAPOR (Lb)':'masa_vapor_lb','TEMP_ACONDICIONADOR':'temperatura_acond_C','AMPERAJE_PEL':'corriente_peletizado_A',
'FEC_INI_PEL_TS':'inicio_peletizado','FEC_FIN_PEL_TS':'fin_peletizado','N_BACHES_MOL':'baches_molienda','FEC_INI_MOL':'inicio_molienda','FEC_FIN_MOL':'fin_molienda','KWH MOL':'energia_molienda_kWh',
'TEMP_DEVANADO':'temperatura_devanado_C','TEMP_ROD_DELANTERO':'temperatura_rod_del_C','TEMP_ROD_TRASERO':'temperatura_rod_tras_C','AMPERAJE_MOL':'corriente_molienda_A',
'N_BACHES_MEZ':'baches_mezclado','FEC_INI_MEZ':'inicio_mezclado','FEC_FIN_MEZ':'fin_mezclado','KWH MEZ':'energia_mezclado_kWh','AMPERAJE_MEZ':'corriente_mezclado_A',
'UNIDADINVENTARIO':'unidad_inventario','AGRUPACION_INVENTARIO':'grupo_inventario','LINEA':'familia','SUBLINEA':'subfamilia','PRESENTACION':'forma','EMPAQUE':'empaque_final'}
d=raw[list(mapping)].rename(columns=mapping).copy()
d.insert(0,'fila_fuente',np.arange(len(d))+2)
issues=[];logs=[]
def issue(i,var,rule,action='Conservar y revisar'):
    issues.append({'fila_fuente':int(d.loc[i,'fila_fuente']),'variable':var,'regla':rule,'accion':action})
for c in ['salida_programada','linea','familia','subfamilia','forma','empaque_final','unidad_inventario','grupo_inventario']:
    for i,v in d[c].items():
        if isinstance(v,str) and v!=v.strip():issue(i,c,'ESPACIOS_EXTREMOS','Eliminar espacios extremos')
    d[c]=d[c].map(lambda x:x.strip() if isinstance(x,str) else x)
for c in [c for c in d if c.startswith(('inicio_','fin_'))]:d[c]=pd.to_datetime(d[c],errors='raise')
for i in d.index[d.drop(columns='fila_fuente').duplicated(keep=False)]:issue(i,'registro','DUPLICADO_EXACTO','Conservar con bandera; no eliminar sin lote_id')
d['masa_dosificada_t']=d.masa_dosificada_kg/1000
d['inicio_registro']=d[[f'inicio_{s}' for s in ['molienda','mezclado','peletizado']]].min(axis=1)
d['fin_registro']=d[[f'fin_{s}' for s in ['molienda','mezclado','peletizado']]].max(axis=1)
d['duracion_total_min']=(d.fin_registro-d.inicio_registro).dt.total_seconds()/60
for i in d.index[d.duracion_total_min.gt(1440)]:issue(i,'duracion_total_min','DURACION_TOTAL_MAYOR_24H')
d['particion']=np.select([d.inicio_registro<pd.Timestamp('2025-02-22'),d.inicio_registro<pd.Timestamp('2025-03-06')],['DESARROLLO','CALIBRACION'],default='PRUEBA')
for s in ['molienda','mezclado','peletizado']:
    dur=(d[f'fin_{s}']-d[f'inicio_{s}']).dt.total_seconds()/60
    d[f'duracion_{s}_min']=dur
    energy=d[f'energia_{s}_kWh']
    d[f'mascara_objetivo_{s}']=(energy.notna()&np.isfinite(energy)&energy.gt(0)).astype(int)
    for i in d.index[~d[f'mascara_objetivo_{s}'].astype(bool)]:issue(i,f'energia_{s}_kWh','OBJETIVO_NO_POSITIVO_O_AUSENTE','No imputar; excluir solo esta salida de perdida y metricas')
    for i in d.index[dur.isna()|dur.le(0)]:issue(i,f'duracion_{s}_min','TIEMPOS_INCOMPLETOS_O_NO_POSITIVOS','No reconstruir fechas; duracion no utilizable')
    for i in d.index[dur.gt(1440)]:issue(i,f'duracion_{s}_min','DURACION_MAYOR_24H')
    d[f'apto_temporal_{s}']=(dur.gt(0)&d[f'mascara_objetivo_{s}'].eq(1)).astype(int)
    for cutoff in ['2025-02-22','2025-03-06']:
        cross=d.inicio_registro.lt(pd.Timestamp(cutoff))&d[f'fin_{s}'].ge(pd.Timestamp(cutoff))
        for i in d.index[cross]:issue(i,f'fin_{s}',f'OBJETIVO_CRUZA_CORTE_{cutoff}','Purgar o reasignar en ese corte; objetivo aun no disponible')
for i in d.index[(60*d.energia_molienda_kWh/d.duracion_molienda_min.where(d.duracion_molienda_min.gt(0))).gt(592.1)]:
    issue(i,'energia_molienda_kWh','POTENCIA_EQUIVALENTE_MAYOR_592_1','Revisar frontera, escala y acumulacion; no corregir ni excluir automaticamente')
for c in ['baches_molienda','baches_mezclado']:
    for i in d.index[d[c].isna()]:issue(i,c,'BACHES_AUSENTES','No sustituir por baches de otra etapa')
    for i in d.index[d[c].notna()&d[c].ne(d.baches_peletizado)]:issue(i,c,'BACHES_DISTINTOS_ENTRE_ETAPAS')
for c in ['masa_dosificada_t','produccion_t_asumida']:
    for i in d.index[d[c].isna()|d[c].le(0)]:issue(i,c,'MASA_NO_POSITIVA_O_AUSENTE','No imputar')
for i in d.index[d.produccion_t_asumida>d.masa_dosificada_t*1.05]:issue(i,'produccion_t_asumida','PRODUCCION_SUPERA_DOSIFICACION_5_PORCIENTO','Bandera exploratoria: revisar vapor, balance y escalado')
for s in ['molienda','mezclado']:
    for i in d.index[d[f'fin_{s}']>d.inicio_peletizado]:issue(i,f'fin_{s}','ETAPA_TERMINA_TRAS_INICIO_PELETIZADO','No usar resumen completo al iniciar peletizado')

# Imputaciones solo para resúmenes operacionales. Nunca implica disponibilidad previa al proceso.
features={'corriente_molienda_A':'molienda','corriente_mezclado_A':'mezclado','corriente_peletizado_A':'peletizado',
'temperatura_devanado_C':'molienda','temperatura_rod_del_C':'molienda','temperatura_rod_tras_C':'molienda',
'presion_vapor_psig':'peletizado','masa_vapor_lb':'peletizado','temperatura_acond_C':'peletizado'}
op=d[['fila_fuente','inicio_registro','particion','linea']].copy()
for c,s in features.items():
    values=pd.to_numeric(d[c],errors='raise').copy()
    bad=values.isna()|~np.isfinite(values)
    if c.startswith('corriente_'):bad|=values.le(0)
    if c in ['presion_vapor_psig','masa_vapor_lb']:bad|=values.lt(0)
    clean=values.mask(bad)
    op[c]=clean
    op[f'imputado_{c}']=0
    for i in d.index[bad]:
        issue(i,c,'ENTRADA_AUSENTE_O_INVALIDA','Copia operacional: mediana historica si hay soporte; original conservado')
        cutoff=min(d.loc[i,'inicio_registro'],pd.Timestamp('2025-02-22'))
        donors=(d.particion.eq('DESARROLLO') & d[f'fin_{s}'].lt(cutoff) & clean.notna() & d[f'apto_temporal_{s}'].eq(1))
        local=donors & d.linea.eq(d.loc[i,'linea'])
        chosen=local if local.sum()>=20 else donors
        stage_observed=bool(d.loc[i,f'apto_temporal_{s}']==1)
        enough=stage_observed and (local.sum()>=20 or donors.sum()>=50)
        value=float(clean[chosen].median()) if enough else None
        if enough:op.loc[i,c]=value;op.loc[i,f'imputado_{c}']=1
        selected=d.loc[chosen,'fila_fuente'].astype(int).tolist() if enough else []
        logs.append({'fila_fuente':int(d.loc[i,'fila_fuente']),'variable':c,'valor_original':values[i] if pd.notna(values[i]) else None,
          'valor_imputado':value,'metodo':('mediana_linea_anterior' if local.sum()>=20 else 'mediana_global_anterior') if enough else ('etapa_no_documentada_o_objetivo_invalido' if not stage_observed else 'sin_soporte_no_imputado'),
          'n_donantes':len(selected),'corte_exclusivo':cutoff,'max_fin_donante':d.loc[chosen,f'fin_{s}'].max() if enough else None,'filas_donantes':','.join(map(str,selected))})
        if enough:assert d.loc[chosen,f'fin_{s}'].max()<cutoff
    if c in ['presion_vapor_psig','masa_vapor_lb']:
        for i in d.index[values.eq(0)]:issue(i,c,'CERO_DE_SIGNIFICADO_NO_CONFIRMADO','Conservar cero; no imputar; requiere sensibilidad')

inc=pd.DataFrame(issues)
imp=pd.DataFrame(logs)
d['n_incidencias']=d.fila_fuente.map(inc.groupby('fila_fuente').size()).fillna(0).astype(int)
assert len(d)==2745 and d.fila_fuente.is_unique
assert d.loc[d.fila_fuente.isin([993,2178]),'mascara_objetivo_peletizado'].eq(1).all()
for s,k in [('molienda','KWH MOL'),('mezclado','KWH MEZ'),('peletizado','KWH PELET')]:assert np.allclose(d[f'energia_{s}_kWh'],raw[k],equal_nan=True)
assert initial_hash==hashlib.file_digest(src.open('rb'),'sha256').hexdigest()
for name,df in [('registros_depurados',d),('entradas_operacionales',op),('incidencias',inc),('imputaciones',imp)]:
    df.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%dT%H:%M:%S.%f')

summary={'fuente':src.name,'sha256':initial_hash,'registros':len(d),'eliminados':0,'registros_con_incidencias':int(d.n_incidencias.gt(0).sum()),
'celdas_imputadas':int(imp.valor_imputado.notna().sum()),'sin_soporte':int(imp.valor_imputado.isna().sum()),
'imputaciones_por_variable':imp[imp.valor_imputado.notna()].variable.value_counts().to_dict(),
'objetivos_positivos':{s:int(d[f'mascara_objetivo_{s}'].sum()) for s in ['molienda','mezclado','peletizado']},
'incidencias_por_regla':inc.regla.value_counts().to_dict(),'particiones':d.particion.value_counts().to_dict(),
'nota':'Depuracion conservadora. Mascara positiva no acredita validez metrologica. Entradas operacionales no son predictores previos.'}
(OUT/'resumen.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
def payload(df):
    return {'headers':list(df.columns),'rows':json.loads(df.to_json(orient='values',date_format='iso',date_unit='ms')),'dates':[c for c in df if pd.api.types.is_datetime64_any_dtype(df[c])]}
book={'Registros':payload(d),'Entradas_operacionales':payload(op),'Incidencias':payload(inc),'Imputaciones':payload(imp.drop(columns='filas_donantes'))}
(OUT/'libro_datos.json').write_text(json.dumps(book,ensure_ascii=False),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
