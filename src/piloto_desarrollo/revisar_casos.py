from pathlib import Path
import ast,json,hashlib
import pandas as pd
import numpy as np
import openpyxl

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'revision_casos_v1';OUT.mkdir(exist_ok=True)
src=ROOT/'Datos Planta Peletizado.xlsx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
initial=sha(src)
d=pd.read_csv(ROOT/'depuracion_v1/registros_depurados.csv').set_index('fila_fuente')
for c in d:
    if c.startswith(('inicio_','fin_')):d[c]=pd.to_datetime(d[c])
a=pd.read_csv(ROOT/'incertidumbre_desarrollo_v1/casos_para_revision.csv')
issues=pd.read_csv(ROOT/'depuracion_v1/incidencias.csv')
imputed=pd.read_csv(ROOT/'depuracion_v1/imputaciones.csv')
tree=ast.parse((ROOT/'depuracion_v1/depurar.py').read_text(encoding='utf8'))
mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mapping' for t in n.targets))
w=openpyxl.load_workbook(src,read_only=True,data_only=True);sheet=w.active
raw=list(sheet.values);headers=list(raw[0]);checks=[]
for row in a.fila_fuente:
    r=d.loc[row]
    for header,col in mapping.items():
        x=raw[row-1][headers.index(header)];y=r[col]
        if col.startswith(('inicio_','fin_')):
            ok=(pd.isna(x) and pd.isna(y)) or (pd.notna(x) and pd.notna(y) and abs((pd.Timestamp(x)-y).total_seconds())<=.001)
        elif pd.isna(x) or pd.isna(y):ok=pd.isna(x) and pd.isna(y)
        elif isinstance(x,(float,int)):ok=np.isclose(x,float(y),rtol=1e-10,atol=1e-9)
        else:ok=str(x).strip()==str(y).strip()
        checks.append({'fila_fuente':int(row),'celda':f'{openpyxl.utils.get_column_letter(headers.index(header)+1)}{row}','variable':col,'coincide':bool(ok)})
w.close()
assert all(z['coincide'] for z in checks)
pd.DataFrame(checks).to_csv(OUT/'cotejo_celdas.csv',index=False)

d['SEC_dosificada']=d.energia_peletizado_kWh/d.masa_dosificada_t
d['potencia_equivalente_peletizado']=60*d.energia_peletizado_kWh/d.duracion_peletizado_min
hist=d[d.particion.eq('DESARROLLO')&d.mascara_objetivo_peletizado.eq(1)&d.apto_temporal_peletizado.eq(1)&d.masa_dosificada_t.gt(0)]
rows=[];donors=[];sections=[]
for _,alert in a.iterrows():
    row=int(alert.fila_fuente);r=d.loc[row]
    peers=hist[hist.linea.eq(r.linea)&hist.subfamilia.eq(r.subfamilia)&hist.fin_peletizado.lt(r.inicio_registro)&hist.masa_dosificada_t.between(.8*r.masa_dosificada_t,1.2*r.masa_dosificada_t)]
    assert row not in peers.index and (peers.fin_peletizado<r.inicio_registro).all()
    flags=issues[issues.fila_fuente.eq(row)]
    z={'fila_fuente':row,'maquina':r.linea,'subfamilia':r.subfamilia,'direccion':alert.direccion,'energia_kWh':r.energia_peletizado_kWh,'prediccion_kWh':alert.pred,'limite_inferior_kWh':alert.inferior,'limite_superior_kWh':alert.superior,'distancia_fuera_intervalo_kWh':alert.exceso_fuera_intervalo_kWh,'masa_dosificada_t':r.masa_dosificada_t,'duracion_peletizado_min':r.duracion_peletizado_min,'corriente_peletizado_A':r.corriente_peletizado_A,'potencia_equivalente_kW':r.potencia_equivalente_peletizado,'SEC_dosificada_kWh_t':r.SEC_dosificada,'n_comparables_previos':len(peers),'soporte_comparables_10':len(peers)>=10,'incidencias_previas':' | '.join(flags.variable+': '+flags.regla),'estado_causa':'PENDIENTE_CONFIRMACION_PLANTA'}
    hints=[]
    for col in ['duracion_peletizado_min','corriente_peletizado_A','SEC_dosificada','potencia_equivalente_peletizado']:
        vals=peers[col].dropna();value=r[col]
        percentile=float(100*((vals<value).sum()+.5*(vals==value).sum())/len(vals)) if len(vals)>=10 and pd.notna(value) else np.nan
        z['percentil_historico_'+col]=percentile
        if pd.notna(percentile) and (percentile>=95 or percentile<=5):hints.append(f'{col}: percentil histórico {percentile:.1f}')
    z['senales_contextuales']=' | '.join(hints) if hints else 'Sin extremos en las cuatro variables o soporte insuficiente'
    z['confirmacion_responsable']='';z['evidencia_planta']='';z['causa_confirmada']='';z['accion_acordada']=''
    rows.append(z)
    for ix in peers.index:donors.append({'fila_caso':row,'fila_comparable':int(ix),'fin_comparable':str(peers.loc[ix,'fin_peletizado'])})
    def f(v):return f'{v:.2f}'
    sections.append(f'''## Fila {row}: {r.linea}, {r.subfamilia}

- Inicio del registro: {r.inicio_registro}. Fin de peletizado: {r.fin_peletizado}.
- Energía original: **{f(r.energia_peletizado_kWh)} kWh**. Predicción: {f(alert.pred)}. Intervalo 95 %: [{f(alert.inferior)}, {f(alert.superior)}] kWh.
- Desviación {alert.direccion}: {f(alert.exceso_fuera_intervalo_kWh)} kWh fuera del intervalo. Este valor no es ahorro recuperable.
- Masa dosificada: {f(r.masa_dosificada_t)} t. Duración: {f(r.duracion_peletizado_min)} min. Corriente media: {f(r.corriente_peletizado_A)} A. Energía/masa dosificada: {f(r.SEC_dosificada)} kWh/t.
- Comparables previos: {len(peers)} de la misma máquina y subfamilia, con masa entre 80 % y 120 % de la del caso. Soporte mínimo exploratorio: 10.
- Señales descriptivas: {z['senales_contextuales']}.
- Incidencias anteriores: {z['incidencias_previas'] or 'Ninguna de las reglas anteriores'}.
- Cotejo: los 32 campos fuente coinciden con Hoja1, fila {row}, tras la normalización documentada.
- Comprobación en planta: contrastar lecturas inicial/final de energía, límites temporales del registro, orden de producción y dosificación; revisar paradas, arranques, cambios de receta y estado del equipo durante este intervalo. Las señales anteriores orientan la consulta y no prueban esas causas.
- Estado: causa pendiente. Responsable, evidencia, causa y acción: campos vacíos para completar en `revision_31_casos.csv`.
''')
out=pd.DataFrame(rows);out.to_csv(OUT/'revision_31_casos.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(donors).to_csv(OUT/'comparables_previos.csv',index=False)
(OUT/'FICHAS_31_CASOS.md').write_text('# Fichas de revisión de desviaciones\n\nRevisión retrospectiva de desarrollo. Ninguna causa industrial se da por confirmada. Los percentiles son descriptivos, con umbrales exploratorios 5 y 95, y no se usaron para modificar el detector ni las energías. La corriente y duración se utilizan aquí después del proceso.\n\n'+'\n'.join(sections),encoding='utf8')
summary={'n_casos':len(out),'celdas_cotejadas':len(checks),'diferencias':0,'casos_con_10_comparables':int(out.soporte_comparables_10.sum()),'casos_sin_10_comparables':int((~out.soporte_comparables_10).sum()),'casos_con_incidencias_previas':int(out.incidencias_previas.ne('').sum()),'casos_con_extremos_contextuales':int(out.senales_contextuales.str.contains('percentil').sum()),'causas_confirmadas':0,'fuente_sha256':initial,'archivo_fuente_intacto':sha(src)==initial,'reserva_modelado_utilizada':False,'criterio_comparables':'Desarrollo, misma maquina y subfamilia, masa +/-20%, fin anterior al inicio del caso; minimo10 para percentiles','limite_disponibilidad':'No hay columnas de inicio/fin de dosificacion ni historial de versiones en los 32 campos fuente'}
assert summary['archivo_fuente_intacto']
(OUT/'verificacion.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2));print(out[['fila_fuente','senales_contextuales']].to_string(index=False))
