"""Intervalos y alertas retrospectivas: exclusivamente desarrollo."""
from pathlib import Path
import hashlib,json,math,platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'incertidumbre_desarrollo_v1'; OUT.mkdir(exist_ok=True)
src=ROOT/'depuracion_v1/registros_depurados.csv'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sha=digest(src)
d=pd.read_csv(src)
d=d[d.particion.eq('DESARROLLO')].copy()
for c in ['inicio_registro','fin_peletizado']: d[c]=pd.to_datetime(d[c])
hour=d.inicio_registro.dt.hour+d.inicio_registro.dt.minute/60
for label,x,period in [('hora',hour,24),('dia',d.inicio_registro.dt.dayofweek,7)]:
    d[label+'_sin']=np.sin(2*np.pi*x/period);d[label+'_cos']=np.cos(2*np.pi*x/period)
cats=['linea','subfamilia','forma','salida_programada']
calendar=['hora_sin','hora_cos','dia_sin','dia_cos']
nums=['masa_dosificada_t','baches_dosificacion']+calendar
d=d[d.mascara_objetivo_peletizado.eq(1)&d.apto_temporal_peletizado.eq(1)&d.masa_dosificada_t.gt(0)&d[nums+cats].notna().all(axis=1)]
config={'alphas':[.1,.05],'ridge_alpha':10,'calibracion_interna_dias':7,'min_cal_maquina':40,'corte_reserva':'2025-02-22','variables':nums+cats,'intervalos':['absoluto_global','absoluto_maquina','por_masa_maquina'],'escenarios':['plan_provisional','sin_masa_baches'],'semana_inicio':['2025-01-25','2025-02-01','2025-02-08','2025-02-15'],'actualizacion':'Modelo y cuantiles fijos durante cada semana','alarma':'Valor observado fuera del intervalo 95%; disponible al finalizar peletizado','seleccion':'Misma muestra elegible que piloto v2; fin de objetivo anterior al final de cada bloque','input_sha256':sha}
(OUT/'configuracion.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')

def qfinite(scores,alpha):
    scores=np.sort(np.asarray(scores));k=math.ceil((len(scores)+1)*(1-alpha))
    return float(scores[k-1]) if k<=len(scores) else float('inf')

assert qfinite(np.arange(1,20),.05)==19
assert math.isinf(qfinite([1,2],.05))
rows=[];logs=[];lineage=[]
for start in config['semana_inicio']:
    a=pd.Timestamp(start); b=a+pd.Timedelta(days=7);c=a-pd.Timedelta(days=7)
    tr=d[d.inicio_registro.lt(c)&d.fin_peletizado.lt(c)]
    ca=d[d.inicio_registro.ge(c)&d.inicio_registro.lt(a)&d.fin_peletizado.lt(a)]
    va=d[d.inicio_registro.ge(a)&d.inicio_registro.lt(b)&d.fin_peletizado.lt(b)]
    assert tr.fin_peletizado.max()<c and ca.fin_peletizado.max()<a and va.fin_peletizado.max()<b<=pd.Timestamp(config['corte_reserva'])
    assert not (set(tr.fila_fuente)&set(ca.fila_fuente) or set(ca.fila_fuente)&set(va.fila_fuente))
    logs.append({'inicio':start,'train':len(tr),'cal_interna':len(ca),'val':len(va),'max_fin_train':str(tr.fin_peletizado.max()),'max_fin_cal':str(ca.fin_peletizado.max()),'cal_por_maquina':ca.linea.value_counts().to_dict()})
    for role,g in [('train',tr),('cal_interna',ca),('validacion',va)]:
        z=g[['fila_fuente','inicio_registro','fin_peletizado']].copy();z['fold']=start;z['rol']=role;lineage.append(z)
    for scenario,numeric in [('plan_provisional',nums),('sin_masa_baches',calendar)]:
        enc=ColumnTransformer([('n',StandardScaler(),numeric),('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
        model=Ridge(alpha=10).fit(enc.fit_transform(tr),tr.energia_peletizado_kWh)
        pc=np.maximum(model.predict(enc.transform(ca)),0);pv=np.maximum(model.predict(enc.transform(va)),0)
        err=np.abs(ca.energia_peletizado_kWh.to_numpy()-pc)
        for method in config['intervalos']:
            if scenario=='sin_masa_baches' and method=='por_masa_maquina':continue
            normalized=method=='por_masa_maquina'
            scores=err/(ca.masa_dosificada_t.to_numpy() if normalized else 1)
            for alpha in config['alphas']:
                widths=[];ns=[];fallbacks=[]
                for _,r in va.iterrows():
                    ix=ca.linea.eq(r.linea).to_numpy()
                    use=method!='absoluto_global' and ix.sum()>=config['min_cal_maquina']
                    s=scores[ix] if use else scores
                    widths.append(qfinite(s,alpha)*(r.masa_dosificada_t if normalized else 1));ns.append(len(s));fallbacks.append(method!='absoluto_global' and not use)
                z=va[['fila_fuente','linea','familia','inicio_registro','fin_peletizado','energia_peletizado_kWh','masa_dosificada_t']].copy()
                z['fold']=start;z['escenario']=scenario;z['metodo']=method;z['nominal']=1-alpha;z['pred']=pv
                z['inferior']=np.maximum(0,pv-np.array(widths));z['superior']=pv+np.array(widths);z['n_cal']=ns;z['fallback_global']=fallbacks
                y=z.energia_peletizado_kWh
                z['cubierto']=y.ge(z.inferior)&y.le(z.superior);z['ancho']=z.superior-z.inferior
                z['alerta_alta']=y.gt(z.superior);z['alerta_baja']=y.lt(z.inferior)
                z['interval_score']=z.ancho+2/alpha*((z.inferior-y).clip(lower=0)+(y-z.superior).clip(lower=0))
                z['error_abs']=abs(y-z.pred);rows.append(z)

p=pd.concat(rows,ignore_index=True)
assert np.isfinite(p[['pred','inferior','superior']]).all().all()
assert not p.duplicated(['fila_fuente','escenario','metodo','nominal']).any()
assert (p.inferior<=p.superior).all()
def metrics(g):return pd.Series({'n':len(g),'cobertura':g.cubierto.mean(),'ancho_medio_kWh':g.ancho.mean(),'interval_score':g.interval_score.mean(),'MAE_kWh':g.error_abs.mean(),'alertas_altas':g.alerta_alta.sum(),'alertas_bajas':g.alerta_baja.sum(),'fallbacks':g.fallback_global.sum()})
keys=['escenario','metodo','nominal']
m=p.groupby(keys).apply(metrics,include_groups=False)
m.to_csv(OUT/'metricas_globales.csv')
for group in ['linea','familia','fold']:p.groupby(keys+[group]).apply(metrics,include_groups=False).to_csv(OUT/f'metricas_por_{group}.csv')
p.to_csv(OUT/'intervalos_y_alertas.csv',index=False)
pd.concat(lineage).to_csv(OUT/'trazabilidad_particiones.csv',index=False)
# Conservative fixed review reference, specified independently of observed coverage.
ref=p[p.escenario.eq('plan_provisional')&p.metodo.eq('absoluto_maquina')&p.nominal.eq(.95)].copy()
alerts=ref[ref.alerta_alta|ref.alerta_baja].copy()
alerts['direccion']=np.where(alerts.alerta_alta,'alta','baja')
alerts['exceso_fuera_intervalo_kWh']=np.maximum(alerts.energia_peletizado_kWh-alerts.superior,alerts.inferior-alerts.energia_peletizado_kWh)
alerts.sort_values('exceso_fuera_intervalo_kWh',ascending=False).to_csv(OUT/'casos_para_revision.csv',index=False)
# Coverage uncertainty: paired days, circular moving blocks, descriptive only.
rng=np.random.default_rng(20260920);ci=[]
for key,g in p.groupby(keys):
    daily=g.groupby(g.inicio_registro.dt.normalize()).cubierto.agg(['sum','size'])
    n=len(daily);idx=((rng.integers(0,n,size=(3000,math.ceil(n/3),1))+np.arange(3))%n).reshape(3000,-1)[:,:n]
    cover=daily['sum'].to_numpy()[idx].sum(axis=1)/daily['size'].to_numpy()[idx].sum(axis=1)
    ci.append(dict(zip(keys,key),IC95_inf=np.quantile(cover,.025),IC95_sup=np.quantile(cover,.975)))
pd.DataFrame(ci).to_csv(OUT/'cobertura_bootstrap.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for method,g in m.loc['plan_provisional'].reset_index().groupby('metodo'):
    axes[0].plot(g.nominal*100,g.cobertura*100,'o-',label=method)
    axes[1].plot(g.nominal*100,g.ancho_medio_kWh,'o-',label=method)
axes[0].plot([90,95],[90,95],'k--',label='Nominal');axes[0].set(xlabel='Cobertura nominal (%)',ylabel='Cobertura observada (%)')
axes[1].set(xlabel='Cobertura nominal (%)',ylabel='Ancho medio (kWh)');axes[0].legend(fontsize=8)
fig.suptitle('Intervalos retrospectivos: desarrollo, 800 registros');fig.tight_layout();fig.savefig(OUT/'cobertura_y_ancho.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(11,8))
for ax,(machine,g) in zip(axes,ref.groupby('linea')):
    g=g.sort_values('inicio_registro');ax.fill_between(g.inicio_registro,g.inferior,g.superior,alpha=.2,label='Intervalo nominal 95%')
    ax.scatter(g.inicio_registro,g.energia_peletizado_kWh,s=7,label='Observado')
    a=g[g.alerta_alta|g.alerta_baja];ax.scatter(a.inicio_registro,a.energia_peletizado_kWh,s=22,color='red',label='Revisar')
    ax.set(title=machine,ylabel='kWh');ax.legend(fontsize=7,loc='upper left')
fig.tight_layout();fig.savefig(OUT/'alertas_por_maquina.png',dpi=160);plt.close(fig)
assert digest(src)==sha
manifest={'config':config,'folds':logs,'python':platform.python_version(),'sklearn':sklearn.__version__,'n_por_configuracion':800,'referencia_revision':'plan_provisional/absoluto_maquina/95%','n_alertas_revision':len(alerts),'reserva_evaluada':False,'verificacion':{'cuantil_finito':True,'sin_solapamiento_train_cal_val':True,'etiquetas_pasadas':True,'intervalos_finitos_ordenados':True,'sin_duplicados':True,'entrada_intacta':True},'fuente_metodo':'https://arxiv.org/abs/2203.15885','script_sha256':digest(Path(__file__))}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
print(m.to_string());print('ALERTAS_REVISION',len(alerts));print(json.dumps(logs))
