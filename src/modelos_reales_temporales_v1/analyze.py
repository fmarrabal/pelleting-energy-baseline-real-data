"""Analyze the frozen comparison only after every requested model has finished."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LABELS={'Ridge_actual':'Ridge: current record','Ridge_ventana':'Ridge: full window',
 'TFT':'TFT (3-seed mean)','NHITS':'N-HiTS (3-seed mean)',
 'TimesFM_2.5_covariates':'TimesFM 2.5 + XReg','TimesFM_2.5_univariate':'TimesFM 2.5: energy only',
 'Chronos_2_covariates':'Chronos-2 + covariates','Chronos_2_univariate':'Chronos-2: energy only'}
ORDER=list(LABELS)
MAIN=['Ridge_actual','Ridge_ventana','TFT','NHITS','TimesFM_2.5_covariates','Chronos_2_covariates']

def metrics(g):
    e=g.prediccion_kWh-g[TARGET]
    return pd.Series({'n':len(g),'MAE_kWh':abs(e).mean(),'RMSE_kWh':np.sqrt((e**2).mean()),
      'bias_kWh':e.mean(),'WAPE':abs(e).sum()/g[TARGET].sum(),'negative_raw':int((g.raw_prediction_kWh<0).sum())})

def figure_save(fig,name):
    p=ROOT/'figures';p.mkdir(exist_ok=True)
    fig.savefig(p/(name+'.png'),dpi=260,bbox_inches='tight')
    fig.savefig(p/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def main():
    p=json.loads((ROOT/'PROTOCOL.json').read_text(encoding='utf8'))
    assert sha(ROOT/'PROTOCOL.json')==json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())['sha256']
    pieces=[];seedpieces=[]
    for start,end in FOLDS:
        for model in ORDER:
            if model in ['TFT','NHITS']:
                seeds=[]
                for seed in SEEDS:
                    q=pd.read_csv(ROOT/'supervised'/f'{model}_{start}_seed{seed}.csv');seeds.append(q);seedpieces.append(q)
                assert all(np.array_equal(seeds[0].fila_fuente,q.fila_fuente) for q in seeds)
                q=seeds[0].drop(columns='seed').copy()
                # Prespecified ensemble averages clipped seed predictions.
                q['prediccion_kWh']=np.mean([v.prediccion_kWh for v in seeds],axis=0)
                q['raw_prediction_kWh']=np.mean([v.raw_prediction_kWh for v in seeds],axis=0)
                q['error']=q.prediccion_kWh-q[TARGET];q['error_abs']=abs(q.error)
            else:
                directory='foundation' if model.startswith(('TimesFM','Chronos')) else 'supervised'
                q=pd.read_csv(ROOT/directory/f'{model}_{start}.csv')
            pieces.append(q)
    pred=pd.concat(pieces,ignore_index=True);seeds=pd.concat(seedpieces,ignore_index=True)
    pred['inicio_registro']=pd.to_datetime(pred.inicio_registro)
    assert len(pred)==p['eligible_total']*len(ORDER)
    assert not pred.duplicated(['modelo','fila_fuente']).any()
    assert pred.groupby('fila_fuente')[TARGET].nunique().max()==1
    matched={k:set(g.fila_fuente) for k,g in pred.groupby('modelo')}
    assert all(s==matched['Ridge_actual'] for s in matched.values())
    pred.to_csv(ROOT/'predictions.csv',index=False)
    seeds.to_csv(ROOT/'predictions_seeds.csv',index=False)
    scores=pred.groupby('modelo').apply(metrics,include_groups=False).reindex(ORDER)
    scores['MAE_change_vs_Ridge_pct']=100*(scores.MAE_kWh/scores.loc['Ridge_actual','MAE_kWh']-1)
    scores.to_csv(ROOT/'metrics.csv')
    byfold=pred.groupby(['modelo','fold']).apply(metrics,include_groups=False)
    byline=pred.groupby(['modelo','linea']).apply(metrics,include_groups=False)
    byfold.to_csv(ROOT/'metrics_by_fold.csv');byline.to_csv(ROOT/'metrics_by_machine.csv')
    seedmetrics=seeds.groupby(['modelo','seed']).apply(metrics,include_groups=False)
    seedmetrics.to_csv(ROOT/'metrics_seeds.csv')
    # Independent check against original donor records, not values copied into audit CSV.
    d=data();links=pd.read_csv(ROOT/'donor_audit.csv')
    donors=d.loc[links.donor.to_numpy()].reset_index(drop=True)
    targets=d.loc[links.fila_fuente.to_numpy()].reset_index(drop=True)
    assert (donors.linea==targets.linea).all()
    assert (donors.fin_peletizado<targets.inicio_registro).all()
    assert (donors.inicio_registro>=pd.to_datetime(links.partition_floor)).all()
    assert (donors.fila_fuente!=targets.fila_fuente).all()
    membership=pd.read_csv(ROOT/'sample_membership.csv')
    for fold,g in membership.groupby('fold'):
        assert not (set(g[g.partition.eq('train')].fila_fuente)&set(g[g.partition.eq('val')].fila_fuente))
        assert not (set(g[g.partition.eq('inner_train')].fila_fuente)&set(g[g.partition.eq('inner_val')].fila_fuente))
    protected={k:sha(BASE/k) for k in p['protected_sha256']}
    assert protected==p['protected_sha256'],'An original artifact changed during experiment'
    # Paired circular bootstrap: resample common observed dates, preserve all machines together.
    pred['date']=pred.inicio_registro.dt.normalize()
    daily=pred.groupby(['date','modelo']).error_abs.sum().unstack().reindex(columns=ORDER)
    count=pred[pred.modelo.eq('Ridge_actual')].groupby('date').size().reindex(daily.index).to_numpy()
    rng=np.random.default_rng(20260922);nd=len(daily);B=5000
    starts=rng.integers(0,nd,size=(B,int(np.ceil(nd/3))))
    ix=((starts[:,:,None]+np.arange(3))%nd).reshape(B,-1)[:,:nd]
    boot=daily.to_numpy()[ix].sum(axis=1)/count[ix].sum(axis=1)[:,None]
    contrasts=[]
    for j,m in enumerate(ORDER):
        delta=boot[:,0]-boot[:,j]
        lo,hi=np.quantile(delta,[.025,.975]);alo,ahi=np.quantile(delta,[.05/(2*7),1-.05/(2*7)])
        contrasts.append({'modelo':m,'MAE_improvement_vs_Ridge_kWh':scores.loc['Ridge_actual','MAE_kWh']-scores.loc[m,'MAE_kWh'],
          'CI95_low':lo,'CI95_high':hi,'CI_family7_low':alo,'CI_family7_high':ahi,
          'MAE_CI95_low':np.quantile(boot[:,j],.025),'MAE_CI95_high':np.quantile(boot[:,j],.975)})
    contrasts=pd.DataFrame(contrasts).set_index('modelo');contrasts.to_csv(ROOT/'paired_contrasts.csv')
    retained=pred[pred.modelo.eq('Ridge_actual')].fila_fuente
    old=pd.read_csv(BASE/'analisis_desarrollo_v2/predicciones_desarrollo.csv')
    old=old[old.modelo.eq('Ridge')&old.fila_fuente.isin(retained)]
    old_mae=float(old.error_abs.mean())
    # Source hashes and numerical evidence, excluding external checkpoint/dependency directories.
    audit={'status':'COMPLETED_EXPLORATORY_REAL_DATA','n_targets':len(retained),'n_models':len(ORDER),
      'n_predictions':len(pred),'n_seed_predictions':len(seeds),'history_relations_checked':len(links),
      'machine_time_partition_violations':0,'matched_sample_verified':True,'protected_originals_unchanged':True,
      'protected_sha256_after':protected,'bootstrap':{'draws':B,'block_observed_dates':3,'n_observed_dates':nd,'seed':20260922,
       'simultaneous_note':'Bonferroni percentile intervals across seven exploratory contrasts; not confirmatory evidence.'},
      'archived_Ridge_MAE_same_subset':old_mae,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),
      'devices':{k:json.loads((ROOT/k/'hardware.json').read_text()) for k in ['supervised','foundation']},
      'main_best_by_observed_MAE':scores.loc[MAIN].MAE_kWh.idxmin(),'all_best_by_observed_MAE':scores.MAE_kWh.idxmin(),
      'code_sha256':{f.name:sha(f) for f in ROOT.glob('*.py')}}
    save_json(ROOT/'AUDIT.json',audit)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#23455d','#64798a','#b35325','#b99630','#168680','#66588e','#709bae','#a18daf']
    # 1: absolute error and paired effects, preserving the zero-shot ablations.
    fig,ax=plt.subplots(1,2,figsize=(12.6,5.4),gridspec_kw={'width_ratios':[1,1.08]},layout='constrained')
    yy=np.arange(len(ORDER));v=scores.MAE_kWh.to_numpy()
    ax[0].barh(yy,v,color=colors,alpha=.9)
    ax[0].errorbar(v,yy,xerr=np.maximum(0,np.stack([v-contrasts.MAE_CI95_low,contrasts.MAE_CI95_high-v])),fmt='none',ecolor='#18232c',capsize=3)
    for i,x in enumerate(v):ax[0].text(x+2,i-.24,f'{x:.1f}',va='center',fontsize=9)
    ax[0].set_yticks(yy,[LABELS[x] for x in ORDER]);ax[0].invert_yaxis();ax[0].set_xlabel('MAE (kWh per record)');ax[0].set_xlim(0,max(contrasts.MAE_CI95_high)*1.12);ax[0].set_title(f'A. Same {len(retained)} real records')
    ef=contrasts.MAE_improvement_vs_Ridge_kWh.to_numpy()
    ax[1].errorbar(ef,yy,xerr=np.maximum(0,np.stack([ef-contrasts.CI95_low,contrasts.CI95_high-ef])),fmt='o',color='#23455d',capsize=4)
    ax[1].axvline(0,color='#b35325',ls='--',lw=1);ax[1].set_yticks(yy,[]);ax[1].invert_yaxis();ax[1].set_xlabel('MAE(Ridge current) - MAE(model), kWh\nPositive values favor the temporal model');ax[1].set_title('B. Paired 95% block-bootstrap intervals')
    ax[1].set_ylim(ax[0].get_ylim())
    figure_save(fig,'01_comparison')
    # 2: all records, a shared linear detail view and full-range inset for extremes.
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    lim=float(np.ceil(pred[TARGET].max()/100)*100)
    mc={'Pellet 1':'#bc612f','Pellet 2':'#168680','Pellet 3':'#66588e'}
    for ax,m in zip(axes.ravel(),MAIN):
        q=pred[pred.modelo.eq(m)]
        for line,g in q.groupby('linea'):ax.scatter(g[TARGET],g.prediccion_kWh,s=11,alpha=.55,color=mc[line],edgecolors='none',label=line)
        ax.plot([0,lim],[0,lim],c='#333333',lw=.8,ls='--');ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Measured energy (kWh)',ylabel='Predicted energy (kWh)',title=LABELS[m]);ax.set_aspect('equal')
        ax.text(.04,.94,f'n={len(q)}; MAE={scores.loc[m,"MAE_kWh"]:.1f}',transform=ax.transAxes,fontsize=9,va='top')
        outside=q[(q[TARGET]>lim)|(q.prediccion_kWh>lim)]
        if len(outside):
            ins=ax.inset_axes([.57,.56,.38,.32])
            full=float(q[[TARGET,'prediccion_kWh']].max().max())*1.03
            for line,g in q.groupby('linea'):ins.scatter(g[TARGET],g.prediccion_kWh,s=3,alpha=.55,color=mc[line],edgecolors='none')
            ins.scatter(outside[TARGET],outside.prediccion_kWh,s=18,color='#66588e',edgecolors='white',linewidths=.3)
            ins.plot([0,full],[0,full],ls='--',color='#444444',lw=.6)
            ins.set(xlim=(0,full),ylim=(0,full));ins.tick_params(labelsize=6)
            ins.set_xticks([0,10000]);ins.set_yticks([0,10000])
            ins.set_title(f'Full range: {len(outside)} extremes',fontsize=7)
            ax.text(.04,.86,f'{len(outside)} points above detail range: inset',transform=ax.transAxes,fontsize=7,va='top')
    axes[0,0].legend(loc='lower right',fontsize=8,frameon=False)
    figure_save(fig,'02_all_record_predictions')
    # 3: performance heterogeneity, cells include the actual MAE, row sample counts in labels.
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),gridspec_kw={'width_ratios':[4,3]},layout='constrained')
    fm=byfold.MAE_kWh.unstack().reindex(ORDER);lm=byline.MAE_kWh.unstack().reindex(ORDER)
    vmax=max(fm.max().max(),lm.max().max())
    for j,(ax,mat,title) in enumerate(zip(axes,[fm,lm],['Chronological outer block','Production line'])):
        im=ax.imshow(mat.to_numpy(),aspect='auto',cmap='YlOrRd',vmin=0,vmax=vmax)
        labels=[str(x)[5:] for x in mat.columns] if j==0 else list(mat.columns)
        ax.set_xticks(np.arange(mat.shape[1]),labels);ax.set_yticks(yy,[LABELS[x] for x in ORDER] if j==0 else [])
        ax.set_title(title)
        for (r,c),val in np.ndenumerate(mat.to_numpy()):ax.text(c,r,f'{val:.1f}',ha='center',va='center',fontsize=9,color='white' if val>vmax*.63 else '#172d3a')
    fig.colorbar(im,ax=axes,label='MAE (kWh per record)',shrink=.8)
    figure_save(fig,'03_stratified_error')
    # 4: complete seed evidence; inner curves are selection evidence, not test curves.
    fig,axes=plt.subplots(1,3,figsize=(12,4),gridspec_kw={'width_ratios':[1,1,1]},layout='constrained')
    fc=['#23455d','#168680','#b35325','#66588e']
    for j,family in enumerate(['TFT','NHITS']):
        for fi,(start,end) in enumerate(FOLDS):
            for seed in SEEDS:
                z=json.loads((ROOT/'supervised'/f'{family}_{start}_seed{seed}_selection.json').read_text())
                curve=z['curve'];axes[j].plot([x['epoch'] for x in curve],[x['inner_MAE_standardized']*z['inner_scale'] for x in curve],color=fc[fi],lw=1,alpha=.62)
        axes[j].set(title=LABELS[family].replace(' (3-seed mean)',''),xlabel='Training epoch',ylabel='Inner validation MAE (kWh)')
    for i,family in enumerate(['TFT','NHITS']):
        ss=seedmetrics.loc[family].MAE_kWh
        axes[2].scatter(np.arange(3)*.13+i-.13,ss,s=35,color=['#23455d','#168680','#b35325'])
        axes[2].scatter(i,scores.loc[family,'MAE_kWh'],marker='D',s=48,color='#111111')
        for seed,x in ss.items():axes[2].annotate(str(seed),(i,x),xytext=(8,0),textcoords='offset points',fontsize=8)
    axes[2].axhline(scores.loc['Ridge_actual','MAE_kWh'],color='#64798a',ls='--',lw=1)
    axes[2].set_xticks([0,1],['TFT','N-HiTS']);axes[2].set_xlim(-.45,1.65);axes[2].set(ylabel='Outer validation MAE (kWh)',title='Seed stability; diamond = ensemble')
    axes[0].legend(handles=[Line2D([0],[0],color=c,label=s[5:]) for c,(s,e) in zip(fc,FOLDS)],title='Outer block start',fontsize=7,title_fontsize=8,frameon=False)
    figure_save(fig,'04_training_and_seeds')
    effects_tft=contrasts.loc['TFT','MAE_improvement_vs_Ridge_kWh'];lo_tft=contrasts.loc['TFT','CI95_low'];hi_tft=contrasts.loc['TFT','CI95_high']
    lines=['# Comparación exploratoria sobre datos industriales reales','',
      f'Completadas las cuatro arquitecturas solicitadas sobre **{len(retained)} registros reales comunes**. No se han utilizado las series sintéticas ni recalculado la prueba final original.','',
      '## Resultado','', '| Modelo | MAE (kWh) | RMSE (kWh) | Cambio de MAE frente a Ridge |','|---|---:|---:|---:|']
    for m,row in scores.iterrows():lines.append(f'| {LABELS[m]} | {row.MAE_kWh:.2f} | {row.RMSE_kWh:.2f} | {row.MAE_change_vs_Ridge_pct:+.1f}% |')
    lines+=['','Un cambio positivo significa más error. TFT y N-HiTS son la media de tres semillas, no la mejor semilla elegida a posteriori.',
      '', '## Qué se ha comparado','',
      'Se conservó el universo de 800 observaciones de las cuatro semanas de desarrollo anteriores. Se excluyeron 216 por no disponer de 16 registros previos completos de la misma máquina dentro de su propia semana. Quedaron 151, 133, 151 y 149 observaciones. Los controles Ridge se reajustaron con las mismas filas objetivo de entrenamiento que las redes. El Ridge de ventana recibe la historia completa además de las variables actuales.',
      '', 'Las redes emplean una configuración compacta por arquitectura y tres semillas. El número de épocas se selecciona en la semana anterior, con un máximo de 40; posteriormente se reentrena desde cero con todo el entrenamiento externo. No es una búsqueda exhaustiva de arquitecturas. Los modelos fundacionales conservan sus pesos. TimesFM utiliza su XReg oficial, con un ajuste separado por origen. Chronos-2 recibe covariables históricas y del registro objetivo, con intercambio entre consultas desactivado.',
      '', '## Interpretación para el artículo','',
      f'La menor MAE observada corresponde a **{LABELS[audit["all_best_by_observed_MAE"]]}**. La comparación informa del rendimiento de estas configuraciones y este contexto; no demuestra una superioridad universal ni descarta otras configuraciones de las arquitecturas.',
      '', f'La ventaja de TFT es de {effects_tft:.2f} kWh (IC95% [{lo_tft:.2f}, {hi_tft:.2f}]), compatible con ausencia de mejora. Su RMSE es mayor que el de Ridge. No hay evidencia suficiente para sustituir el modelo congelado por TFT a partir de este experimento.',
      '', 'TimesFM + XReg genera tres predicciones superiores a 2000 kWh, hasta 12202,66 kWh. El diagnóstico posterior reproduce los resultados y localiza la contribución extrema en XReg: la masa actual queda muy lejos de un contexto de 16 masas casi constantes. La resolución independiente en float64 confirma que no es un simple error de redondeo. Se conservan los tres casos y todas las métricas originales; esta observación no se utiliza para ajustar ni sustituir las predicciones. Es una limitación del adaptador local en este diseño, no una demostración de que TimesFM falle en cualquier aplicación.',
      '', 'El horizonte es un registro industrial condicionado a sus variables consolidadas, no un intervalo fijo en minutos. La masa dosificada, el número de baches y los metadatos del producto no tienen acreditada su disponibilidad antes del proceso. Por tanto, este experimento sigue siendo una línea base retrospectiva condicional. No equivale a una predicción prospectiva validada, ni prueba ahorros energéticos.',
      '', 'Las ventanas no cruzan máquinas ni límites de las particiones internas o externas. El fin de peletizado se usa como aproximación de la disponibilidad de la etiqueta; falta confirmar su publicación efectiva. No existe una identidad de ciclo o lote físico verificada que permita afirmar separación adicional por lotes. Los registros inactivos no forman parte de esta evaluación.',
      '', 'Los intervalos pareados usan 5000 remuestreos de bloques de tres fechas observadas. Se ofrecen intervalos marginales del 95% y una corrección de Bonferroni para siete contrastes; son evidencia exploratoria en desarrollo, no una confirmación independiente. La selección de ventanas reduce y puede cambiar la composición de la muestra: no debe compararse esta MAE con la MAE anterior de 800 casos como si fueran iguales.',
      '', f'Como referencia adicional, el Ridge archivado alcanza {old_mae:.2f} kWh sobre estas mismas 584 filas, pero había usado más objetivos de entrenamiento; no sustituye al control emparejado de esta tabla.',
      '', '## Trazabilidad','',
      f'- {len(links):,} relaciones de historia verificadas contra el archivo depurado; cero cruces de máquina, accesos a etiquetas futuras o cruces del límite de partición.',
      '- Originales, modelo congelado, predicciones de la prueba final y los dos PDF anteriores conservan sus SHA-256.',
      '- Ejecución real registrada en `supervised/hardware.json` y `foundation/hardware.json`; CPU porque la GPU estaba ocupada.',
      '- `predictions.csv`: cada predicción; `predictions_seeds.csv`: todas las semillas; `paired_contrasts.csv`: contrastes; `AUDIT.json`: comprobaciones.',
      '- Figuras PNG y SVG con todos los puntos, desgloses por máquina/semana y curvas de selección.',
      '', '## Fuentes técnicas','']
    lines.extend('- '+x for x in p['sources'])
    (ROOT/'INFORME.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(scores.to_string());print(json.dumps({k:audit[k] for k in ['n_targets','n_predictions','history_relations_checked','protected_originals_unchanged']}))

if __name__=='__main__':main()
