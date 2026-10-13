"""Compare the preserved CPU experiment with the new CUDA replication."""
from common import *
from analyze import ORDER,LABELS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    cpu=BASE/'modelos_reales_temporales_v1'
    plan=json.loads((ROOT/'GPU_RUN_PLAN.json').read_text(encoding='utf8'))
    assert sha(ROOT/'GPU_RUN_PLAN.json')==json.loads((ROOT/'GPU_RUN_LOCK.json').read_text())['sha256']
    assert all(sha(BASE/p)==h for p,h in plan['cpu_artifact_sha256'].items())
    assert all(sha(BASE/p)==h for p,h in plan['original_protected_sha256'].items())
    proc=json.loads((ROOT/'PROCESS_COMPLETION.json').read_text())
    assert len(proc)==2 and all(x['exit_code']==0 for x in proc)
    completions={k:json.loads((ROOT/k/'completion.json').read_text()) for k in ['supervised','foundation']}
    assert all(x['status']=='COMPLETE' and x['device']=='cuda' for x in completions.values())
    assert completions['supervised']['new_seed_refits']==24 and completions['foundation']['new_fold_variants']==16
    a=pd.read_csv(cpu/'predictions.csv');b=pd.read_csv(ROOT/'predictions.csv')
    merged=b.merge(a,on=['modelo','fila_fuente'],suffixes=('_gpu','_cpu'),validate='one_to_one')
    assert len(merged)==len(a)==len(b)==4672
    assert np.array_equal(merged[TARGET+'_gpu'],merged[TARGET+'_cpu'])
    assert np.array_equal(merged.fold_gpu,merged.fold_cpu)
    merged['prediction_abs_difference_kWh']=abs(merged.prediccion_kWh_gpu-merged.prediccion_kWh_cpu)
    merged.to_csv(ROOT/'paired_cpu_gpu_predictions.csv',index=False)
    drift=merged.groupby('modelo').prediction_abs_difference_kWh.agg(['mean','max','median']).reindex(ORDER)
    drift.to_csv(ROOT/'numerical_differences.csv')
    c=pd.read_csv(cpu/'metrics.csv',index_col=0);g=pd.read_csv(ROOT/'metrics.csv',index_col=0)
    comp=pd.DataFrame({'CPU_MAE_kWh':c.MAE_kWh,'GPU_MAE_kWh':g.MAE_kWh,'CPU_RMSE_kWh':c.RMSE_kWh,
       'GPU_RMSE_kWh':g.RMSE_kWh,'GPU_minus_CPU_MAE_kWh':g.MAE_kWh-c.MAE_kWh}).reindex(ORDER)
    comp.to_csv(ROOT/'comparison_cpu_gpu.csv')
    foundation=merged[merged.modelo.str.startswith(('TimesFM','Chronos'))]
    foundation.sort_values('prediction_abs_difference_kWh',ascending=False).head(32).to_csv(ROOT/'foundation_largest_device_differences.csv',index=False)
    sr=json.loads((ROOT/'supervised/runtime.json').read_text())
    fr=json.loads((ROOT/'foundation/runtime.json').read_text())
    assert len(sr)==24 and len(fr)==16
    assert all(x['device']=='cuda' and x['peak_cuda_allocated_MiB']>0 for x in sr+fr)
    epochs=[]
    for record in sr:
        stem=f"{record['model']}_{record['fold']}_seed{record['seed']}_selection.json"
        old=json.loads((cpu/'supervised'/stem).read_text());new=json.loads((ROOT/'supervised'/stem).read_text())
        # Verify each epoch really minimizes its own chronological inner-validation curve.
        best=min(new['curve'],key=lambda v:v['inner_MAE_standardized'])['epoch']
        assert best==new['selected_epoch']
        epochs.append({'model':record['model'],'fold':record['fold'],'seed':record['seed'],
                       'CPU_selected_epoch':old['selected_epoch'],'GPU_selected_epoch':best})
    pd.DataFrame(epochs).to_csv(ROOT/'epoch_selection_cpu_gpu.csv',index=False)
    effects=pd.read_csv(ROOT/'paired_contrasts.csv',index_col=0)
    normdiag=json.loads((ROOT/'CHRONOS_NORMALIZATION_DIAGNOSTIC.json').read_text(encoding='utf8'))
    audit={'status':'GPU_RETRAINING_COMPLETE','preserved_cpu_files':len(plan['cpu_artifact_sha256']),
      'cpu_artifacts_unchanged':True,'originals_and_frozen_final_unchanged':True,'matched_targets':584,'matched_predictions':len(merged),
      'parent_protocol_sha256':sha(ROOT/'PROTOCOL.json'),'gpu_plan_sha256':sha(ROOT/'GPU_RUN_PLAN.json'),
      'selection_checked':24,'new_supervised_refits':24,'supervised_inner_fits':24,'foundation_fold_variants':16,
      'all_neural_execution_cuda_verified':True,'completions':completions,
      'peak_cuda_allocated_MiB':max(x['peak_cuda_allocated_MiB'] for x in sr+fr),
      'frozen_foundation_prediction_max_difference_kWh':{k:float(drift.loc[k,'max']) for k in ORDER if k.startswith(('TimesFM','Chronos'))},
      'frozen_foundation_cross_device_equivalence_at_0_01kWh':{k:bool(drift.loc[k,'max']<=.01) for k in ORDER if k.startswith(('TimesFM','Chronos'))},
      'chronos_normalization_diagnostic':normdiag,
      'fundamental_scope':'Repetition on the same development sample; GPU/CPU training RNG trajectories can differ, and frozen-model numerical differences are reported separately. No new independent final test.',
      'duration_note':'Measured operational times are not a controlled hardware speed benchmark; initial CPU jobs shared the CPU.'}
    save_json(ROOT/'GPU_AUDIT.json',audit)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(10.5,5.5),layout='constrained')
    yy=np.arange(len(ORDER));ax.barh(yy-.18,comp.CPU_MAE_kWh,height=.33,label='CPU run',color='#8695a1')
    ax.barh(yy+.18,comp.GPU_MAE_kWh,height=.33,label='CUDA replication',color='#147f83')
    ax.set_yticks(yy,[LABELS[x] for x in ORDER]);ax.invert_yaxis();ax.set_xlabel('MAE (kWh per industrial record; same 584 records)')
    ax.legend(frameon=False);ax.set_title('Same protocol: CPU versus Blackwell GPU')
    for i,m in enumerate(ORDER):
        ax.text(comp.loc[m,'GPU_MAE_kWh']+.8,i+.18,f"{comp.loc[m,'GPU_MAE_kWh']:.2f}",va='center',fontsize=8)
    ax.set_xlim(0,max(comp.CPU_MAE_kWh.max(),comp.GPU_MAE_kWh.max())*1.15)
    fig.savefig(ROOT/'figures/05_cpu_gpu_comparison.png',dpi=260,bbox_inches='tight')
    fig.savefig(ROOT/'figures/05_cpu_gpu_comparison.svg',bbox_inches='tight');plt.close(fig)
    t=effects.loc['TFT'];conclusion=('El intervalo de la diferencia de MAE entre Ridge y TFT incluye cero; la repetición no aporta una mejora concluyente.'
      if t.CI95_low<=0<=t.CI95_high else 'El intervalo exploratorio de la diferencia Ridge-TFT no incluye cero; sigue siendo la misma muestra de desarrollo, no una confirmación independiente.')
    lines=['# Reentrenamiento con GPU sobre datos reales','',
      '**Completado en NVIDIA RTX PRO 5000 Blackwell, con CUDA.** Se han vuelto a entrenar desde cero TFT y N-HiTS y se ha repetido la inferencia de TimesFM 2.5 y Chronos-2 en GPU.','',
      '| Modelo | MAE CPU (kWh) | MAE GPU (kWh) | RMSE GPU (kWh) |','|---|---:|---:|---:|']
    for m,r in comp.iterrows():lines.append(f'| {LABELS[m]} | {r.CPU_MAE_kWh:.2f} | {r.GPU_MAE_kWh:.2f} | {r.GPU_RMSE_kWh:.2f} |')
    lines+=['','## Qué se ha ejecutado','',
      '- TFT: 12 reajustes externos (cuatro semanas por tres semillas), precedidos de 12 entrenamientos de selección cronológica interna.',
      '- N-HiTS: otros 12 reajustes externos y 12 entrenamientos internos, con el mismo presupuesto que TFT.',
      '- TimesFM 2.5 y Chronos-2: las variantes univariantes y con covariables, con pesos preentrenados congelados y cálculo neuronal en CUDA. No se ha realizado fine-tuning.',
      '- Ridge sigue siendo un control CPU; el solucionador JAX de XReg también permanece en CPU. Los pasos neuronales y las redes entrenadas sí usan GPU, verificado en los manifiestos.',
      '- Mismos 584 registros reales, ventanas de 16 registros completos, horizonte de un registro, cuatro particiones cronológicas y semillas 11, 29 y 47. La selección de épocas usa exclusivamente validación interna.',
      '', '## Interpretación','',conclusion,
      f'La mejora de MAE de TFT frente a Ridge en GPU es {t.MAE_improvement_vs_Ridge_kWh:.3f} kWh, con IC95% [{t.CI95_low:.3f}, {t.CI95_high:.3f}]. Un valor positivo favorece TFT.',
      '', 'La GPU cambia la ejecución y puede producir trayectorias estocásticas diferentes aun usando los mismos números de semilla. No se selecciona a posteriori la mejor ejecución entre CPU y GPU. En modelos con pesos congelados, las diferencias se analizan como efectos numéricos del dispositivo, no como aprendizaje adicional.',
      '', 'Los resultados siguen siendo exploratorios, retrospectivos y condicionales a las variables del registro. No se ha usado ni recalculado la prueba final original. El modelo de línea base congelado se mantiene; este ensayo no constituye una nueva validación independiente.',
      '', '## Diferencias numéricas entre dispositivos','',
      f"TimesFM 2.5 conserva sus predicciones con diferencias máximas inferiores a 0,0001 kWh. Chronos-2 univariante difiere como máximo {drift.loc['Chronos_2_univariate','max']:.6f} kWh. En cambio, Chronos-2 con covariables **no es numéricamente equivalente entre CPU y GPU**: diferencia absoluta media {drift.loc['Chronos_2_covariates','mean']:.3f} kWh y máxima {drift.loc['Chronos_2_covariates','max']:.3f} kWh, con pesos congelados. La estabilidad al recargar o cambiar el lote dentro de GPU no elimina esta discrepancia entre dispositivos.",
      'La diferencia de MAE de Chronos-2 entre CPU y GPU no se interpreta como aprendizaje ni mejora del modelo. Sus predicciones se conservan tal como las produce la implementación en cada dispositivo. El diagnóstico posterior del normalizador se encuentra en CHRONOS_NORMALIZATION_DIAGNOSTIC.json; no modifica ninguna predicción principal. La comparación con Ridge sigue siendo desfavorable a Chronos-2 en ambos dispositivos.',
      f"El diagnóstico del preprocesamiento nativo comprueba {normdiag['native_prepared_channels_checked']} canales, incluyendo las categorías codificadas a partir de la historia. De {normdiag['constant_channels']} canales constantes, {normdiag['constant_channels_context_difference_over_0_1']} presentan diferencias normalizadas CPU/GPU superiores a 0,1; ningún canal no constante supera ese umbral. El normalizador sustituye la escala únicamente cuando es exactamente cero, de modo que una varianza residual casi nula puede amplificar diferencias de redondeo. Es un mecanismo compatible con la discrepancia observada; no se ha aplicado ni validado una corrección de extremo a extremo.",
      '', '## Integridad y ejecución','',
      f"Se han comprobado {audit['preserved_cpu_files']} archivos de la ejecución CPU sin cambios, además de los originales y la evaluación final congelada. Las {len(merged)} predicciones principales coinciden en identificadores y objetivos entre ambos dispositivos.",
      f"El pico de memoria CUDA asignada registrado fue {audit['peak_cuda_allocated_MiB']:.1f} MiB. Los tiempos operativos constan en PROCESS_COMPLETION.json y los manifiestos por modelo; no se interpretan como un benchmark controlado de velocidad.",
      '', '## Archivos','',
      '- `comparison_cpu_gpu.csv`: comparación directa de métricas.',
      '- `predictions.csv` y `predictions_seeds.csv`: predicciones GPU agregadas e individuales.',
      '- `paired_cpu_gpu_predictions.csv`: emparejamiento fila a fila; `numerical_differences.csv`: magnitudes de las diferencias.',
      '- `paired_contrasts.csv`: incertidumbre exploratoria frente a Ridge, con el procedimiento de remuestreo original.',
      '- `GPU_RUN_PLAN.json`, `GPU_AUDIT.json`, `NUMERICAL_CHECKS.json`: protocolo, prueba de ejecución CUDA y comprobaciones.',
      '- `CHRONOS_NORMALIZATION_DIAGNOSTIC.json` y `chronos_normalization_diagnostic.csv`: análisis posterior de la sensibilidad del normalizador, separado del contraste principal.',
      '- `supervised/`: checkpoints `.pt`, codificadores, selección de épocas y tiempos.',
      '- `figures/`: métricas, todas las predicciones, desgloses, semillas y comparación CPU/GPU.',
      '', 'Para reproducir: se utiliza el entorno `modelos_gpu_v1/venv/Scripts/python.exe`. `run_all.py` comprueba el código de entrenamiento congelado, ejecuta los trabajos CUDA secuencialmente y conserva los códigos de salida. Los scripts omiten resultados existentes; una repetición nueva requiere otro directorio. `analyze.py` y `compare_cpu_gpu.py` reconstruyen el análisis sin reentrenar. Los datos originales, los checkpoints preentrenados y las dependencias permanecen en el proyecto local.']
    (ROOT/'INFORME_GPU.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(comp.to_string());print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
