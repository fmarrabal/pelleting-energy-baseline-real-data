# Explainable energy baselines for industrial feed pelleting: a chronological comparison of regression, temporal and foundation models

[English](README.md) | **Español**

Archivo de reproducción del manuscrito **Paper_Peletizado_Datos_Reales_v3.pdf**. Incluye los datos originales, código, resultados congelados y proyecto LaTeX/Overleaf.

Estudio complementario: [GitHub](https://github.com/fmarrabal/pelleting-energy-baselines-synthetic).

## Pregunta científica y alcance

El estudio evalúa hasta qué punto una línea base energética explicable predice la electricidad de un registro industrial de peletizado y si los modelos temporales o fundacionales aportan una mejora. La unidad de análisis es un **registro industrial irregular**, no una observación de sensores cada diez segundos. La salida es la **electricidad de peletizado en kWh por registro**. Se trata de una línea base retrospectiva condicionada a descriptores consolidados de producción: no se ha verificado de forma independiente que sus valores planificados estuvieran disponibles antes de comenzar la operación.

El Excel original `Datos Planta Peletizado.xlsx` contiene 2.745 registros: 730 de Pellet 1, 1.148 de Pellet 2 y 867 de Pellet 3. Se identifican 32 campos de origen. Los tres registros con energía cero se conservan en la auditoría y se excluyen de la población válida para la respuesta; no se imputa la energía objetivo. El archivo conserva los filtros e imputaciones auxiliares. El Excel anterior `Datos_Peletizado_Base.xlsx` pertenece a la evidencia de calibración/fidelidad del estudio sintético y no debe sustituir silenciosamente a este archivo.

### Variables de entrada y salida

| Función | Variable | Significado |
|---|---|---|
| Respuesta | `energia_peletizado_kWh` | Electricidad atribuida al peletizado, kWh por registro |
| Entrada numérica | `masa_dosificada_t` | Masa dosificada, convertida de kg a toneladas |
| Entrada numérica | `baches_dosificacion` | Número de baches de dosificación registrados |
| Entradas numéricas | `hora_sin`, `hora_cos` | Codificación cíclica de la hora de origen |
| Entradas numéricas | `dia_sin`, `dia_cos` | Codificación cíclica del día de la semana |
| Entradas categóricas | `linea`, `subfamilia`, `forma`, `salida_programada` | Máquina, subfamilia de producto, presentación y salida programada |
| Historia temporal | Los 16 registros anteriores completados de la misma máquina | Energía pasada y descriptores; su finalización precede estrictamente al origen objetivo |
| Salidas | Predicción puntual e intervalos del 90% y 95% | Línea base condicional e incertidumbre calibrada con residuos |

La normalización y codificación categórica se ajustan exclusivamente con entrenamiento. El diseño estático congelado contiene 22 columnas codificadas. Se excluyen como entradas la energía actual, la duración actual, los promedios operacionales actuales y la masa producida posteriormente. La finalización aproxima el momento de disponibilidad; no es un registro verificado de publicación en la base de datos. `fila_fuente` conserva la fila original del Excel.

### Particiones, selección e incertidumbre

La comparación estática utiliza cuatro particiones temporales expansivas, con validaciones que comienzan el 25 de enero y el 1, 8 y 15 de febrero de 2025: 800 registros de validación en conjunto. Se comparan Ridge, histogram gradient boosting, ExtraTrees, MLP, consumo específico por máquina, consumo específico reciente y mediana por máquina. Ridge se congela con 1.542 registros de desarrollo, se calibra con 380 y se evalúa sobre 798 registros de prueba desde el 6 de marzo. Los CSV y manifiestos contienen la pertenencia y exclusiones exactas; no deben reconstruirse únicamente a partir de la posición de las filas.

La comparación temporal moderna es un experimento separado y **limitado al desarrollo**. Las historias pertenecen a la misma máquina y se reinician en cada partición. La exigencia de 16 antecedentes completados deja 584 registros de validación comunes. TFT y N-HiTS se entrenan con semillas 11, 29 y 47; la validación interna selecciona épocas, con máximo 40, mínimo 10 antes de la parada temprana y paciencia 8. Los pesos fundacionales permanecen congelados. Las ocho variantes comparten los mismos registros, distintos de los 798 de la prueba final.

Con `e = predicción - observación`, MAE es `mean(abs(e))`, RMSE es `sqrt(mean(e**2))` y el sesgo es `mean(e)`. WAPE es `sum(abs(e))/sum(observación)` y se almacena como fracción. Los intervalos fijos emplean cuantiles de residuos de calibración y límites inferiores no negativos. La cobertura es empírica: la dependencia temporal y la deriva limitan una interpretación basada en intercambiabilidad. La recalibración móvil y adaptación residual posteriores son exploratorias.

### Resultados de referencia

| Población evaluada | Modelo | n | MAE (kWh) | RMSE (kWh) |
|---|---|---:|---:|---:|
| Prueba final congelada | Ridge | 798 | 52,4489 | 80,4186 |
| Prueba final congelada | Consumo específico | 798 | 61,2872 | 92,0540 |
| Desarrollo temporal común | Ridge con entradas actuales | 584 | 53,5457 | 77,0140 |
| Desarrollo temporal común | TFT | 584 | 53,6507 | 79,1050 |
| Desarrollo temporal común | N-HiTS | 584 | 65,2784 | 92,5581 |
| Desarrollo temporal común | Chronos-2 con covariables | 584 | 85,5268 | 120,8235 |
| Desarrollo temporal común | TimesFM 2.5 univariante | 584 | 96,3980 | 130,7237 |
| Desarrollo temporal común | Chronos-2 univariante | 584 | 100,0225 | 136,4695 |
| Desarrollo temporal común | TimesFM 2.5 con XReg | 584 | 112,1213 | 644,6068 |

Ridge reduce aproximadamente un 14,42% el MAE final respecto al consumo específico. TFT no demuestra una mejora sobre Ridge con entradas actuales en la comparación temporal común. Los errores extremos de XReg se conservan y se diagnostican; no se eliminan para modificar la clasificación. La cobertura final de los intervalos nominales del 90% y 95% es aproximadamente 85,96% y 93,23%.

### Correspondencia entre artículo y archivos

| Componente | Evidencia congelada | Cálculo original |
|---|---|---|
| Depuración, exclusiones y diccionario | `research/depuracion_v1/`, `analysis/data/variables_source.csv` | `src/depuracion_v1/depurar.py` |
| Comparación estática | `research/analisis_desarrollo_v2/` | `src/piloto_desarrollo/continuar_analisis.py` |
| Ridge congelado y calibración | `research/calibracion_final_v1/` | `src/piloto_desarrollo/calibrar_final.py` |
| Prueba final y cobertura | `research/prueba_final_v1/` | `src/piloto_desarrollo/evaluar_prueba_final.py` |
| Hipótesis exploratorias | `research/hipotesis_exploratorias_v1/` | `src/piloto_desarrollo/contrastar_hipotesis.py` |
| Protocolo temporal CPU e historias | `research/modelos_reales_temporales_v1/` | `src/modelos_reales_temporales_v1/` |
| Ajustes temporales GPU y fundacionales | `research/modelos_reales_gpu_v1/` | `src/modelos_reales_gpu_v1/` |
| Las 13 figuras finales | `analysis/data/`, `analysis/DESCRIPTIVE_STATISTICS.json` | `analysis/make_figures.py`, `analysis/figure_design.py` |
| Siete tablas, ecuaciones y bibliografía | `latex/sections/`, `latex/references.bib` | Fuentes congeladas; las entradas numéricas están archivadas arriba |

El protocolo histórico también protege mediante hash un PDF real anterior y el PDF sintético. Se conservan en el archivo completo para verificar procedencia; el PDF sintético no aporta observaciones ni modelos ajustados a este análisis real.

### Límites de interpretación

Los resultados sustentan una línea base para las condiciones registradas. No demuestran ahorro energético medido, efectos causales, detección de fallos, predicción en línea validada ni generalización a otra planta. Se documentan los solapamientos de registros y la identidad física de lote no verificada. La prueba final ya había sido examinada antes de las extensiones temporales exploratorias; los análisis nuevos no constituyen otra prueba independiente. El manuscrito es una versión de investigación preparada para envío, no una aceptación editorial.

## Qué contiene el paquete

```text
README.md / README.es.md      Guías completas en inglés y español
reproduce.py                 Verificación, métricas, figuras, PDF y preparación de entrenamientos
package.json                 Modelos, recetas y revisiones fundacionales inmutables
analysis/                    Scripts de figuras y datos numéricos suficientes para reconstruirlas
latex/                       main.tex, bibliografía, figuras, tablas/secciones y PDF original
overleaf/                    ZIP listo para importar con main.tex en la raíz
src/                         Copia literal del código científico original, para inspección en GitHub
research/                    Datos originales, resultados, modelos y protocolos completos
manifests/                   Inventarios por archivo y SHA-256 de los assets
environment/                 Versiones observadas y dependencias
provenance/                  Trazabilidad y verificaciones del manuscrito original
verification/                Comprobaciones realizadas durante la publicación del paquete
runs/                        Salidas nuevas: nunca sustituyen a los resultados congelados
```

`research/` está completo en la carpeta local entregada. En GitHub se distribuye como asset de la release `v3-reproducibility`: **582 archivos**, 135.3 MB descomprimidos y 69.4 MB comprimidos. El repositorio conserva los datos derivados necesarios para regenerar las figuras sin descargar el archivo completo. No se requieren Git LFS ni GitHub Actions. Los pesos preentrenados se descargan de sus proveedores con una revisión exacta; no se redistribuyen las cachés ni los entornos virtuales.

## Inicio rápido: reconstruir el artículo

```bash
git clone https://github.com/fmarrabal/pelleting-energy-baseline-real-data.git
cd pelleting-energy-baseline-real-data
python -m venv .venv
```

En Windows PowerShell, activa con `.venv\Scripts\Activate.ps1`; en Linux/macOS, `source .venv/bin/activate`. El entorno observado es Python 3.13.13. Se registran versiones exactas en `environment/observed_versions.json`; el programa de verificación usa Python 3.11 o posterior. Usa Python 3.13 para aproximar el entorno archivado.

```bash
python -m pip install -r requirements-replay.txt
python reproduce.py verify
python reproduce.py metrics
python reproduce.py figures
python reproduce.py paper
```

El último comando necesita **pdfLaTeX y BibTeX** de TeX Live o MiKTeX. `python reproduce.py all` ejecuta los cuatro pasos. Las salidas aparecen en `runs/replay/`, incluidas las métricas recalculadas y el PDF compilado. El PDF original de `latex/` se conserva. La reconstrucción no entrena ni selecciona modelos. Las tablas científicas originales se conservan en LaTeX; el estudio sintético también regenera sus tablas numéricas mediante el script de figuras.

Para descargar y verificar todos los datos, modelos y predicciones:

```bash
python reproduce.py download
python reproduce.py verify
python reproduce.py metrics
```

La descarga emplea HTTPS, comprueba SHA-256 antes de extraer y verifica cada archivo. Si la descarga falla, puede obtenerse el ZIP desde [Releases](https://github.com/fmarrabal/pelleting-energy-baseline-real-data/releases/tag/v3-reproducibility) y guardarse en `downloads/`; el mismo comando valida y extrae el ZIP. La reconstrucción completa de métricas del estudio sintético añade la prueba de 169.876 orígenes cuando `research/` está instalado.

## Repetir entrenamiento e inferencia

Primero instala las dependencias de entrenamiento y una compilación de PyTorch compatible con la GPU. El experimento archivado utilizó **torch 2.14.0+cu130**, CUDA 13.0 y una NVIDIA RTX PRO 5000 Blackwell de 48 GB. No se presupone que cualquier GPU o versión produzca los mismos bits. Consulta el [instalador oficial de PyTorch](https://pytorch.org/get-started/locally/) para tu plataforma; `requirements-training.txt` fija el resto de dependencias directas observadas. Las dependencias transitivas y el hardware pueden afectar trayectorias estocásticas.

```bash
python -m pip install -r requirements-training.txt
python reproduce.py download
python reproduce.py prepare-training --recipe real-gpu --destination runs/fresh-gpu
python reproduce.py fetch-models --destination runs/fresh-gpu
python runs/fresh-gpu/execute.py
```

La carpeta de destino debe ser nueva y estar dentro de `runs/`. Se copia la evidencia necesaria y se excluyen los resultados previos de la etapa que se va a repetir. `REPRODUCTION_PLAN.json` deja constancia de las órdenes y hashes del código; `execute.py` comprueba esos hashes antes de ejecutar. `REPRODUCTION_COMPLETE.json` solo se crea si todos los procesos terminan correctamente. En el código GPU sintético se elimina únicamente la exigencia del nombre del directorio del entorno original; se conserva la obligación de CUDA y se registra el cambio.

| Receta | Secuencia ejecutada desde la carpeta independiente |
|---|---|
| `real-static` | `depuracion_v1/depurar.py` / `piloto_desarrollo/continuar_analisis.py` / `piloto_desarrollo/calibrar_final.py` / `piloto_desarrollo/evaluar_prueba_final.py` |
| `real-gpu` | `modelos_reales_gpu_v1/run_supervised.py` / `modelos_reales_gpu_v1/run_foundation.py` / `modelos_reales_gpu_v1/analyze.py` / `modelos_reales_gpu_v1/compare_cpu_gpu.py` |

Las recetas GPU comienzan desde los datos/tensores congelados: permiten reproducir esa etapa sin volver a seleccionar la comparación anterior. La receta estática real y la receta principal sintética comienzan en el Excel original correspondiente. Los experimentos exploratorios adicionales se conservan con sus scripts y salidas en `research/`; para repetirlos, ejecuta el script de `src/` correspondiente dentro de una copia nueva de ese árbol y documenta una nueva ejecución. No ejecutes scripts históricos directamente sobre la evidencia congelada, porque algunos escriben sus resultados en la misma carpeta.

TimesFM: `google/timesfm-2.5-200m-pytorch`, revisión `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`. Chronos-2: `amazon/chronos-2`, revisión `29ec3766d36d6f73f0696f85560a422f50e8498c`. `fetch-models` los descarga en la caché de la ejecución independiente; la inferencia histórica utiliza después modo local. Esta descarga necesita red y espacio adicional. Entrenar TFT/N-HiTS y ejecutar pesos fundacionales son operaciones diferentes; los fundacionales no se reajustan.

## Overleaf y compilación local

Descarga `overleaf/pelleting-energy-baseline-real-data-Overleaf.zip` o el asset homónimo de la release. En Overleaf usa **New Project → Upload Project**, selecciona el ZIP y establece **main.tex** como documento principal con **pdfLaTeX**. El ZIP incluye la bibliografía, su estilo, todas las figuras vectoriales empleadas, las tablas/secciones y el PDF de referencia con nombre distinto de `main.pdf`. No contiene Python, datos masivos ni rutas absolutas necesarias para compilar. No requiere shell escape. [Guía oficial de importación](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project).

También puedes compilar desde `latex/` con `pdflatex main.tex`, `bibtex main`, y dos pasadas más de `pdflatex main.tex`. Esas órdenes generan `main.pdf`; el PDF original con nombre largo permanece intacto. `reproduce.py paper` realiza las pasadas en una copia dentro de `runs/` y comprueba errores de referencias, citas, desbordamiento horizontal y número de páginas. Las fechas y metadatos de compilación pueden cambiar el hash de un PDF visualmente equivalente.

## Integridad, alcance de las verificaciones y resolución de problemas

| Fuente original | SHA-256 |
|---|---|
| `Datos Planta Peletizado.xlsx` | `d5968030da3e704e7505e9512151824d9384ae5e68259b71d01e1e9ee7647395` |

El inventario `manifests/research_files.json` identifica cada archivo científico; `manifests/tracked_files.json` protege el contenido de reconstrucción publicado en Git. Los manifiestos históricos se conservan como evidencia: algunos contienen rutas del equipo original o hashes de una entrega anterior. Los manifiestos nuevos son la referencia para esta distribución portátil.

La verificación numérica recalcula métricas desde predicciones congeladas y las contrasta con las tablas archivadas. La reproducción local de figuras/PDF verifica la compilación; no constituye una nueva validación estadística. `verification/` documenta las comprobaciones realmente ejecutadas. Las recetas de entrenamiento se suministran para repetir los ajustes; salvo que un informe lo indique expresamente, **el empaquetado no vuelve a entrenar toda la GPU**.

Si falta `pdflatex`, instala una distribución TeX o usa el ZIP de Overleaf. Si falta `research/`, ejecuta `download`. Si falla un hash, conserva el archivo modificado y usa una copia limpia de la release: no edites el manifiesto para ocultar la diferencia. Si no hay CUDA disponible, aún puedes verificar métricas, regenerar figuras y compilar el artículo; las recetas GPU fallan explícitamente. Si faltan pesos en modo offline, ejecuta `fetch-models` con el mismo destino de la receta. Solo deben cargarse archivos joblib/PT de procedencia fiable y hash verificado.

## Citación, derechos y contacto

La ficha `CITATION.cff` identifica esta entrega del archivo de reproducción y su mantenedor de GitHub. El manuscrito suministrado no incorpora todavía autores ni DOI; no se inventan esos datos. Para citar resultados científicos, utiliza el título del manuscrito, la versión v3 y esta release, y actualiza la referencia cuando exista publicación/DOI. El acceso público no concede por sí mismo una licencia general de reutilización: véase `RIGHTS.md`. Los modelos, bibliotecas y archivos de terceros conservan sus propios términos. Usa Issues para comunicar problemas de reproducción, indicando versión, sistema operativo, receta y mensaje de error sin incluir credenciales. GitHub Actions permanece desactivado; los cálculos de esta entrega se verifican localmente.


## Verificación de esta entrega (23 de septiembre de 2026)

Se recalcularon 12 filas de resultados desde las predicciones y se regeneraron todas las figuras. Las 24 páginas reconstruidas coinciden con el PDF original tanto en texto extraído como en píxeles a 72 dpi. Se verificaron los hashes del archivo completo, se analizaron sintácticamente 40 scripts científicos y se preparó una ejecución GPU independiente sin iniciar entrenamiento. Los PDF originales permanecen idénticos byte a byte. Los informes están en `verification/`.
