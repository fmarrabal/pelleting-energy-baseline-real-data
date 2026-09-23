# Explainable energy baselines for industrial feed pelleting: a chronological comparison of regression, temporal and foundation models

**English** | [Español](README.es.md)

Reproducibility archive for **Paper_Peletizado_Datos_Reales_v3.pdf**, including original data, code, frozen results and a complete LaTeX/Overleaf project.

Companion study: [GitHub](https://github.com/fmarrabal/pelleting-energy-baselines-synthetic).

## Scientific question and scope

This study asks how well an explainable, conditional energy baseline predicts the electricity associated with an industrial pelleting record, and whether more complex temporal and foundation models improve that baseline. Its unit of analysis is an **irregular industrial record**, not a ten-second sensor observation. The response is **pelleting electricity in kWh per record**. The baseline is retrospective and conditional on consolidated production descriptors: their availability as planned values before production has not been independently established.

The original workbook `Datos Planta Peletizado.xlsx` contains 2,745 records: 730 from Pellet 1, 1,148 from Pellet 2 and 867 from Pellet 3. The analysis identifies 32 source fields. Three zero-energy responses are retained in the audit but excluded from the valid energy-target population; targets are not imputed. The archive records all additional filters and auxiliary imputations. The older `Datos_Peletizado_Base.xlsx` belongs to the synthetic study's calibration/fidelity evidence and must not be silently substituted for this workbook.

### Inputs and outputs

| Role | Variable | Interpretation |
|---|---|---|
| Response | `energia_peletizado_kWh` | Electricity attributed to the pelleting stage, kWh per record |
| Numeric input | `masa_dosificada_t` | Dosed mass, converted from kg to tonnes |
| Numeric input | `baches_dosificacion` | Number of dosing batches recorded for the operation |
| Numeric inputs | `hora_sin`, `hora_cos` | Cyclic representation of the record-origin hour |
| Numeric inputs | `dia_sin`, `dia_cos` | Cyclic representation of the weekday |
| Categorical inputs | `linea`, `subfamilia`, `forma`, `salida_programada` | Machine, product subgroup, presentation and planned output type |
| Temporal history | Previous 16 completed records of the same machine | Past energy and associated descriptors; completion strictly precedes the target origin |
| Outputs | Point prediction and 90%/95% intervals | Conditional energy baseline and residual-calibrated uncertainty |

The numerical/categorical pipeline is fitted only on the training population. The frozen static design has 22 encoded columns. Current measured energy, current operating duration, current operational averages and subsequently produced mass are excluded as predictors. Completion timestamps proxy observation availability; they are not a verified database-publication log. `fila_fuente` preserves the original Excel row identity.

### Chronology, model selection and uncertainty

The static comparison uses four expanding development folds starting on 25 January, 1 February, 8 February and 15 February 2025, with 800 validation records in total. Seven model families/baselines are compared: Ridge, histogram gradient boosting, ExtraTrees, MLP, machine-level specific-energy baseline, recent specific-energy baseline and machine median. The frozen Ridge model is fitted on 1,542 eligible development records, calibrated on 380 records and evaluated on 798 final-test records from 6 March onwards. Exact membership and exclusions are in the CSVs and manifests; do not reconstruct membership by row number alone.

The modern temporal comparison is a separate **development-only** experiment. Strict same-machine histories reset at partition boundaries. Requiring 16 completed predecessors leaves 584 common validation records. TFT and N-HiTS are refitted with seeds 11, 29 and 47; inner validation chooses epochs with a maximum of 40, a minimum of 10 before early stopping and patience 8. Foundation weights remain frozen. All eight variants are compared on the same common records. This population is different from the 798-record final test.

For residuals `e = prediction - observed`, MAE is `mean(abs(e))`, RMSE is `sqrt(mean(e**2))` and bias is `mean(e)`. WAPE is `sum(abs(e))/sum(observed)` and is stored as a fraction. Fixed intervals use calibration residual quantiles with nonnegative lower bounds. Coverage is empirical; temporal dependence and drift limit exchangeability-based interpretations. Post-hoc rolling calibration and residual adaptation remain exploratory.

### Reference results to reproduce

| Evaluation population | Model | n | MAE (kWh) | RMSE (kWh) |
|---|---|---:|---:|---:|
| Frozen final test | Ridge | 798 | 52.4489 | 80.4186 |
| Frozen final test | Specific-energy baseline | 798 | 61.2872 | 92.0540 |
| Common temporal development | Ridge, current inputs | 584 | 53.5457 | 77.0140 |
| Common temporal development | TFT | 584 | 53.6507 | 79.1050 |
| Common temporal development | N-HiTS | 584 | 65.2784 | 92.5581 |
| Common temporal development | Chronos-2 with covariates | 584 | 85.5268 | 120.8235 |
| Common temporal development | TimesFM 2.5 univariate | 584 | 96.3980 | 130.7237 |
| Common temporal development | Chronos-2 univariate | 584 | 100.0225 | 136.4695 |
| Common temporal development | TimesFM 2.5 with XReg | 584 | 112.1213 | 644.6068 |

The final-test Ridge MAE is approximately 14.42% lower than the specific-energy baseline. TFT does not establish an improvement over current-input Ridge in the common temporal comparison. The unusually large XReg errors are retained, diagnosed and reported; they are not removed to improve the ranking. Fixed 90% and 95% intervals cover approximately 85.96% and 93.23% of final-test observations.

### Evidence map

| Manuscript component | Frozen evidence | Original computation |
|---|---|---|
| Cleaning, exclusions, variable dictionary | `research/depuracion_v1/` and `analysis/data/variables_source.csv` | `src/depuracion_v1/depurar.py` |
| Static development comparison | `research/analisis_desarrollo_v2/` | `src/piloto_desarrollo/continuar_analisis.py` |
| Frozen Ridge and calibration | `research/calibracion_final_v1/` | `src/piloto_desarrollo/calibrar_final.py` |
| Final test and interval coverage | `research/prueba_final_v1/` | `src/piloto_desarrollo/evaluar_prueba_final.py` |
| Exploratory hypotheses | `research/hipotesis_exploratorias_v1/` | `src/piloto_desarrollo/contrastar_hipotesis.py` |
| CPU temporal protocol, donor audit | `research/modelos_reales_temporales_v1/` | `src/modelos_reales_temporales_v1/` |
| GPU temporal fits and foundation inference | `research/modelos_reales_gpu_v1/` | `src/modelos_reales_gpu_v1/` |
| All 13 final figures | `analysis/data/`, `analysis/DESCRIPTIVE_STATISTICS.json` | `analysis/make_figures.py`, `analysis/figure_design.py` |
| Seven tables, equations, bibliography | `latex/sections/`, `latex/references.bib` | Frozen manuscript sources; numerical inputs are archived above |

Historical protocol hashes also protect an earlier real-data PDF and the synthetic manuscript PDF. Those small files remain in the research asset solely to satisfy provenance checks; the synthetic PDF contributes no observations or fitted model to this real-data analysis.

### Interpretation limits

These observations support an energy baseline for the recorded operating conditions. They do not establish measured energy savings, causal variable effects, fault-detection accuracy, a verified online forecast, or performance on an independent plant. Record overlap and uncertain physical batch identity are documented. The final test had already been inspected before the exploratory temporal extensions; new analyses must not be presented as a new untouched test. The manuscript is a research version prepared for submission, not evidence of journal acceptance.

## Package contents

```text
README.md / README.es.md      Complete English and Spanish guides
reproduce.py                 Integrity, metrics, figures, PDF and fresh-run preparation
package.json                 Models, recipes and immutable foundation revisions
analysis/                    Plotting scripts and self-contained numerical figure inputs
latex/                       main.tex, bibliography, figures, tables/sections and original PDF
overleaf/                    Import-ready ZIP with main.tex at its root
src/                         Verbatim original scientific scripts, browsable on GitHub
research/                    Complete raw data, outputs, checkpoints and protocols
manifests/                   Per-file inventories and release-asset SHA-256
environment/                 Observed versions and dependencies
provenance/                  Original manuscript provenance and QA records
verification/                Checks actually performed for this distribution
runs/                        New outputs, separate from the frozen scientific evidence
```

The delivered local folder already contains `research/`. On GitHub this tree is distributed as an asset of release `v3-reproducibility`: **582 files**, 135.3 MB uncompressed and 69.4 MB compressed. The Git repository itself includes the derived data needed for all final figures. Neither Git LFS nor GitHub Actions is required. Pretrained weights are downloaded from their providers at exact revisions; environment installations and model caches are not redistributed.

## Quick start: rebuild the paper

```bash
git clone https://github.com/fmarrabal/pelleting-energy-baseline-real-data.git
cd pelleting-energy-baseline-real-data
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS. The observed environment is Python 3.13.13. Exact observed direct-package versions are in `environment/observed_versions.json`; the integrity utility requires Python 3.11 or newer. Use Python 3.13 to approximate the archived environment.

```bash
python -m pip install -r requirements-replay.txt
python reproduce.py verify
python reproduce.py metrics
python reproduce.py figures
python reproduce.py paper
```

The last command requires **pdfLaTeX and BibTeX** from TeX Live or MiKTeX. `python reproduce.py all` runs all four stages. New outputs are written to `runs/replay/`, including recomputed metrics and the compiled PDF; the original PDF in `latex/` is preserved. Replay does not fit or select models. Original scientific tables remain in the manuscript sources; the synthetic plotting script also regenerates its numerical LaTeX tables.

To obtain all raw data, checkpoints and complete predictions:

```bash
python reproduce.py download
python reproduce.py verify
python reproduce.py metrics
```

The HTTPS downloader checks SHA-256 before extraction and then validates every extracted file. Alternatively obtain the research ZIP from [Releases](https://github.com/fmarrabal/pelleting-energy-baseline-real-data/releases/tag/v3-reproducibility), place it in `downloads/`, and run the same command to verify/extract it. For the synthetic study, metric replay additionally checks the 169,876-origin original test when `research/` is installed.

## Repeat training and inference

Install training dependencies and a PyTorch build appropriate for the GPU. The archived GPU environment used **torch 2.14.0+cu130**, CUDA 13.0 and an NVIDIA RTX PRO 5000 Blackwell with 48 GB. Different devices or versions need not reproduce identical floating-point bits. Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) for your platform; `requirements-training.txt` pins the other observed direct dependencies. Transitive dependencies and hardware can affect stochastic training trajectories.

```bash
python -m pip install -r requirements-training.txt
python reproduce.py download
python reproduce.py prepare-training --recipe real-gpu --destination runs/fresh-gpu
python reproduce.py fetch-models --destination runs/fresh-gpu
python runs/fresh-gpu/execute.py
```

The destination must be new and inside `runs/`. The utility copies required evidence while excluding prior outputs of the stage being repeated. `REPRODUCTION_PLAN.json` records commands and code hashes; `execute.py` validates those hashes before execution. `REPRODUCTION_COMPLETE.json` is written only after every child process succeeds. For synthetic GPU scripts, a portability patch removes only the original environment-directory-name assertion, preserving the CUDA requirement and recording before/after hashes.

| Recipe | Commands executed from the independent run directory |
|---|---|
| `real-static` | `depuracion_v1/depurar.py` / `piloto_desarrollo/continuar_analisis.py` / `piloto_desarrollo/calibrar_final.py` / `piloto_desarrollo/evaluar_prueba_final.py` |
| `real-gpu` | `modelos_reales_gpu_v1/run_supervised.py` / `modelos_reales_gpu_v1/run_foundation.py` / `modelos_reales_gpu_v1/analyze.py` / `modelos_reales_gpu_v1/compare_cpu_gpu.py` |

GPU recipes start from the exact archived cleaned data/tensors, reproducing that stage without repeating upstream model selection. The real static and synthetic primary recipes start from the corresponding original workbook. Additional exploratory experiments retain their scripts and outputs in `research/`; rerun the corresponding `src/` script inside a fresh copy of that tree and record a new execution. Do not run historical scripts directly against frozen evidence: some write results beside the source script.

TimesFM uses `google/timesfm-2.5-200m-pytorch` at `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`. Chronos-2 uses `amazon/chronos-2` at `29ec3766d36d6f73f0696f85560a422f50e8498c`. `fetch-models` populates the independent run's model cache; original inference scripts then load locally. Downloads require network access and additional disk space. Training TFT/N-HiTS and executing frozen foundation weights are different operations; the latter does not fine-tune weights.

## Overleaf and local LaTeX

Download `overleaf/pelleting-energy-baseline-real-data-Overleaf.zip` or the identically named release asset. In Overleaf select **New Project → Upload Project**, upload the ZIP and choose **main.tex** as the main document with **pdfLaTeX**. The ZIP contains the bibliography and style, all required vector figures, tables/sections, and the reference PDF under a name distinct from `main.pdf`. It excludes Python, bulk datasets and machine-specific compilation paths. Shell escape is not required. See [Overleaf's official import guide](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project).

Local compilation from `latex/` is `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. These commands create `main.pdf` without overwriting the named reference PDF. `reproduce.py paper` performs the passes in a copy under `runs/` and checks undefined references/citations, horizontal overflows and page count. Build dates and graphic metadata can change PDF bytes despite equivalent visual content.

## Integrity, verification boundaries and troubleshooting

| Original source | SHA-256 |
|---|---|
| `Datos Planta Peletizado.xlsx` | `d5968030da3e704e7505e9512151824d9384ae5e68259b71d01e1e9ee7647395` |

`manifests/research_files.json` identifies every archived scientific file; `manifests/tracked_files.json` protects the tracked replay inputs. Historical manifests remain as evidence and can reference the original workstation or earlier delivery hashes. The new manifests govern this portable distribution.

Numerical verification recomputes metrics from frozen predictions and reconciles them with archived results. Figure/PDF replay checks reconstruction and compilation; it is not new statistical validation. `verification/` records checks actually run. Fresh-training recipes are provided for repeating fits; unless a verification report explicitly says otherwise, **packaging does not rerun the entire GPU experiment**.

If `pdflatex` is unavailable, install TeX or use the Overleaf ZIP. If `research/` is missing, run `download`. If a hash differs, preserve the changed file and obtain a clean release copy instead of changing the expected hash to conceal the difference. Without CUDA, metric/figure/PDF replay still works; GPU recipes explicitly fail. If an offline model is missing, run `fetch-models` with the same destination as the recipe. Only load trusted, hash-verified joblib/PT artifacts.

## Citation, rights and contact

`CITATION.cff` identifies this reproducibility release and its GitHub maintainer. The supplied manuscript currently has no author byline or DOI; neither is invented here. Cite the manuscript title, v3 and this release for these archived results, and update the reference when publication metadata becomes available. Public access alone does not grant a blanket reuse license: see `RIGHTS.md`. Third-party models, libraries and assets retain their own terms. Report reproducibility problems through Issues, specifying version, OS, recipe and error without credentials. GitHub Actions remains disabled; release checks run locally.


## Checks for this release (23 September 2026)

12 result rows were recomputed from predictions and every figure was regenerated. All 24 rebuilt pages match the original PDF in extracted text and pixel-for-pixel at 72 dpi. Full-archive hashes were checked, 40 scientific scripts were syntax-parsed, and an independent GPU run was prepared without launching training. Original PDFs remain byte-identical. Reports are in `verification/`.
