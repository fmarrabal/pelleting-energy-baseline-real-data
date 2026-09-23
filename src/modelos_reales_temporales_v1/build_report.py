"""Create a separate LaTeX supplement; preserve all existing manuscripts."""
from common import *
import shutil,subprocess
from analyze import LABELS,ORDER,MAIN

def tex_escape(s):
    return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')

def main():
    report=ROOT/'report';report.mkdir(exist_ok=True)
    scores=pd.read_csv(ROOT/'metrics.csv',index_col=0)
    effects=pd.read_csv(ROOT/'paired_contrasts.csv',index_col=0)
    seed=pd.read_csv(ROOT/'metrics_seeds.csv')
    audit=json.loads((ROOT/'AUDIT.json').read_text(encoding='utf8'))
    numeric=json.loads((ROOT/'NUMERICAL_CHECKS.json').read_text(encoding='utf8'))
    assert numeric['status']=='PASS'
    counts=json.loads((ROOT/'PROTOCOL.json').read_text(encoding='utf8'))['sample_counts']
    baseline=scores.loc['Ridge_actual','MAE_kWh']
    best=min(['TFT','NHITS','TimesFM_2.5_covariates','Chronos_2_covariates'],key=lambda x:scores.loc[x,'MAE_kWh'])
    direction='lower' if scores.loc[best,'MAE_kWh']<baseline else 'higher'
    change=abs(100*(scores.loc[best,'MAE_kWh']/baseline-1))
    table='\n'.join(f'{tex_escape(LABELS[m])} & {r.MAE_kWh:.2f} & {r.RMSE_kWh:.2f} & {r.bias_kWh:+.2f} & {100*r.WAPE:.1f} & {r.MAE_change_vs_Ridge_pct:+.1f} \\\\' for m,r in scores.iterrows())
    flow='\n'.join(f"{r['fold']} & {r['inner_train']} & {r['inner_val']} & {r['train']} & {r['val']} \\\\" for r in counts)
    stability='; '.join(f"{tex_escape(LABELS[m].split(' (')[0])}: {seed[seed.modelo.eq(m)].MAE_kWh.min():.2f}--{seed[seed.modelo.eq(m)].MAE_kWh.max():.2f} kWh" for m in ['TFT','NHITS'])
    comparative=(r'None of the four tested modern-model specifications reduced aggregate MAE relative to the matched current-record Ridge control.'
      if all(scores.loc[m,'MAE_kWh']>=baseline for m in ['TFT','NHITS','TimesFM_2.5_covariates','Chronos_2_covariates'])
      else r'At least one tested modern-model specification reduced the observed aggregate MAE; the paired uncertainty intervals and independent future validation remain necessary for interpreting this difference.')
    text=r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype,amsmath,amssymb,booktabs,array,graphicx,xcolor,geometry,fancyhdr,hyperref}
\geometry{margin=20mm,top=23mm,bottom=23mm}
\hypersetup{colorlinks=true,urlcolor=blue!55!black,linkcolor=blue!55!black,citecolor=blue!55!black}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Industrial pelleting: real-record temporal benchmark}\fancyhead[R]{\small Exploratory supplement}\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
\newcommand{\fig}[3]{\begin{center}\includegraphics[width=#2\linewidth]{../figures/#1.png}\end{center}{\small #3}\par}
\begin{document}
{\LARGE\bfseries Modern temporal models for a real industrial pelleting energy baseline}\par
{\large TFT, N-HiTS, TimesFM 2.5 and Chronos-2}\par
{\small Reproducible exploratory supplement; 22 September 2026.}\par

\textbf{Scope.} This experiment uses real industrial records from \textit{Datos Planta Peletizado.xlsx}, through the previously audited cleaned table. It contains no synthetic high-frequency measurements. The previous frozen baseline, final-test predictions and manuscripts are unchanged. The new comparison uses development data only and must not be described as an additional independent final test.

\section*{1. Question, target and chronological design}
The question is whether short histories of completed operations improve conditional estimates of the energy of a nominated industrial record. The target is pelleting energy in kWh per record. Four successive weekly outer validation blocks start on 25 January, 1 February, 8 February and 15 February 2025. The horizon is one record, not one fixed interval of physical time.

For record $i$, with origin $o_i$, machine $m_i$ and partition interval $[a,b)$, the context is
\begin{equation}
\mathcal H_i=\operatorname{last}_{16}\{j:m_j=m_i,\ a\leq o_j<b,\ f_j<o_i\},
\end{equation}
where $f_j$ is pelleting completion time and records are ordered by $(f_j,\text{source row})$. The target also satisfies $a\leq o_i<b$ and $f_i<b$. We discard targets with fewer than 16 eligible historical records. Every context remains inside its own partition; later validation queries may use earlier completed validation records, without updating learned model weights. A week-end completion restriction preserves the original development evaluation policy.

Of 800 original validation targets, 216 lack enough within-partition history. All eight specifications below use exactly the remaining \textbf{584 targets}. Training targets are also matched between Ridge, TFT and N-HiTS. The preceding seven-day block supplies inner validation; its histories independently restart at that block's boundary.

\begin{center}\small
\begin{tabular}{lrrrr}\toprule
Outer block start & Inner train & Inner valid. & Refit train & Outer valid.\\\midrule
@@FLOW@@
\bottomrule\end{tabular}\end{center}

\textbf{Inputs.} Six numeric variables comprise dosified mass, dosing batch count, and sine/cosine encodings of origin hour and weekday. Four categorical variables describe machine, product subfamily, product form and scheduled output. Temporal models receive these variables for the context and target record, plus historical energy. Current energy, completion time, duration and operational averages are not predictors. All training-based scaling and category encoding use training records only.

The resulting estimand is $\widehat E_i=g(\mathbf x_i,\{E_j,\mathbf x_j\}_{j\in\mathcal H_i})$. The table contains consolidated process values: the availability of planned mass, batch count and product metadata at origin has not been established. Results are therefore \textbf{retrospective conditional estimates}, not validated pre-operation forecasts. Completion is a proxy for label availability. Unique physical batch/cycle identity remains unverified.

\clearpage
\section*{2. Models and matched-sample results}
TFT and N-HiTS use the official NeuralForecast implementations\cite{tft,nhits}, a single compact configuration per family, and seeds 11, 29 and 47. TFT has hidden size 32 and four attention heads. N-HiTS has three blocks with two 64-unit layers each. Both use Adam, learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 64 and standardized-target MSE. Each seed selects its epoch on inner validation (maximum 40, patience eight, at least ten epochs evaluated) and is refitted from scratch on the full outer training set. Their reported predictions average the three clipped seed predictions.

TimesFM 2.5 and Chronos-2 retain frozen pretrained weights\cite{timesfm,chronos}. Energy-only variants are ablation controls. The main TimesFM variant uses native \texttt{XReg + TimesFM}, ridge penalty 10, fitted separately for every query to prevent cross-query leakage. Chronos-2 uses native historical and future covariates with \texttt{cross\_learning=False}. Their weights receive no plant-specific fine-tuning. Both Ridge controls have $\alpha=10$; the window control receives all flattened historical inputs and the current covariates. Negative final energy predictions are clipped to zero; raw values remain in the exported tables.

\begin{center}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}\toprule
Specification & MAE & RMSE & Bias & WAPE (\%) & $\Delta$MAE (\%)\\\midrule
@@TABLE@@
\bottomrule\end{tabular}\end{center}
{\small Energy errors are in kWh per record. $\Delta$MAE is relative to current-record Ridge; positive values indicate worse performance. Each row uses $n=584$.}

\fig{01_comparison}{1}{\textbf{Figure 1.} Absolute accuracy and paired error differences on identical observations. Error bars are marginal 95\% intervals from 5,000 circular bootstrap resamples of blocks of three observed dates, with all machines sampled together. Positive paired differences favor the temporal model. The results directory also supplies Bonferroni-adjusted intervals across seven contrasts.}

@@COMPARATIVE@@ Among the four modern covariate specifications, @@BEST@@ has the smallest MAE, @@BESTMAE@@ kWh, which is @@CHANGE@@\% @@DIRECTION@@ than the matched current-record Ridge control (@@BASELINE@@ kWh).

TFT improves MAE by only @@TFTGAIN@@ kWh (95\% paired interval: @@TFTLOW@@ to @@TFTHIGH@@ kWh), compatible with no improvement. Its RMSE and absolute bias are higher than Ridge's. These results do not support replacing the frozen Ridge baseline with TFT.

\clearpage
\section*{3. Predictions at the individual-record level}
\fig{02_all_record_predictions}{1}{\textbf{Figure 2.} The same 584 matched observations are plotted for every model; colors identify the three production lines. The main axes share a 0--900 kWh detail range. Three TimesFM + XReg predictions exceed that range and appear in a full-range inset. No records are removed from the results. The two energy-only foundation-model ablations are retained in Figure 1 and the machine-readable tables.}

These plots evaluate the conditional estimate of recorded energy. They do not assess trajectories within a production cycle, seconds-ahead responses or hypothetical energy savings. Differences between recorded and estimated energy are residuals for review, not diagnosed equipment faults. The new temporal models have not replaced the frozen operational baseline.

\textbf{TimesFM + XReg extremes.} Three predictions exceed 2,000 kWh, reaching 12,202.66 kWh. A post-result diagnostic reconstructs these outputs and locates the extreme contribution in XReg: current mass lies far outside the narrow range of 16 nearly constant historical masses. Solving the same regression independently in float64 changes its normalized prediction by less than 0.0003 in these cases. The result reflects local extrapolation, rather than a simple floating-point failure. All cases and primary predictions are retained; no residual-driven correction or fallback is fitted after inspection.

\textbf{A comparison that must be kept separate.} The archived Ridge model achieves @@OLDMAE@@ kWh MAE on these same retained rows, but had used more training targets. That value is supplied as context only. The primary Ridge controls in this supplement are refitted on the same eligible training targets as the neural models. Likewise, MAEs from the previous 800-row experiment cannot be compared directly with the 584-row values without acknowledging the changed population.

\textbf{Short-history limitation.} Sixteen completed records preserve 73\% of the original validation targets while enforcing independent partitions. This is a deliberately limited context for a first real-data experiment. The results do not establish the performance of longer contexts, alternative pooling choices, larger neural networks or fine-tuned foundation models. The architectures can be applied to the real records, but their high-frequency forecasting claims require different data and a different validation design.

\clearpage
\section*{4. Temporal heterogeneity and seed stability}
\fig{03_stratified_error}{1}{\textbf{Figure 3.} MAE by outer week and machine. Each cell uses the shared eligible records of that stratum. Warm-up exclusions differ among machines and weeks; detailed sample counts accompany the numeric tables.}

\fig{04_training_and_seeds}{1}{\textbf{Figure 4.} All inner-validation learning curves (four folds, three seeds per architecture), and aggregate outer-validation MAE for each seed. Curve colors encode the outer block. The black diamond denotes the prespecified three-seed ensemble, and the dashed line the current-record Ridge control. Inner curves guide epoch selection only; outer metrics did not guide training.}

Across individual seeds, aggregate MAE ranges are @@STABILITY@@. Seed variability is part of the result. Choosing the lowest-error seed after inspecting outer validation would produce a different, optimistically selected estimate. No such seed selection is used here.

\clearpage
\section*{5. Interpretation and reproducibility}
@@COMPARATIVE@@ This finding concerns the specific short-context experiment and its selection budget. It cannot establish that these architectures are intrinsically unsuitable for industrial energy forecasting. The foundation models were used in zero-shot mode, whereas the small supervised models learned from hundreds of labeled plant records. Those are different adaptation regimes, explicitly reported rather than treated as equal training budgets.

The comparison is exploratory because the development data and earlier baselines have already been inspected. Bootstrap intervals describe uncertainty within this development sample; even multiplicity correction does not create independent confirmation. One active-record horizon has been evaluated. Multihorizon physical-time forecasting, inactive states, prospective covariate availability, physical monotonicity and calibrated uncertainty for these new models remain untested. Architectures described as interpretable do not, by themselves, provide a validated process explanation; no causal interpretation of attention weights is claimed.

The contribution suitable for the manuscript is a transparent empirical comparison and its boundary conditions. The frozen final test retains its original evidential status. Any subsequent selection, longer-context study or model adaptation should be declared as development work and confirmed on newly acquired chronological observations before supporting a deployment claim.

\textbf{Execution and checks.} Both runs used CPU because the 48 GB GPU was occupied by another workload. No competing process was stopped. The protocol and source hashes were saved before training. An independent post-run join checked @@DONORS@@ historical donor relations against source timestamps and machine identities. Zero temporal, machine or partition violations were found. Checkpoint reload checks for TFT and N-HiTS and a Chronos-2 batch-size consistency check passed. Original data, the frozen model, final-test predictions and the previous real and synthetic manuscripts retained their SHA-256 hashes.

\textbf{Artifacts.} The package contains the frozen protocol, source code, individual predictions, all seed predictions, sample membership, exclusions, donor audit, metrics, bootstrap contrasts, vector/raster figures and this LaTeX source. Trained small-model checkpoints remain in the local experiment directory. Foundation checkpoints are identified by immutable revisions; weights and the original plant workbook are not bundled in the supplement archive. Reproduction uses local files and does not require uploading plant records to a service.

\begin{thebibliography}{9}\small
\bibitem{tft} Nixtla. \emph{Temporal Fusion Transformer: NeuralForecast documentation}. \url{https://nixtlaverse.nixtla.io/neuralforecast/models.tft.html}. Accessed 22 September 2026.
\bibitem{nhits} Nixtla. \emph{N-HiTS: NeuralForecast documentation}. \url{https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html}. Accessed 22 September 2026.
\bibitem{timesfm} Google Research. \emph{TimesFM: official implementation, including the 2.5 XReg interface}. \url{https://github.com/google-research/timesfm}. Checkpoint \texttt{google/timesfm-2.5-200m-pytorch}, revision \texttt{1d952420fba87f3c6dee4f240de0f1a0fbc790e3}.
\bibitem{chronos} Amazon Science. \emph{Chronos forecasting: official implementation of Chronos-2}. \url{https://github.com/amazon-science/chronos-forecasting}. Checkpoint \texttt{amazon/chronos-2}, revision \texttt{29ec3766d36d6f73f0696f85560a422f50e8498c}.
\end{thebibliography}
\end{document}
'''
    substitutions={'FLOW':flow,'TABLE':table,'COMPARATIVE':comparative,'BEST':tex_escape(LABELS[best]),
      'BESTMAE':f'{scores.loc[best,"MAE_kWh"]:.2f}','CHANGE':f'{change:.1f}','DIRECTION':direction,
      'BASELINE':f'{baseline:.2f}','OLDMAE':f'{audit["archived_Ridge_MAE_same_subset"]:.2f}',
      'TFTGAIN':f'{effects.loc["TFT","MAE_improvement_vs_Ridge_kWh"]:.2f}',
      'TFTLOW':f'{effects.loc["TFT","CI95_low"]:.2f}','TFTHIGH':f'{effects.loc["TFT","CI95_high"]:.2f}',
      'STABILITY':stability,'DONORS':f'{audit["history_relations_checked"]:,}'}
    for k,v in substitutions.items():text=text.replace('@@'+k+'@@',v)
    assert '@@' not in text
    (report/'main.tex').write_text(text,encoding='utf8')
    for _ in range(2):
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'],cwd=report,capture_output=True,text=True,encoding='utf8',errors='replace')
        if result.returncode:raise RuntimeError(result.stdout[-6000:])
    log=(report/'main.log').read_text(errors='replace')
    for token in ['undefined references','undefined citations','Overfull \\hbox','Missing character:']:
        if token in log:raise RuntimeError('PDF review required: '+token)
    out=BASE/'output/pdf/Anexo_Modelos_Temporales_Datos_Reales.pdf'
    shutil.copyfile(report/'main.pdf',out)
    print('PDF_CREATED',out,flush=True)

if __name__=='__main__':main()
