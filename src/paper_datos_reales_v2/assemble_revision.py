"""Assemble expanded manuscript from preserved v1 and independently editable sections."""
from pathlib import Path
import re
import pandas as pd
H=Path(__file__).resolve().parent
s=(H.parent/'paper_datos_reales_v1/main.tex').read_text(encoding='utf8')
s=s.replace('temporal evaluation and uncertainty across three production lines','chronological comparison with temporal and foundation models')
a=s.index('Energy assessment in feed pelleting requires');b=s.index('\\end{abstract}',a)
s=s[:a]+r'''An industrial energy baseline should explain what consumption it estimates, which production inputs it uses and what information was available at each evaluation origin. We examine 2,745 real operational records from three feed-pelleting lines, with recorded pelleting electricity as the response in kWh per record. The inputs describe dosed mass, dosing batch count, machine, product subfamily, presentation, bag/bulk category and cyclic calendar coordinates. An original chronological comparison evaluates seven estimators on 800 development records. Its frozen Ridge baseline, fitted on 1,542 records and calibrated on 380, reduces MAE from 61.29 to 52.45 kWh on the original 798-record test, a 14.42% reduction relative to machine-specific energy intensity. A subsequent exploratory GPU benchmark constructs strictly separated, completed same-machine histories of 16 records. On 584 matched development targets, TFT has MAE 53.65 kWh versus 53.55 for its matched Ridge control; the paired difference does not demonstrate improvement. N-HiTS and native TimesFM 2.5 and Chronos-2 variants also fail to improve this control under the tested configuration. An exact additive decomposition explains the frozen Ridge predictions, while residual and interval diagnostics expose machine-dependent bias and imperfect coverage. Updating interval calibration mainly widens the bounds, with little improvement in interval score. The evidence supports a transparent retrospective conditional baseline. Irregular aggregated records, unverified planned-input availability and unresolved meter/order boundaries limit interpretation as pre-operation, multi-horizon forecasting. Predictive gains and out-of-interval records do not establish energy savings or faults.
''' +s[b:]
s=s.replace('interpretable regression; prediction intervals.','interpretable regression; temporal foundation models; prediction intervals.')
s=s.replace('Their availability does not establish that aggregated industrial records contain the trajectories, future inputs or repeated cycles needed for a useful comparison.',
            'The relevant industrial question is whether these models add predictive value after measurement units, inputs, training eligibility and evaluation targets have been matched. Aggregated operations provide different information from dense trajectories of power or equipment state.')
a=s.index('We address these issues using real records');b=s.index('\\section{Industrial records',a)
s=s[:a]+r'''We address these issues through complementary analyses of the same industrial source. The original frozen-model test assesses a transparent baseline against machine-specific energy intensity. A subsequent exploratory comparison evaluates temporal and foundation models on matched event histories. Residual adaptation and calibration updates examine how completed recent operations could modify a baseline. These analyses share a source but have different evaluation populations and evidential roles.

\input{sections/objectives}

'''+s[b:]
s=s.replace('Three pelleting outcomes are non-positive or absent and are not imputed.','Three pelleting outcomes are zero and are not imputed.')
s=s.replace('physical presentation, dispatch category and cyclic encodings','physical presentation, bag/bulk category and cyclic encodings')
s=s.replace('Machine and product/dispatch categories','Machine and product/bag-bulk categories')
# Append the detailed dictionary after the general availability overview.
anchor='\\section{Methods}'
s=s.replace(anchor,'\\input{sections/variables}\n\n'+anchor)
# Explicit status of the actual work, retaining the original final test.
s=s.replace('The current manuscript reconstructs its figures and tables from archived predictions, without new model fitting or new test predictions.',
            'The temporal CPU/GPU benchmark was subsequently conducted on development records only. This manuscript revision reconstructs the original and GPU results from their archived predictions; it performs no additional model selection, fitting or final-test prediction.')
anchor='\\subsection{Exploratory product and temporal adjustments}'
s=s.replace(anchor,'\\input{sections/temporal_methods}\n\n'+anchor)
# Accurate figure captions for revised point and interval representations.
s=s.replace('Electricity and dosed mass for every positive-energy record, followed by the product-family distribution for each machine. Counts printed beside the bars refer to all source records, including the three excluded from the positive-energy scatter. Shared scatter limits retain the complete range. Product labels are family classifications rather than verified recipes.',
 'Detailed input/output distributions. Panels (a)--(c) retain all 2,742 positive-energy records, with common axes and machine-specific summaries. Panel (d) uses every source mass in empirical distribution curves; panel (e) shows the full dosing-count distribution. Panel (f) shows product-family shares across all 2,745 rows, with counts printed for major segments. The ratio of totals in the upper panels is a descriptive aggregate, not an intrinsic efficiency ranking.')
s=s.replace('Figure~\\ref{fig:production} shows all 2,742 positive-energy records and the product-family composition of all 2,745 source records.',
 'Figure~\\ref{fig:production} shows all 2,742 positive-energy records, the complete mass and dosing-count distributions, and the product-family composition of all 2,745 source records.')
anchor='\\subsection{Original frozen-model test}'
s=s.replace(anchor,'\\input{sections/temporal_results}\n\n'+anchor)
s=s.replace('The shaded region connects adjacent record-wise bounds only as a visual guide; it is not a continuous-time energy envelope.',
 'Faint vertical segments are individual record-wise intervals; there is no continuous-time uncertainty envelope. Red open circles identify every out-of-interval record. Shared vertical axes permit direct comparison of scale across machines.')
s=s.replace('The lower panel reports retained and candidate sample sizes.','The lower panel reports retained and candidate sample sizes and marks the actual calibration and test boundaries.')
s=s.replace('Lower panels show signed errors against actual irregular origin dates. All 798 observations and the complete residual range are retained.',
 'Lower panels show signed errors against actual irregular origin dates, with red lines marking each machine\'s mean error. All 798 observations and the complete residual range are retained; identical limits make subgroup differences visible.')
s=s.replace('Transformer and foundation-model results from separately simulated series cannot fill that evidential gap. Their future evaluation should begin from a plant historian and verified machine/cycle identifiers, with horizons defined on the recorded sampling grid.',
 'The present real-record TFT, N-HiTS, TimesFM and Chronos experiments address a one-record task and cannot fill that evidential gap. A future multi-horizon evaluation should begin from a plant historian and verified machine/cycle identifiers, with horizons defined on the recorded sampling grid. No explicit active/inactive labels or elapsed-time positional model were introduced into the present benchmark.')
anchor='\\subsection{Forecasting scope and physical interpretation}'
s=s.replace(anchor,r'''\subsection{Why temporal complexity did not improve this comparison}
The matched benchmark provides a more specific result than a general preference for simple models. A compact TFT and the linear current-input control have similar MAE on this particular 584-record population; no paired improvement is established. Both the full-window linear control and N-HiTS perform worse. This pattern is consistent with much of the predictable variation being associated with current production characteristics, while short recent histories add noise or require a more suitable representation. It is not a causal demonstration of why a given architecture fails.

Several features constrain transfer from standard time-series benchmarks. Each event aggregates a different amount of production over a different duration; event position is not a uniform time coordinate. Product mix changes between operations, and histories reset at week boundaries. The strict policy removes 27% of the original validation targets as history warm-up, so the modern comparison excludes some early-week operating conditions. These restrictions improve clarity of the information flow but also narrow the population and may disadvantage methods that benefit from longer histories. A future study should prespecify context-length and elapsed-time representations, with a prospective history policy that reflects actual deployment.

The local XReg extremes and the Chronos constant-channel numerical sensitivity, detailed in Appendix~A, further show that a native interface is not automatically robust under every industrial input distribution. Foundation weights, covariate adaptation, normalization and record semantics all contribute to the delivered prediction. The primary results preserve those native computations so that these limitations remain visible. They do not justify silently repairing an adapter and reporting the repaired result as if it had been prespecified.

'''+anchor)
# Operational interface with required scientific interpretation rather than implementation detail alone.
s=s.replace('Its interface should accept a stable record identifier, machine, timestamped plan values and later outcome publications.',
 'Its interface should accept a stable record identifier, machine, timestamped plan values and later outcome publications. For an issued estimate, the interface must retain the input version actually used; when the operation finishes, its energy outcome can be compared with that stored estimate. Subsequent completed residuals may update a separately versioned calibration policy. This sequence prevents retrospectively amended production totals from being mistaken for the inputs of a genuine earlier forecast.')
a=s.index('Using real industrial records from three pelleting lines,',s.index('\\section{Conclusions}'));b=s.index('\\section*{Data and reproducibility',a)
s=s[:a]+r'''A production-adjusted Ridge baseline reduced MAE by 14.42\% relative to machine-specific energy intensity in the original 798-record chronological test. Explicit input/output definitions and an exact additive decomposition make this result interpretable as expected pelleting electricity per recorded operation. The subsequent matched GPU benchmark did not establish an improvement from TFT, N-HiTS or the tested TimesFM 2.5 and Chronos-2 variants on 584 development records with 16-record contexts. The frozen Ridge therefore remains the reference model; it is not replaced by the best-looking seed or device result.

The remaining limitations are substantive. Pellet~1 retains marked underestimation, interval coverage varies by machine and week, and a post-hoc rolling update obtains higher coverage largely through wider intervals. Product-dependent mass effects and stabilized residual correction show modest exploratory signals that warrant a new independent period. Before prospective use, the plant must establish planned-input availability, outcome publication times and consistent meter/order boundaries. The present evidence supports retrospective conditional assessment and a defined interface for a future digital twin, not measured savings, fault diagnosis or validated high-frequency multi-horizon forecasting.

'''+s[b:]
a=s.index('The analytical package retains');b=s.index('\\bibliographystyle',a)
s=s[:a]+r'''The analytical package retains source-row identifiers, fingerprints of the original workbook and protected artifacts, archived prediction tables, detailed plotting code, source dictionaries and an editable LaTeX manuscript. The previous 52 original point-metric reconciliations are complemented by independent recalculation of all 32 main GPU point-metric values. The frozen Ridge coefficients exactly reconstruct the archived test outputs, without refitting or a new prediction call. Source data, earlier predictions, trained models and previous manuscripts remain unchanged by this revision. Figures use the actual observations and archived predictions; no extra points are synthesized to increase apparent sample density.

The packaged manuscript includes record-level industrial values and is therefore intended for authorized local review. Industrial data and source documents have not been publicly released through this work; public sharing conditions require agreement with the data owner. Author affiliations, funding, conflicts of interest and any journal-specific declarations must be supplied by the authors before submission. The reproducibility appendix records computational details and distinguishes numerical consistency from scientific generalization.

\input{sections/numerical_appendix}
\clearpage
'''+s[b:]
# Raw literals above use standard LaTeX backslashes. Normalize only erroneous doubled command prefix.
s=s.replace('14.42\\\\%','14.42\\%')
s=re.sub(r'(?<!\\)%',r'\\%',s)
(H/'main.tex').write_text(s,encoding='utf8')
for name in ['build.py','qa_pdf.py']:
    script=(H.parent/'paper_datos_reales_v1'/name).read_text(encoding='utf8').replace('Paper_Peletizado_Datos_Reales_v1.pdf','Paper_Peletizado_Datos_Reales_v2.pdf')
    if name=='qa_pdf.py':
        script=script.replace("tex=(HERE/'main.tex').read_text(encoding='utf8')", "tex=(HERE/'main.tex').read_text(encoding='utf8')+'\\n'+'\\n'.join(p.read_text(encoding='utf8') for p in sorted((HERE/'sections').glob('*.tex')))")
        script=script.replace("assert all(len(p.extract_text())>500 for p in reader.pages)","assert all(len(p.extract_text())>100 for p in reader.pages)")
    (H/name).write_text(script,encoding='utf8')
# Derive the GPU result table from archived values to prevent transcription drift.
p=H/'sections/temporal_results.tex';s=p.read_text(encoding='utf8')
metrics=pd.read_csv(H/'data/gpu/metrics.csv').set_index('modelo')
names=[('Ridge_actual','Ridge: current'),('TFT','TFT'),('Ridge_ventana','Ridge: full window'),('NHITS','N-HiTS'),
       ('Chronos_2_covariates','Chronos-2 + covariates'),('TimesFM_2.5_univariate','TimesFM 2.5: energy only'),
       ('Chronos_2_univariate','Chronos-2: energy only'),('TimesFM_2.5_covariates','TimesFM 2.5 + XReg')]
rows=[]
for key,label in names:
    r=metrics.loc[key];rows.append(f'{label} & {r.MAE_kWh:.2f} & {r.RMSE_kWh:.2f} & ${r.bias_kWh:.2f}$ & {100*r.WAPE:.2f}'+r'\\')
a=s.index('Ridge: current & 53.55');b=s.index('\\bottomrule',a)
s=s[:a]+'\n'.join(rows)+s[b:]
p.write_text(s,encoding='utf8')
print('Assembled expanded manuscript and GPU table from archived values.')
