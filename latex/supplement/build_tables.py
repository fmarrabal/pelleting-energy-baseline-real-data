"""Generate the supplementary LaTeX tables straight from the archived aggregate results (no manual transcription)."""
from pathlib import Path
import json
import pandas as pd

D = Path(r"E:/ARTICULOS-CIENTIFICOS/Paletizado_Sousa/Reproducibilidad_Datos_Reales/analysis/data")
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def table(name, caption, label, colspec, header, rows, size="small", note=None, env="tabular", extra=""):
    lines = [f"\\begin{{table}}[!htbp]\\centering\\{size}{extra}", f"\\caption{{{caption}}}\\label{{{label}}}"]
    if env == "tabularx":
        lines.append(f"\\begin{{tabularx}}{{\\linewidth}}{{{colspec}}}\\toprule")
    else:
        lines.append(f"\\begin{{tabular}}{{{colspec}}}\\toprule")
    lines.append(" & ".join(header) + " \\\\\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}" if env == "tabularx" else "\\end{tabular}")
    if note:
        lines.append(f"\\par\\vspace{{3pt}}\\footnotesize {note}")
    lines.append("\\end{table}")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf8")


def f(x, nd=2):
    return f"{x:.{nd}f}"


def neg(x, nd=2):
    s = f"{x:.{nd}f}"
    return f"$-{s[1:]}$" if s.startswith("-") else s


MODEL = {"Ridge": "Ridge", "HistGradientBoosting": "Gradient boosting", "ExtraTrees": "ExtraTrees", "MLP_3_semillas": "MLP, three-seed mean",
         "SEC_maquina": "Line-specific SEC", "SEC_ultimos5": "Recent SEC, last five records", "Mediana_maquina": "Line-specific median",
         "H1_masa_maquina": "H1: mass by pelleting line", "H2_masa_subfamilia": "H2: mass by product subfamily", "H3_sesgo_reciente": "H3: recent-residual correction",
         "Ridge_actual": "Ridge, current inputs", "Ridge_ventana": "Ridge, full 16-record window", "TFT": "TFT", "NHITS": "N-HiTS",
         "Chronos_2_covariates": "Chronos-2 + covariates", "Chronos_2_univariate": "Chronos-2, energy only",
         "TimesFM_2.5_covariates": "TimesFM 2.5 + XReg", "TimesFM_2.5_univariate": "TimesFM 2.5, energy only"}
ORDER = ["Ridge_actual", "Ridge_ventana", "TFT", "NHITS", "Chronos_2_covariates", "TimesFM_2.5_univariate", "Chronos_2_univariate", "TimesFM_2.5_covariates"]
ROLE = {"entrada_modelo_original": "model input", "objetivo": "target", "cronologia": "chronology: origin", "cronologia_etiquetas": "chronology: labels",
        "resumen_operacional_no_usado": "op. summary (unused)", "contexto_no_usado": "context (unused)"}
AVAIL = {"no_acreditada_antes_del_origen": "not verified before origin", "posterior_o_no_documentada": "after origin / undocumented",
         "posterior": "after origin", "observacion_del_evento": "event observation"}

# S1 variables -----------------------------------------------------------------------------------------------------
v = pd.read_csv(D / "disponibilidad_variables.csv")
fid = pd.read_csv(D / "fidelidad_32_variables.csv").set_index("original")
def brk(s):  # typewriter names may break after underscores
    return "\\texttt{" + esc(s).replace("\\_", "\\_\\allowbreak{}") + "}"


rows = [[brk(r.campo_excel), brk(r.campo_analitico), ROLE[r.rol], AVAIL[r.disponibilidad], str(int(r.ausentes_original)),
         f"{int(fid.loc[r.campo_excel, 'coinciden']):,}/{int(fid.loc[r.campo_excel, 'n']):,}"] for r in v.itertuples()]
table("S_variables", "The 32 source fields of the plant workbook, their analytical names, role in the study, availability at the prediction origin, number of missing values in the source and cells matched in the fidelity reconciliation.",
      "tab:s_variables", "@{}>{\\raggedright\\arraybackslash}p{.16\\linewidth}>{\\raggedright\\arraybackslash}p{.23\\linewidth}>{\\raggedright\\arraybackslash}p{.15\\linewidth}>{\\raggedright\\arraybackslash}Xrr@{}",
      ["Source field", "Analytical name", "Role", "Availability at origin", "Missing", "Matched"], rows, size="scriptsize", env="tabularx",
      note="Original Spanish field names are retained for traceability. Model inputs are admitted for the retrospective baseline; prospective use would require a plan version and a timestamp before the origin. The fidelity reconciliation compared every cell of the analytical table with the workbook under the documented tolerances: 32 fields $\\times$ 2,745 records = 87,840 comparisons, all matched.")

# S2 numeric summary, S3 categories -----------------------------------------------------------------------------------
ds = json.load(open(D.parent / "DESCRIPTIVE_STATISTICS.json", encoding="utf8"))
NAMES = {"masa_dosificada_t": "Dosed mass (t)", "baches_dosificacion": "Dosing batches", "energia_peletizado_kWh": "Pelleting electricity (kWh)"}
rows = []
for k, s in ds["numeric_summary"].items():
    rows.append([NAMES[k], f"{int(s['count']):,}", f(s["mean"]), f(s["std"]), f(s["min"]), f(s["25%"]), f(s["50%"]), f(s["75%"]), f(s["max"])])
table("S_numeric", f"Descriptive statistics of the numerical variables over all 2,745 source records (including the three zero-energy records). Pearson correlation between dosed mass and dosing-batch count: {ds['mass_batches_pearson']:.3f}.",
      "tab:s_numeric", "@{}lrrrrrrrr@{}", ["Variable", "$n$", "Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"], rows)
CAT = {"linea": "Pelleting line", "familia": "Product family", "subfamilia": "Product subfamily", "forma": "Presentation", "salida_programada": "Bag/bulk"}
rows = []
for field in ["linea", "familia", "subfamilia", "forma", "salida_programada"]:
    for cat, n in ds["category_counts"][field].items():
        rows.append([CAT[field], f"\\texttt{{{esc(cat)}}}", f"{n:,}"])
table("S_categories", "Record counts of every categorical level in the 2,745 source records. Family (\\texttt{LINEA}) is context only; line, subfamily, presentation and bag/bulk are model inputs.",
      "tab:s_categories", "@{}llr@{}", ["Field", "Level", "Records"], rows, size="footnotesize")

# S4 temporal structure, S5 partition flow ----------------------------------------------------------------------------
t = pd.read_csv(D / "estructura_temporal.csv")
rows = [[r.linea, f"{int(r.n):,}", f"{int(r.n_objetivos_positivos):,}", str(int(r.n_solapa_intervalo_anterior)), str(int(r.n_duracion_mayor24h)), f(r.mediana_min_entre_inicios, 1), f(r.duracion_mediana_min, 1)] for r in t.itertuples()]
table("S_temporal", "Temporal structure of the records by pelleting line: records, positive targets, within-line overlaps with the previous interval, operations longer than 24 h, median minutes between consecutive starts and median recorded pelleting duration.",
      "tab:s_temporal", "@{}lrrrrrr@{}", ["Line", "Records", "Positive targets", "Overlaps", "$>$24 h", "Median gap (min)", "Median duration (min)"], rows)
fl = pd.read_csv(D / "flujo_muestra.csv")
PART = {"DESARROLLO": "Development", "CALIBRACION": "Calibration", "PRUEBA": "Test"}
rows = [[PART[r.particion], f"{int(r.originales):,}", f"{int(r.incluidos):,}", str(int(r.excluidos)), r.min_origen[:16], r.max_origen[:16]] for r in fl.itertuples()]
table("S_flow", "Chronological partitions: candidate records, records retained after the eligibility rules, exclusions and the range of analytical origins (earliest recorded stage start).",
      "tab:s_flow", "@{}lrrrll@{}", ["Partition", "Candidates", "Retained", "Excluded", "First origin", "Last origin"], rows)

# S6 coefficients, S7 numeric effects -----------------------------------------------------------------------------------
cf = pd.read_csv(D / "ridge_coefficients.csv")
LAB = {"num__masa_dosificada_t": ("Dosed mass", "numerical (per training SD)"), "num__baches_dosificacion": ("Dosing batches", "numerical (per training SD)"),
       "num__hora_sin": ("Origin hour, sine", "numerical (per training SD)"), "num__hora_cos": ("Origin hour, cosine", "numerical (per training SD)"),
       "num__dia_sin": ("Weekday, sine", "numerical (per training SD)"), "num__dia_cos": ("Weekday, cosine", "numerical (per training SD)")}
rows = []
for r in cf.itertuples():
    if r.encoded_variable in LAB:
        name, kind = LAB[r.encoded_variable]
    else:
        field, level = r.encoded_variable.replace("cat__", "").split("_", 1)
        name, kind = f"\\texttt{{{esc(level)}}}", {"linea": "pelleting line", "subfamilia": "product subfamily", "forma": "presentation", "salida": "bag/bulk"}[field.split("_")[0]] + " (one-hot)"
    rows.append([name, kind, neg(r.coefficient_kWh, 4)])
table("S_coefficients", f"All 22 coefficients of the frozen Ridge model (penalty 10, fitted on 1,542 development records). Intercept: {ds['ridge']['intercept_kWh']:.2f} kWh. Numerical coefficients are per training standard deviation; one-hot coefficients are per indicator.",
      "tab:s_coefficients", "@{}llr@{}", ["Encoded variable", "Type", "Coefficient (kWh)"], rows, size="footnotesize",
      note="With an unpenalized intercept and complete one-hot sets, the penalized solution centres each categorical set, which is why the two-level fields (presentation, bag/bulk) carry opposite coefficients of equal magnitude.")
ne = pd.read_csv(D / "ridge_numeric_effects.csv")
NN = {"masa_dosificada_t": "Dosed mass (t)", "baches_dosificacion": "Dosing batches", "hora_sin": "Origin hour, sine", "hora_cos": "Origin hour, cosine", "dia_sin": "Weekday, sine", "dia_cos": "Weekday, cosine"}
rows = [[NN[r.variable], neg(r.coefficient_per_training_sd_kWh, 4), neg(r.training_mean, 4), f(r.training_sd, 4), neg(r.coefficient_per_original_unit, 4)] for r in ne.itertuples()]
table("S_effects", "Conversion of the numerical Ridge coefficients from standardized to original units: $\\beta_{\\mathrm{unit}}=\\beta_{\\mathrm{SD}}/s_{\\mathrm{train}}$, with the training mean and standard deviation used for standardization.",
      "tab:s_effects", "@{}lrrrr@{}", ["Variable", "Coef. per training SD (kWh)", "Training mean", "Training SD", "Coef. per original unit (kWh)"], rows)

# S8 development + extensions, S9 contrasts ----------------------------------------------------------------------------
md = pd.read_csv(D / "modelos_desarrollo.csv")
hm = pd.read_csv(D / "hipotesis_metricas.csv")
rows = [[MODEL[r.modelo], str(int(r.n)), f(r.MAE_kWh, 4), f(r.RMSE_kWh, 4), neg(r.sesgo_kWh, 4), f(100 * r.WAPE, 2)] for r in md.itertuples()]
rows += [[MODEL[r.modelo], str(int(r.n)), f(r.MAE, 4), f(r.RMSE, 4), neg(r.sesgo, 4), f(100 * r.WAPE, 2)] for r in hm.itertuples() if r.modelo != "Ridge"]
table("S_development", "Development performance on the same 800 chronological targets at full archived precision: the seven original methods and the three post-hoc extensions of the Ridge model. MAE, RMSE and bias in kWh per record.",
      "tab:s_development", "@{}lrrrrr@{}", ["Method", "$n$", "MAE", "RMSE", "Bias", "WAPE (\\%)"], rows)
hc = pd.read_csv(D / "hipotesis_contrastes.csv")
rows = [[MODEL[r.hipotesis], f(r.reduccion_MAE, 3), f(r.reduccion_pct, 2), f"[{neg(r.IC95_inf, 3)}, {f(r.IC95_sup, 3)}]", f"[{neg(r.IC98_333_inf, 3)}, {f(r.IC98_333_sup, 3)}]"] for r in hc.itertuples()]
table("S_contrasts", "Paired MAE reductions of the exploratory extensions relative to the frozen Ridge specification on the 800 development targets: block-bootstrap percentile intervals (5,000 replicates, blocks of three observed origin dates) at the marginal 95\\% level and at the Bonferroni-adjusted 98.33\\% level for the three comparisons.",
      "tab:s_contrasts", "@{}lrrll@{}", ["Extension", "Reduction (kWh)", "Reduction (\\%)", "95\\% interval (kWh)", "98.33\\% interval (kWh)"], rows, extra="\\setlength{\\tabcolsep}{4pt}")

# S10 intervals by line, S11 by week -------------------------------------------------------------------------------------
METH = {"fijo": "Fixed", "actualizado_200": "Rolling, 200 residuals"}
il = pd.read_csv(D / "intervalos_linea.csv").sort_values(["metodo", "nominal", "linea"], ascending=[False, True, True])
rows = [[METH[r.metodo], f"{int(r.nominal)}\\%", r.linea, str(int(r.n)), f(100 * r.cobertura, 2), f(r.ancho_medio_kWh, 2), f(r.interval_score, 2), str(int(r.alertas_altas)), str(int(r.alertas_bajas))] for r in il.itertuples()]
table("S_intervals_line", "Residual intervals by pelleting line on the 798 test records: empirical coverage, mean width, interval score and the number of records above (high) and below (low) the interval, for the fixed calibration and the post-hoc rolling update.",
      "tab:s_intervals_line", "@{}lllrrrrrr@{}", ["Policy", "Nominal", "Line", "$n$", "Coverage (\\%)", "Width (kWh)", "Score (kWh)", "High", "Low"], rows, size="footnotesize")
iw = pd.read_csv(D / "intervalos_semana.csv").sort_values(["metodo", "nominal", "semana"], ascending=[False, True, True])
rows = [[METH[r.metodo], f"{int(r.nominal)}\\%", r.semana, str(int(r.n)), f(100 * r.cobertura, 2), f(r.ancho_medio_kWh, 2), f(r.interval_score, 2), str(int(r.alertas_altas)), str(int(r.alertas_bajas))] for r in iw.itertuples()]
table("S_intervals_week", "Residual intervals by test week (week starting on the Monday indicated; the first and last weeks are partial).",
      "tab:s_intervals_week", "@{}lllrrrrrr@{}", ["Policy", "Nominal", "Week of", "$n$", "Coverage (\\%)", "Width (kWh)", "Score (kWh)", "High", "Low"], rows, size="footnotesize")

# S12 sample counts, S13 by fold, S14 by line, S15 seeds, S16 contrasts ---------------------------------------------------
proto = json.load(open(D / "gpu" / "PROTOCOL.json", encoding="utf8"))
rows = [[s["fold"], str(s["train"]), str(s["inner_train"]), str(s["inner_val"]), str(s["val"])] for s in proto["sample_counts"]]
table("S_folds", "Sample sizes of the matched temporal benchmark for each outer validation week (week starting on the date shown): outer training records, inner training and inner validation records used for epoch selection, and outer validation targets.",
      "tab:s_folds", "@{}lrrrr@{}", ["Validation week", "Outer training", "Inner training", "Inner validation", "Targets"], rows)
mf = pd.read_csv(D / "gpu" / "metrics_by_fold.csv")
rows = []
for m in ORDER:
    for r in mf[mf.modelo == m].sort_values("fold").itertuples():
        rows.append([MODEL[m], r.fold, str(int(r.n)), f(r.MAE_kWh), f(r.RMSE_kWh), neg(r.bias_kWh), f(100 * r.WAPE, 2), str(int(r.negative_raw))])
table("S_by_fold", "Matched temporal benchmark by outer validation week (584 targets in total). MAE, RMSE and bias in kWh per record; the last column counts raw predictions below zero before clipping.",
      "tab:s_by_fold", "@{}llrrrrrr@{}", ["Method", "Week", "$n$", "MAE", "RMSE", "Bias", "WAPE (\\%)", "Neg."], rows, size="scriptsize")
mm = pd.read_csv(D / "gpu" / "metrics_by_machine.csv")
rows = []
for m in ORDER:
    for r in mm[mm.modelo == m].sort_values("linea").itertuples():
        rows.append([MODEL[m], r.linea, str(int(r.n)), f(r.MAE_kWh), f(r.RMSE_kWh), neg(r.bias_kWh), f(100 * r.WAPE, 2), str(int(r.negative_raw))])
table("S_by_line", "Matched temporal benchmark by pelleting line (149, 257 and 178 targets on Pellet~1, 2 and 3).",
      "tab:s_by_line", "@{}llrrrrrr@{}", ["Method", "Line", "$n$", "MAE", "RMSE", "Bias", "WAPE (\\%)", "Neg."], rows, size="scriptsize")
ms = pd.read_csv(D / "gpu" / "metrics_seeds.csv")
rows = [[MODEL[r.modelo], str(int(r.seed)), str(int(r.n)), f(r.MAE_kWh), f(r.RMSE_kWh), neg(r.bias_kWh), f(100 * r.WAPE, 2)] for r in ms.itertuples()]
table("S_seeds", "Individual seeds of the locally trained temporal models on the 584 matched targets. The reported ensemble averages the three clipped predictions, so its MAE is not the mean of the three seed MAEs.",
      "tab:s_seeds", "@{}lrrrrrr@{}", ["Method", "Seed", "$n$", "MAE", "RMSE", "Bias", "WAPE (\\%)"], rows)
pc = pd.read_csv(D / "gpu" / "paired_contrasts.csv").set_index("modelo").loc[ORDER]
rows = [[MODEL[m], neg(r.MAE_improvement_vs_Ridge_kWh, 3), f"[{neg(r.CI95_low, 3)}, {neg(r.CI95_high, 3)}]", f"[{neg(r.CI_family7_low, 3)}, {neg(r.CI_family7_high, 3)}]", f"[{f(r.MAE_CI95_low, 2)}, {f(r.MAE_CI95_high, 2)}]"] for m, r in pc.iterrows()]
table("S_paired", "Paired MAE differences relative to current-input Ridge on the 584 matched targets (positive values would favour the alternative): marginal 95\\% and family-adjusted 99.29\\% block-bootstrap intervals for the seven contrasts (5,000 replicates, blocks of three observed origin dates, seed 20260922), and the 95\\% interval of each method's own MAE.",
      "tab:s_paired", "@{}lrlll@{}", ["Method", "Improvement (kWh)", "95\\% interval", "99.29\\% interval", "MAE 95\\% interval"], rows, size="footnotesize")

# S17 CPU-GPU, S18 epochs, S19 numerical differences ---------------------------------------------------------------------
cg = pd.read_csv(D / "gpu" / "comparison_cpu_gpu.csv").set_index("modelo").loc[ORDER]
rows = [[MODEL[m], f(r.CPU_MAE_kWh, 4), f(r.GPU_MAE_kWh, 4), f(r.CPU_RMSE_kWh, 4), f(r.GPU_RMSE_kWh, 4), neg(r.GPU_minus_CPU_MAE_kWh, 4)] for m, r in cg.iterrows()]
table("S_cpu_gpu", "CPU and GPU runs of the same protocol on the 584 matched targets. TFT and N-HiTS were trained afresh on each device; foundation-model weights were frozen on both.",
      "tab:s_cpu_gpu", "@{}lrrrrr@{}", ["Method", "CPU MAE", "GPU MAE", "CPU RMSE", "GPU RMSE", "GPU $-$ CPU MAE"], rows, size="footnotesize")
ep = pd.read_csv(D / "gpu" / "epoch_selection_cpu_gpu.csv")
rows = [[r.model.replace("NHITS", "N-HiTS"), r.fold, str(int(r.seed)), str(int(r.CPU_selected_epoch)), str(int(r.GPU_selected_epoch))] for r in ep.itertuples()]
table("S_epochs", "Epoch selected on the inner validation week for every model, outer week and seed, on CPU and on GPU (maximum 40 epochs, patience 8).",
      "tab:s_epochs", "@{}llrrr@{}", ["Model", "Week", "Seed", "CPU epoch", "GPU epoch"], rows, size="footnotesize")
nd = pd.read_csv(D / "gpu" / "numerical_differences.csv").set_index("modelo").loc[ORDER]
rows = [[MODEL[m], f"{r['mean']:.6f}", f"{r['max']:.6f}", f"{r['median']:.6f}"] for m, r in nd.iterrows()]
table("S_numdiff", "Absolute CPU--GPU prediction differences per method over the 584 matched targets (kWh).",
      "tab:s_numdiff", "@{}lrrr@{}", ["Method", "Mean", "Maximum", "Median"], rows)
print("tables written:", len(list(OUT.glob("*.tex"))))
