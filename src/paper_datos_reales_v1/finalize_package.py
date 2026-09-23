"""Package the reviewed PDF, source, figures and data without altering originals."""
from pathlib import Path
import argparse, hashlib, json, re, shutil, zipfile
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
p=argparse.ArgumentParser();p.add_argument('--confirm-visual-review',action='store_true');a=p.parse_args()
if not a.confirm_visual_review:raise SystemExit('Render and inspect every current PDF page before confirming visual review.')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
audit=json.loads((HERE/'AUDITORIA.json').read_text(encoding='utf8'))
assert all(sha(ROOT/p)==h for p,h in audit['sha256_antes'].items())
additional={
 'Series_temporales_peletizado_4_semanas_Pellets_1_2_3_consolidado.xlsx':'56cab6baf079871f0bf881502a37ef745fbfd83e22ec422c3488f3cabe100e2c',
 'paper_latex_v3/Paper_Pelleting_Energy_Baselines_v3.pdf':'2b8e58be66c6041e85be161d98ac5185761519ec2bdd3ddb644600460c6376d9'}
assert all(sha(ROOT/p)==h for p,h in additional.items())
pdf=HERE/'Paper_Peletizado_Datos_Reales_v1.pdf'
q=json.loads((HERE/'qa/structural_checks.json').read_text(encoding='utf8'))
assert not q['compile_errors'] and q['pdf_sha256']==sha(pdf)
for i in range(1,q['pages']+1):assert (HERE/f'qa/page-{i:02d}.png').is_file()
q.update({'visual_review_completed':True,'visual_review_scope':'every rendered page; source figures separately inspected',
 'review_date_utc':datetime.now(timezone.utc).isoformat(),'preserved_original_sources':True,
 'source_cells_reconciled':audit['celdas_originales_cotejadas'],'archived_point_metrics_reconciled':audit['metricas_puntuales_verificadas'],
 'donor_relationships_verified':audit['relaciones_donante_H3_verificadas']+audit['relaciones_donante_intervalos_verificadas'],
 'test_model_refit':False,'new_test_predictions':False,'submission_ready':False,
 'pending':['industrial record identity','input and label publication times','independent period for extensions','authors and submission declarations']})
(HERE/'QA_VERIFIED.json').write_text(json.dumps(q,indent=2,ensure_ascii=False),encoding='utf8')
readme=(HERE/'README.md').read_text(encoding='utf8')
readme=re.sub(r'manuscrito de \d+ páginas',f'manuscrito de {q["pages"]} páginas',readme)
(HERE/'README.md').write_text(readme,encoding='utf8')
out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
shutil.copyfile(pdf,out/pdf.name)
root_files=['main.tex','main.bbl','references.bib','elsarticle-num.bst','build.py','make_figures.py',
 'auditar_y_preparar.py','README.md','PROTOCOLO_DATOS_REALES.md','ESTRUCTURA_ARTICULO.md',
 'protocolo.json','AUDITORIA.json','FIGURE_MANIFEST.json','QA_VERIFIED.json',pdf.name]
files=[HERE/f for f in root_files]+sorted((HERE/'data').glob('*'))+sorted((HERE/'figures').glob('*.pdf'))+sorted((HERE/'figures').glob('*.png'))
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'pdf_sha256':sha(pdf),
 'contents':{f.relative_to(HERE).as_posix():{'sha256':sha(f),'bytes':f.stat().st_size} for f in files},
 'industrial_records_are_real':True,'synthetic_comparisons_excluded_from_industrial_results':True,
 'sources_and_previous_manuscript_preserved':True,'additional_preservation_checks':additional}
(HERE/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
files.append(HERE/'PACKAGE_MANIFEST.json')
target=HERE/'Peletizado_Datos_Reales_LaTeX_v1.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,f.relative_to(HERE))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for n,item in manifest['contents'].items():assert hashlib.sha256(z.read(n)).hexdigest()==item['sha256']
assert sha(out/pdf.name)==sha(pdf)
result={'pdf':str(out/pdf.name),'zip':str(target),'zip_sha256':sha(target),'zip_bytes':target.stat().st_size,
        'zip_files':len(files),'pages':q['pages'],'figures':q['figures'],'references':q['references'],
        'source_preservation':True,'qa_pass':True}
(HERE/'DELIVERY.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
