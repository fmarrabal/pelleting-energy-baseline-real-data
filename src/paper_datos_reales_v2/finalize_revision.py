"""Verify preservation, reviewed PDF identity, and package manuscript/figure sources."""
from pathlib import Path
import argparse, hashlib, json, re, shutil, zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;BASE=H.parent
parser=argparse.ArgumentParser();parser.add_argument('--confirm-visual-review',action='store_true');args=parser.parse_args()
if not args.confirm_visual_review:raise SystemExit('Inspect all rendered pages before passing --confirm-visual-review.')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
sm=json.loads((H/'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
for rel,h in sm['sources'].items():assert sha(BASE/rel)==h,rel
pdf=H/'Paper_Peletizado_Datos_Reales_v2.pdf'
qa=json.loads((H/'qa/structural_checks.json').read_text())
assert qa['pdf_sha256']==sha(pdf)
assert (qa['pages'],qa['figures'],qa['tables'],qa['references'],qa['numbered_equations'])==(28,13,8,40,14)
assert not qa['compile_errors']
tex=(H/'main.tex').read_text(encoding='utf8')+'\n'+'\n'.join(p.read_text(encoding='utf8') for p in (H/'sections').glob('*.tex'))
assert not re.search(r'(?<!\\)%',tex),'Unescaped percentage would silently truncate prose.'
assert not any(t in tex for t in ['TODO','TBD','INSERT FIGURE'])
figs=json.loads((H/'FIGURE_MANIFEST.json').read_text())
for name in figs:
    for ext in ['pdf','svg','png']:assert (H/'figures'/f'{name}.{ext}').stat().st_size>1000
p=pd.read_csv(H/'data/gpu/predictions.csv')
assert len(p)==8*584 and p.groupby('modelo').fila_fuente.nunique().eq(584).all()
for model,q in p.groupby('modelo'):
    assert (q.energia_peletizado_kWh<=1100).all()
    if model!='TimesFM_2.5_covariates':assert q.prediccion_kWh.between(0,1100).all()
    else:assert int((q.prediccion_kWh>1100).sum())==3 and q.prediccion_kWh.max()<13000
parts=pd.read_csv(H/'data/ridge_additive_components.csv')
cs=['Dosed mass','Dosing batches','Hour','Weekday','Machine','Product subfamily','Presentation','Bag / bulk']
assert np.max(abs(parts.intercept_kWh+parts[cs].sum(axis=1)-parts.archived_prediction_kWh))<1e-9
qa.update({'visual_review_completed':True,'pages_reviewed':list(range(1,29)),
           'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
           'all_source_hashes_unchanged':True,'source_hashes_checked':len(sm['sources']),
           'visual_review_notes':['All 28 rendered pages inspected.','All 13 scientific figures inspected.','Overlapping diagram titles and coverage annotations corrected.','TimesFM extreme outputs retained in full-range inset and metrics.'],
           'new_training':False,'new_final_test_prediction':False})
dump(H/'QA_VERIFIED.json',qa)
out=BASE/'output/pdf';out.mkdir(parents=True,exist_ok=True);dest=out/pdf.name;shutil.copyfile(pdf,dest);assert sha(dest)==sha(pdf)
files=[H/x for x in ['main.tex','main.bbl','references.bib','elsarticle-num.bst',pdf.name,'build.py','make_figures.py','qa_pdf.py','README.md','SOURCE_MANIFEST.json','DESCRIPTIVE_STATISTICS.json','FIGURE_MANIFEST.json','QA_VERIFIED.json']]
for sub in ['sections','data','figures']:files+=sorted(p for p in (H/sub).rglob('*') if p.is_file())
mapping={str(p.relative_to(H)).replace('\\','/'):sha(p) for p in files}
dump(H/'PACKAGE_MANIFEST.json',{'files':mapping,'purpose':'Editable manuscript and figures from archived real-data results; no standalone training environment.'})
files.append(H/'PACKAGE_MANIFEST.json')
zpath=H/'Peletizado_Datos_Reales_LaTeX_v2.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
    for p in files:z.write(p,p.relative_to(H).as_posix())
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None
    for rel,h in mapping.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
delivery={'pdf':str(dest),'pdf_sha256':sha(dest),'zip':str(zpath),'zip_bytes':zpath.stat().st_size,
          'zip_sha256':sha(zpath),'files_in_zip':len(files),'pages':28,'figures':13,'tables':8,'equations':14,'references':40,
          'protected_originals_unchanged':True,'visual_review':True}
dump(H/'DELIVERY.json',delivery);print(json.dumps(delivery,indent=2))
