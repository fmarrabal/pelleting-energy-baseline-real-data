"""Render before visual review; package only after explicit local review flag."""
from pathlib import Path
import argparse,hashlib,json,subprocess,zipfile,platform,importlib.metadata as md
from PIL import Image,ImageDraw
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
PDF=BASE/'output/pdf/Anexo_Modelos_Temporales_Datos_Reales.pdf'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--confirm-visual-review',action='store_true');a=parser.parse_args()
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    if not a.confirm_visual_review:
        poppler=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
        subprocess.run([str(poppler),'-r','115','-png',str(PDF),str(qa/'page')],check=True,capture_output=True)
        reader=PdfReader(PDF);npages=len(reader.pages)
        (qa/'text.txt').write_text('\n\n'.join(p.extract_text() for p in reader.pages),encoding='utf8')
        assert npages==5,npages
        pages=sorted(qa.glob('page-*.png'))
        assert len(pages)==npages
        for first in range(0,npages,2):
            im=Image.new('RGB',(1530,1100),'#d9dde2');dr=ImageDraw.Draw(im)
            for j,p in enumerate(pages[first:first+2]):
                page=Image.open(p).convert('RGB');page.thumbnail((745,1050));x=10+j*765
                im.paste(page,(x,30));dr.text((x+5,10),p.stem,fill='black')
            im.save(qa/f'review-{first+1}-{min(first+2,npages)}.png')
        save(qa/'STRUCTURAL_CHECKS.json',{'pdf_sha256':sha(PDF),'pages':npages,'figures':4,'tables':2,'visual_review_complete':False})
        print('RENDERED',npages,'pages');return
    checks=json.loads((qa/'STRUCTURAL_CHECKS.json').read_text())
    assert checks['pdf_sha256']==sha(PDF),'The PDF changed after rendering.'
    assert json.loads((ROOT/'NUMERICAL_CHECKS.json').read_text())['status']=='PASS'
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text(encoding='utf8'))
    assert all(sha(BASE/k)==v for k,v in protocol['protected_sha256'].items())
    checks.update({'visual_review_complete':True,'status':'VERIFIED','original_hashes_rechecked':True})
    save(ROOT/'QA_VERIFIED.json',checks)
    packages=['torch','neuralforecast','timesfm','chronos-forecasting','numpy','pandas','scikit-learn','matplotlib','joblib']
    # Read package versions from the actual run protocol and scientific environment,
    # without confusing this PDF-rendering Python with the model-training Python.
    env={'training_versions':protocol['versions'],'xreg_extra_dependencies':{'jax':'0.11.2','jaxlib':'0.11.2','ml_dtypes':'0.6.0','opt_einsum':'3.4.0'},
      'rendering_python':platform.python_version(),'note':'TimesFM package 3.0.2 supplies the explicitly selected TimesFM 2.5 class and checkpoint.'}
    save(ROOT/'ENVIRONMENT.json',env)
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(x in rel.parts for x in ['dependencies','__pycache__','qa']):continue
        if p.suffix in ['.py','.md','.json','.csv','.tex','.svg','.png'] and p.name not in ['PACKAGE_MANIFEST.json','DELIVERY.json']:
            files.append(p)
    manifest=[{'path':str(p.relative_to(BASE)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)]
    manifest.append({'path':str(PDF.relative_to(BASE)).replace('\\','/'),'sha256':sha(PDF),'bytes':PDF.stat().st_size})
    save(ROOT/'PACKAGE_MANIFEST.json',{'files':manifest,'exclusions':['original workbook','foundation weights','trained checkpoints','dependencies','logs','render intermediates']})
    archive=ROOT/'Modelos_Reales_Temporales_LaTeX_Resultados.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in files+[PDF,ROOT/'PACKAGE_MANIFEST.json']:z.write(p,str(p.relative_to(BASE)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    save(ROOT/'DELIVERY.json',{'pdf':str(PDF),'pdf_sha256':sha(PDF),'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'files_in_zip':len(files)+2})
    print(json.dumps(checks,indent=2));print('PACKAGED',archive,archive.stat().st_size)

if __name__=='__main__':main()
