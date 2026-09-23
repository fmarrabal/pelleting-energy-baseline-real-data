"""Render all pages and prepare visual-review sheets; not a visual PASS by itself."""
from pathlib import Path
import re, json, subprocess, hashlib
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
QA=HERE/'qa';QA.mkdir(exist_ok=True)
PDF=HERE/'Paper_Peletizado_Datos_Reales_v2.pdf'
poppler=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
subprocess.run([str(poppler),'-r','110','-png',str(PDF),str(QA/'page')],check=True,capture_output=True)
reader=PdfReader(PDF)
text='\n\n'.join(f'PAGE {i+1}\n'+p.extract_text() for i,p in enumerate(reader.pages))
(QA/'extracted_text.txt').write_text(text,encoding='utf8')
pages=[QA/f'page-{i:02d}.png' for i in range(1,len(reader.pages)+1)]
assert len(pages)==len(reader.pages)
for start in range(0,len(pages),3):
    canvas=Image.new('RGB',(1830,890),'#d9dde2')
    draw=ImageDraw.Draw(canvas)
    for j,p in enumerate(pages[start:start+3]):
        im=Image.open(p).convert('RGB');im.thumbnail((594,842))
        x=10+610*j;canvas.paste(im,(x,34));draw.text((x+8,10),p.stem,fill='black')
    canvas.save(QA/f'review-{start+1:02d}-{min(start+3,len(pages)):02d}.png')
tex=(HERE/'main.tex').read_text(encoding='utf8')+'\n'+'\n'.join(p.read_text(encoding='utf8') for p in sorted((HERE/'sections').glob('*.tex')))
bbl=(HERE/'main.bbl').read_text(encoding='utf8')
log=(HERE/'main.log').read_text(encoding='utf8',errors='replace')
report={'pages':len(reader.pages),'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'references':len(re.findall(r'\\bibitem',bbl)),
        'figures':len(re.findall(r'\\fig\{',tex)),'tables':len(re.findall(r'\\begin\{table\}',tex)),
        'numbered_equations':len(re.findall(r'\\newlabel\{eq:',(HERE/'main.aux').read_text(encoding='utf8')))+2,
        'compile_errors':any(token.lower() in log.lower() for token in ['undefined references','undefined citations','Overfull \\hbox','Missing character:']),
        'visual_review_completed':False}
assert not report['compile_errors']
assert all(len(p.extract_text())>100 for p in reader.pages)
(QA/'structural_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
