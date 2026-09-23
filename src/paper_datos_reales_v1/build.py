"""Compile this manuscript only; never train a model or alter archived evidence."""
from pathlib import Path
import shutil, subprocess
HERE=Path(__file__).resolve().parent
for executable,args in [('pdflatex',['-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex']),
 ('bibtex',['main']),('pdflatex',['-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex']),
 ('pdflatex',['-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'])]:
    command=shutil.which(executable)
    if not command: raise SystemExit(f'{executable} required (TeX Live or MiKTeX).')
    result=subprocess.run([command,*args],cwd=HERE,capture_output=True,text=True,encoding='utf8',errors='replace')
    if result.returncode:
        print(result.stdout[-7000:]); print(result.stderr[-2000:]); raise SystemExit(result.returncode)
log=(HERE/'main.log').read_text(errors='replace')
for token in ['undefined references','undefined citations','Overfull \\hbox','Missing character:']:
    if token in log: raise SystemExit('Review required: '+token)
out=HERE/'Paper_Peletizado_Datos_Reales_v1.pdf'
shutil.copyfile(HERE/'main.pdf',out)
print('Compiled:',out)
