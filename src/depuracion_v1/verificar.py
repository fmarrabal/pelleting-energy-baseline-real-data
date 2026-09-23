from pathlib import Path
import json, math, datetime, hashlib
import openpyxl
p=Path(__file__).resolve().parent
data=json.loads((p/'libro_datos.json').read_text(encoding='utf8'))
w=openpyxl.load_workbook(p/'Datos_depurados_v1.xlsx',read_only=True,data_only=True)
n=0;errors=[]
for name,t in data.items():
    s=w[name]
    assert list(next(s.values))[:len(t['headers'])]==t['headers']
    for i,row in enumerate(s.iter_rows(min_row=2,max_row=len(t['rows'])+1,max_col=len(t['headers']),values_only=True)):
        for j,v in enumerate(row):
            expected=t['rows'][i][j]
            if t['headers'][j] in t['dates'] and expected is not None:expected=datetime.datetime.fromisoformat(expected)
            ok=(v==expected) if not isinstance(expected,(float,int)) else isinstance(v,(float,int)) and math.isclose(v,expected,rel_tol=1e-12,abs_tol=1e-9)
            if not ok:errors.append((name,i+2,j,str(v),str(expected)))
            n+=1
assert not errors,errors[:5]
summary=json.loads((p/'resumen.json').read_text(encoding='utf8'))
assert hashlib.file_digest((p.parent/summary['fuente']).open('rb'),'sha256').hexdigest()==summary['sha256']
assert all(c.data_type!='e' for s in w for row in s for c in row)
(p/'verificacion.json').write_text(json.dumps({'celdas_cotejadas':n,'diferencias':0,'errores_excel':0,'original_sin_cambios':True}),encoding='utf8')
print('Verificadas',n,'celdas, incluidas fechas, sin diferencias. Original sin cambios.')
