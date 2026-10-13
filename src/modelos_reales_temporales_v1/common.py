"""Real-record benchmark. Importing this module never trains or changes old artifacts."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(ROOT / 'dependencies'))
os.environ.setdefault('HF_HOME', str(BASE / 'modelos_avanzados_v1/hf_cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM = ['masa_dosificada_t','baches_dosificacion','hora_sin','hora_cos','dia_sin','dia_cos']
CAT = ['linea','subfamilia','forma','salida_programada']
TARGET = 'energia_peletizado_kWh'
L = 16
SEEDS = [11,29,47]
FOLDS = [('2025-01-25','2025-02-01'),('2025-02-01','2025-02-08'),('2025-02-08','2025-02-15'),('2025-02-15','2025-02-22')]
PROTECTED = ['Datos Planta Peletizado.xlsx','depuracion_v1/registros_depurados.csv',
 'calibracion_final_v1/modelo_calibrado.joblib','prueba_final_v1/predicciones_prueba.csv',
 'paper_datos_reales_v1/Paper_Peletizado_Datos_Reales_v1.pdf',
 'paper_latex_v3/Paper_Pelleting_Energy_Baselines_v3.pdf']

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf8')

def device_policy():
    """Do not evict or compete with an almost-full GPU."""
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        free=float(raw.split(',')[2]); device='cuda' if free>=4096 else 'cpu'
    except Exception as ex: raw=str(ex);device='cpu'
    if device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']='-1'
    return device, raw

def data():
    d=pd.read_csv(BASE/'depuracion_v1/registros_depurados.csv')
    d=d[d.particion.eq('DESARROLLO')].copy()  # no calibration or final labels in downstream arrays
    for c in ['inicio_registro','fin_peletizado']: d[c]=pd.to_datetime(d[c])
    h=d.inicio_registro.dt.hour+d.inicio_registro.dt.minute/60
    d['hora_sin']=np.sin(2*np.pi*h/24);d['hora_cos']=np.cos(2*np.pi*h/24)
    d['dia_sin']=np.sin(2*np.pi*d.inicio_registro.dt.dayofweek/7)
    d['dia_cos']=np.cos(2*np.pi*d.inicio_registro.dt.dayofweek/7)
    good=d.mascara_objetivo_peletizado.eq(1)&d.apto_temporal_peletizado.eq(1)&d.masa_dosificada_t.gt(0)&d[NUM+CAT].notna().all(axis=1)
    return d[good].set_index('fila_fuente',drop=False).rename_axis('record_id').sort_values(['inicio_registro','fila_fuente']).copy()

def windows(d, floor, ceiling, target_ids=None):
    floor,ceiling=pd.Timestamp(floor),pd.Timestamp(ceiling)
    pool=d[(d.inicio_registro>=floor)&(d.inicio_registro<ceiling)&(d.fin_peletizado<ceiling)].copy()
    cand=pool if target_ids is None else pool[pool.fila_fuente.isin(target_ids)]
    ids=[]; donors=[];excluded=[]
    for r in cand.itertuples():
        h=pool[(pool.linea==r.linea)&(pool.fin_peletizado<r.inicio_registro)].sort_values(['fin_peletizado','fila_fuente']).tail(L)
        if len(h)<L:
            excluded.append({'fila_fuente':r.fila_fuente,'n_historia':len(h),'partition_start':str(floor),'partition_end':str(ceiling)})
            continue
        assert (h.fin_peletizado<r.inicio_registro).all() and (h.linea==r.linea).all()
        assert (h.inicio_registro>=floor).all() and r.fila_fuente not in h.index
        ids.append(r.fila_fuente);donors.append(h.fila_fuente.to_numpy())
    return {'ids':np.array(ids,dtype=np.int64),'donors':np.array(donors,dtype=np.int64).reshape(-1,L),'excluded':excluded,
            'floor':str(floor),'ceiling':str(ceiling)}

def fold_windows(d, start,end):
    old=pd.read_csv(BASE/'analisis_desarrollo_v2/predicciones_desarrollo.csv')
    expected=old[old.modelo.eq('Ridge')&old.fold.eq(start)].fila_fuente.to_numpy()
    inner=(pd.Timestamp(start)-pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    return {'train':windows(d,'2025-01-01',start),
            'inner_train':windows(d,'2025-01-01',inner),
            'inner_val':windows(d,inner,start),
            'val':windows(d,start,end,expected)}

def encode(d, train_windows, apply_windows):
    # Only records that enter training windows, including completed historical donors.
    fit_ids=np.unique(np.r_[train_windows['ids'],train_windows['donors'].ravel()])
    enc=ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
    enc.fit(d.loc[fit_ids])
    ytrain=d.loc[train_windows['ids'],TARGET].to_numpy(float)
    mu=float(ytrain.mean());sd=float(ytrain.std())
    result={}
    for name,w in apply_windows.items():
        rows=np.column_stack([w['donors'],w['ids']])
        x=enc.transform(d.loc[rows.ravel()]).reshape(len(rows),L+1,-1).astype('float32')
        hist=d.loc[w['donors'].ravel(),TARGET].to_numpy(float).reshape(-1,L)
        truth=d.loc[w['ids'],TARGET].to_numpy(float)
        result[name]={'x':x,'hist':((hist-mu)/sd).astype('float32'),
                      'y':((truth-mu)/sd).astype('float32'),'truth':truth,'ids':w['ids']}
    return enc,mu,sd,result

def prediction_frame(d,ids,p,model,fold,**extra):
    raw=np.asarray(p,dtype=float).reshape(-1)
    assert len(raw)==len(ids) and np.isfinite(raw).all()
    q=d.loc[ids,['fila_fuente','linea','familia','subfamilia','inicio_registro',TARGET]].copy().reset_index(drop=True)
    q['modelo']=model;q['fold']=fold;q['raw_prediction_kWh']=raw;q['prediccion_kWh']=np.maximum(0,raw)
    q['error']=q.prediccion_kWh-q[TARGET];q['error_abs']=abs(q.error)
    for k,v in extra.items():q[k]=v
    return q
