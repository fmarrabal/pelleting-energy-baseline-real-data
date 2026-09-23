"""Detailed vector figures from complete recorded data and archived experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

H=Path(__file__).resolve().parent;D=H/'data';F=H/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':11,
 'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlepad':10,
 'legend.fontsize':9.2,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':320,'axes.linewidth':.65,'text.color':'#203D50','axes.labelcolor':'#203D50','xtick.color':'#53636F','ytick.color':'#53636F'})
C={'Pellet 1':'#2367A0','Pellet 2':'#C77C19','Pellet 3':'#27816B'}
dark='#17364D';grey='#6E7881';red='#B74538'
d=pd.read_csv(D/'variables_source.csv',parse_dates=['inicio_registro','inicio_peletizado','fin_peletizado'])
test=pd.read_csv(D/'prueba_original.csv',parse_dates=['inicio_registro','fin_peletizado'])
gpu=pd.read_csv(D/'gpu/predictions.csv',parse_dates=['inicio_registro'])
stats=json.loads((H/'DESCRIPTIVE_STATISTICS.json').read_text(encoding='utf8'))
manifest={}
def save(fig,name,info):
    for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig);manifest[name]=info
def stamp(ax,text,loc=(.04,.96),size=9.2):
    ax.text(*loc,text,transform=ax.transAxes,va='top',fontsize=size,
            bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=3))
def dateaxis(ax,interval=1):
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO,interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'));ax.tick_params(axis='x',labelsize=8)
def box(ax,x,y,w,h,title,body,color='#E9F0F5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',
                             facecolor=color,edgecolor='#8DA2B3',lw=.8))
    ax.text(x+.015,y+h-.045,title,va='top',weight='bold',color=dark,fontsize=10)
    ax.text(x+.015,y+h-.115,body,va='top',color='#253642',fontsize=9,linespacing=1.5)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=grey,lw=1.2))

from figure_design import framework, allocation
framework(d,test,save)
allocation(d,C,save)

# Input / output distributions, all real points and empirical distribution curves.
fig,axes=plt.subplots(2,3,figsize=(8.2,7.3),layout='constrained')
v=d[d.energia_peletizado_kWh.gt(0)]
for j,(line,col) in enumerate(C.items()):
    g=v[v.linea.eq(line)];ax=axes[0,j]
    ax.scatter(g.masa_dosificada_t,g.energia_peletizado_kWh,s=9,color=col,alpha=.44,linewidths=0)
    ax.set(xlim=(0,39),ylim=(0,1230),xlabel='Dosed mass (t)',ylabel='Electricity (kWh / record)' if j==0 else '',title=f'({chr(97+j)}) {line}: n = {len(g)}')
    stamp(ax,f'Median mass {g.masa_dosificada_t.median():.2f} t\nMedian energy {g.energia_peletizado_kWh.median():.1f} kWh')
    ax.grid(alpha=.15)
for line,col in C.items():
    g=d[d.linea.eq(line)]
    x=np.sort(g.masa_dosificada_t);axes[1,0].step(x,np.arange(1,len(x)+1)/len(x),where='post',color=col,label=line)
    counts=g.baches_dosificacion.value_counts().reindex(range(1,11),fill_value=0)
    axes[1,1].plot(counts.index,100*counts/len(g),marker='o',ms=3.5,color=col,label=line)
axes[1,0].set(title='(d) Dosed mass',xlabel='Dosed mass (t)',ylabel='Empirical cumulative fraction',ylim=(0,1.04));axes[1,0].legend(loc='lower right')
axes[1,1].set(title='(e) Dosing batch count',xlabel='Recorded batches / operation',ylabel='Share within machine (%)',xticks=[1,3,5,7,10]);axes[1,1].grid(alpha=.15)
families=['LINEA AVICULTURA','LINEA PORCICULTURA','LINEA GANADERIA','LINEA EQUINOS','LINEA OTROS']
fc=['#517FA4','#D89E57','#579B87','#9A82AF','#A9AFB6'];left=np.zeros(3)
for fam,col,lab in zip(families,fc,['Poultry','Swine','Cattle','Equine','Other']):
    n=np.array([d[(d.linea==line)&(d.familia==fam)].shape[0] for line in C]);tot=np.array([int((d.linea==line).sum()) for line in C]);pct=100*n/tot
    axes[1,2].barh(list(C),pct,left=left,color=col,label=lab)
    for j,(p,l,nn) in enumerate(zip(pct,left,n)):
        if p>12:axes[1,2].text(l+p/2,j,str(nn),ha='center',va='center',fontsize=8,color='white' if col!=fc[1] else '#222')
    left+=pct
axes[1,2].set(title='(f) Product mix',xlabel='Share of all records (%)',xlim=(0,100));axes[1,2].legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncols=2,fontsize=8)
save(fig,'02_production_and_energy',{'positive_energy_points':len(v),'mass_cdf_records':len(d),'batch_count_records':len(d),'family_records':len(d),'truncation':False})

# Temporal design: exact fold counts and one genuine context/target, labelled by source row.
fig,axes=plt.subplots(2,1,figsize=(8.2,8.3),height_ratios=[1,1.8],layout='constrained')
proto=json.loads((D/'gpu/PROTOCOL.json').read_text());ax=axes[0]
for j,((a,b),n) in enumerate(zip(proto['folds'],proto['sample_counts'])):
    start,end=pd.Timestamp(a),pd.Timestamp(b);first=pd.Timestamp('2025-01-02')
    ax.barh(3-j,(start-first).days,left=mdates.date2num(first),height=.62,color='#CCDDEA')
    ax.barh(3-j,7,left=mdates.date2num(start),height=.62,color='#C3DFD1')
    ax.text(mdates.date2num(first)+.6,3-j,f'Train {n["train"]:,}  |  inner {n["inner_train"]:,} / {n["inner_val"]}',va='center',fontsize=8.5)
    ax.text(mdates.date2num(start)+3.5,3-j,f'Val {n["val"]}',ha='center',va='center',fontsize=8.5)
ax.set(yticks=[3,2,1,0],yticklabels=['Week 1','Week 2','Week 3','Week 4'],title='(a) Expanding training and weekly validation',xlabel='2025; green bars mark outer validation');dateaxis(ax)
w=pd.read_csv(D/'example_window.csv',parse_dates=['inicio_peletizado','fin_peletizado','origin','partition_floor'])
t=stats['example_window_target'];origin=pd.Timestamp(t['inicio_registro']);ax=axes[1]
for j,r in enumerate(w.itertuples()):
    a=mdates.date2num(r.inicio_peletizado);b=mdates.date2num(r.fin_peletizado)
    ax.plot([a,b],[j,j],color=C['Pellet 1'],lw=3,solid_capstyle='round');ax.scatter(b,j,s=13,color=C['Pellet 1'],zorder=3)
ax.axvline(origin,color=red,ls='--',lw=1.3)
ax.axvline(pd.Timestamp('2025-01-25'),color=grey,ls=':',lw=1)
ax.set(yticks=range(16),yticklabels=[f'{i+1:02d} | row {r.donor}' for i,r in enumerate(w.itertuples())],
       title='(b) A real 16-record window on Pellet 1',ylabel='History position | original Excel row',xlabel='Recorded pelleting intervals; dots mark completion (2025)')
ax.tick_params(axis='y',labelsize=8);ax.xaxis.set_major_locator(mdates.DayLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.set_xlim(pd.Timestamp('2025-01-24 18:00'),(origin+pd.Timedelta(hours=24)).to_pydatetime());ax.set_ylim(-1,18)
ax.text(mdates.date2num(origin)+.08,11,'Target origin\n28 Jan 04:24\n\nEnergy becomes\navailable no earlier\nthan 28 Jan 11:04',fontsize=8.5,va='center',color=red)
ax.text(mdates.date2num(pd.Timestamp('2025-01-25 03:00')),16.6,'Same machine; all donor outcomes completed before the origin',fontsize=8.5,color=dark)
save(fig,'07_windows',{'target_source_row':823,'donors':w.donor.tolist(),'actual_timestamps':True,'context':16,'horizon_records':1})

# Original development and exploratory contrasts (same records).
fig,aa=plt.subplots(2,2,figsize=(8.2,7.6),layout='constrained',width_ratios=[1.15,1])
axes=aa[0]
names={'Ridge':'Ridge','HistGradientBoosting':'Gradient boosting','ExtraTrees':'ExtraTrees','MLP_3_semillas':'MLP (3-seed mean)','SEC_maquina':'Machine SEC','SEC_ultimos5':'Recent SEC (5)','Mediana_maquina':'Machine median'}
m=pd.read_csv(D/'modelos_desarrollo.csv').sort_values('MAE_kWh')
axes[0].barh([names[n] for n in m.modelo],m.MAE_kWh,color=[dark if n=='Ridge' else '#94ABBD' for n in m.modelo])
for j,r in enumerate(m.itertuples()):axes[0].text(r.MAE_kWh+1,j,f'{r.MAE_kWh:.2f}',va='center',fontsize=8.5)
axes[0].invert_yaxis();axes[0].set(xlim=(0,96),xlabel='MAE (kWh / record)',title='(a) Original comparison (n = 800)')
h=pd.read_csv(D/'hipotesis_contrastes.csv')
for j,r in enumerate(h.itertuples()):
    axes[1].plot([r.IC98_333_inf,r.IC98_333_sup],[j,j],color='#A6BBC9',lw=6)
    axes[1].plot([r.IC95_inf,r.IC95_sup],[j,j],color=dark,lw=2)
    axes[1].scatter(r.reduccion_MAE,j,color=dark,s=30,zorder=4)
    axes[1].text(r.reduccion_MAE,j-.16,f'{r.reduccion_MAE:.3f}',ha='center',fontsize=9)
axes[1].axvline(0,color=grey,ls='--',lw=.9)
axes[1].set(yticks=range(3),yticklabels=['H1: mass x machine','H2: mass x subfamily','H3: recent residuals'],ylim=(3,-.6),xlim=(-.5,3.65),xlabel='Paired MAE reduction (kWh)',title='(b) Exploratory MAE reductions')
stamp(axes[1],'Dark: 95%; light: 98.33% intervals\n5,000 resamples of 3 observed dates\nPositive values favour the update',(.02,.22),8)
hp=pd.read_csv(D/'hipotesis_predicciones.csv')
for ax,key,title in [(aa[1,0],'linea','(c) Reduction by machine'),(aa[1,1],'fold','(d) Reduction by validation week')]:
    means=hp.groupby(['modelo',key])['abs'].mean().unstack(key)
    counts=hp[hp.modelo.eq('Ridge')].groupby(key).size()
    positions=np.arange(len(means.columns))
    for name,col,mark,label in [('H1_masa_maquina',C['Pellet 1'],'o','H1'),('H2_masa_subfamilia',C['Pellet 2'],'s','H2'),('H3_sesgo_reciente',C['Pellet 3'],'^','H3')]:
        ax.plot(positions,means.loc['Ridge']-means.loc[name],color=col,marker=mark,ms=4,label=label,lw=1.2)
    labs=[str(v).replace('Pellet ','P') if key=='linea' else f'{pd.Timestamp(v):%d %b}' for v in means.columns]
    ax.set_xticks(positions,[lab+f'\nn={int(counts[v])}' for lab,v in zip(labs,means.columns)],fontsize=9)
    ax.set(title=title,ylabel='MAE reduction (kWh)',ylim=(-1.2,5.1))
    ax.axhline(0,ls='--',color=grey,lw=.8);ax.grid(axis='y',alpha=.15)
    ax.legend(ncols=3,loc='upper left',fontsize=8)
save(fig,'03_development_and_hypotheses',{'records_per_method':800,'methods':7,'hypotheses':3,'archived_intervals':True,'new_panels':'machine and week MAE differences recalculated from unchanged archived predictions'})

# GPU comparison: all methods; CI of differences and fold/line detail.
ORDER=['Ridge_actual','TFT','Ridge_ventana','NHITS','Chronos_2_covariates','TimesFM_2.5_univariate','Chronos_2_univariate','TimesFM_2.5_covariates']
LABEL={'Ridge_actual':'Ridge: current','Ridge_ventana':'Ridge: full window','TFT':'TFT (3 seeds)','NHITS':'N-HiTS (3 seeds)',
 'Chronos_2_covariates':'Chronos-2 + cov.','Chronos_2_univariate':'Chronos-2: energy only',
 'TimesFM_2.5_covariates':'TimesFM 2.5 + XReg','TimesFM_2.5_univariate':'TimesFM 2.5: energy only'}
ct=pd.read_csv(D/'gpu/paired_contrasts.csv').set_index('modelo').loc[ORDER]
mt=pd.read_csv(D/'gpu/metrics.csv').set_index('modelo').loc[ORDER]
fig,axes=plt.subplots(1,2,figsize=(8.2,5.4),layout='constrained',width_ratios=[1,1])
for j,model in enumerate(ORDER):
    r=ct.loc[model];v=mt.loc[model,'MAE_kWh'];col=dark if model=='Ridge_actual' else '#577F9E'
    axes[0].plot([r.MAE_CI95_low,r.MAE_CI95_high],[j,j],color=col,lw=1.8);axes[0].scatter(v,j,color=col,s=27)
    axes[0].text(v+3,j-.12,f'{v:.2f}',fontsize=8.5,color=dark)
    if j:
        axes[1].plot([r.CI_family7_low,r.CI_family7_high],[j,j],color='#B6C7D3',lw=5)
        axes[1].plot([r.CI95_low,r.CI95_high],[j,j],color=dark,lw=1.6);axes[1].scatter(r.MAE_improvement_vs_Ridge_kWh,j,s=25,color=dark,zorder=4)
axes[0].set(yticks=range(8),yticklabels=[LABEL[x] for x in ORDER],ylim=(7.6,-.6),xlim=(40,224),xlabel='MAE (kWh / record), 95% interval',title='(a) Overall error (n = 584)')
axes[1].set(yticks=range(8),yticklabels=[],ylim=(7.6,-.6),xlim=(-215,15),xlabel='Paired MAE reduction (kWh)',title='(b) Difference from Ridge')
axes[1].axvline(0,ls='--',lw=.9,color=red);stamp(axes[1],'TFT: -0.11 kWh\n95% CI [-2.20, 2.32]\n\nDark: 95% intervals\nLight: 99.29% (7 contrasts)',(.04,.94),8.5)
for ax in axes:ax.grid(axis='x',alpha=.17)
save(fig,'08_gpu_comparison',{'records':584,'methods':8,'resamples':5000,'unit':'kWh per record','all_intervals_full_range':True})

fig,axes=plt.subplots(1,2,figsize=(8.2,5.8),layout='constrained',width_ratios=[1.16,1])
for ax,filename,key,title in [(axes[0],'metrics_by_fold.csv','fold','(a) Validation week'),(axes[1],'metrics_by_machine.csv','linea','(b) Machine')]:
    q=pd.read_csv(D/'gpu'/filename);pivot=q.pivot(index='modelo',columns=key,values='MAE_kWh').loc[ORDER]
    im=ax.imshow(pivot,vmin=35,vmax=130,cmap='YlGnBu',aspect='auto')
    cols=list(pivot.columns);counts=q[q.modelo=='Ridge_actual'].set_index(key).n
    ax.set(yticks=range(8),yticklabels=[LABEL[x] for x in ORDER] if ax==axes[0] else [],xticks=range(len(cols)),
           xticklabels=[(str(x)[5:] if key=='fold' else x.replace('Pellet ','P'))+f'\nn={int(counts[x])}' for x in cols],title=title)
    for a in range(8):
        for b in range(len(cols)):
            val=pivot.iloc[a,b];ax.text(b,a,f'{val:.1f}',ha='center',va='center',fontsize=9,color='white' if val>90 else '#17364D')
fig.colorbar(im,ax=axes,shrink=.72,label='MAE (kWh / record); saturation at 130',extend='max')
save(fig,'09_gpu_stratification',{'records':584,'every_fold_and_machine':True,'cell_text_untruncated':True})

# Every prediction from each primary model; no sampling or omission of extreme XReg errors.
fig,axes=plt.subplots(4,2,figsize=(8.2,11),layout='constrained')
for j,(ax,model) in enumerate(zip(axes.flat,ORDER)):
    q=gpu[gpu.modelo==model]
    for line,col in C.items():
        g=q[q.linea==line];ax.scatter(g.energia_peletizado_kWh,g.prediccion_kWh,s=7,alpha=.5,color=col,label=line,linewidths=0)
    ax.plot([0,1100],[0,1100],ls='--',color=grey,lw=.8)
    ax.set(xlim=(-20,1100),ylim=(-20,1100),xlabel='Observed energy (kWh)',ylabel='Predicted energy (kWh)',title=f'({chr(97+j)}) {LABEL[model]}')
    stamp(ax,f'MAE {mt.loc[model,"MAE_kWh"]:.2f}  |  RMSE {mt.loc[model,"RMSE_kWh"]:.2f} kWh',(.03,.95),8)
    if model=='TimesFM_2.5_covariates':
        inset=ax.inset_axes([.57,.48,.41,.37])
        for line,col in C.items():
            g=q[q.linea==line];inset.scatter(g.energia_peletizado_kWh,g.prediccion_kWh,s=3,alpha=.4,color=col)
        inset.set(xlim=(0,1100),ylim=(0,13000),xticks=[0,1000],yticks=[0,6000,12000]);inset.tick_params(labelsize=6.5)
        inset.set_title('Full range: all 584',fontsize=7,pad=2,bbox=dict(facecolor='white',edgecolor='none',pad=.5))
        nout=int((q.prediccion_kWh>1100).sum());ax.text(.03,.72,f'{nout} predictions\n> 1,100 kWh\nsee full-range inset',transform=ax.transAxes,fontsize=8,color=red,va='top')
axes[0,0].legend(loc='lower right',fontsize=8)
save(fig,'10_gpu_all_predictions',{'points':len(gpu),'records_per_method':584,'no_subsampling':True,'TimesFM_full_range_inset':True})

# Frozen Ridge interpretation: exact coefficients + all-record contributions and example.
cf=pd.read_csv(D/'ridge_coefficients.csv');parts=pd.read_csv(D/'ridge_additive_components.csv')
fig,axes=plt.subplots(2,2,figsize=(8.2,8.1),layout='constrained')
num=cf.iloc[:6]
axes[0,0].barh(['Dosed mass','Dosing batches','Hour: sine','Hour: cosine','Weekday: sine','Weekday: cosine'],num.coefficient_kWh,color=['#407F9B' if x>=0 else '#C77C63' for x in num.coefficient_kWh])
axes[0,0].invert_yaxis();axes[0,0].set(xlim=(-12,88),xlabel='Coefficient (kWh / training SD)',title='(a) Numerical coefficients')
for j,v in enumerate(num.coefficient_kWh):axes[0,0].text(max(v,0)+1,j,f'{v:.2f}',va='center',fontsize=8)
cat=cf.iloc[6:].copy();labs=['P1','P2','P3','Poultry: broiler','Poultry: layer','Equine','Cattle: meat','Cattle: dairy','Rabbit','Swine: finishing','Swine: breeding','Swine: prestarter','Crumbled','Pelleted','Bag','Bulk']
axes[0,1].barh(labs,cat.coefficient_kWh,color=['#407F9B' if x>=0 else '#C77C63' for x in cat.coefficient_kWh]);axes[0,1].invert_yaxis()
axes[0,1].axvline(0,color=grey,lw=.7);axes[0,1].set(xlim=(-65,80),xlabel='One-hot coefficient (kWh)',title='(b) Categorical coefficients');axes[0,1].tick_params(axis='y',labelsize=8)
groups=['Dosed mass','Dosing batches','Hour','Weekday','Machine','Product subfamily','Presentation','Bag / bulk']
rng=np.random.default_rng(20260922)
for j,col in enumerate(groups):
    vals=parts[col].to_numpy();axes[1,0].scatter(vals,j+rng.uniform(-.15,.15,len(vals)),s=2,alpha=.12,color=dark)
    q=np.quantile(vals,[.25,.5,.75]);axes[1,0].plot([q[0],q[2]],[j,j],color='#C77C19',lw=3);axes[1,0].scatter(q[1],j,color=red,s=12,zorder=3)
axes[1,0].set(yticks=range(8),yticklabels=groups,ylim=(7.6,-.6),xlabel='Contribution to the estimate (kWh)',title='(c) Contributions across 798 records');axes[1,0].axvline(0,color=grey,lw=.7)
# Choose deterministically by observed energy nearest the test median, tie by row ID.
row=parts.assign(dist=abs(parts.observed_kWh-parts.observed_kWh.median())).sort_values(['dist','fila_fuente']).iloc[0]
vals=[row.intercept_kWh]+[row[g] for g in groups];labels=['Intercept']+groups
axes[1,1].barh(labels,vals,color=[grey]+['#407F9B' if x>=0 else '#C77C63' for x in vals[1:]])
axes[1,1].invert_yaxis();axes[1,1].set(xlabel='Additive term (kWh)',title=f'(d) Worked example: row {int(row.fila_fuente)}')
for j,val in enumerate(vals):axes[1,1].text(max(val,0)+3,j,f'{val:.1f}',va='center',fontsize=8)
axes[1,1].set_xlim(min(vals)-20,360)
stamp(axes[1,1],f'Sum = {row.archived_prediction_kWh:.2f} kWh\nObserved = {row.observed_kWh:.2f} kWh',(.46,.35),8.5)
save(fig,'11_explanation',{'coefficient_count':22,'decomposed_test_records':798,'example_source_row':int(row.fila_fuente),'example_rule':'nearest median observed test energy, then source row','jitter_only_display':True,'causal_effects':False})

# Final-test error diagnostics: all records, robust annotations and full residual ranges.
fig,axes=plt.subplots(2,3,figsize=(8.2,7),layout='constrained')
errors=test.Ridge-test.energia_peletizado_kWh
for j,(line,col) in enumerate(C.items()):
    g=test[test.linea==line].sort_values('inicio_registro');e=g.Ridge-g.energia_peletizado_kWh;ax=axes[0,j]
    ax.scatter(g.energia_peletizado_kWh,g.Ridge,s=13,alpha=.6,color=col,linewidths=0);ax.plot([0,1150],[0,1150],ls='--',color=grey,lw=.8)
    ax.set(xlim=(0,1150),ylim=(0,1150),xlabel='Observed energy (kWh)',ylabel='Frozen estimate (kWh)' if j==0 else '',title=f'({chr(97+j)}) {line}: n = {len(g)}')
    stamp(ax,f'MAE {e.abs().mean():.2f}\nRMSE {np.sqrt(np.mean(e**2)):.2f}\nBias {e.mean():.2f} kWh',size=8.5)
    ax=axes[1,j];ax.scatter(g.inicio_registro,e,s=12,alpha=.62,color=col,linewidths=0);ax.axhline(0,ls='--',color=grey,lw=.8)
    ax.axhline(e.mean(),color=red,lw=1,label='Machine mean error')
    ax.set(ylim=(min(errors.min()*1.1,-10),max(errors.max()*1.15,10)),xlabel='Origin date in 2025',ylabel='Estimate - observed (kWh)' if j==0 else '',title=f'({chr(100+j)}) Signed errors over time');dateaxis(ax,2)
    if j==0:ax.legend(loc='lower left',fontsize=8)
save(fig,'04_original_test_scatter',{'all_test_records':798,'shared_axis_full_range':True,'test_reused_for_selection':False})

fig,axes=plt.subplots(3,1,figsize=(8.2,8.5),layout='constrained')
for j,(ax,(line,col)) in enumerate(zip(axes,C.items())):
    g=test[test.linea==line].sort_values(['inicio_registro','fila_fuente']);x=np.arange(1,len(g)+1)
    outside=(g.energia_peletizado_kWh<g.L95)|(g.energia_peletizado_kWh>g.U95)
    # Individual vertical intervals avoid suggesting a continuous energy envelope.
    ax.vlines(x,g.L95,g.U95,color=col,alpha=.28,lw=.75,label='Fixed 95% interval')
    ax.scatter(x,g.Ridge,s=7,color=col,alpha=.8,label='Ridge estimate');ax.scatter(x,g.energia_peletizado_kWh,s=9,color='#30343A',alpha=.65,label='Observed')
    ax.scatter(x[outside],g.loc[outside,'energia_peletizado_kWh'],s=27,facecolor='none',edgecolor=red,lw=.8,label='Outside interval')
    ax.set(title=f'({chr(97+j)}) {line}: n = {len(g)}; 95% coverage {100*(1-outside.mean()):.1f}%',xlabel='Record order within machine (irregular events)',ylabel='Pelleting energy (kWh)',xlim=(-1,len(g)+2),ylim=(-15,1220))
    ax.grid(axis='y',alpha=.15);ax.xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0].legend(ncols=4,loc='upper left',fontsize=8)
save(fig,'05_all_test_records',{'records':798,'bounds':1596,'individual_intervals_not_continuous_band':True,'outside_95':54})

fig,axes=plt.subplots(2,2,figsize=(8.2,7),layout='constrained')
iv=pd.read_csv(D/'intervalos_linea.csv')
for j,nom in enumerate([90,95]):
    ax=axes[0,j]
    for line,col in C.items():
        g=iv[(iv.linea==line)&(iv.nominal==nom)].set_index('metodo').loc[['fijo','actualizado_200']]
        x=g.ancho_medio_kWh.to_numpy();y=100*g.cobertura.to_numpy()
        ax.annotate('',xy=(x[1],y[1]),xytext=(x[0],y[0]),arrowprops=dict(arrowstyle='->',color=col,lw=1.2))
        ax.scatter(x[0],y[0],s=35,color=col,marker='o',label=f'{line} (n={int(g.n.iloc[0])})');ax.scatter(x[1],y[1],s=40,color=col,marker='^')
        fixed_offset=(-6,8) if nom==95 and line=='Pellet 2' else ((-3,10) if nom==95 and line=='Pellet 3' else (-5,-13))
        ax.annotate(f'{y[0]:.1f}%',(x[0],y[0]),xytext=fixed_offset,textcoords='offset points',fontsize=7.5,ha='right',color=col)
        offset=(-5,10) if nom==90 and line=='Pellet 1' else ((5,-12) if nom==90 and line=='Pellet 3' else ((-4,-15) if nom==95 and line=='Pellet 1' else (5,2)))
        ax.annotate(f'{y[1]:.1f}%',(x[1],y[1]),xytext=offset,textcoords='offset points',fontsize=7.5,color=col,ha='right' if offset[0]<0 else 'left')
    ax.axhline(nom,ls='--',color=grey,lw=.8);ax.set(xlabel='Mean interval width (kWh)',ylabel='Coverage (%)',title=f'({chr(97+j)}) Nominal {nom}%: width / coverage',ylim=(78,100),xlim=(140,335) if nom==90 else (225,352))
axes[0,0].legend(loc='lower right',fontsize=8);stamp(axes[0,1],'Circle: fixed\nTriangle: updated (200 residuals)',(.04,.20),8)
weekly=pd.read_csv(D/'intervalos_semana.csv')
for method,marker,col,label in [('fijo','o',C['Pellet 1'],'Fixed'),('actualizado_200','^',C['Pellet 2'],'Updated (200)')]:
    g=weekly[(weekly.nominal==95)&(weekly.metodo==method)].sort_values('semana');dates=pd.to_datetime(g.semana)
    axes[1,0].plot(dates,100*g.cobertura,marker=marker,color=col,label=label)
    axes[1,1].plot(dates,g.interval_score,marker=marker,color=col,label=label)
    if method=='fijo':
        axes[1,0].set_xticks(dates,[f'{dt:%d %b}\nn={int(n)}' for dt,n in zip(dates,g.n)],fontsize=8)
        axes[1,1].set_xticks(dates,[f'{dt:%d %b}\nn={int(n)}' for dt,n in zip(dates,g.n)],fontsize=8)
axes[1,0].axhline(95,ls='--',color=grey,lw=.8);axes[1,0].set(ylabel='Coverage (%)',ylim=(83,102),title='(c) Weekly 95% coverage')
axes[1,1].set(ylabel='Mean interval score (kWh)',title='(d) Interval score (lower is better)')
for ax in axes[1]:ax.legend(fontsize=8);ax.set_xlabel('Week beginning in 2025');ax.grid(alpha=.15)
save(fig,'06_uncertainty_tradeoff',{'records_per_policy':798,'posthoc_update':True,'nominal_levels':[90,95],'point_predictions_unchanged':True})

# Secondary numerical diagnostic with every CPU/GPU prediction and all supervised seeds.
paired=pd.read_csv(D/'gpu/paired_cpu_gpu_predictions.csv');seeds=pd.read_csv(D/'gpu/metrics_seeds.csv')
fig,axes=plt.subplots(1,2,figsize=(8.2,4.8),layout='constrained')
q=paired[paired.modelo=='Chronos_2_covariates']
for line,col in C.items():
    g=q[q.linea_gpu==line];axes[0].scatter(g.prediccion_kWh_cpu,g.prediccion_kWh_gpu-g.prediccion_kWh_cpu,s=10,color=col,alpha=.55,label=line)
axes[0].axhline(0,color=grey,ls='--',lw=.8);axes[0].legend(loc='upper right')
axes[0].set(xlabel='CPU prediction (kWh)',ylabel='GPU - CPU prediction (kWh)',title='(a) Chronos-2: device differences')
stamp(axes[0],f'Median absolute difference: {q.prediction_abs_difference_kWh.median():.2f} kWh\nMaximum: {q.prediction_abs_difference_kWh.max():.2f} kWh',(.03,.90),8)
for j,model in enumerate(['TFT','NHITS']):
    g=seeds[seeds.modelo==model].sort_values('seed');x=j+np.array([-.17,0,.17])
    axes[1].scatter(x,g.MAE_kWh,s=42,color=C['Pellet 1'],label='Individual GPU seed' if j==0 else None)
    for xx,r in zip(x,g.itertuples()):axes[1].annotate(str(r.seed),(xx,r.MAE_kWh),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    axes[1].scatter(j,mt.loc[model,'MAE_kWh'],marker='D',s=42,color=red,label='Mean prediction (3 seeds)' if j==0 else None)
axes[1].axhline(mt.loc['Ridge_actual','MAE_kWh'],color=grey,ls='--',lw=1,label='Matched Ridge')
axes[1].set(xticks=[0,1],xticklabels=['TFT','N-HiTS'],xlim=(-.5,1.5),ylim=(50,80),ylabel='MAE (kWh / record)',title='(b) Training seeds and ensemble');axes[1].legend(loc='upper left',fontsize=8)
save(fig,'12_numerical_and_seeds',{'Chronos_paired_records':584,'seed_predictions':3504,'not_a_new_model_selection':True})
(H/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Generated',len(manifest),'figures, PDF + SVG + PNG; no fabricated observations.')
