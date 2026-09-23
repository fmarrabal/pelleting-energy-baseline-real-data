"""Publication figures from complete archived industrial observations/predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
D=HERE/'data'; F=HERE/'figures'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10.5,
 'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
 'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
colors={'Pellet 1':'#386cb0','Pellet 2':'#d17c20','Pellet 3':'#368369'}
d=pd.read_csv(D/'registros_reales_figuras.csv',parse_dates=['inicio_registro','inicio_peletizado','fin_peletizado'])
test=pd.read_csv(D/'prueba_original.csv',parse_dates=['inicio_registro','fin_peletizado'])
allmetrics=pd.read_csv(D/'modelos_desarrollo.csv')
iv=pd.read_csv(D/'intervalos_registro.csv',parse_dates=['inicio','fin'])
figmanifest={}
def save(fig,name,info):
    fig.savefig(F/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(F/f'{name}.png',bbox_inches='tight')
    plt.close(fig);figmanifest[name]=info
def dates(ax):
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO,interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='x',labelsize=8)

# Actual counts for every calendar date; a missing record is not evidence of zero load.
fig,axes=plt.subplots(2,1,figsize=(8.8,5.6),height_ratios=[2,1.2],layout='constrained')
daily=d.assign(date=d.inicio_registro.dt.normalize()).groupby(['date','linea']).size().unstack(fill_value=0)
daily=daily.reindex(pd.date_range(d.inicio_registro.min().normalize(),d.inicio_registro.max().normalize()),fill_value=0)
bottom=np.zeros(len(daily))
for line in colors:
    axes[0].bar(daily.index,daily[line],bottom=bottom,width=.88,color=colors[line],label=line)
    bottom+=daily[line].to_numpy()
axes[0].legend(ncols=3,loc='upper left');axes[0].set(ylabel='Recorded operations / day',title='(a) All 2,745 industrial records')
for ax in axes:
    for dt in ['2025-02-22','2025-03-06']: ax.axvline(pd.Timestamp(dt),ls='--',lw=1,color='#555555')
    ax.set_xlim(pd.Timestamp('2025-01-01'),pd.Timestamp('2025-04-03'));dates(ax)
for y,(start,end,label,col) in enumerate([
 ('2025-01-02','2025-02-22','Development: 1,542 retained / 1,556','#dbe7f5'),
 ('2025-02-22','2025-03-06','Calibration: 380 / 391','#fae5c9'),
 ('2025-03-06','2025-04-02','Original test: 798 / 798','#d5eade')]):
    a,b=pd.Timestamp(start),pd.Timestamp(end)
    axes[1].barh(2-y,(b-a).days,left=mdates.date2num(a),height=.62,color=col,edgecolor='#999999',lw=.5)
    # Labels placed at fixed left margin avoid squeezing the short calibration interval.
    axes[1].text(mdates.date2num(pd.Timestamp('2025-01-04')),2-y,label,va='center',fontsize=8,
                  bbox=dict(facecolor='white',edgecolor='none',alpha=.90,pad=2))
axes[1].set(yticks=[],title='(b) Preserved allocation; 25 development/calibration exclusions',xlabel='2025; earliest recorded stage start')
save(fig,'01_records_and_partitions',{'source_rows':len(d),'calendar_days':len(daily),'unit':'records, not interpolated samples'})

fig,axes=plt.subplots(2,3,figsize=(8.8,6.4),layout='constrained',height_ratios=[1.5,1])
v=d[d.energia_peletizado_kWh.gt(0)&d.masa_dosificada_t.gt(0)]
families=['LINEA AVICULTURA','LINEA PORCICULTURA','LINEA GANADERIA']
for col,(line,color) in enumerate(colors.items()):
    g=v[v.linea.eq(line)]
    axes[0,col].scatter(g.masa_dosificada_t,g.energia_peletizado_kWh,s=10,alpha=.35,color=color,rasterized=False)
    axes[0,col].set(title=f'{line}\n{len(g):,} positive-energy records',xlabel='Dosed mass (t)',ylabel='Pelleting electricity (kWh)' if col==0 else '',xlim=(0,v.masa_dosificada_t.max()*1.04),ylim=(0,v.energia_peletizado_kWh.max()*1.04))
    gg=d[d.linea.eq(line)]; f=gg.familia.value_counts()
    major=[int(f.get(k,0)) for k in families]; major.append(len(gg)-sum(major))
    axes[1,col].barh(['Poultry','Swine','Cattle','Other'],np.array(major)/len(gg)*100,color=color,alpha=.82)
    axes[1,col].invert_yaxis();axes[1,col].set(xlim=(0,105),xlabel='Share of all records (%)')
    for y,(n,pct) in enumerate(zip(major,np.array(major)/len(gg)*100)):
        axes[1,col].text(min(pct+1.5,92),y,str(n),va='center',fontsize=8)
save(fig,'02_production_and_energy',{'n_scatter':len(v),'n_family':len(d),'all_positive_records':True,
 'max_mass':float(v.masa_dosificada_t.max()),'max_energy':float(v.energia_peletizado_kWh.max())})

fig,axes=plt.subplots(1,2,figsize=(8.8,4.5),layout='constrained',width_ratios=[1.15,1])
names={'Ridge':'Ridge','HistGradientBoosting':'Gradient boosting','ExtraTrees':'ExtraTrees',
 'MLP_3_semillas':'MLP (3-seed mean)','SEC_maquina':'Machine SEC','SEC_ultimos5':'Recent SEC (5)','Mediana_maquina':'Machine median'}
m=allmetrics.sort_values('MAE_kWh')
axes[0].barh([names[n] for n in m.modelo],m.MAE_kWh,color=['#386cb0' if n=='Ridge' else '#94a6b7' for n in m.modelo])
axes[0].invert_yaxis()
for y,val in enumerate(m.MAE_kWh):axes[0].text(val+1,y,f'{val:.2f}',va='center',fontsize=8)
axes[0].set(xlim=(0,95),xlabel='MAE (kWh / record)',title='(a) Original comparison\n800 development records')
h=pd.read_csv(D/'hipotesis_contrastes.csv')
for y,r in enumerate(h.itertuples()):
    axes[1].plot([r.IC98_333_inf,r.IC98_333_sup],[y,y],color='#8fa3b5',lw=5,alpha=.6)
    axes[1].plot([r.IC95_inf,r.IC95_sup],[y,y],color='#386cb0',lw=2)
    axes[1].plot(r.reduccion_MAE,y,'o',color='#17344f',ms=5)
axes[1].axvline(0,color='black',lw=.8,ls='--')
axes[1].set(yticks=range(3),yticklabels=['H1: mass × machine','H2: mass × product','H3: recent residuals'],xlabel='MAE reduction (kWh)',
 title='(b) Exploratory paired differences\nSame 800 records',ylim=(2.6,-.6),xlim=(-.5,3.6))
axes[1].text(.03,.04,'Dark: 95% CI\nLight: 98.33% CI\n3-day block bootstrap',transform=axes[1].transAxes,fontsize=8,
             bbox=dict(facecolor='white',edgecolor='none',pad=3),zorder=5)
save(fig,'03_development_and_hypotheses',{'n_per_method':800,'original_methods':7,'hypotheses':3,'independent_validation':False})

fig,axes=plt.subplots(2,3,figsize=(8.8,6.6),layout='constrained')
maxval=float(max(test.energia_peletizado_kWh.max(),test.Ridge.max()))*1.04
for col,(line,color) in enumerate(colors.items()):
    g=test[test.linea.eq(line)].sort_values('inicio_registro');e=g.Ridge-g.energia_peletizado_kWh
    axes[0,col].scatter(g.energia_peletizado_kWh,g.Ridge,s=14,alpha=.65,color=color)
    axes[0,col].plot([0,maxval],[0,maxval],ls='--',color='#555555',lw=.8)
    axes[0,col].set(title=f'{line}: n = {len(g)}',xlabel='Observed (kWh)',ylabel='Ridge estimate (kWh)' if col==0 else '',xlim=(0,maxval),ylim=(0,maxval))
    axes[0,col].text(.05,.94,f'MAE {e.abs().mean():.2f} kWh\nBias {e.mean():.2f} kWh',transform=axes[0,col].transAxes,va='top',fontsize=8)
    axes[1,col].scatter(g.inicio_registro,e,s=12,alpha=.65,color=color)
    axes[1,col].axhline(0,color='#555555',ls='--',lw=.8)
    axes[1,col].set(ylabel='Estimate minus observation (kWh)' if col==0 else '',xlabel='2025; record origin')
    er=test.Ridge-test.energia_peletizado_kWh
    axes[1,col].set_ylim(min(er.min()*1.08,-10),max(er.max()*1.08,10))
    axes[1,col].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO,interval=2))
    axes[1,col].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
save(fig,'04_original_test_scatter',{'n':len(test),'each_point_one_archived_record':True,'min_residual':float((test.Ridge-test.energia_peletizado_kWh).min()),'max_residual':float((test.Ridge-test.energia_peletizado_kWh).max())})

fig,axes=plt.subplots(3,1,figsize=(8.8,7.8),layout='constrained')
for ax,(line,color) in zip(axes,colors.items()):
    g=test[test.linea.eq(line)].sort_values(['inicio_registro','fila_fuente']);x=np.arange(len(g))
    # Record rank is explicit: no suggestion of a regular sampling interval.
    ax.fill_between(x,g.L95,g.U95,color=color,alpha=.16,label='Fixed 95% interval')
    ax.scatter(x,g.Ridge,s=7,color=color,alpha=.8,label='Ridge estimate')
    ax.scatter(x,g.energia_peletizado_kWh,s=10,color='#252525',alpha=.65,label='Measured energy')
    ax.set(title=f'{line}: every original test record (n = {len(g)})',xlabel='Chronological record rank within machine (irregular event time)',ylabel='Energy (kWh)',xlim=(-2,len(g)+1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',alpha=.15)
axes[0].legend(ncols=3,loc='upper left',fontsize=8)
save(fig,'05_all_test_records',{'n':len(test),'n_interval_bounds':2*len(test),'subsampling':False,'x_axis':'within-machine record rank; not elapsed time'})

fig,axes=plt.subplots(2,2,figsize=(8.8,6.8),layout='constrained')
lineiv=pd.read_csv(D/'intervalos_linea.csv')
for j,nominal in enumerate([90,95]):
    ax=axes[0,j]
    for line,col in colors.items():
        g=lineiv[lineiv.linea.eq(line)&lineiv.nominal.eq(nominal)].set_index('metodo')
        x=g.loc[['fijo','actualizado_200'],'ancho_medio_kWh'];y=100*g.loc[['fijo','actualizado_200'],'cobertura']
        ax.plot(x,y,color=col,lw=1,alpha=.7)
        ax.scatter(x.iloc[0],y.iloc[0],marker='o',s=38,color=col,label=line)
        ax.scatter(x.iloc[1],y.iloc[1],marker='^',s=45,color=col)
    ax.axhline(nominal,ls='--',color='#555555',lw=.8)
    ax.set(xlabel='Mean interval width (kWh)',ylabel='Observed coverage (%)',title=f'({chr(97+j)}) Nominal {nominal}%\nMachine-level trade-off',ylim=(78,100))
axes[0,0].legend(loc='lower right');axes[0,1].text(.03,.04,'Circle: fixed\nTriangle: updated (200)',transform=axes[0,1].transAxes,fontsize=8)
weekly=pd.read_csv(D/'intervalos_semana.csv')
for method,marker,col,label in [('fijo','o','#386cb0','Fixed'),('actualizado_200','^','#b56625','Updated (200)')]:
    g=weekly[weekly.nominal.eq(95)&weekly.metodo.eq(method)].copy()
    g['fecha']=pd.to_datetime(g.semana);g=g.sort_values('fecha')
    axes[1,0].plot(g.fecha,100*g.cobertura,marker=marker,color=col,label=label)
    axes[1,1].plot(g.fecha,g.interval_score,marker=marker,color=col,label=label)
    if method=='fijo':
        for r in g.itertuples(): axes[1,0].annotate(f'n={int(r.n)}',(r.fecha,r.cobertura*100),xytext=(0,-13),textcoords='offset points',ha='center',fontsize=7)
for ax in axes[1]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'));ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO));ax.set_xlabel('Week beginning in 2025\nEdge weeks are partial');ax.tick_params(axis='x',labelsize=8)
axes[1,0].axhline(95,ls='--',color='#555555',lw=.8);axes[1,0].set(ylabel='Coverage (%)',ylim=(83,102),title='(c) Weekly 95% coverage');axes[1,0].legend(loc='lower left')
axes[1,1].set(ylabel='Mean interval score (kWh)',title='(d) Weekly 95% interval score\nLower is better');axes[1,1].legend()
save(fig,'06_uncertainty_tradeoff',{'n_per_method':798,'source':'archived fixed and post-hoc updated intervals','independent_update_validation':False})
(HERE/'FIGURE_MANIFEST.json').write_text(json.dumps(figmanifest,indent=2),encoding='utf8')
print(json.dumps(figmanifest,indent=2))
