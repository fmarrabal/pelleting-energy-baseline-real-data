"""Editorial designs for the opening figures; all numeric examples are archived data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK='#203D50'; BLUE='#2367A0'; TEAL='#27816B'; GREY='#687680'; PALE='#EDF3F7'
def frame(ax,x,y,w,h,fill='white',edge='#CDD9E1',radius=.045):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.012,rounding_size={radius}',
                             linewidth=.8,edgecolor=edge,facecolor=fill))
def txt(ax,x,y,s,size=10,weight='normal',color=INK,ha='left',va='top',**kw):
    return ax.text(x,y,s,fontsize=size,weight=weight,color=color,ha=ha,va=va,linespacing=1.35,**kw)
def arr(ax,a,b,col=GREY,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,linewidth=1,
                                color=col,linestyle='--' if dashed else '-'))

def framework(d,test,save):
    fig=plt.figure(figsize=(7.5,5.95));ax=fig.add_axes([.025,.015,.95,.97]);ax.set(xlim=(0,7.1),ylim=(0,5.15));ax.axis('off')
    txt(ax,0,5.12,'a   From production conditions to an energy baseline',11.3,'bold')
    # Three concise, aligned regions: inputs, calculation, returned quantities.
    frame(ax,.015,2.54,2.205,2.21,PALE)
    txt(ax,.14,4.59,'PRODUCTION RECORD',9,'bold',BLUE)
    for y,head,body in [(4.24,'Quantity','Dosed mass (t)\nDosing batch count'),
                        (3.60,'Product and equipment','Pelleting line and subfamily\nPresentation; bag / bulk'),
                        (2.96,'Calendar','Origin hour and weekday')]:
        txt(ax,.14,y,head,9.7,'bold');txt(ax,.14,y-.20,body,9.2)
    arr(ax,(2.25,3.72),(2.73,3.72))
    frame(ax,2.76,3.04,1.68,1.40,'#F6F9FB',BLUE)
    txt(ax,3.60,4.23,'RIDGE BASELINE',9.3,'bold',BLUE,ha='center')
    txt(ax,3.60,3.91,'Training-based\nscaling and encoding',9.3,ha='center')
    txt(ax,3.60,3.38,r'$\hat y=\max(0,\,b+z^{\mathsf{T}}\beta)$',10,ha='center')
    arr(ax,(4.47,3.72),(4.96,3.72))
    txt(ax,5.12,4.56,'ENERGY PER RECORD',9,'bold',BLUE)
    txt(ax,5.12,4.22,'Point estimate',10.2,'bold')
    txt(ax,5.12,3.99,'Expected electricity (kWh)',9.2)
    txt(ax,5.12,3.62,'Prediction intervals',10.2,'bold')
    txt(ax,5.12,3.39,'90% and 95% limits (kWh)',9.2)
    txt(ax,5.12,3.04,'Variable contributions',10.2,'bold')
    txt(ax,5.12,2.82,'Additive terms in the estimate',9.2)
    txt(ax,3.58,2.74,'Calibrated on later\ncompleted operations',9.1,ha='center',color=GREY)
    arr(ax,(4.42,2.64),(4.97,3.32),col=GREY,dashed=True)
    ax.plot([0,7.05],[2.26,2.26],color='#CED9E0',lw=.8)

    txt(ax,0,2.08,'b   Reading the output: an actual operation',11.3,'bold')
    r=test[test.fila_fuente.eq(2080)].iloc[0]
    src=d[d.fila_fuente.eq(2080)].iloc[0]
    txt(ax,0,1.73,'Pellet 2  |  10 March 2025',10,'bold')
    txt(ax,0,1.46,f'{src.masa_dosificada_t:.3f} t; {int(src.baches_dosificacion)} dosing batches\nSwine-finishing product\nPelleted; bagged\nOrigin 08:31; completion 11:21',9.5)
    txt(ax,0,.32,'Source row 2080',9,color=GREY)
    # Numeric output drawn on a genuine kWh axis, not a schematic uncertainty width.
    plot=fig.add_axes([.435,.135,.51,.17])
    plot.plot([r.L95,r.U95],[0,0],color='#B9CFE0',lw=10,solid_capstyle='round',zorder=1)
    plot.plot([r.L90,r.U90],[0,0],color=BLUE,lw=4,solid_capstyle='round',zorder=2)
    plot.scatter([r.Ridge],[0],color=INK,s=45,zorder=4)
    plot.scatter([r.energia_peletizado_kWh],[.15],color=TEAL,marker='D',s=33,zorder=4)
    plot.annotate(f'Estimate {r.Ridge:.2f}',(r.Ridge,0),xytext=(-8,-24),textcoords='offset points',ha='right',fontsize=9.2,color=INK,
                  arrowprops=dict(arrowstyle='-',color=INK,lw=.6))
    plot.annotate(f'Observed {r.energia_peletizado_kWh:.2f}',(r.energia_peletizado_kWh,.15),xytext=(7,21),textcoords='offset points',ha='left',fontsize=9.2,color=TEAL,
                  arrowprops=dict(arrowstyle='-',color=TEAL,lw=.6))
    plot.set(xlim=(80,420),ylim=(-.65,.90),yticks=[],xticks=[100,200,300,400],xlabel='Pelleting electricity (kWh)')
    plot.spines[['top','left','right']].set_visible(False);plot.spines['bottom'].set_color('#A8B5BF');plot.tick_params(axis='x',labelsize=9)
    txt(ax,3.11,1.70,f'95% interval: {r.L95:.2f} - {r.U95:.2f} kWh',9.5,color=GREY)
    txt(ax,3.11,.075,f'Observed excess: +{r.energia_peletizado_kWh-r.Ridge:.2f} kWh; inside the interval',9.2,'bold',TEAL)
    save(fig,'00_framework',{'type':'baseline inputs and outputs with actual worked example','example_source_row':2080,'example_prediction_kWh':float(r.Ridge),'example_observed_kWh':float(r.energia_peletizado_kWh),'native_intervals_90_95':True})

def allocation(d,C,save):
    fig=plt.figure(figsize=(7.5,5.55))
    gs=fig.add_gridspec(4,1,height_ratios=[1,1,1,1.32],left=.11,right=.98,top=.94,bottom=.03,hspace=.36)
    daily=d.assign(day=d.inicio_registro.dt.normalize()).groupby(['day','linea']).size().unstack(fill_value=0)
    daily=daily.reindex(pd.date_range('2025-01-02','2025-04-01'),fill_value=0)
    a,b=pd.Timestamp('2025-02-22'),pd.Timestamp('2025-03-06')
    axes=[]
    ymax=int(np.ceil(daily.max().max()/5)*5)
    for j,(line,col) in enumerate(C.items()):
        ax=fig.add_subplot(gs[j]);axes.append(ax)
        ax.axvspan(pd.Timestamp('2025-01-01'),a,color='#EBF2F7',zorder=0)
        ax.axvspan(a,b,color='#FBF0DC',zorder=0)
        ax.axvspan(b,pd.Timestamp('2025-04-03'),color='#E8F3ED',zorder=0)
        ax.bar(daily.index,daily[line],color=col,width=.83,zorder=3)
        for boundary in [a,b]:ax.axvline(boundary,color='#80919D',ls=(0,(3,3)),lw=.8,zorder=4)
        ax.set(xlim=(pd.Timestamp('2025-01-01'),pd.Timestamp('2025-04-03')),ylim=(0,ymax+4),yticks=[0,10,20])
        ax.set_ylabel('Records / day',fontsize=9.5,labelpad=8)
        ax.text(.01,.87,f'{line}   |   n = {int(daily[line].sum()):,}',transform=ax.transAxes,fontsize=9.5,color=INK,weight='bold')
        ax.tick_params(axis='y',labelsize=9);ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#A9B6BF')
        ticks=pd.to_datetime(['2025-01-02','2025-01-20','2025-02-08','2025-02-22','2025-03-06','2025-03-20','2025-04-01'])
        ax.set_xticks(ticks)
        if j<2:ax.tick_params(axis='x',labelbottom=False,length=0)
        else:ax.set_xticklabels([f'{x:%d %b}' for x in ticks],fontsize=9)
    fig.text(.11,.976,'a   Daily records by pelleting line',fontsize=11,weight='bold',color=INK)
    ax=fig.add_subplot(gs[3]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(0,.98,'b   Chronological allocation',fontsize=11,weight='bold',color=INK,va='top')
    for x,w,title,dates,ns,foot,fill in [
        (0,.31,'Development','2 Jan - 21 Feb','1,542 retained','1,556 candidates; 14 excluded','#EBF2F7'),
        (.345,.31,'Calibration','22 Feb - 5 Mar','380 retained','391 candidates; 11 excluded','#FBF0DC'),
        (.690,.31,'Frozen test','6 Mar - 1 Apr','798 retained','798 candidates; none excluded','#E8F3ED')]:
        frame(ax,x+.012,.03,w-.024,.72,fill)
        ax.text(x+.018,.65,title,transform=ax.transAxes,fontsize=10.1,weight='bold',color=INK)
        ax.text(x+.018,.49,dates,transform=ax.transAxes,fontsize=9,color=GREY)
        ax.text(x+.018,.29,ns,transform=ax.transAxes,fontsize=10.2,weight='bold',color=INK)
        ax.text(x+.018,.10,foot,transform=ax.transAxes,fontsize=8.0,color=GREY)
    save(fig,'01_records_and_partitions',{'source_records':2745,'machine_counts':{k:int(daily[k].sum()) for k in C},'retained':[1542,380,798],'candidate':[1556,391,798],'calendar_days':len(daily),'zero_record_day_not_zero_load':True})
