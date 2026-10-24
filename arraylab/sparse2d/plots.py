"""Static scientific figures shared by desktop and batch entrypoints."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from .core import Config, available_mask, candidates, coarray, db, field, response_field, scan_uv, structured_arrays
from .scenarios import case_config,scenarios
from .elements import build_element
from ..parallel import ordered_map


def _plot_response(*args):
    """Plot entrypoints run after synthesis, outside a numerical worker batch."""
    rows, _ = ordered_map(lambda _: response_field(*args), [None], workers=1)
    return rows[0]

NAMES = {
    'Dense reference':'满阵参考', 'Regular thinning':'规则抽稀',
    'Paired reference (center excluded)':'对称配对参考（不含中心）',
    'Random symmetric':'随机选点', 'Greedy exchange':'贪心交换',
    'Simulated annealing':'模拟退火', 'L1 top-K uniform':'ℓ1 取整等幅',
    'L1 weighted solution':'ℓ1 加权解',
    'Hollow uniform':'掏洞全可用等幅', 'Hollow amplitude LP':'掏洞固定位置调幅',
    'SA amplitude LP':'退火选点后调幅',
    'Feasible packing reference':'间距可行参考', 'Packing amplitude LP':'可行参考联合调幅',
    'Joint amplitude LP':'全可用位置联合调幅',
}
COLORS=['#8795a5','#b4a4d5','#d48a61','#00998b','#2978b8','#e2a626','#ba477c','#6452a0','#688745','#375e7b']


def result_name(result):
    if 'weighting' not in result: return NAMES.get(result['name'],result['name'])
    from .weighting import METHODS
    w=result['weighting']
    return f"{NAMES.get(w['layout_name'],w['layout_name'])} + {METHODS[w['method']]} #{result['name'].rsplit('#',1)[-1]}"


def plot_name(result):
    if 'weighting' not in result: return result['name']
    w=result['weighting']
    short={'Simulated annealing':'SA','Greedy exchange':'Greedy','Random symmetric':'Random',
           'Regular thinning':'Regular','L1 top-K uniform':'L1 top-K','L1 weighted solution':'L1 weighted',
           'Dense reference':'Dense','Hollow amplitude LP':'Hollow LP','Hollow uniform':'Hollow',
           'SA amplitude LP':'SA + LP'}
    return f"{short.get(w['layout_name'],w['layout_name'])} + {w['method'].title()} #{result['name'].rsplit('#',1)[-1]}"


def style():
    import matplotlib as mpl
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':11,
        'axes.labelcolor':'#34435a','axes.edgecolor':'#cad3df',
        'grid.color':'#e6ebf1','figure.facecolor':'white','savefig.facecolor':'white'})


def display_case(bundle,result,case_index=None):
    c=Config(**bundle['config']); cases=bundle.get('scenarios') or scenarios(c)
    index=result['metrics'].get('worst_case_index',0) if case_index is None else case_index
    case=cases[index]
    return c,case_config(c,case),case


DETAIL_PLOTS=(('layout','阵列布局'),('pattern','二维方向图'),('cuts','扫描切线'),('coarray','差分共阵'))


def _draw_detail_axes(fig,ax,kind,bundle,result,case_index=None,view_mode='total',phase_mode='ideal'):
    c,cc,case=display_case(bundle,result,case_index)
    xy=result['xy']; w=result['weights']; center=scan_uv(cc)
    if kind=='layout':
        allxy=candidates(c)[1]
        ax.scatter(allxy[:,0],allxy[:,1],s=12,color='#dce3ec',label='Candidate')
        forbidden=allxy[~available_mask(c)]
        if len(forbidden):
            ax.scatter(forbidden[:,0],forbidden[:,1],s=18,color='#cb6573',marker='x',label='Forbidden')
            shape=(Circle((0,0),c.hole_x/2) if c.hole_shape=='circle' else
                   Rectangle((-c.hole_x/2,-c.hole_y/2),c.hole_x,c.hole_y))
            shape.set(facecolor='#f9e1e4',edgecolor='#cb6573',alpha=.45,zorder=0); ax.add_patch(shape)
        ax.scatter(xy[:,0],xy[:,1],s=20+60*w/w.max(),color='#00998b',edgecolors='white',linewidths=.4,label='Active')
        ax.set(xlabel='x / λ0',ylabel='y / λ0',title=f"{plot_name(result)} | N={len(xy)}",aspect='equal')
        ax.grid(alpha=.5); ax.legend(loc='upper right',fontsize=7)
    elif kind=='pattern':
        element=build_element(c,bundle.get('element_pattern'))
        axis=np.linspace(-1,1,181); uu,vv=np.meshgrid(axis,axis)
        uv=np.column_stack((uu.ravel(),vv.ravel())); visible=np.sum(uv**2,axis=1)<=1
        values=np.full(len(uv),np.nan); values[visible]=db(_plot_response(xy,w,uv[visible],c,case,element,view_mode,phase_mode))
        im=ax.imshow(values.reshape(uu.shape),origin='lower',extent=(-1,1,-1,1),vmin=-40,vmax=max(0.,float(np.nanmax(values))),cmap='viridis')
        rx,ry=1/(c.nx*cc.spacing),1/(c.ny*cc.spacing)
        if view_mode!='element': ax.add_patch(Rectangle(center-[rx,ry],2*rx,2*ry,fill=False,edgecolor='white',linestyle='--',linewidth=1))
        ax.plot(*center,'w+',ms=8)
        view_label={'total':'Total','array_factor':'Array factor','element':'Element'}[view_mode]
        prefix=f'{c.phase_bits}-bit ' if phase_mode=='quantized' and c.phase_bits and view_mode!='element' else ''
        ax.set(xlabel='u',ylabel='v',title=f"{prefix}{view_label}: {case['frequency_ghz']:g} GHz, {case['theta']:g}/{case['phi']:g} deg")
        fig.colorbar(im,ax=ax,label='Element relative magnitude / dB' if view_mode=='element' else 'Field / ideal steering reference dB',shrink=.83,pad=.02)
    elif kind=='cuts':
        element=build_element(c,bundle.get('element_pattern'))
        for along,color,label in [(0,'#00998b','u cut, v=v0'),(1,'#2978b8','v cut, u=u0')]:
            extent=np.sqrt(1-center[1-along]**2)
            axis=np.linspace(-extent,extent,1401); uv=np.tile(center,(len(axis),1)); uv[:,along]=axis
            ax.plot(axis,db(_plot_response(xy,w,uv,c,case,element,view_mode,phase_mode)),color=color,label=label,lw=1.3)
            if phase_mode=='quantized' and c.phase_bits and view_mode!='element':
                ax.plot(axis,db(_plot_response(xy,w,uv,c,case,element,view_mode)),color=color,ls=':',alpha=.7,lw=1,label=('u ideal' if along==0 else 'v ideal'))
            if 'weighting' in result:
                source=bundle['results'][result['weighting']['source_index']]
                ax.plot(axis,db(_plot_response(source['xy'],source['weights'],uv,c,case,element,view_mode,phase_mode)),color=color,ls='--',alpha=.5,lw=.9)
        if view_mode=='array_factor' or element.model=='isotropic': ax.axhline(-3.0103,color='#adb6c1',ls=':',lw=.8)
        maximum=result['scenario_metrics'][case['index']].get('total_peak_db',0) if view_mode=='total' else 0
        if phase_mode=='quantized' and c.phase_bits and view_mode=='total':
            maximum=result['phase_quantization']['scenarios'][case['index']]['total_peak_db']
        ax.set(xlabel='Direction cosine u or v',ylabel='Relative field / dB',ylim=(-45,max(1,maximum+1)),
           title=('Quantized cuts (dotted: ideal)' if phase_mode=='quantized' and c.phase_bits else
                  'Cuts (dashed: original)' if 'weighting' in result else 'Cuts through steering direction'))
        ax.legend(fontsize=7); ax.grid(alpha=.6)
    elif kind=='coarray':
        ca=coarray(result['integer']); lags=ca['lags']*c.spacing
        ax.scatter(lags[:,0],lags[:,1],s=8,c=np.log1p(ca['multiplicity']),cmap='Blues')
        hx,hy=ca['central_half_widths']
        ax.add_patch(Rectangle((-hx*c.spacing,-hy*c.spacing),2*hx*c.spacing,2*hy*c.spacing,fill=False,color='#e2a626',lw=1.5))
        ax.set(xlabel='Δx / λ0',ylabel='Δy / λ0',aspect='equal',
               title=f"Difference coarray | lags={ca['unique_lags']}, holes={ca['bounding_holes']}")
    else: raise ValueError('Unknown detail plot: '+str(kind))
    return ax


def draw_detail(fig,bundle,result,case_index=None,view_mode='total',phase_mode='ideal'):
    """Four-panel report figure; uses the same drawing routines as independent views."""
    fig.clear(); fig.set_layout_engine(None)
    for index,(kind,_) in enumerate(DETAIL_PLOTS):
        _draw_detail_axes(fig,fig.add_subplot(2,2,index+1),kind,bundle,result,case_index,view_mode,phase_mode)
    fig.subplots_adjust(left=.075,right=.96,bottom=.13,top=.93,wspace=.36,hspace=.56)
    _metric_footer(fig,bundle,result,case_index,view_mode,phase_mode)


def draw_detail_plot(fig,bundle,result,kind,case_index=None,view_mode='total',compact=False,phase_mode='ideal'):
    """Redraw one vector plot; never crop a raster from the overview."""
    if kind not in dict(DETAIL_PLOTS): raise ValueError('Unknown detail plot: '+str(kind))
    fig.clear()
    fig.set_layout_engine('constrained',w_pad=.05 if compact else .24,h_pad=.05 if compact else .12)
    ax=fig.add_subplot(111)
    _draw_detail_axes(fig,ax,kind,bundle,result,case_index,view_mode,phase_mode)
    if compact:
        if kind=='layout': ax.set_title(f'Physical layout | N={len(result["xy"])}',fontsize=9)
        elif kind=='coarray':
            ca=coarray(result['integer']); ax.set_title(f"Lags={ca['unique_lags']} | holes={ca['bounding_holes']}",fontsize=9)
        else: ax.set_title(ax.get_title(),fontsize=9)
        ax.tick_params(labelsize=8); ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
        if kind=='pattern':
            fig.axes[1].set_ylabel('Element / dB' if view_mode=='element' else 'Field / dB',fontsize=8)
            fig.axes[1].tick_params(labelsize=7)
    if kind in ('pattern','cuts'):
        _metric_footer(fig,bundle,result,case_index,view_mode,phase_mode)
    return ax


def detail_figure(bundle,result,phase_mode='ideal'):
    fig=Figure(figsize=(12,8),dpi=130)
    draw_detail(fig,bundle,result,phase_mode=phase_mode)
    return fig


def _metric_footer(fig,bundle,result,case_index,view_mode,phase_mode):
    from .pattern_metrics import sparse_pattern_summary, sparse_metric_lines
    from ..pattern_metrics import draw_metric_footer
    index=result['metrics'].get('worst_case_index',0) if case_index is None else case_index
    summary=sparse_pattern_summary(bundle,result,index,view_mode,phase_mode)
    draw_metric_footer(fig,sparse_metric_lines(summary))


def comparison_figure(bundle):
    fig=Figure(figsize=(12,max(5.8,len(bundle['results'])*.42+2.2)),dpi=140)
    ax=fig.add_subplot(121)
    rows=bundle['results']; x=np.arange(len(rows))
    psl=[r['metrics']['validation_psl_db'] for r in rows]
    colors=[COLORS[i%len(COLORS)] for i in range(len(rows))]
    ax.barh(x,psl,color=colors,height=.65)
    ax.set_yticks(x,[plot_name(r) for r in rows]); ax.invert_yaxis()
    for j,(value,r) in enumerate(zip(psl,rows)):
        ax.text(value-.15,j,f"{value:.2f} dB | N={r['metrics']['elements']}",ha='right',va='center',fontsize=8)
    ax.set(xlim=(min(min(psl)-9,-1),max(0,max(psl)+1)),xlabel='Total-field peak outside fixed mask / dB',title='Sampled validation; lower is better')
    ax.grid(axis='x',alpha=.5)
    ax=fig.add_subplot(122)
    for r,color in zip(rows,colors):
        if r['history']:
            count=len(r['history'])-1
            ax.plot(np.linspace(0,1,len(r['history'])),db(np.asarray(r['history'])),
                    label=f"{r['name']} ({count} steps)",color=color,lw=1.5)
    ax.set(xlabel='Fraction of recorded history (not equal work)',ylabel='Best sampled objective / dB',title='Optimization history (different budgets)')
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    ax.grid(alpha=.5)
    fig.suptitle(f"{bundle['config']['nx']} x {bundle['config']['ny']} candidates | target K={bundle['config']['k']} | seed={bundle['config']['seed']}",fontsize=13)
    fig.subplots_adjust(left=.19,right=.97,bottom=.14,top=.85,wspace=.42)
    return fig


def draw_weight_comparison(fig,bundle,result,case_index=None):
    fig.clear()
    if 'weighting' not in result:
        ax=fig.add_subplot(111); ax.axis('off')
        ax.text(.5,.5,'Choose a weighting method and apply it to the selected layout.',ha='center',va='center')
        return
    c,cc,case=display_case(bundle,result,case_index); center=scan_uv(cc)
    element=build_element(c,bundle.get('element_pattern'))
    source=bundle['results'][result['weighting']['source_index']]
    for along in (0,1):
        ax=fig.add_subplot(1,2,along+1)
        extent=np.sqrt(1-center[1-along]**2)
        axis=np.linspace(-extent,extent,1601); uv=np.tile(center,(len(axis),1)); uv[:,along]=axis
        for r,color,label,ls in [(source,'#8b96a5','Original','--'),(result,'#00998b','Weighted','-')]:
            ax.plot(axis,db(_plot_response(r['xy'],r['weights'],uv,c,case,element)),color=color,label=label,ls=ls,lw=1.4)
        maximum=max(r['scenario_metrics'][case['index']].get('total_peak_db',0) for r in (source,result))
        ax.set(xlabel='u' if along==0 else 'v',ylabel='Total field / steering dB',ylim=(-45,max(1,maximum+1)),title='u cut, v=v0' if along==0 else 'v cut, u=u0')
        ax.grid(alpha=.6); ax.legend(fontsize=8)
    fig.suptitle(f"{plot_name(result)} | Same {len(result['indices'])} positions | {case['frequency_ghz']:g} GHz, {case['theta']:g}/{case['phi']:g} deg",fontsize=11)
    fig.subplots_adjust(left=.09,right=.97,bottom=.18,top=.80,wspace=.25)


def weight_comparison_figure(bundle,result):
    fig=Figure(figsize=(12,4.6),dpi=140)
    draw_weight_comparison(fig,bundle,result)
    return fig


def draw_case_results(fig,bundle,result):
    fig.clear(); ax=fig.add_subplot(111)
    rows=result['scenario_metrics']; x=np.arange(len(rows))
    ax.bar(x,[row['validation_psl_db'] for row in rows],color='#00998b')
    ax.set_xticks(x,[f"{row['frequency_ghz']:g}G\n{row['theta']:g}/{row['phi']:g}" for row in rows],fontsize=8)
    ax.set(ylabel='Validated peak / dB',xlabel='Frequency GHz; theta/phi degrees',title=f'{plot_name(result)} | one shared layout and amplitude vector')
    ax.grid(axis='y',alpha=.5); fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.84)


def scenario_figure(bundle):
    fig=Figure(figsize=(13,5.5),dpi=140); ax=fig.add_subplot(111)
    for i,result in enumerate(bundle['results']):
        rows=result['scenario_metrics']
        ax.plot(np.arange(len(rows)),[r['validation_psl_db'] for r in rows],marker='.',lw=1,label=plot_name(result),color=COLORS[i%len(COLORS)])
    cases=bundle['scenarios']; ax.set_xticks(np.arange(len(cases)),[f"{r['frequency_ghz']:g}G\n{r['theta']:g}/{r['phi']:g}" for r in cases],fontsize=8)
    ax.set(ylabel='Validated peak / dB',xlabel='Frequency GHz; theta/phi degrees',title='Joint synthesis: one physical layout and amplitude vector per method')
    ax.legend(fontsize=8,ncol=3); ax.grid(alpha=.4)
    fig.subplots_adjust(left=.08,right=.97,bottom=.19,top=.89)
    return fig


def draw_element(fig,c,element,frequency):
    fig.clear(); ax=fig.add_subplot(121)
    axis=np.linspace(-1,1,181); u,v=np.meshgrid(axis,axis)
    uv=np.column_stack((u.ravel(),v.ravel())); valid=np.sum(uv**2,axis=1)<=1
    values=np.full(len(uv),np.nan); values[valid]=db(element.response(uv[valid],frequency))
    im=ax.imshow(values.reshape(u.shape),origin='lower',extent=(-1,1,-1,1),vmin=-40,vmax=0,cmap='viridis')
    ax.set(xlabel='u',ylabel='v',title='Element envelope: +z hemisphere')
    fig.colorbar(im,ax=ax,label='Element relative field / dB',shrink=.8,pad=.02)
    ax=fig.add_subplot(122)
    cut=np.linspace(-1,1,1601)
    for uv,label,color in [(np.column_stack((cut,np.zeros_like(cut))),'xz cut (v=0)','#00998b'),
                           (np.column_stack((np.zeros_like(cut),cut)),'yz cut (u=0)','#2978b8')]:
        ax.plot(np.rad2deg(np.arcsin(cut)),db(element.response(uv,frequency)),label=label,color=color)
    ax.set(xlabel='Signed angle from +z / degree',ylabel='Element relative field / dB',ylim=(-45,1),title='Principal cuts; scalar magnitude')
    ax.legend(fontsize=8); ax.grid(alpha=.5)
    fig.suptitle(f'Element: {element.model} | {frequency:g} GHz | scalar envelope, no absolute gain',fontsize=11)
    fig.subplots_adjust(left=.07,right=.97,bottom=.17,top=.82,wspace=.3)


def element_figure(c,element,frequency):
    fig=Figure(figsize=(12,5),dpi=140); draw_element(fig,c,element,frequency); return fig


def draw_phase_comparison(fig,bundle,result,case_index=None):
    from .phase_quantization import excitation_phases
    fig.clear()
    c,cc,case=display_case(bundle,result,case_index)
    if not c.phase_bits:
        ax=fig.add_subplot(111); ax.axis('off')
        ax.text(.5,.5,'Enable phase_bits (1..12) and calculate to compare ideal and quantized phases.',ha='center',va='center')
        return
    element=build_element(c,bundle.get('element_pattern')); center=scan_uv(cc)
    for along in (0,1):
        ax=fig.add_subplot(1,3,along+1)
        extent=np.sqrt(max(0.,1-center[1-along]**2)); axis=np.linspace(-extent,extent,1601)
        uv=np.tile(center,(len(axis),1)); uv[:,along]=axis
        for mode,label,color,ls in [('ideal','Ideal','#8795a5','--'),('quantized',f'{c.phase_bits}-bit','#00998b','-')]:
            ax.plot(axis,db(_plot_response(result['xy'],result['weights'],uv,c,case,element,'total',mode)),color=color,ls=ls,label=label,lw=1.3)
        ax.axvline(center[along],color='#a6b3c0',ls=':',lw=.7)
        maximum=max(result['scenario_metrics'][case['index']]['total_peak_db'],result['phase_quantization']['scenarios'][case['index']]['total_peak_db'])
        ax.set(xlabel='u' if along==0 else 'v',ylabel='Field / ideal steering dB',ylim=(-45,max(1.,maximum+1)),title='u cut' if along==0 else 'v cut')
        ax.grid(alpha=.4); ax.legend(fontsize=8)
    ax=fig.add_subplot(133); phases=excitation_phases(result['xy'],case,c.phase_bits)
    ax.plot(np.arange(len(result['xy'])),phases['error_deg'],'.-',color='#2978b8',lw=.8)
    half=180/2**c.phase_bits; ax.axhline(half,color='#a6b3c0',ls='--'); ax.axhline(-half,color='#a6b3c0',ls='--')
    ax.set_ylim(-1.15*half,1.15*half)
    ax.set(xlabel='Element order in selected layout',ylabel='Circular phase error / degree',title=f'Nearest-code error | step={2*half:g} deg')
    ax.grid(alpha=.4)
    fig.suptitle(f"{plot_name(result)} | {case['frequency_ghz']:g} GHz, {case['theta']:g}/{case['phi']:g} deg | fixed support and amplitudes",fontsize=10)
    fig.subplots_adjust(left=.065,right=.98,bottom=.2,top=.80,wspace=.4)


def phase_comparison_figure(bundle,result,case_index=None):
    fig=Figure(figsize=(15,4.8),dpi=140); draw_phase_comparison(fig,bundle,result,case_index); return fig


def structure_figure():
    fig=Figure(figsize=(12,7),dpi=140)
    for column,(name,integer) in enumerate(structured_arrays().items()):
        ca=coarray(integer)
        ax=fig.add_subplot(2,3,column+1)
        ax.scatter(integer[:,0],integer[:,1],s=25,color='#00998b')
        ax.set(title=f'{name}\nPhysical elements: {len(integer)}',xlabel='x / base spacing',ylabel='y / base spacing',aspect='equal')
        ax.grid(alpha=.5)
        ax=fig.add_subplot(2,3,column+4)
        ax.scatter(ca['lags'][:,0],ca['lags'][:,1],s=8,color='#2978b8')
        ax.set(title=f"Lags: {ca['unique_lags']} | holes: {ca['bounding_holes']}\nCentral rectangle: {ca['central_rectangle_lags']}",
               xlabel='lag x / base spacing',ylabel='lag y / base spacing',aspect='equal')
    fig.suptitle('Structured geometry examples | different counts and apertures; no DOA experiment',fontsize=12)
    fig.subplots_adjust(left=.07,right=.97,top=.85,bottom=.09,wspace=.4,hspace=.65)
    return fig
