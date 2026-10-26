"""Render saved phased-array data; browsing views never invokes a solver."""
import numpy as np
from arraylab.shared.plotting.metrics import draw_metric_footer
from arraylab.shared.plotting.layout import responsive_subplots
from arraylab.shared.plotting.surface import clear_figure

def _number(value):return '未解析' if value is None else f'{float(value):.5g}'

def selections(result):
    return [(i,f"{case['frequencyGHz']:g} GHz · X {case['scanX']:g}° / Y {case['scanY']:g}°") for i,case in enumerate((result.get('synthesis') or {}).get('after',{}).get('cases',[]))]

def draw(figure,result,arrays,view,selection=0):
    clear_figure(figure);figure.set_layout_engine('constrained',w_pad=.08,h_pad=.08,wspace=.13,hspace=.16)
    analysis=result['analysis'];prefix=''
    if result.get('caseAnalyses'):
        selection=max(0,min(len(result['caseAnalyses'])-1,int(selection)));analysis=result['caseAnalyses'][selection];prefix=f'case_{selection}_'
    get=lambda key:np.asarray(arrays.get(prefix+key,arrays.get(key)))
    c=analysis['config'];m=analysis['metrics'];axes=[]
    if view=='layout':
        axes=list(figure.subplots(1,2));points=get('elements')
        for ax,column,title,cmap in [(axes[0],2,'相对电压幅度','viridis'),(axes[1],3,'实际馈相 / °','twilight')]:
            sc=ax.scatter(points[:,0],points[:,1],c=points[:,column],cmap=cmap,s=32)
            figure.colorbar(sc,ax=ax);ax.set(xlabel='x / λ0',ylabel='y / λ0',title=title);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    elif view=='cuts':
        axes=list(figure.subplots(1,2))
        for ax,key,label in zip(axes,('cut_x','cut_y'),('X','Y')):
            data=get(key);ax.plot(data[:,0],data[:,1],color='#168b9c');ax.set(xlabel=label+'方向角 / °',ylabel='相对功率 / dB',title=label+'切面',ylim=(-60,3));ax.grid(alpha=.25)
    elif view=='uv':
        ax=figure.add_subplot(111);axes=[ax];grid=get('uv_db');u=np.asarray(arrays['uv_u']);v=np.asarray(arrays['uv_v'])
        mesh=ax.pcolormesh(u,v,np.ma.masked_invalid(grid),vmin=-40,vmax=0,cmap='viridis',shading='auto');figure.colorbar(mesh,ax=ax,label='相对功率 / dB')
        ax.plot(np.sin(c['steerX']*np.pi/180),np.sin(c['steerY']*np.pi/180),'wx');ax.set(xlabel='u=sin(X)',ylabel='v=sin(Y)',title='二维可见圆盘');ax.set_aspect('equal')
    elif view=='scan' and result.get('scan'):
        axes=list(figure.subplots(2,2).flat);rows=result['scan']
        for ax,key,title in zip(axes,('hpbw','sll','pointingPowerDb','pointingOffset'),('HPBW / °','峰值旁瓣 / dB','命令方向功率 / dB','局部峰指向偏移 / °')):
            ax.plot([r['angle'] for r in rows],[np.nan if r[key] is None else r[key] for r in rows],marker='o',ms=3)
            ax.set(xlabel='X命令扫描角 / °',ylabel=title,title=title);ax.grid(alpha=.25)
    elif view=='optimization' and result.get('synthesis'):
        ax=figure.add_subplot(111);axes=[ax];s=result['synthesis'];history=s['history']
        ax.plot([r['iteration'] for r in history],[r.get('bestSll',r.get('worstSll',np.nan)) for r in history]);ax.set(xlabel='迭代',ylabel='粗网格旁瓣 / dB',title='搜索过程（最终结论见独立细验证）');ax.grid(alpha=.25)
    else:
        ax=figure.add_subplot(111);ax.set_axis_off();ax.text(.5,.5,'请先运行对应任务；逐工况与阵元明细见表格。',ha='center',va='center',transform=ax.transAxes);return figure
    lines=[f"{m['count']}阵元 · 参考 {c['referenceGHz']:g} GHz / 分析 {c['frequencyGHz']:g} GHz · {'固定移相' if c['phaseMode']=='phase' else '理想真时延'}",
           f"切面HPBW X {_number(m['hpbwX'])}° / Y {_number(m['hpbwY'])}° · 切面SLL X {_number(m['sllX'])} dB / Y {_number(m['sllY'])} dB",
           f"加权效率 {_number(100*m['taperEfficiency'])}% · 相位量化RMS {_number(m['phaseRmsDeg'])}° · 理想相干合成 {_number(m['coherentGainDb'])} dB（不是辐射增益）"]
    if view=='scan':lines.append('空值保留为未解析，不当作0；理想相移的SLL可能近乎不变。')
    if result.get('synthesis'):lines.append('最终离散约束达标。' if result['synthesis']['feasible'] else '最终离散约束未全部达标，仅为候选。')
    draw_metric_footer(figure,lines);responsive_subplots(figure,axes,2 if len(axes)>1 else 1,column_width=330,row_height=220)
    return figure

def table(result,arrays,view,selection=0):
    if view=='scan':
        rows=result.get('scan') or []
        return ['X角/°','HPBW/°','SLL/dB','指向功率/dB','峰偏移/°','状态','原因'],[[r['angle'],r['hpbw'],r['sll'],r['pointingPowerDb'],r['pointingOffset'],r['status'],r['reason']] for r in rows],'固定命令v切面，状态和原因不能用零值替代。'
    if view=='cases':
        s=result.get('synthesis') or {};after=s.get('after',{})
        if after.get('cases'):
            return ['序号','GHz','X/°','Y/°','区域SLL/dB','HPBW-X/°','HPBW-Y/°','通过','原因'],[[r['index']+1,r['frequencyGHz'],r['scanX'],r['scanY'],r['sll2D'],r['hpbwX'],r['hpbwY'],r['feasible'],'；'.join(r['issues'])] for r in after['cases']],'区域旁瓣：全可见圆盘排除椭圆之外；HPBW：穿实际峰值的两条切面。'
        return ['扫描角/°','SLL/dB','HPBW/°','局部峰/°','状态','原因'],[[r['angle'],r['sll'],r['hpbw'],r['peakAngle'],r.get('status','旧记录未提供'),r.get('reason','旧记录未提供')] for r in s.get('scanResults',[])],'一维最终细验证；不可见期望主阶保留空值与原因，不由其他瓣替代。'
    if view=='optimization':
        s=result.get('synthesis') or {};before=s.get('before',{});after=s.get('after',{})
        keys=[k for k in ('worstSll','hpbw','maxHpbwX','maxHpbwY','minSpacing','width','height','taperEfficiency','amplitudeDynamicRangeDb','passCount','caseCount') if k in after]
        return ['指标','初值','最终'],[[k,before.get(k),after.get(k)] for k in keys],'；'.join(s.get('notes',[]))
    analysis=result['analysis'];case_label=''
    if result.get('caseAnalyses'):
        selection=max(0,min(len(result['caseAnalyses'])-1,int(selection)))
        analysis=result['caseAnalyses'][selection];case_label=f'工况 {selection+1} · '
    config=analysis['config'];elements=analysis['elements']
    note=(f"{case_label}参考 {config['referenceGHz']:g} GHz / 分析 {config['frequencyGHz']:g} GHz · "
          f"命令 X {config['steerX']:g}° / Y {config['steerY']:g}°。"
          '参考波长坐标；幅度是相对电压，馈相读取该工况已保存的实际激励。')
    return ['阵元','x/λ0','y/λ0','电压幅度','馈相/°'],[[i+1,p['x'],p['y'],p['amplitude'],p['phaseDeg']] for i,p in enumerate(elements)],note
