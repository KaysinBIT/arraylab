"""Qt-free views of saved WaveLab-derived waveform and ambiguity results."""
from pathlib import Path

import numpy as np

from .pattern_metrics import draw_metric_footer
from .plot_layout import responsive_subplots
from .surface_plot import clear_figure


VIEWS = ('overview', 'phase', 'spectra', 'rd', 'range_recovery', 'pri_recovery')
COLORS = {'A': '#168b9c', 'B': '#c78728'}


def value(value, digits=5):
    return '—' if value is None else f'{float(value):.{digits}g}'


def _profile(report, profile):
    profiles = report['analysis']['profiles']
    return profile if profiles.get(profile) else 'A'


def waveform_metric_lines(report, profile='A', chinese=False):
    profile = _profile(report, profile)
    item = report['analysis']['profiles'][profile]
    m, p = item['result']['metrics'], item['params']
    combined = report['analysis']['metrics']
    if chinese:
        return [f"{profile}组 · {p['mode'].upper()} · 距离分辨率 {value(m['rangeResolutionM'])} m · 距离栅格 {value(m['rangeBinM'])} m",
                f"保守速度 ±{value(m['maxVelocityMps'])} m/s（±{value(m['maxVelocityMps']*3.6)} km/h） · 速度分辨率 {value(m['velocityResolutionMps'])} m/s",
                f"有效采样带宽 {value(m['observedBandwidthMHz'])} MHz · 同组同Tx重访 {value(m['txIntervalUs'])} μs · 总采集 {value(combined['burstMs'])} ms",
                '理想单目标频谱；补零不提高物理分辨率，IF距离上限不等于探测距离。']
    return [f"Profile {profile} | {p['mode'].upper()} | range resolution {value(m['rangeResolutionM'])} m | FFT bin {value(m['rangeBinM'])} m",
            f"Conservative speed +/-{value(m['maxVelocityMps'])} m/s (+/-{value(m['maxVelocityMps']*3.6)} km/h) | resolution {value(m['velocityResolutionMps'])} m/s",
            f"Observed bandwidth {value(m['observedBandwidthMHz'])} MHz | same-profile/Tx interval {value(m['txIntervalUs'])} us | total acquisition {value(combined['burstMs'])} ms",
            'Ideal single-target spectra; zero padding does not improve physical resolution. IF-limited range is not detection range.']


def _empty(axis, text):
    axis.set_axis_off()
    axis.text(.5, .5, text, transform=axis.transAxes, ha='center', va='center', wrap=True)


def _spectrum(axis, rows, color, label=None):
    if rows:
        axis.plot([row['x'] for row in rows], [max(-100., row['y']) for row in rows], color=color, lw=1.3, label=label)
    axis.set_ylim(-100, 3); axis.grid(alpha=.25)


def draw_waveform_figure(figure, report, view='overview', profile='A', chinese=False):
    """Render saved arrays and sequence metadata only; never run an analysis."""
    if view not in VIEWS:
        raise ValueError('Unknown waveform plot: ' + str(view))
    clear_figure(figure)
    if hasattr(figure, '_arraylab_responsive_plots'):
        del figure._arraylab_responsive_plots
    figure.set_layout_engine('constrained', w_pad=.08, h_pad=.08, wspace=.13, hspace=.16)
    profile = _profile(report, profile)
    analysis = report['analysis']; item = analysis['profiles'][profile]
    p, m = item['params'], item['result']['metrics']
    lines = waveform_metric_lines(report, profile, chinese)
    if view == 'overview':
        chirp, timeline = figure.subplots(2, 1)
        for name in ('A', 'B'):
            current = analysis['profiles'].get(name)
            if current is None: continue
            q, metrics = current['params'], current['result']['metrics']
            chirp.plot([0, q['rampUs']], [q['startGHz'], metrics['endGHz']], color=COLORS[name], lw=1.8, label=name)
            x = np.array([q['adcStartUs'], q['adcStartUs'] + metrics['adcUs']])
            chirp.plot(x, q['startGHz'] + q['slopeMHzUs']*x/1000, color=COLORS[name], lw=5, alpha=.5)
        chirp.set(xlabel='Chirp时间 / μs' if chinese else 'Time within chirp / us', ylabel='GHz',
                  title='单个Chirp：粗线为ADC采样窗口' if chinese else 'Individual chirps\nThick segment: ADC window')
        chirp.grid(alpha=.25); chirp.legend(fontsize=8)
        alternating = analysis['options']['enabled'] and analysis['options']['layout'] == 'alternating'
        if alternating:
            events = analysis['previewEvents']
            for event in events:
                name = event['profile']; row = 1 if name == 'A' else 0
                timeline.broken_barh([(event['startUs'], event['endUs']-event['startUs'])], (row-.28,.56), facecolors='#e5edf3')
                timeline.broken_barh([(event['startUs'], event['rampUs'])], (row-.28,.56), facecolors=COLORS[name])
                label = 'Tx*' if event['txIndex'] is None else f"T{event['txIndex']+1}"
                timeline.text(event['startUs']+event['rampUs']/2, row, label, ha='center', va='center', fontsize=7, color='white')
            timeline.set(yticks=[0,1], yticklabels=['B','A'], xlabel='帧内时间 / μs' if chinese else 'Time in frame / us',
                         title='整轮交替：仅示意开头若干Chirp' if chinese else 'Whole-round alternation\nChronological prefix only')
        else:
            blocks = analysis['timelineBlocks']
            for block in blocks:
                name = block['profile']; row = 1 if name == 'A' else 0
                start, length = block['startUs']/1000, (block['endUs']-block['startUs'])/1000
                timeline.broken_barh([(start,length)], (row-.28,.56), facecolors=COLORS[name])
                timeline.text(start+length/2,row,f"{name} · {block['chirpCount']} chirps",ha='center',va='center',fontsize=8,color='white')
            timeline.set(xlim=(0,analysis['metrics']['frameMs']), yticks=[0,1], yticklabels=['B','A'],
                         xlabel='帧内时间 / ms' if chinese else 'Time in frame / ms',
                         title='整帧采集时序（色块包含Chirp空闲）' if chinese else 'Frame acquisition blocks (including chirp idle)')
        timeline.set_ylim(-.6,1.6); timeline.grid(axis='x',alpha=.25)
        draw_metric_footer(figure, lines)
        responsive_subplots(figure,[chirp,timeline],1,row_height=190)
    elif view == 'phase':
        axis=figure.add_subplot(111)
        if p['mode'] != 'ddm':
            _empty(axis, '当前为TDM：各Tx轮流发射，不使用DDM相位编码。' if chinese else 'TDM transmits one Tx at a time; no DDM phase coding.')
        else:
            from matplotlib.colors import LinearSegmentedColormap
            count=min(12,p['loops']); data=np.zeros((p['ddmSlots'],count))
            for tx in range(p['txCount']): data[tx]=np.arange(count)*360*tx/p['ddmSlots']%360
            cmap=LinearSegmentedColormap.from_list('readable_phase',['#f5fbff','#c4e1ef'])
            axis.imshow(data,cmap=cmap,vmin=0,vmax=360,aspect='auto',interpolation='nearest')
            for slot in range(p['ddmSlots']):
                for n in range(count):
                    text=f'{data[slot,n]:g}°' if slot<p['txCount'] else ('空' if chinese else '—')
                    axis.text(n,slot,text,color='#16334a',fontsize=8,fontweight='bold',ha='center',va='center')
            axis.set(xticks=np.arange(count),xticklabels=np.arange(count),yticks=np.arange(p['ddmSlots']),
                     yticklabels=[f'Tx{k+1}' if k<p['txCount'] else (f'空槽{k+1}' if chinese else f'Empty{k+1}') for k in range(p['ddmSlots'])],
                     xlabel=f'{profile}组内Chirp序号' if chinese else f'Chirp index within profile {profile}',
                     title=f'{profile}组DDM相位 / °（A、B分别计数）' if chinese else f'Profile {profile}: DDM phase / deg\nIndependent A/B counters')
            axis.tick_params(labelsize=8)
        draw_metric_footer(figure, lines[:3]+(['空槽仅为未分配编码槽；图示不代表已完成盲解码。'] if chinese else ['An empty slot is an unassigned code, not a completed blind decoder.']))
    elif view == 'spectra':
        axes=figure.subplots(1,2); spectrum=report['spectra'][profile]
        _spectrum(axes[0],spectrum['rangeProfile'],COLORS[profile])
        _spectrum(axes[1],spectrum['dopplerProfile'],COLORS[profile], '编码频谱' if chinese else 'Encoded spectrum')
        if p['mode']=='ddm':
            _spectrum(axes[1],spectrum['baseDopplerProfile'],'#8495a4','已知未编码参考' if chinese else 'Known uncoded reference')
            axes[1].legend(fontsize=7)
        axes[0].set(xlabel='表观距离 / m' if chinese else 'Apparent range / m',ylabel='dB',title=f'{profile} · 距离谱' if chinese else f'{profile}: range spectrum')
        axes[1].set(xlabel='速度 / m/s' if chinese else 'Velocity / m/s',ylabel='dB',title=f'{profile} · 多普勒谱' if chinese else f'{profile}: Doppler spectrum')
        top=axes[1].secondary_xaxis('top',functions=(lambda v:v*3.6,lambda v:v/3.6));top.set_xlabel('km/h');top.tick_params(labelsize=8)
        for axis in axes:axis.tick_params(labelsize=8)
        detected=spectrum.get('detected') or {}
        lines.append((f"距离栅格读数：{value(detected.get('rangeM'))} m；超出规划IF时不报告目标距离。" if chinese else
                      f"Range-bin reading: {value(detected.get('rangeM'))} m; out-of-IF targets are not reported as valid range."))
        draw_metric_footer(figure,lines);responsive_subplots(figure,axes,2,column_width=330,row_height=210)
    elif view == 'rd':
        axis=figure.add_subplot(111); data=report['spectra'][profile]['rangeDoppler']
        x=np.asarray(data['ranges']);y=np.asarray(data['velocities']);z=np.asarray(data['values']).reshape(data['height'],data['width'])
        mesh=axis.pcolormesh(x,y,z,shading='auto',cmap='viridis',vmin=-80,vmax=0)
        bar=figure.colorbar(mesh,ax=axis,pad=.15);bar.set_label('dB')
        axis.set(xlabel='表观距离 / m' if chinese else 'Apparent range / m',ylabel='速度 / m/s' if chinese else 'Velocity / m/s',
                 title=f'{profile} · 理想单目标距离–速度图' if chinese else f'{profile}: ideal single-target range-Doppler map')
        secondary=axis.secondary_yaxis('right',functions=(lambda v:v*3.6,lambda v:v/3.6));secondary.set_ylabel('km/h')
        draw_metric_footer(figure,lines[:3]+(['热图做区间最大值汇聚；DDM编码峰未作多目标盲解码。'] if chinese else ['Display uses max pooling; DDM code peaks are not blind-decoded targets.']))
    elif view=='range_recovery':
        result=report['range_recovery'][profile];axes=figure.subplots(2,1)
        observations=result.get('measurements',[])
        if observations:
            for index,measurement in enumerate(observations):
                _spectrum(axes[0],measurement.get('observedSpectrum',[]),('#168b9c','#c78728')[index%2],f'{index+1}')
            axes[0].legend(fontsize=8)
            axes[0].set(xlabel='折叠拍频 / MHz' if chinese else 'Folded beat / MHz',ylabel='dB',title='实数ADC观测谱' if chinese else 'Real-ADC observed spectra')
        else:_empty(axes[0],result.get('reason','') if chinese else 'ADC recovery: '+result.get('status','unknown'))
        _draw_candidates(axes[1],result,'minRangeM','maxRangeM',chinese)
        draw_metric_footer(figure,_recovery_lines(result,chinese,adc=True));responsive_subplots(figure,axes,1,row_height=190)
    else:
        result=report['pri_recovery'];axis=figure.add_subplot(111)
        _draw_candidates(axis,result,'minM','maxM',chinese)
        draw_metric_footer(figure,_recovery_lines(result,chinese,adc=False))
    return figure


def _draw_candidates(axis,result,minimum,maximum,chinese):
    candidates=result.get('candidates',[])
    if not candidates:
        _empty(axis,result.get('reason','无候选') if chinese else 'Recovery status: '+result.get('status','unknown'))
        return
    shown=candidates[:80]
    for index,candidate in enumerate(shown,1):
        lo,hi=candidate[minimum],candidate[maximum]
        center=candidate['rangeM']
        axis.errorbar(center,index,xerr=[[max(0,center-lo)],[max(0,hi-center)]],fmt='o',color='#168b9c',capsize=3,markersize=4)
    axis.set(xlabel='距离 / m' if chinese else 'Range / m',ylabel='候选编号' if chinese else 'Candidate',
             title=(f"一致候选区间：{len(candidates)}个" if chinese else f"Consistent candidate intervals: {len(candidates)}"))
    axis.grid(axis='x',alpha=.25)
    if len(candidates)>80:
        axis.text(.02,.98,'图示前80个，完整列表见表格/导出' if chinese else 'First 80 shown; export retains all',transform=axis.transAxes,va='top',fontsize=8)


def _recovery_lines(result,chinese,adc):
    status=result.get('status','unknown');resolved=value(result.get('resolvedRangeM',result.get('rangeM')))
    if chinese:
        return [f"状态：{status} · 唯一恢复距离：{resolved} m",result.get('reason',''),
                'ADC折叠实验：要求模拟IF保留该回波，速度先验由用户提供。' if adc else '跨周期实验：独立时延观测级模型，不表示实际探测距离。',
                '唯一性仅针对给定搜索范围和容差；多候选不自动选为真值，区间不是统计置信区间。']
    # The saved explanation is preserved verbatim in JSON; exported English figures
    # use short labels so Chinese fonts are not a headless dependency.
    return [f'Status: {status} | unique recovered range: {resolved} m',
            'ADC folding: analog IF must retain the echo; velocity is an external prior.' if adc else 'Independent delay-observation model; not an operational detection-range claim.',
            'Uniqueness is conditional on search limits and tolerances. Multiple candidates remain unresolved.']


def render_waveform_plots(directory,report,mode='all'):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if mode=='none':return
    kinds=VIEWS if mode=='all' else (mode,)
    if any(kind not in VIEWS for kind in kinds):raise ValueError('Unknown waveform plot mode')
    directory=Path(directory)
    for profile in ('A','B'):
        if not report['analysis']['profiles'].get(profile):continue
        for kind in kinds:
            with matplotlib.rc_context({'font.family':'DejaVu Sans','font.size':9,'text.usetex':False}):
                figure=Figure(figsize=(9,6),dpi=140);FigureCanvasAgg(figure)
                draw_waveform_figure(figure,report,kind,profile)
                figure.savefig(directory/f'waveform-{profile}-{kind}.png');figure.clear()
