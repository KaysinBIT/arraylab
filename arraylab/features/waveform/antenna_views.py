"""Render saved radar design arrays and diagnostics without recalculation."""
from arraylab.shared.plotting.metrics import draw_metric_footer
from arraylab.shared.plotting.layout import responsive_subplots


def number(value, digits=4):
    return '—' if value is None else f'{float(value):.{digits}g}'


def _empty(axis, text):
    axis.set_axis_off(); axis.text(.5, .5, text, ha='center', va='center', wrap=True, transform=axis.transAxes)


def _line(axis, points, label=None):
    if points: axis.plot([p['x'] for p in points], [p['y'] for p in points], lw=1.4, label=label)
    axis.grid(alpha=.25); axis.set_ylim(-45, 2)


def draw_radar_figure(figure, report, view, profile, chinese=False):
    def text(zh, en): return zh if chinese else en
    block = report.get(view)
    if view == 'antenna': block = (block or {}).get(profile)
    if not block or not block.get('valid'):
        axis = figure.add_subplot(111)
        explanation = '\n'.join(block.get('errors', [])) if block else ''
        _empty(axis, explanation if chinese and explanation else text('尚无结果；在“天线设计”中运行相应操作。', 'No result. Run the corresponding radar operation.'))
        return figure
    if view == 'antenna':
        layout, pattern = figure.subplots(1, 2)
        for key, marker, color in [('tx', '^', '#c78728'), ('rx', 's', '#168b9c')]:
            positions = block[key]
            layout.scatter([p[0] for p in positions], [p[1] for p in positions], marker=marker, c=color, label=key.upper())
        virtual = block['virtual']
        layout.scatter([p['x'] for p in virtual], [p['z'] for p in virtual], facecolors='none', edgecolors='#6c58cd',
                       s=[20+10*p['multiplicity'] for p in virtual], label=text('虚拟位置', 'Virtual'))
        layout.set(xlabel='x / λ', ylabel='z / λ', title=text('Tx / Rx与虚拟布局', 'Tx / Rx and virtual layout'))
        layout.grid(alpha=.25); layout.legend(fontsize=8)
        _line(pattern, block['azimuth'], text('方位', 'Azimuth')); _line(pattern, block['elevation'], text('俯仰', 'Elevation'))
        pattern.set(xlabel=text('角度 / °', 'Angle / deg'), ylabel='dB', title=text('数字阵列因子切面', 'Digital array factor cuts')); pattern.legend(fontsize=8)
        lines = [text(f"{block['channelCount']}个Tx–Rx通道 · {block['uniqueCount']}个独立虚拟位置",
                      f"{block['channelCount']} Tx-Rx channels | {block['uniqueCount']} unique positions"),
                 text(f"方位HPBW {number(block['beamwidthAzDeg'])}° · 俯仰HPBW {number(block['beamwidthElDeg'])}°",
                      f"Azimuth HPBW {number(block['beamwidthAzDeg'])} deg | Elevation HPBW {number(block['beamwidthElDeg'])} deg"),
                 text('单频理想阵列因子；不等于算法分辨率或实际增益。', 'Ideal single-frequency array factor; not algorithm resolution or realized gain.')]
    elif view == 'sparse_design':
        pattern, response = figure.subplots(1, 2)
        _line(pattern, block['pattern']); pattern.set(xlabel=text('角度 / °', 'Angle / deg'), ylabel='dB', title=text('物理阵列因子', 'Physical array factor'))
        sr = block.get('superresolution'); method = block['options']['criterion']
        if sr and not sr.get('valid'):
            _empty(response, text('所选算法不适用或超出预算\n详情见检查表；未回退其他算法', 'Algorithm unsupported or over budget. No fallback.'))
        elif sr and method in ('dml', 'esprit'):
            representative = sr.get('representative') or {}
            truth = representative.get('truth', representative.get('truthAngles', []))
            estimates = representative.get('estimates', [])
            for i, angle in enumerate(truth): response.axvline(angle, color='#c78728', ls='--', label=text('真角', 'Truth') if i == 0 else None)
            for i, angle in enumerate(estimates): response.plot(angle, .5, marker='x', ms=9, color='#168b9c', label=text('估计', 'Estimate') if i == 0 else None)
            all_angles = truth + estimates
            if all_angles: response.set_xlim(min(all_angles)-3, max(all_angles)+3)
            response.set_ylim(0, 1); response.set_yticks([]); response.grid(alpha=.25)
            response.set(xlabel=text('角度 / °', 'Angle / deg'), title=method.upper()+text('：首次验证角度', ': first validation estimates'))
            if all_angles: response.legend(fontsize=8)
        else:
            _line(response, block.get('twoTargetPattern', []))
            title = method.upper()+text('归一化谱（首次验证）', ' spectrum (first validation)') if sr else text('常规双目标响应', 'Conventional two-target response')
            response.set(xlabel=text('角度 / °', 'Angle / deg'), ylabel='dB', title=title)
        m = block['metrics']
        lines = [text(('所采样工况达标' if block['status'] == 'met' else '候选未满足全部约束')+f" · {m['uniqueCount']}/{m['channelCount']} 独立位置/通道",
                      f"Status: {block['status']} | {m['uniqueCount']}/{m['channelCount']} unique / channels"),
                 text(f"最差HPBW {number(m['worstHpbwDeg'])}° · 最差旁瓣 {number(m['worstSidelobeDb'])} dB",
                      f"Worst HPBW {number(m['worstHpbwDeg'])} deg | worst sidelobe {number(m['worstSidelobeDb'])} dB")]
        if sr:
            lines.append(text(f"最差经验通过率 {number(None if sr.get('successRate') is None else 100*sr['successRate'])}% · RMSE {number(sr.get('rmseDeg'))}° · 处理维数 {sr.get('usedChannelCount', '—')}",
                              f"Worst empirical success {number(sr.get('successRate'))} | RMSE {number(sr.get('rmseDeg'))} deg | dimension {sr.get('usedChannelCount', 'n/a')}"))
        lines.append(text('有限离散验证；不保证连续扫描域、全局最优或硬件性能。', 'Finite sampled validation; no continuous-domain, global-optimum or hardware guarantee.'))
        if block.get('retainedFromPreviousCalculation'):
            lines.append(text(f"保留的独立综合快照 · 原参考频率 {number(block.get('referenceFrequencyGHz'),7)} GHz · 未重新验证",
                              f"Retained design snapshot | reference {number(block.get('referenceFrequencyGHz'),7)} GHz | not rerun"))
    else:
        gains, energy = figure.subplots(1, 2)
        values = [block['requiredTxGainDbi'], block['requiredRxGainDbi']]
        bars = gains.bar(['Tx', 'Rx'], values, color=['#c78728', '#168b9c']); gains.bar_label(bars, fmt='%.3f')
        gains.set(ylabel='dBi', title=text('每物理通道所需增益', 'Required per-channel gain')); gains.grid(axis='y', alpha=.25)
        profiles = block['profiles']; xs = list(range(len(profiles)))
        energy.bar([x-.18 for x in xs], [p['energyTimeMs'] for p in profiles], width=.36, label=text('ADC积累', 'ADC integration'))
        energy.bar([x+.18 for x in xs], [p['effectiveTimeMs'] for p in profiles], width=.36, label=text('计窗效率后', 'After window efficiency'))
        energy.set_xticks(xs, [p['name'] for p in profiles]); energy.set(ylabel='ms', title=text('实际能量积累时间', 'Actual integration time')); energy.legend(fontsize=8)
        lines = [text(('目标超出波形通带；数学增益不能恢复观测' if block['status'] == 'waveform_limited' else '波形通带内的理想能量预算')+f" · 限制组 {block['limitingProfile']}",
                      f"Status: {block['status']} | limiting profile {block['limitingProfile']}"),
                 text(f"Tx {number(values[0])} dBi · Rx {number(values[1])} dBi · 所需SNR含裕量 {number(block['requiredSnrWithMarginDb'])} dB",
                      f"Tx {number(values[0])} dBi | Rx {number(values[1])} dBi | required SNR + margin {number(block['requiredSnrWithMarginDb'])} dB"),
                 text('Rx合并单独计入；不额外添加Tx数量收益；等效口径不是物理设计。', 'Rx combining is explicit; no hidden Tx gain. Equivalent aperture is a budget estimate.')]
    draw_metric_footer(figure, lines)
    responsive_subplots(figure, list(figure.axes), 2, row_height=250)
    return figure


def radar_table(report, view, profile):
    block = report.get(view)
    if view == 'antenna': block = (block or {}).get(profile)
    if not block:
        return ['说明'], [], [], '尚无本项结果。F5计算波形、天线方向图与增益；综合和算法验证使用独立按钮。'
    checks = [[c.get('label', ''), {'pass': '通过', 'warn': '提示', 'fail': '未通过'}.get(c.get('status'), c.get('status', '')), c.get('detail', '')] for c in block.get('checks', [])]
    checks += [['输入检查', '未通过', e] for e in block.get('errors', [])]
    if not block.get('valid'): return ['说明'], [], checks, '本项未形成有效结果，详见检查；其他已保存结果保留。'
    if view == 'antenna':
        rows = [[kind.upper(), i, number(p[0]), number(p[1]), 1] for kind in ('tx', 'rx') for i, p in enumerate(block[kind], 1)]
        rows += [['虚拟', i, number(p['x']), number(p['z']), p['multiplicity']] for i, p in enumerate(block['virtual'], 1)]
        checks += [['模型边界', '提示', e] for e in block['warnings']]
        return ['类别', '序号', 'x / λ', 'z / λ', '重数'], rows, checks, '坐标以当前结果组的波长表示。虚拟位置=Tx+Rx；重合位置保留通道重数。'
    if view == 'gain_budget':
        rows = [[p['name'], number(p.get('requiredTxGainDbi')), number(p.get('requiredRxGainDbi')),
                 number(p.get('energyTimeMs')), number(p.get('windowLossDb')), number(p.get('rxCombiningGainDb')),
                 number(p.get('deliveredSnrDb')), number(p.get('txEquivalentSquareSideMm')), number(p.get('rxEquivalentSquareSideMm'))] for p in block['profiles']]
        checks += [['等效口径容纳', '提示', e] for e in block.get('layoutWarnings', [])]
        return ['组', 'Tx/dBi', 'Rx/dBi', 'ADC积累/ms', '窗损耗/dB', 'Rx合并/dB', 'SNR/dB', 'Tx等效边/mm', 'Rx等效边/mm'], rows, checks, '增益为目标方向每物理通道realized gain。等效面积/边长仅作预算；SNR由用户指定，不自动推导Pd/Pfa。'
    for key in ('tx', 'rx'):
        coordinates='; '.join(f"({number(p[0])}, {number(p[1])})" for p in block['geometry'][key])
        checks.append([key.upper()+'候选坐标', 'x,z / λ', coordinates])
    m=block['metrics']
    checks.append(['孔径与虚拟格点', '指标', f"物理孔径 {number(m['physicalApertureLambda'])}λ；虚拟孔径 {number(m['virtualApertureLambda'])}λ；格点占用 {number(100*m['sparseFillRatio'])}%"])
    if block.get('retainedFromPreviousCalculation'):
        checks.insert(0,['保留的独立综合快照','未重算',f"原参考频率 {number(block.get('referenceFrequencyGHz'),7)} GHz。当前波形F5未重新执行本次综合或验证。"])
    sr = block.get('superresolution')
    if not sr:
        rows = [[p['centerDeg'], number(p.get('hpbwDeg')), number(p.get('pairDipDb')), '通过' if p['resolved'] else '未通过'] for p in block['scanChecks']]
        return ['扫描中心/°', 'HPBW/°', '双峰谷深/dB', '判据'], rows, checks, '常规双目标判据与HPBW代理不同；结果仅针对列出的离散扫描中心。'
    rows = []
    for case in sr.get('validation', []):
        rows.append([number(case.get('centerDeg')), case.get('seed'), f"{case.get('successes', 0)}/{case.get('trials', 0)}",
                     number(case.get('successRate')), number(case.get('rmseDeg')), number(case.get('worstErrorDeg')),
                     case.get('usedChannelCount', '—'), case.get('status', '')])
    checks += [['算法适用性', '提示', e] for e in sr.get('warnings', [])]
    representative = sr.get('representative') or {}; subset = representative.get('subset') or {}
    truth=representative.get('truth',representative.get('truthAngles',[]))
    checks.append(['首次独立验证', '真角 / 估计', f"真角 {', '.join(number(x) for x in truth)}°；估计 {', '.join(number(x) for x in representative.get('estimates',[]))}°；误差 {', '.join(number(x) for x in representative.get('errors',[]))}°"])
    note = f"{sr['method'].upper()} · 已知K=2；表中是各扫描中心的独立验证，不挑选最佳试验。实际处理维数 {sr.get('usedChannelCount', '—')}，有效角误差限 {number(sr.get('effectiveToleranceDeg'))}°。"
    if subset: note += f" 子阵位置数 {subset.get('count', '—')}，处理孔径 {number(subset.get('processedAperture', subset.get('aperture')))}λ。"
    diagnostics = representative.get('diagnostics') or {}
    if sr['method'] == 'dml': note += f" DML首次验证投影残差 {number(diagnostics.get('residualPower'))}，Gram条件数 {number(diagnostics.get('conditionNumber'))}。"
    return ['扫描中心/°', '种子', '成功/次数', '通过率', 'RMSE/°', '最大误差/°', '处理维数', '状态'], rows, checks, note
