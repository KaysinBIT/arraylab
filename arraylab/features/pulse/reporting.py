import html
import json
from .provenance import SOURCE_VERSION as __version__

from arraylab.shared.simulation_io import clean_json


def render_html_report(summary):
    """Readable standalone report; machine-readable provenance stays expandable."""
    def esc(value):return html.escape(str(value))
    def number(value):
        if value is None:return '—'
        if isinstance(value,(int,float)):return f'{value:.6g}'
        return str(value)
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(number(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    labels={'pulse_count':'总脉冲数','duration_s':'采集时长 / s','range_resolution_m':'理想距离尺度 / m',
            'velocity_resolution_mps':'速度分辨基准 / m/s','blind_range_m':'近距完整接收盲区 / m',
            'receive_range_max_m':'完整接收距离上限 / m','unambiguous_range_m':'最小 PRI 距离周期 / m',
            'unambiguous_velocity_mps':'同通道无模糊速度幅值 / m/s','peak_power_w':'总传导峰值功率 / W',
            'average_power_w':'平均功率 / W','tx_count':'物理 Tx 数','rx_count':'物理 Rx 数',
            'independent_virtual_positions':'独立虚拟位置数','single_pulse_snr_db':'需求距离单脉冲 SNR 基准 / dB',
            'estimated_memory_mib':'预计工作内存 / MiB','detections':'实际检测点数'}
    metrics=summary.get('metrics',{});parts=[]
    if metrics:parts.append('<h2>系统与处理结果</h2>'+table(['指标','结果'],[(v,metrics[k]) for k,v in labels.items() if k in metrics]))
    if summary.get('kind')=='statistics':
        rows=[]
        for key,label in [('pd','固定目标单元 Pd'),('pfa','固定无目标单元 Pfa'),('scene_detection_rate','场景目标关联率')]:
            val=summary.get(key,{})
            rows.append([label,val.get('estimate'),val.get('successes'),val.get('trials'),' – '.join(number(x) for x in val.get('ci95',[]))])
        parts.append('<h2>独立统计结果</h2>'+table(['量','估计值','计数','独立试验数','95% 置信区间'],rows))
        if summary.get('pfa',{}).get('zero_count_upper95') is not None:
            parts.append('<p>零虚警时的 95% 单侧上界：'+esc(number(summary['pfa']['zero_count_upper95']))+'。零计数不表示零概率。</p>')
    if summary.get('kind')=='inverse':
        rows=[]
        for candidate in summary.get('candidates',[]):
            m=candidate['metrics'];v=candidate.get('verification',{})
            rows.append([candidate['rank'],'通过' if candidate['passed'] else '未证实',m.get('average_power_w'),m.get('duration_s'),v.get('pd_estimate'),candidate['reason']])
        parts.append('<h2>有界候选搜索</h2><p>已筛选 '+esc(summary.get('screened_count',0))+' 个候选；未找到方案不等于不可实现。</p>'+table(['排序','状态','平均功率 W','采集时间 s','实测 Pd','依据'],rows))
    if summary.get('detections'):
        range_label='表观距离 m' if summary.get('waveform')=='lfm' else '距离 m'
        parts.append('<h2>实际检测点</h2><p>检测点可包含虚警、旁瓣或杂波，不等同于输入目标清单。角度采用阵列坐标；空白角度不代表零度。</p>'+table(['序号',range_label,'速度 m/s','阵列方位 °','阵列俯仰 °','单元/训练噪声 dB'],[[i+1]+[d.get(k) for k in ('range_m','velocity_mps','azimuth_deg','elevation_deg','snr_db')] for i,d in enumerate(summary['detections'])]))
    for key,label in [('assumptions','模型与约定'),('warnings','本次结果的适用条件')]:
        if summary.get(key):parts.append('<h2>'+label+'</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in summary[key])+'</ul>')
    record=esc(json.dumps(summary,ensure_ascii=False,indent=2))
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ArrayLab · 脉冲雷达计算报告</title><style>body{font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif;max-width:1050px;margin:40px auto;color:#17283a;padding:0 20px}h1,h2{color:#147b83}h2{font-size:21px;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d5e0e3;text-align:left;padding:8px}th{background:#edf5f5}tr{break-inside:avoid}pre{white-space:pre-wrap;word-break:break-word;font:12px monospace}.meta{font-size:13px;color:#526373;overflow-wrap:anywhere}@media print{body{margin:0;padding:0}h2{break-after:avoid}details:not([open]) pre{display:none}}</style><h1>ArrayLab · 脉冲雷达计算报告</h1><p>'+esc(summary.get('project_name','工程'))+'</p><p class="meta">版本 '+esc(summary.get('version',__version__))+' · 任务 '+esc({'simulate':'信号仿真','statistics':'独立统计','inverse':'反向设计'}.get(summary.get('kind'),'计算'))+' · 计算用时 '+esc(number(summary.get('elapsed_s')))+' s<br>输入 SHA-256：'+esc(summary.get('input_hash',''))+'</p>'+''.join(parts)+'<details><summary>展开完整数值记录与发射事件表</summary><pre>'+record+'</pre></details></html>'
