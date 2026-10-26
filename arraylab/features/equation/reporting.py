"""Pure report formatting extracted from the frozen EquationLab service."""
import csv
import html
import io
import json
import numpy as np
from arraylab.shared.simulation_io import clean_json
from .provenance import SOURCE_VERSION

def _curve_svg(xs,ys,title,xlabel,ylabel):
    values=[(float(x),float(y)) for x,y in zip(xs,ys) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(values)<2:return ""
    x=np.array([v[0] for v in values]);y=np.array([v[1] for v in values]);xmin,xmax=float(x.min()),float(x.max());ymin,ymax=float(y.min()),float(y.max())
    if xmax==xmin:xmax=xmin+1
    if ymax==ymin:ymax=ymin+1
    coords=" ".join(f"{70+690*(xx-xmin)/(xmax-xmin):.2f},{245-190*(yy-ymin)/(ymax-ymin):.2f}" for xx,yy in values)
    ticks="".join(f'<text x="{70+690*i/4}" y="270" text-anchor="middle">{xmin+(xmax-xmin)*i/4:.3g}</text><text x="58" y="{250-190*i/4}" text-anchor="end">{ymin+(ymax-ymin)*i/4:.3g}</text>' for i in range(5))
    return f'<svg viewBox="0 0 820 315" role="img" aria-label="{html.escape(title)}" style="width:100%;max-height:360px;border:1px solid #dae5eb;background:white;font:13px sans-serif"><text x="70" y="27" font-size="18">{html.escape(title)}</text><path d="M70 48V245H760" fill="none" stroke="#789"/><polyline points="{coords}" fill="none" stroke="#14858b" stroke-width="2.5"/>{ticks}<text x="415" y="301" text-anchor="middle">{html.escape(xlabel)}</text><text x="18" y="155" transform="rotate(-90,18,155)" text-anchor="middle">{html.escape(ylabel)}</text></svg>'


def _heatmap_svg(arrays):
    if "rd_power" not in arrays:return ""
    raw=arrays["rd_power"];sy=max(1,math_ceil(raw.shape[0]/48));sx=max(1,math_ceil(raw.shape[1]/96));a=10*np.log10(np.maximum(raw[::sy,::sx],1e-300))
    maximum=float(np.nanmax(a));minimum=maximum-60;ny,nx=a.shape;rect=[]
    for i in range(ny):
        for j in range(nx):
            v=float(np.clip((a[i,j]-minimum)/60,0,1));color=f"rgb({int(20+225*v)},{int(50+160*v)},{int(90-30*v)})"
            rect.append(f'<rect x="{70+690*j/nx:.2f}" y="{45+190*(ny-1-i)/ny:.2f}" width="{690/nx+.2:.2f}" height="{190/ny+.2:.2f}" fill="{color}"/>')
    ranges=arrays["ranges_m"];velocities=arrays["velocities_mps"]
    return '<svg viewBox="0 0 820 290" role="img" aria-label="实际距离速度功率图" style="width:100%;font:13px sans-serif"><text x="70" y="25" font-size="18">实际距离–速度功率图</text>'+''.join(rect)+f'<text x="70" y="260">{ranges[0]:.2f} m</text><text x="680" y="260">{ranges[-1]:.2f} m</text><text x="4" y="55">{velocities[-1]:.2f}</text><text x="4" y="235">{velocities[0]:.2f}</text><text x="290" y="282">表观距离；速度 m/s；显示降采样 {sy}×{sx}</text></svg>'


def _report(result,plots=""):
    name=html.escape(result["config"].get("name","雷达工程"));cfg=result["config"]
    b=result if result.get('kind')=='budget' else result.get('budget',{})
    def number(key,unit):return f"{b[key]:.3f} {unit}" if b.get(key) is not None else "不可达或未定义"
    task_names={"simulation":"单次回波仿真","snr":"固定 CUT SNR 校核","calibration":"独立 H0 校准","statistics":"Pd / Pfa 统计","distance":"距离网格统计","roc":"经验 ROC"}
    actual=result.get("noise_trials",result.get("trials",result.get("pd",{}).get("trials")))
    if result["kind"]=="simulation":actual=1
    if actual is None and result.get("points"):actual=max(((x.get("pd") or {}).get("trials",0) for x in result["points"]),default=0)
    rows=[("实际任务",task_names.get(result["kind"],result["kind"])),("独立试验数（每个条件）",str(actual) if actual is not None else "见逐条件记录"),
        ("波形 / 通道",f"{cfg['waveform']['kind'].upper()} / {cfg['array']['mode'].upper()}"),
        ("自由空间接收功率",number('received_power_dbm','dBm')),("理想能量 SNR",number('output_snr_db','dB')),
        ("理论参考距离",number('max_range_m','m')),("算法版本",result["algorithm_version"]),("随机种子",str(result["seed"])),
        ("配置 SHA-256",result["config_hash"]),("标准", "ITU-R P.676-13 / P.838-3")]
    table="".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k,v in rows)
    warnings="".join("<li>"+html.escape(str(x))+"</li>" for x in result.get("warnings",[]))
    summary={k:v for k,v in result.items() if k not in ("config","warnings")}
    if b.get("valid"):plots=_curve_svg(b["curve"]["range_m"],b["curve"]["snr_db"],"自由空间能量参考","距离 / m","SNR / dB")+plots
    if result["kind"] in ("distance","roc"):
        points=[x for x in result["points"] if x.get("pd") and x.get("pfa")]
        plots+=_curve_svg([x["range_m"] if result["kind"]=="distance" else x["pfa"]["estimate"] for x in points],
            [x["pd"]["estimate"] for x in points],"Pd–距离采样" if result["kind"]=="distance" else "经验 ROC", "距离 / m" if result["kind"]=="distance" else "Pfa","Pd")
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{name} · 验证报告</title>
<style>body{{max-width:1100px;margin:45px auto;padding:0 32px;color:#183042;font:16px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}}h1{{font-size:32px}}h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dbe4e9;padding:10px;text-align:left}}th{{width:220px;background:#f1f6f8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:20px;font-size:13px}}.tag{{color:#087e83}}@media print{{body{{margin:0}}button{{display:none}}pre{{font-size:10px}}}}</style>
<p class="tag">EquationLab · 合成场景 / 可复现计算</p><h1>{name}</h1><p>雷达方程与检测性能验证报告 · <button onclick="window.print()">打印 / 保存 PDF</button></p><table>{table}</table>
<h2>计算图表</h2>{plots}<h2>结果与统计证据</h2><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre><h2>适用条件</h2><ul>{warnings}</ul>
<p>自由空间预算、场景处理和统计置信区间有各自适用条件。合成模型结果不代表硬件验收。</p>
<h2>完整输入与单位</h2><pre>{html.escape(json.dumps(cfg,ensure_ascii=False,indent=2))}</pre>
<h2>公式来源</h2><p><a href="https://www.mathworks.com/help/radar/ug/connecting-the-radar-equation-to-waveform-level-simulation.html">雷达方程与波形处理</a> · <a href="https://www.itu.int/rec/R-REC-P.676/en">ITU-R P.676</a> · <a href="https://www.itu.int/rec/R-REC-P.838/en">ITU-R P.838</a></p></html>'''


def _csv_result(result):
    output=io.StringIO(newline="");writer=csv.writer(output)
    legacy=result.get('algorithm_version')!=SOURCE_VERSION
    if result["kind"] in ("distance","roc"):
        prefix='legacy_' if legacy else ''
        writer.writerow(["range_m" if result["kind"]=="distance" else "threshold_db",*[prefix+x for x in ("pd","pd_lower","pd_upper","pfa","pfa_lower","pfa_upper","status")]])
        for p in result["points"]:
            pd=p.get("pd") or {};pf=p.get("pfa") or {}
            writer.writerow([p.get("range_m",p.get("threshold_db")),pd.get("estimate"),pd.get("lower"),pd.get("upper"),pf.get("estimate"),pf.get("lower"),pf.get("upper"),p.get("status","")])
    elif result.get("detections"):
        fields=["range_m","velocity_mps","power_dbw","cell_power_to_noise_db"];writer.writerow(fields)
        for d in result["detections"]:writer.writerow([d.get(k,d.get('cell_snr_db') if k=='cell_power_to_noise_db' else None) for k in fields])
    else:
        writer.writerow(["metric","value"])
        if legacy:writer.writerow(['snapshot_notice','旧版或未知算法保存口径，须重新计算；原始数值未修改。'])
        for k,v in result.items():
            if isinstance(v,(str,int,float,bool)):writer.writerow([k,v])
            elif isinstance(v,dict) and "estimate" in v:
                for field in ("estimate","lower","upper","trials"):writer.writerow([f"{k}.{field}",v[field]])
    return "\ufeff"+output.getvalue()


def math_ceil(x):return int(np.ceil(x))

def render_report(result, arrays=None):
    saved=clean_json(result,complex_style='pair')
    if result.get('algorithm_version')!=SOURCE_VERSION:
        saved['warnings']=[*saved.get('warnings',[]),'旧版或未知算法保存快照；其中统计和验收口径不是当前1.0.1模型的结论，请重新计算。']
    return _report(saved,_heatmap_svg(arrays or {})).replace(
        '<p class="tag">EquationLab · 合成场景 / 可复现计算</p>',
        '<p class="tag">ArrayLab · 雷达方程验证 · EquationLab '+html.escape(str(result.get('algorithm_version','未说明')))+'计算快照 · 合成场景</p>')


def result_csv(result):
    return _csv_result(clean_json(result,complex_style='pair'))
