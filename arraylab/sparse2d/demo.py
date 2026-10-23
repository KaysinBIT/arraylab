"""Legacy sparse command options and complete offline report renderer."""
import os
# Bound numerical-library worker counts before importing NumPy.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
from datetime import datetime
import html
import json
from pathlib import Path
import sys

import numpy as np
from .core import Config, run, scan_uv, structured_arrays, coarray
from .scenarios import LIGHT_MM_GHZ,case_config,scenarios
from .elements import MODELS,build_element
from .runtime_paths import data_directory
from .phase_quantization import excitation_phases


def export_bundle(bundle, target):
    import matplotlib
    with matplotlib.rc_context():
        return _export_bundle(bundle, target)


def _export_bundle(bundle, target):
    """Write into a new directory. Existing exports are never overwritten."""
    import matplotlib
    matplotlib.use('Agg') if 'matplotlib.pyplot' not in sys.modules and 'PySide6.QtWidgets' not in sys.modules else None
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from .plots import comparison_figure, detail_figure, structure_figure, style, weight_comparison_figure, scenario_figure, element_figure,phase_comparison_figure
    style()
    target=Path(target)
    target.mkdir(parents=True,exist_ok=False)
    serial={k:v for k,v in bundle.items() if k!='results'}
    config=Config(**bundle['config']); cases=bundle.get('scenarios') or scenarios(config)
    first=cases[0]; first_config=case_config(config,first)
    wavelength_mm=LIGHT_MM_GHZ/config.reference_ghz
    element=build_element(config,bundle.get('element_pattern'))
    (target/'element_pattern.json').write_text(json.dumps(element.snapshot,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (target/'element_samples.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['frequency_ghz','theta_deg','phi_deg','response_db'])
        for frequency in dict.fromkeys(case['frequency_ghz'] for case in cases):
            for theta in range(0,91,5):
                phi=np.arange(0,361,10); t=np.deg2rad(theta); p=np.deg2rad(phi)
                uv=np.column_stack((np.sin(t)*np.cos(p),np.sin(t)*np.sin(p)))
                amplitude=element.response(uv,frequency)
                for angle,value in zip(phi,amplitude):
                    writer.writerow([frequency,theta,int(angle),'-inf' if value==0 else float(20*np.log10(value))])
    fig=element_figure(config,element,first['frequency_ghz']); FigureCanvasAgg(fig)
    fig.savefig(target/'element_preview.png'); fig.clear()
    serial['results']=[]
    for i,r in enumerate(bundle['results']):
        serial['results'].append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()})
        with (target/f'layout_{i+1}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f); writer.writerow(['candidate_index','x_lambda','y_lambda','amplitude','steering_phase_deg','steering_frequency_ghz','theta_deg','phi_deg','x_mm','y_mm','ideal_steering_phase_deg','phase_error_deg','phase_code','phase_bits'])
            phases=excitation_phases(r['xy'],first,config.phase_bits)
            for j,(ix,xy,w) in enumerate(zip(r['indices'],r['xy'],r['weights'])):
                writer.writerow([int(ix),*xy,float(w),float(phases['command_phase_deg'][j]),first['frequency_ghz'],first['theta'],first['phi'],*(xy*wavelength_mm),
                    float(phases['ideal_phase_deg'][j]),float(phases['error_deg'][j]),int(phases['phase_code'][j]) if config.phase_bits else '',config.phase_bits])
        fig=detail_figure(bundle,r); FigureCanvasAgg(fig); fig.savefig(target/f'layout_{i+1}.png'); fig.clear()
        if config.phase_bits:
            fig=detail_figure(bundle,r,phase_mode='quantized'); FigureCanvasAgg(fig); fig.savefig(target/f'layout_{i+1}_quantized.png'); fig.clear()
            fig=phase_comparison_figure(bundle,r); FigureCanvasAgg(fig); fig.savefig(target/f'phase_quantization_{i+1}.png'); fig.clear()
        if 'weighting' in r:
            fig=weight_comparison_figure(bundle,r); FigureCanvasAgg(fig)
            fig.savefig(target/f'weighting_{i+1}.png'); fig.clear()
    structured={}
    for name,integer in structured_arrays().items():
        ca=coarray(integer)
        structured[name]={'integer_coordinates':integer.tolist(),'elements':len(integer),
                         **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in ca.items()}}
    serial['structured_examples']=structured
    serial['runtime']={'python':sys.version.split()[0],'numpy':np.__version__}
    import scipy
    serial['runtime']['scipy']=scipy.__version__
    (target/'summary.json').write_text(json.dumps(serial,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (target/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['elements','fill','available_fill','forbidden_elements','amplitude_dynamic_range_db','validation_psl_db','optimization_psl_db','grid_gap_db','array_factor_validation_psl_db','element_at_steering_db','total_peak_db','mainlobe_peak_db','relative_to_main_peak_psl_db','hpbw_u','hpbw_v','case_count','worst_case_label','min_distance_lambda_ref','min_distance_mm','spacing_violations','unique_lags','bounding_holes','central_rectangle_lags','taper_efficiency','aperture_x_lambda','aperture_y_lambda']
        writer=csv.DictWriter(f,fieldnames=['method']+keys); writer.writeheader()
        for r in bundle['results']: writer.writerow({'method':r['name'],**{k:r['metrics'][k] for k in keys}})
    with (target/'scenario_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['method','index','theta','phi','frequency_ghz','frequency_ratio','optimization_psl_db','validation_psl_db','grid_gap_db','array_factor_psl_db','element_at_steering_db','total_peak_db','mainlobe_peak_db','relative_to_main_peak_psl_db','hpbw_u','hpbw_v','min_distance_lambda_frequency','hpbw_definition']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
        for r in bundle['results']:
            for case in r['scenario_metrics']: writer.writerow({'method':r['name'],**case})
    with (target/'excitation_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['method','case_index','frequency_ghz','theta_deg','phi_deg','candidate_index','x_mm','y_mm','shared_amplitude','steering_phase_deg','ideal_steering_phase_deg','phase_error_deg','phase_code','phase_bits'])
        for r in bundle['results']:
            for case in cases:
                cc=case_config(config,case)
                phases=excitation_phases(r['xy'],case,config.phase_bits)
                for j,(ix,xy,w) in enumerate(zip(r['indices'],r['xy'],r['weights'])):
                    writer.writerow([r['name'],case['index'],case['frequency_ghz'],case['theta'],case['phi'],int(ix),*(xy*wavelength_mm),float(w),
                        float(phases['command_phase_deg'][j]),float(phases['ideal_phase_deg'][j]),float(phases['error_deg'][j]),int(phases['phase_code'][j]) if config.phase_bits else '',config.phase_bits])
    if config.phase_bits:
        fields=['method','index','theta','phi','frequency_ghz','bits','step_deg','steering_response_defined','steering_field_linear',
            'ideal_validation_psl_db','validation_psl_db','delta_psl_db','validation_psl_ideal_reference_db','relative_to_main_peak_psl_db',
            'steering_loss_db','max_phase_error_deg','rms_phase_error_deg','ideal_pointing_error_deg','pointing_error_deg',
            'beam_shift_from_ideal_deg','global_pointing_error_deg','global_peak_outside_mainlobe']
        with (target/'phase_quantization.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
            for r in bundle['results']:
                for row in r['phase_quantization']['scenarios']: writer.writerow({'method':r['name'],**row})
    weighted=[r for r in bundle['results'] if 'weighting' in r]
    if weighted:
        with (target/'weighting_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['result','source','weight_method','elements','source_psl_db','weighted_psl_db','delta_psl_db','actual_dr_db','support_preserved'])
            for r in weighted:
                w=r['weighting']; source=bundle['results'][w['source_index']]
                writer.writerow([r['name'],source['name'],w['method'],r['metrics']['elements'],
                    source['metrics']['validation_psl_db'],r['metrics']['validation_psl_db'],
                    r['metrics']['validation_psl_db']-source['metrics']['validation_psl_db'],
                    r['metrics']['amplitude_dynamic_range_db'],True])
    for filename,fig in [('comparison.png',comparison_figure(bundle)),('structures.png',structure_figure())]:
        FigureCanvasAgg(fig); fig.savefig(target/filename); fig.clear()
    if len(cases)>1:
        fig=scenario_figure(bundle); FigureCanvasAgg(fig); fig.savefig(target/'joint_cases.png'); fig.clear()
    rows=''.join(f"<tr><td>{html.escape(r['name'])}</td><td>{r['metrics']['elements']}</td><td>{r['metrics']['validation_psl_db']:.2f}</td><td>{r['metrics']['grid_gap_db']:.2f}</td><td>{r['metrics']['amplitude_dynamic_range_db']:.2f}</td><td>{r['metrics']['unique_lags']}</td><td>{r['metrics']['central_rectangle_lags']}</td></tr>" for r in bundle['results'])
    images=''
    for i,r in enumerate(bundle['results']):
        images+=f'<h2>{html.escape(r["name"])}</h2><p>{html.escape(r["notes"])}</p><img src="layout_{i+1}.png">'
        if 'weighting' in r:
            images+=f'<p>加权参数：<code>{html.escape(json.dumps(r["weighting"],ensure_ascii=False))}</code></p><img src="weighting_{i+1}.png">'
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>二维稀疏阵列实验结果</title><style>body{font:16px/1.7 "Segoe UI","Microsoft YaHei",sans-serif;color:#243349;background:#f3f6fa;max-width:1120px;margin:36px auto;padding:0 24px}img{width:100%;background:white;border-radius:12px;margin:12px 0 28px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:10px 14px;text-align:left;border-bottom:1px solid #e5eaf0}h1{font-size:30px}h2{margin-top:36px}.note{background:#e0f2ed;padding:16px 22px;border-radius:10px}code{font-size:13px}</style><h1>二维稀疏阵列实验结果</h1>'''
    report+=f'<p>参数快照：<code>{html.escape(json.dumps(bundle["config"],ensure_ascii=False))}</code></p>'
    report+=f'<p>候选位置：{bundle["candidate_counts"]["total"]}；可用：{bundle["candidate_counts"]["available"]}；中心禁布：{bundle["candidate_counts"]["forbidden"]}。孔尺寸按 λ 计，圆孔使用 X 字段作为直径。幅度动态范围设置只约束固定位置的两种幅度 LP，不约束重加权 ℓ1 参考。</p>'
    report+=f'<p>参考频率 {config.reference_ghz:g} GHz（λ0={wavelength_mm:.6g} mm）；固定物理坐标，不同频率使用 f/f0 缩放相位。联合 {len(cases)} 个离散工况，共同位置和共同幅度，逐频理想校相。最小欧氏中心距要求 {config.min_spacing:g} λ0；所有输出均检查间距违规数为 0。PSL 汇总取跨工况最差值，HPBW 属于该最差工况。逐工况数据见 scenario_metrics.csv，共同幅度和逐工况相位见 excitation_cases.csv。</p>'
    if bundle.get('spacing',{}).get('pool_groups') is not None:
        report+='<p>间距约束采用可行支撑及冲突检查；参考布局和重加权 ℓ1 仅在记录的可行候选池内工作，该池不是最大装填或全局最优保证。贪心和退火在所有可用对称位置对上搜索，并拒绝违反距离的交换。</p>'
    report+=f'<h2>单元方向图</h2><p>已计算单元：{html.escape(element.snapshot["name"])}。总场=共同标量单元包络×阵因子；总场按各工况扫描点归一化。单元预览使用自身相对幅度参考。导入文件的校验信息、角网格和数值快照保存在 element_pattern.json，后续加权及导出不再依赖源文件；element_samples.csv 是前半球重采样数据。</p><img src="element_preview.png">'
    report+='<div class="note">独立研究 demo。逐工况窄带远场、相同朝向的标量单元、无互耦和极化模型；不据此估算实物增益。PSL 为固定参考主瓣矩形外相对扫描点的总场采样峰值，不是连续角域保证。单元方向性可造成波束峰偏移，另导出单元扫描响应、总峰及相对主瓣峰值的副瓣。HPBW 为穿过扫描点的切线在参考主瓣内取峰后的半功率区间宽度，单位是方向余弦，不是三维全局峰波束宽度或角度。Dense 与 ℓ1 加权解的阵元数不同；差分 lag 数不等于可分辨信源数。</div>'
    if weighted:
        report+='<p>稀疏后加权保持所选布局的全部位置和阵元数，按本报告的已计算参数重新评价；新权重替换原幅度。Taylor/Chebyshev 的参考抑制参数不保证稀疏阵达到该副瓣，幅度下限裁剪也会改变原窗。加权前后数据见 weighting_comparison.csv，逐阵元幅度见 layout_*.csv。</p>'
    report+='<h2>比较结果</h2><table><tr><th>方法</th><th>阵元数</th><th>复核 PSL / dB</th><th>复核差 / dB</th><th>实际幅度范围 / dB</th><th>不同 lag</th><th>中心无孔矩形 lag</th></tr>'+rows+'</table><img src="comparison.png">'+images
    if config.phase_bits:
        report+=f'<h2>移相器量化复核：{config.phase_bits} bit</h2><p>先完成理想相位综合与调幅，再在相同位置/幅度上最近码量化。步进 {360/2**config.phase_bits:g}°，最大误差上界为半步。每个频率分别编程，不是固定码宽带模型。上方主比较表和理想图属于理想综合；以下曲线共用理想扫描点参考，保留量化损失。量化 PSL 另按实际扫描点归一化，扫描点响应≤−160 dB 时记未定义；主瓣区/全局峰指向为采样估计。</p>'
        for i,r in enumerate(bundle['results']):
            q=r['phase_quantization']['metrics']; psl='未定义（存在扫描零陷）' if q['validation_psl_db'] is None else f"{q['validation_psl_db']:.3f} dB"
            report+=f'<h3>{html.escape(r["name"])}</h3><p>理想/量化最差 PSL：{r["metrics"]["validation_psl_db"]:.3f} dB / {psl}；最大扫描损失 {q["max_steering_loss_db"]:.3f} dB；最大相位误差 {q["max_phase_error_deg"]:.3f}°；最大主瓣区采样峰偏差 {q["max_pointing_error_deg"]:.3f}°。</p><img src="layout_{i+1}_quantized.png"><img src="phase_quantization_{i+1}.png">'
        report+='<p>逐工况指标见 phase_quantization.csv；layout_*.csv 和 excitation_cases.csv 中 steering_phase_deg 为实际下发相位，另附理想相位、环形误差和码字。实际相位按 0–360° 码本输出，关闭量化时保留原连续相位；空码字表示理想模式。</p>'
    if len(cases)>1:
        report+='<h2>联合工况复核</h2><p>图中每条方法曲线使用同一个物理布局和同一个幅度向量，各点只改变扫描相位与频率。</p><img src="joint_cases.png">'
    report+='<h2>结构化共阵示例</h2><p>三种结构的阵元数、孔径不同，只比较几何覆盖。此处未执行 MUSIC、空间平滑或测向性能实验。</p><img src="structures.png"><p>完整参数、阵元、权重和模型见 summary.json；逐阵元激励见 layout_*.csv。</p></html>'
    (target/'report.html').write_text(report,encoding='utf-8')
    return target


def parser():
    p=argparse.ArgumentParser(description='Standalone planar sparse-array research demo',
        epilog='JSON headless control: python demo.py --headless --help (or python headless.py --help)')
    p.add_argument('--gui',action='store_true',help='Open independent desktop experiment window')
    p.add_argument('--output',type=Path,help='Create a new result directory; never overwrite')
    p.add_argument('--weight-method',choices=['none','minimax','taylor','chebyshev','gaussian','uniform'],default='none')
    p.add_argument('--weight-source',choices=['dense','regular','random','greedy','annealing','l1-topk','l1-weighted'],default='annealing')
    p.add_argument('--weight-dr-db',type=float,default=20.)
    p.add_argument('--weight-sll',type=float,default=30.)
    p.add_argument('--weight-nbar',type=int,default=4)
    p.add_argument('--weight-sigma',type=float,default=.35)
    defaults=Config()
    for key in Config.__dataclass_fields__:
        value=getattr(defaults,key)
        choices={'hole_shape':['none','rectangle','circle'],'element_model':list(MODELS),
                 'element_import_mode':['grid','axisymmetric'],'element_dipole_axis':['x','y','z']}
        extra={'choices':choices[key]} if key in choices else {}
        p.add_argument('--'+key.replace('_','-'),type=type(value),default=value,**extra)
    return p


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--headless':
        from .headless import main as control_main
        return control_main(sys.argv[2:])
    p=parser(); args=p.parse_args()
    values={k:getattr(args,k) for k in Config.__dataclass_fields__}
    try: config=Config(**values).validate()
    except ValueError as exc: p.error(str(exc))
    from .weighting import Weighting,add_weighting
    settings=Weighting(method=args.weight_method,dr_db=args.weight_dr_db,sll_db=args.weight_sll,nbar=args.weight_nbar,sigma=args.weight_sigma)
    if args.weight_method!='none':
        try: settings.validate(config)
        except ValueError as exc: p.error(str(exc))
    if args.gui:
        if args.weight_method!='none': p.error('Use the desktop post-weighting tab; --weight-method applies to batch runs')
        from .gui import launch
        return launch(config)
    target=args.output or data_directory()/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    if target.exists():
        print(f'Output already exists: {target}',file=sys.stderr); return 3
    try:
        bundle=run(config,lambda msg:print(msg,file=sys.stderr,flush=True))
        if args.weight_method!='none':
            source_name={'dense':'Hollow uniform' if config.hole_shape!='none' else 'Dense reference',
                'regular':'Regular thinning','random':'Random symmetric','greedy':'Greedy exchange',
                'annealing':'Simulated annealing','l1-topk':'L1 top-K uniform','l1-weighted':'L1 weighted solution'}[args.weight_source]
            source_index=0 if args.weight_source=='dense' else next((i for i,r in enumerate(bundle['results']) if r['name']==source_name),None)
            if source_index is None: raise ValueError('Selected weighting source is unavailable for this geometry')
            bundle=add_weighting(bundle,source_index,settings,lambda msg:print(msg,file=sys.stderr,flush=True))
        export_bundle(bundle,target)
    except (OSError,RuntimeError,ValueError) as exc:
        print(str(exc),file=sys.stderr); return 1
    print(json.dumps({'output':str(target.resolve()),'results':[
        {'method':r['name'],**r['metrics'],'phase_quantization':r.get('phase_quantization',{}).get('metrics')} for r in bundle['results']]},ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
