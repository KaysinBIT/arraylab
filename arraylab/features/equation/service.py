"""Native services over the unchanged, independent budget and receiver cores."""
from copy import deepcopy
import json
import math
from pathlib import Path

from arraylab import __version__
from .provenance import SOURCE_VERSION, SOURCE_ARCHIVE_SHA256
from .config import default_config, normalize_config, checked_config, validate_config, config_hash, resource_estimate
from .budget import calculate
from .detection import calibration_fingerprint

ACTIONS = ('budget', 'simulation', 'snr', 'calibration', 'statistics', 'distance', 'roc')
ACTION_LABELS = {'budget': '系统预算与反算', 'simulation': '真实回波与检测', 'snr': '固定CUT的SNR验证',
                 'calibration': '独立H0门限校准', 'statistics': '独立Pd/Pfa统计', 'distance': '距离网格统计', 'roc': '经验ROC'}


def execution_resources(kind, configuration):
    """Include retained trial records as well as the unchanged receiver estimate."""
    native=resource_estimate(configuration)
    bytes_per_trial={'snr':128,'calibration':96,'statistics':256,'distance':256,'roc':160}.get(kind,0)
    extra=int(configuration['experiment']['trials'])*bytes_per_trial
    if kind=='distance':extra+=int(configuration['experiment']['distance_count'])*4096
    return {**native,'receiver_estimated_bytes':native['estimated_bytes'],'trial_record_estimated_bytes':extra,
            'estimated_bytes':native['estimated_bytes']+extra,
            'estimated_mib':(native['estimated_bytes']+extra)/1024**2}


def _configuration(value):
    if value is None:
        return default_config()
    if type(value) is not dict:
        raise ValueError('雷达方程工程须为JSON对象。')
    if value.get('format') == 'arraylab.equation':
        if set(value) != {'format', 'schema_version', 'configuration'} or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
            raise ValueError('不支持的ArrayLab雷达方程工程版本。')
        value = value['configuration']
    if type(value) is not dict:
        raise ValueError('雷达方程配置须为JSON对象。')
    if 'schema_version' in value and (type(value['schema_version']) is not int or value['schema_version'] != 1):
        raise ValueError('雷达方程工程版本须为整数1。')
    return value


def default_configuration(kind='fmcw', frequency_ghz=77):
    return checked_config(default_config(kind, frequency_ghz))


def normalize_configuration(configuration=None):
    """Normalize/validate source schema 1, never launch numerical experiments."""
    return checked_config(_configuration(configuration))


def validate_configuration(configuration=None):
    try:
        normalized = normalize_config(_configuration(configuration))
        errors = validate_config(normalized)
        metrics = resource_estimate(normalized) if not errors else {}
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
        errors, metrics = [str(exc)], {}
    return {'valid': not errors, 'errors': errors, 'warnings': [], 'metrics': metrics}


def budget(configuration=None):
    """Independent energy reference; never calls the waveform receiver engine."""
    return calculate(_configuration(configuration))


def execute(kind, configuration=None, *, progress=None, cancel=None):
    """Return a replayable result and full NumPy arrays for an external worker.

    The host owns the process/resource lifecycle. Source cancellations remain
    InterruptedError; no embedded HTTP server or browser runtime is started.
    """
    if kind not in ACTIONS:
        raise ValueError('未知雷达方程操作：' + str(kind))
    normalized = normalize_configuration(configuration)
    if cancel and cancel():
        raise InterruptedError('计算已取消')
    resources=execution_resources(kind,normalized)
    if resources['estimated_mib']>normalized['experiment']['memory_mib']:
        raise ValueError(f"包含逐次统计记录的预计内存 {resources['estimated_mib']:.1f} MiB 超过预算 {normalized['experiment']['memory_mib']} MiB。")
    if kind == 'budget':
        result, arrays = calculate(normalized), {}
        result.update(kind='budget', schema_version=1, algorithm_version=SOURCE_VERSION,
                      seed=normalized['experiment']['seed'], config=normalized,
                      config_hash=config_hash(normalized), elapsed_s=0.)
        if progress: progress(1., '预算完成；尚未运行回波或统计验证')
    else:
        from .statistics import execute as run_experiment
        result, arrays = run_experiment(kind, normalized, progress, cancel)
        result['budget'] = calculate(normalized)
    if cancel and cancel():
        raise InterruptedError('计算已取消')
    result['arraylab_version'] = __version__
    result['resource_guard'] = resources
    result['source'] = {'name': 'EquationLab', 'version': SOURCE_VERSION, 'archive_sha256': SOURCE_ARCHIVE_SHA256}
    return result, arrays


def presets():
    directory = Path(__file__).with_name('data')/'presets'
    return [{'id': path.stem, 'name': configuration['name'], 'configuration': configuration}
            for path in sorted(directory.glob('*.json'))
            for configuration in [checked_config(json.loads(path.read_text(encoding='utf-8-sig')))]]


def reset_array(configuration):
    """Explicitly rebuild both geometries/weights after a channel-count edit.

    Does not mutate the input or silently select another channel/waveform mode.
    A previously applied calibration remains subject to its source fingerprint.
    """
    updated=normalize_config(_configuration(configuration))
    for side in ('tx','rx'):
        updated['array'][side+'_positions_m']=[]
        updated['array'][side+'_weights']=[]
    return checked_config(updated)


def apply_calibration(configuration, calibration_result):
    """Apply only a sufficiently sampled H0 threshold for the same receiver/CUT."""
    normalized = normalize_configuration(configuration)
    report = calibration_result
    if type(report) is not dict or report.get('kind') != 'calibration' or report.get('ready') is not True:
        raise ValueError('该H0校准尚未满足尾部样本条件，不能应用。')
    trials,pfa=report.get('trials'),report.get('pfa')
    if (type(trials) is not int or trials<1 or type(pfa) not in (int,float) or not math.isfinite(pfa) or
            pfa!=normalized['detection']['pfa'] or trials*pfa<20):
        raise ValueError('H0校准的尾部样本计数或Pfa不一致。')
    factor = report.get('factor')
    if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
        raise ValueError('校准门限不是有限正数。')
    if report.get('calibration_hash') != calibration_fingerprint(normalized):
        raise ValueError('校准与当前配置或CUT不匹配，请重新校准。')
    updated = deepcopy(normalized)
    updated['detection'].update(threshold_mode='calibrated', calibration_factor=factor,
                                calibration_hash=report['calibration_hash'])
    return checked_config(updated)
