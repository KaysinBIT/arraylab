"""Prepare a clean feed design from a selected, computed uniform linear array."""
from dataclasses import asdict, dataclass

from arraylab.features.linear import ArrayParameters, ArrayResult
from arraylab.features.feed.model import FeedSettings, LEGACY_SETTINGS, validate_feed_inputs


@dataclass(frozen=True)
class LinearFeedImport:
    parameters: ArrayParameters
    settings: FeedSettings
    source_label: str
    notes: tuple[str, ...] = ()


def prepare_linear_feed_import(data, network=FeedSettings()):
    """Use the saved result, never uncomputed edits or unrelated feed excitation.

    Uniform analysis has no feed impedance information, so the existing valid
    network impedances and topology are retained. Everything that could change
    its target field is reset to an ideal uniform, continuously steered design.
    This function performs validation only and does not synthesize a network.
    """
    result = getattr(data, "result", None)
    if not isinstance(result, ArrayResult):
        raise ValueError("所选曲线没有可恢复的阵列参数。请先选择一次直线阵计算结果；外部曲线不能直接导入馈电设计。")
    label = getattr(data, "label", "")
    if not isinstance(label, str):
        raise ValueError("所选直线阵记录的标签无效。")
    if not isinstance(network, FeedSettings):
        raise ValueError("当前馈电网络设置无效。")
    parameters = ArrayParameters(**asdict(result.parameters))
    parameters.validate()
    values = {key: getattr(network, key) for key in LEGACY_SETTINGS}
    notes = ()
    if values["topology"] == "symmetric_series" and parameters.elements % 2:
        values["topology"] = "corporate"
        notes = ("原1分2对称串馈仅支持偶数阵元，已改为并联馈电。",)
    settings = FeedSettings(**values)
    validate_feed_inputs(parameters, settings)
    return LinearFeedImport(parameters, settings, label or "所选直线阵结果", notes)
