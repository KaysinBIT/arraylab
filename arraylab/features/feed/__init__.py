"""Public feed API; implementation imports are lazy and owned by the feature."""

__all__ = ['ElectricalControls', 'FeedPanel', 'FeedSettings', 'design_feed', 'diagram_svg', 'electrical_excitation', 'feed_settings_from_args', 'finite_number', 'load_feed_configuration', 'normalized_branch_delays', 'parser', 'port_list', 'save_feed_configuration', 'settings_from_payload', 'settings_payload', 'synthesize_impedance', 'target_amplitudes', 'validate_feed_record', 'validate_saved_pattern', 'write_feed_files']


def __getattr__(name):
    if name == 'ElectricalControls':
        from .controls import ElectricalControls
        return ElectricalControls
    if name == 'FeedPanel':
        from .desktop import FeedPanel
        return FeedPanel
    if name == 'FeedSettings':
        from .model import FeedSettings
        return FeedSettings
    if name == 'design_feed':
        from .model import design_feed
        return design_feed
    if name == 'diagram_svg':
        from .diagram import diagram_svg
        return diagram_svg
    if name == 'electrical_excitation':
        from .excitation import electrical_excitation
        return electrical_excitation
    if name == 'feed_settings_from_args':
        from .cli_options import feed_settings_from_args
        return feed_settings_from_args
    if name == 'finite_number':
        from .excitation import finite_number
        return finite_number
    if name == 'load_feed_configuration':
        from .model import load_feed_configuration
        return load_feed_configuration
    if name == 'normalized_branch_delays':
        from .model import normalized_branch_delays
        return normalized_branch_delays
    if name == 'parser':
        from .cli_options import parser
        return parser
    if name == 'port_list':
        from .cli_options import port_list
        return port_list
    if name == 'save_feed_configuration':
        from .model import save_feed_configuration
        return save_feed_configuration
    if name == 'settings_from_payload':
        from .model import settings_from_payload
        return settings_from_payload
    if name == 'settings_payload':
        from .model import settings_payload
        return settings_payload
    if name == 'synthesize_impedance':
        from .model import synthesize_impedance
        return synthesize_impedance
    if name == 'target_amplitudes':
        from .excitation import target_amplitudes
        return target_amplitudes
    if name == 'validate_feed_record':
        from .records import validate_feed_record
        return validate_feed_record
    if name == 'validate_saved_pattern':
        from .records import validate_saved_pattern
        return validate_saved_pattern
    if name == 'write_feed_files':
        from .model import write_feed_files
        return write_feed_files
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
