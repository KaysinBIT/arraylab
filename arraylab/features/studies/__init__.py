"""Public studies API; implementation imports are lazy and owned by the feature."""

__all__ = ['ScanStudyDialog', 'StudySettings', 'load_study_configuration', 'main', 'run_study', 'save_study_configuration']


def __getattr__(name):
    if name == 'ScanStudyDialog':
        from .desktop import ScanStudyDialog
        return ScanStudyDialog
    if name == 'StudySettings':
        from .model import StudySettings
        return StudySettings
    if name == 'load_study_configuration':
        from .model import load_study_configuration
        return load_study_configuration
    if name == 'main':
        from .cli import main
        return main
    if name == 'run_study':
        from .model import run_study
        return run_study
    if name == 'save_study_configuration':
        from .model import save_study_configuration
        return save_study_configuration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
