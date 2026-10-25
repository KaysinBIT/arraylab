"""Only the task-tree navigation changed in this iteration is exercised."""
import os
from pathlib import Path

import pytest


ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())


@pytest.fixture(scope='module')
def navigation_window():
    pytest.importorskip('PySide6')
    output=ROOT/'tmp'/'navigation-tests';output.mkdir(parents=True,exist_ok=True)
    settings={'QT_QPA_PLATFORM':'offscreen','MPLCONFIGDIR':str(output/'mpl'),
              'ARRAYLAB_HISTORY_DIR':str(output/'linear'),'ARRAYLAB_SPARSE_HISTORY_DIR':str(output/'sparse'),
              'TEMP':str(output/'temp'),'TMP':str(output/'temp')}
    for key,value in settings.items():
        if key!='QT_QPA_PLATFORM':Path(value).mkdir(parents=True,exist_ok=True)
    old={key:os.environ.get(key) for key in settings};os.environ.update(settings)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase,QFont
    from arraylab.application.desktop.window import MainWindow
    from arraylab.application.verification.navigation import settle
    from arraylab.framework.parallel import requested_workers,set_workers
    app=QApplication.instance() or QApplication([])
    assert app.platformName()=='offscreen'
    if not QFontDatabase.families():
        font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
        if font.is_file():QFontDatabase.addApplicationFont(str(font));app.setFont(QFont('Microsoft YaHei',9))
    app.setQuitOnLastWindowClosed(False);previous=requested_workers();set_workers(1)
    window=MainWindow(auto_compute=False);window.resize(1536,900);window.show();settle(app)
    report={'navigation':{'screenshots':[]}}
    def check(value,name):assert value,name
    try:yield app,window,output,report,check
    finally:
        window.close();window.deleteLater();app.processEvents();set_workers(previous)
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def test_all_nodes_navigate_without_computing_or_editing(navigation_window):
    from arraylab.application.verification.navigation import check_all_leaves
    check_all_leaves(*navigation_window)


def test_controls_sync_back_and_refresh_does_not_steal_selection(navigation_window):
    from arraylab.application.verification.navigation import check_reverse_and_refresh
    check_reverse_and_refresh(*navigation_window)


def test_modules_expand_without_collapsing_other_manual_groups(navigation_window):
    from arraylab.application.verification.navigation import check_expansion
    check_expansion(*navigation_window)


def test_existing_linear_and_sparse_navigation_contracts(navigation_window):
    from arraylab.application.verification.navigation import check_legacy
    check_legacy(*navigation_window)


def test_feed_records_rename_restore_and_same_item_reselection(navigation_window):
    from arraylab.application.verification.navigation import check_feed_history
    check_feed_history(*navigation_window)


def test_native_linear_views_override_history_selection(navigation_window):
    from arraylab.application.verification.navigation import check_linear_reverse_history
    check_linear_reverse_history(*navigation_window)


def test_sparse_busy_only_signal_does_not_take_parameter_focus(navigation_window):
    from arraylab.application.verification.navigation import check_sparse_busy_preserves_parameter
    check_sparse_busy_preserves_parameter(*navigation_window)


def test_bad_feed_record_keeps_actual_selection_and_result(navigation_window):
    from arraylab.application.verification.navigation import check_feed_bad_record_rolls_back
    check_feed_bad_record_rolls_back(*navigation_window)


@pytest.mark.parametrize('size',[(1100,640),(1536,900)])
def test_navigation_layout_and_scroll_reachability(navigation_window,size):
    from arraylab.application.verification.navigation import check_layout
    check_layout(*navigation_window,*size)
