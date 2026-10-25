"""Compatibility contracts for application-owned desktop task registration."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from arraylab.application.desktop.task_registry import (
    TASKS, TASK_REGISTRY, task_key_for_page, task_page, task_spec,
)
from arraylab.framework.registry import TaskRegistry, TaskSpec


ROOT = next(path for path in Path(__file__).resolve().parents if (path/'pyproject.toml').is_file())

# These values are the existing public workflow contract, not inferred from
# the new registry: labels, hint text and navigation indices must not change.
EXPECTED_TASKS = {
    'home': ('开始', '选择一项任务，或打开以前保存的计算记录。', 7),
    'linear': ('直线阵分析', '设置参数 → F5计算并自动留存 → 选择历史结果 → 将该阵列导入馈电设计', 0),
    'feed': ('直线阵馈电', '右侧设置阵列、馈电拓扑与幅相 → 运行设计 → 核对端口与方向图 → 保存记录', 4),
    'planar': ('平面阵馈电', '右侧设置二维布局、目标角与馈电 → 运行设计 → 查看结果 → 导出结果', 5),
    'sparse': ('二维稀疏设计', '右侧设置候选阵列与约束 → 运行综合 → 选择方案 → 调幅或继续设计馈电', 6),
    'subarray': ('子阵中心优化', '右侧设置内部模板、孔径与扫描 → 优化中心位置 → 核对实际阵元及方向图 → 导出结果', 8),
    'mimo': ('稀疏MIMO分析', '分别设置Tx/Rx → F5计算 → 核对虚拟位置与通道 → 查看数字波束 → 保存记录', 9),
    'waveform': ('雷达波形设计', 'F5分析波形与增益 → 天线设计页独立综合/验证 → 应用坐标 → F5核对 → 保存记录', 10),
}


def test_legacy_tuple_dictionary_and_menu_order_are_unchanged():
    assert type(TASKS) is dict and TASKS == EXPECTED_TASKS
    assert list(TASKS) == list(EXPECTED_TASKS)
    assert [task.key for task in TASK_REGISTRY] == list(EXPECTED_TASKS)
    assert all(type(row) is tuple for row in TASKS.values())


def test_inspector_shortcut_and_run_text_contract():
    expected = {
        'home': (0, 'Ctrl+Home', '运行计算'),
        'linear': (0, None, '计算直线阵'),
        'feed': (0, 'F6', '生成馈电设计'),
        'planar': (2, 'F7', '生成平面阵设计'),
        'sparse': (1, 'F8', '开始稀疏综合'),
        'subarray': (3, 'F9', '优化子阵中心'),
        'mimo': (4, 'F10', '计算MIMO阵列'),
        'waveform': (5, 'F11', '计算雷达波形'),
    }
    for key, values in expected.items():
        task = task_spec(key)
        assert (task.inspector, task.shortcut, task.run_label) == values
        assert task.tree_attr == key + '_tree_item'
        assert task_page(key) == EXPECTED_TASKS[key][2]


def test_all_linear_subviews_and_old_unknown_page_fallback():
    assert [task_key_for_page(page) for page in range(11)] == [
        'linear', 'linear', 'linear', 'linear', 'feed', 'planar', 'sparse',
        'home', 'subarray', 'mimo', 'waveform',
    ]
    assert task_spec('linear').page_indices == (0, 1, 2, 3)
    assert task_key_for_page(-1) == task_key_for_page(99) == 'linear'


def test_duplicate_keys_or_pages_are_rejected_before_ui_creation():
    first = TaskSpec('first', 'First', 'hint', 0, 0, 'first_tree', pages=(0, 1))
    with pytest.raises(ValueError, match='Duplicate task key'):
        TaskRegistry((first, replace(first, page=3, pages=())))
    with pytest.raises(ValueError, match='Duplicate task page'):
        TaskRegistry((first, TaskSpec('second', 'Second', 'hint', 1, 0, 'second_tree')))
    registry = TaskRegistry((first,))
    assert registry.for_page(1) is first
    with pytest.raises(KeyError): registry.for_page(10)


@pytest.mark.parametrize('changes', [
    {'page': True}, {'inspector': -1}, {'pages': (1,)}, {'pages': (0, 0)},
])
def test_invalid_page_metadata_is_rejected(changes):
    values = {'key': 'demo', 'title': 'Demo', 'hint': 'hint', 'page': 0,
              'inspector': 0, 'tree_attr': 'demo_tree', **changes}
    with pytest.raises(ValueError): TaskSpec(**values)


def test_task_metadata_import_does_not_load_qt_or_features():
    code = '''
import sys
from arraylab.application.desktop.task_registry import TASK_REGISTRY, task_page
assert task_page('mimo') == 9 and len(TASK_REGISTRY) == 8
assert not any(name.startswith(('PySide6', 'arraylab.features.')) for name in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stderr


def test_workflow_public_page_interface_and_legacy_export():
    pytest.importorskip('PySide6')
    from types import SimpleNamespace
    from arraylab.application.desktop.workflow import WorkflowActions, TASKS as canonical
    from arraylab.workflow import TASKS as legacy
    assert legacy is canonical is TASKS
    seen = []
    host = SimpleNamespace(task_page=lambda key: WorkflowActions.task_page(None, key),
                           show_view=lambda page, **options: seen.append(page))
    for key in EXPECTED_TASKS:
        WorkflowActions.activate_task(host, key)
    assert seen == [row[2] for row in EXPECTED_TASKS.values()]
    for page in range(11):
        probe = SimpleNamespace(tabs=SimpleNamespace(currentIndex=lambda value=page: value))
        assert WorkflowActions.active_task(probe) == task_key_for_page(page)


def test_native_task_routing_keeps_headers_inspectors_and_shortcuts(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtWidgets import QApplication
    from arraylab.application.desktop.window import MainWindow

    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'mpl'))
    monkeypatch.setenv('ARRAYLAB_HISTORY_DIR', str(tmp_path/'linear'))
    monkeypatch.setenv('ARRAYLAB_SPARSE_HISTORY_DIR', str(tmp_path/'sparse'))
    app = QApplication.instance() or QApplication([])
    window = MainWindow(auto_compute=False)
    try:
        for task in TASK_REGISTRY:
            window.activate_task(task.key)
            assert window.active_task() == task.key
            assert window.tabs.currentIndex() == task.page
            assert window.inspector_stack.currentIndex() == task.inspector
            assert window.task_title.text() == task.title
            assert window.task_hint.text() == task.hint
            assert window.task_actions[task.key].shortcut().toString() == (task.shortcut or '')
            assert tuple(getattr(window, task.tree_attr).data(0, Qt.ItemDataRole.UserRole)) == ('view', task.page)
            assert {i for i in range(window.tabs.count()) if window.tabs.isTabVisible(i)} == set(task.page_indices)
        assert not window.datasets and window.sparse_panel.bundle is None
        assert all(panel.report is None for panel in (
            window.feed_panel, window.planar_panel, window.subarray_panel,
            window.mimo_panel, window.waveform_panel,
        ))
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(window, QEvent.Type.DeferredDelete)
        app.processEvents()
