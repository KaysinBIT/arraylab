"""Edit matched branch delays without calculating or replacing a result snapshot."""
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)


class BranchDelayDialog(QDialog):
    DELAY_COLUMN = 5

    def __init__(self, report, values, parent=None, reset_reason=""):
        super().__init__(parent)
        self.setWindowTitle("支路电长度")
        self.resize(930, 610)
        self.setMinimumSize(650, 420)
        self.result_delays = None
        self.sections = report["sections"]
        layout = QVBoxLayout(self)
        frequency = report["array_parameters"]["frequency_ghz"]
        explanation = QLabel(
            f"设计频率 {frequency:g} GHz。每支路先经过固定90°变换段，再经过与下游等效阻抗匹配的延迟段。\n"
            "只编辑“附加延迟”：0–3600°，正值产生相位滞后。扫描或等相模式会重算端口补偿，输出可能保持目标不变。\n"
            "查看所加延迟的影响，可选择“网络原始输出”方向图或“不补偿”模式。应用后按 F5 生成新设计。\n"
            "电长度/360°是当前频点的导波波长比例，不换算为物理尺寸；当前结果保持原快照。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.reset_notice = QLabel(reset_reason)
        self.reset_notice.setWordWrap(True)
        self.reset_notice.setStyleSheet("color: #a45c14; font-weight: bold")
        self.reset_notice.setVisible(bool(reset_reason))
        layout.addWidget(self.reset_notice)
        self.table = QTableWidget(len(self.sections), 7)
        self.table.setObjectName("branchDelayTable")
        self.table.setHorizontalHeaderLabels([
            "支路", "起点 → 终点", "变换阻抗 Zₜ (Ω)", "匹配阻抗 Zc (Ω)",
            "固定变换 (°)", "附加延迟 (°)", "两段合计 (°)",
        ])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, (section, delay) in enumerate(zip(self.sections, values, strict=True)):
            numbers = [section["id"], section["parent"] + " → " + section["child"],
                       f'{section["line_ohm"]:.7g}', f'{section["child_equivalent_ohm"]:.7g}',
                       f'{section["electrical_length_deg"]:.7g}', f"{delay:.12g}",
                       f'{section["electrical_length_deg"] + delay:.12g}']
            for column, value in enumerate(numbers):
                item = QTableWidgetItem(value)
                if column != self.DELAY_COLUMN:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                else:
                    item.setToolTip("该支路的匹配延迟段，0–3600°；与90°变换段采用不同阻抗。")
                self.table.setItem(row, column, item)
        self.table.itemChanged.connect(self.update_total)
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.clear_button = QPushButton("全部清零")
        buttons.addButton(self.clear_button, QDialogButtonBox.ButtonRole.ResetRole)
        self.clear_button.clicked.connect(self.clear_values)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Apply)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.apply_button.setText("应用到待计算设置")
        self.apply_button.clicked.connect(self.apply_values)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def update_total(self, item):
        if item.column() != self.DELAY_COLUMN:
            return
        try:
            value = float(item.text().strip())
            valid = math.isfinite(value) and 0 <= value <= 3600
        except ValueError:
            valid = False
        total = (f'{self.sections[item.row()]["electrical_length_deg"] + value:.12g}'
                 if valid else "输入无效")
        self.table.item(item.row(), 6).setText(total)

    def clear_values(self):
        for row in range(self.table.rowCount()):
            self.table.item(row, self.DELAY_COLUMN).setText("0")

    def apply_values(self):
        # Moving focus commits a still-open table editor before reading items.
        self.apply_button.setFocus()
        values = []
        for row in range(self.table.rowCount()):
            try:
                value = float(self.table.item(row, self.DELAY_COLUMN).text().strip())
                if not math.isfinite(value) or not 0 <= value <= 3600:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "电长度无效", f"{self.sections[row]['id']} 的附加延迟须为0–3600°的有限数值。")
                self.table.setCurrentCell(row, self.DELAY_COLUMN)
                return
            values.append(value)
        self.result_delays = tuple(values)
        self.accept()
