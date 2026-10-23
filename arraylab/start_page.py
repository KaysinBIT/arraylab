"""Task selection and a short, actionable explanation of saving."""
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QLabel,QPushButton,QGroupBox


def build_start_page(window):
    page=QWidget();layout=QVBoxLayout(page);layout.setContentsMargins(28,20,28,20);layout.setSpacing(15)
    title=QLabel('先选择要完成的工作');title.setStyleSheet('font-size:23px;font-weight:600;color:#185986;');layout.addWidget(title)
    note=QLabel('参数在右侧，结果在中间，计算与保存集中在顶部。进入任务后不会自动计算。')
    note.setWordWrap(True);layout.addWidget(note)
    grid=QGridLayout();grid.setSpacing(14);window.task_cards={}
    for index,(kind,title,description,button_text) in enumerate([
        ('linear','直线阵分析','查看均匀直线阵的方向图、扫描角和阵元激励。','分析直线阵'),
        ('feed','直线阵馈电','设计并联、串行、1分2对称串馈的阻抗与端口相位。','设计直线阵馈电'),
        ('planar','平面阵馈电','为规则或任意x-y布局设计二维波束指向和馈电。','设计平面阵馈电'),
        ('sparse','二维稀疏设计','在候选位置中选择阵元，比较方案，再转入馈电设计。','综合稀疏阵列'),
        ('subarray','子阵中心优化','固定子阵内部模板，优化中心位置，并核对多工况方向图。','优化子阵位置'),
    ]):
        card=QGroupBox(title);body=QVBoxLayout(card);body.setContentsMargins(16,18,16,14)
        label=QLabel(description);label.setWordWrap(True);body.addWidget(label);body.addStretch()
        button=QPushButton(button_text);button.clicked.connect(lambda checked=False,k=kind:window.activate_task(k))
        button.setMinimumHeight(34);body.addWidget(button);window.task_cards[kind]=button;grid.addWidget(card,index//2,index%2)
    layout.addLayout(grid,1)
    flow=QLabel('选择任务  →  设置参数  →  运行计算  →  核对结果  →  保存或导出')
    flow.setStyleSheet('padding:10px;background:#eaf2f8;color:#245c85;font-weight:600;');flow.setWordWrap(True);layout.addWidget(flow)
    save=QGroupBox('保存怎么选');row=QHBoxLayout(save)
    for title,text in [('保存参数','保存当前输入；下次打开后重新计算。'),('保存计算记录','直线阵/直线阵馈电的结果、标签和选中项可以一起恢复。'),('导出结果','导出所选结果的表格、曲线或图像，供外部使用。')]:
        box=QVBoxLayout();label=QLabel(title);label.setStyleSheet('font-weight:600;');box.addWidget(label)
        content=QLabel(text);content.setWordWrap(True);box.addWidget(content);box.addStretch();row.addLayout(box,1)
    layout.addWidget(save)
    resume=QPushButton('打开以前保存的计算记录…');resume.clicked.connect(window.load_history_dialog);layout.addWidget(resume)
    window.resume_button=resume
    return page
