"""Application-owned desktop task metadata; no business modules are imported."""
from arraylab.framework.registry import TaskRegistry, TaskSpec


TASK_REGISTRY = TaskRegistry((
    TaskSpec('home', '开始', '选择一项任务，或打开以前保存的计算记录。',
             7, 0, 'home_tree_item', 'Ctrl+Home'),
    TaskSpec('linear', '直线阵分析', '设置参数 → F5计算并自动留存 → 选择历史结果 → 将该阵列导入馈电设计',
             0, 0, 'linear_tree_item', run_label='计算直线阵', pages=(0, 1, 2, 3)),
    TaskSpec('feed', '直线阵馈电', '右侧设置阵列、馈电拓扑与幅相 → 运行设计 → 核对端口与方向图 → 保存记录',
             4, 0, 'feed_tree_item', 'F6', '生成馈电设计'),
    TaskSpec('planar', '平面阵馈电', '右侧设置二维布局、目标角与馈电 → 运行设计 → 查看结果 → 导出结果',
             5, 2, 'planar_tree_item', 'F7', '生成平面阵设计'),
    TaskSpec('sparse', '二维稀疏设计', '右侧设置候选阵列与约束 → 运行综合 → 选择方案 → 调幅或继续设计馈电',
             6, 1, 'sparse_tree_item', 'F8', '开始稀疏综合'),
    TaskSpec('subarray', '子阵中心优化', '右侧设置内部模板、孔径与扫描 → 优化中心位置 → 核对实际阵元及方向图 → 导出结果',
             8, 3, 'subarray_tree_item', 'F9', '优化子阵中心'),
    TaskSpec('mimo', '稀疏MIMO分析', '分别设置Tx/Rx → F5计算 → 核对虚拟位置与通道 → 查看数字波束 → 保存记录',
             9, 4, 'mimo_tree_item', 'F10', '计算MIMO阵列'),
    TaskSpec('waveform', '雷达波形设计', 'F5分析波形与增益 → 天线设计页独立综合/验证 → 应用坐标 → F5核对 → 保存记录',
             10, 5, 'waveform_tree_item', 'F11', '计算雷达波形'),
    TaskSpec('phased','相控阵综合','选择阵列/扫描/优化操作 → 设置约束 → F5 → 核对各工况 → 应用候选或送入雷达',
             11,6,'phased_tree_item',None,'运行相控阵操作'),
    TaskSpec('equation','雷达方程验证','先看预算 → 回波仿真 → 固定CUT核对 → 独立校准与检测统计 → 保存完整记录',
             12,7,'equation_tree_item',None,'运行方程验证'),
    TaskSpec('pulse','脉冲雷达设计','需求与预算 → 波形/时序/阵列/场景 → F5仿真或反向设计 → 核对并保存',
             13,8,'pulse_tree_item',None,'运行脉冲雷达操作'),
))

# Keep the established public tuple dictionary available through workflow.py.
TASKS = {task.key: (task.title, task.hint, task.page) for task in TASK_REGISTRY}


def task_spec(key):
    return TASK_REGISTRY[key]


def task_page(key):
    return task_spec(key).page


def task_key_for_page(page):
    # The old workbench treats unknown/non-business pages as a linear view.
    return TASK_REGISTRY.for_page(page, default='linear').key
