# ArrayLab 0.27.0 功能验证、算法纠错与模块重构报告

审查日期：2026-10-28。基线为已发布的0.26.1，提交`70f21ca89747b50e4ae9119d92c25e49affb0da7`。本轮按现有代码、仓库说明、既有测试和可独立核算的数学不变量开展验证，不将“测试全通过”解释为全部公式、输入域或硬件性能已经穷尽验证。

本轮确认并修复6类产品缺陷：2类数值尺度错误、4类记录/数据完整性错误。先补充修前失败的反例，再修复；之后提取重复计算与后台命令配置。最终现有与新增测试共78个入口文件、2007个用例通过，逐文件在独立子进程运行，均退出0；无失败、错误、跳过或超时。累计子进程耗时441.667秒，不等于并行执行的端到端墙钟时间。

## 1. 依据、范围与风险优先级

### 1.1 依据

| 依据 | 本轮用途 |
| --- | --- |
| `application/desktop/task_registry.py`、`application/desktop/workflow.py` | 确认10个业务模块、任务切换、快捷键和统一操作链 |
| `application/cli.py`、各业务`cli.py` | 确认CLI路由、JSON输出和无头入口 |
| 各`features/*`模型、记录实现及就近测试 | 确认已有规则和边界，不根据同类产品补造功能 |
| [快速开始](quickstart.md)、[任务树](task-navigation.md)、[结构说明](structure-refactor.md) | 操作意图、框架/业务/公共层边界；历史版本说明只按其原有范围使用 |
| [馈电阻抗](feed-impedance.md)、[支路电长度](branch-electrical-lengths.md)、[方向图指标](pattern-metrics.md) | 馈电与幅相、3dB宽度等主要数值口径 |
| [FMCW波形](radar-waveforms.md)、[雷达组件](radar-components.md)、[子模块同步](upstream-sync-0261.md) | 雷达模型适用范围、旧记录和旧校准语义 |
| 根目录`tests/test_*.py` | 稳定测试入口；实现位于就近`tests`目录，避免重复收集同一批测试 |

外部参考项目并未在本轮重新完整审计；此前来源对照的126/232等数字不计入本轮通过数。原独立项目、`mbfn_sparse_mimo`、`sparse2d_demo`和`交付`目录均不属于本轮修改范围。

### 1.2 模块、流程及对外接口

| 业务模块 | 核心输入与操作链 | 主要输出/接口 |
| --- | --- | --- |
| 直线阵分析 | 阵元/频率/间距/指向/加权 → F5 → 选择计算历史 → 比较或导入馈电 | 切面、三维、主要指标、阵元激励；JSON/CSV、线阵自动历史 |
| 直线阵馈电 | 目标阵列 → 并联/串行/1分2对称串行 → 阻抗和支路延迟 → 端口/方向图核对 | 支路/端口表、连接图、目标/未补偿/最终方向图、设计记录 |
| 平面阵馈电 | 规则或任意坐标 → 指向及馈电设置 → F5 → 核对端口与方向图 | 二维/三维、端口/支路、参数与结果文件 |
| 二维稀疏设计 | 候选网格/目标K/约束 → 综合 → 复核扫描工况 → 调幅或继续馈电 | 稀疏布局、切面、各工况指标、自动历史 |
| 子阵中心优化 | 固定内部模板 → 子阵中心/孔径/扫描约束 → 优化与复核 | 中心与实际阵元坐标、工况结果、图形/表格 |
| 稀疏MIMO | 分别设置Tx/Rx → 虚拟几何与通道统计 → 波束指标 | 物理与虚拟布局、重数、通道表、完整记录 |
| FMCW雷达波形设计 | 共用参数及A/B → F5 → 时序/频谱/解模糊 → 天线设计、预算或场景 | 波形报告、对比记录、CSV/图像；`waveform`命令 |
| 相控阵综合 | 阵列/相位模型 → 分析、扫描或综合 → 逐工况复核 | 场、切面、馈相与可行性；`phased`命令、NPZ记录 |
| 雷达方程验证 | 预算 → 回波仿真 → 同CUT SNR/校准/统计 | 预算、原始与处理数组、统计及报告；`equation`命令 |
| 脉冲雷达设计 | 波形/时序/阵列/场景 → 预算、仿真、统计或反向设计 | 接收/脉压/距离速度/角谱、场景与工程；`pulse`命令、`.prlab` |

桌面入口为`ArrayLab.exe`或`python -m arraylab`；无头入口为`ArrayLab-cli.exe`或`python -m arraylab --headless`。现有默认线阵/馈电命令，以及`mimo`、`subarray`、`waveform`、`phased`、`equation`、`pulse`、`--sparse`、`--planar`、`--study`路由保留。具体参数以对应`--help`和操作手册为准。

“保存参数”保存当前输入；“保存记录”保存成功计算的输入与结果；“导出”提供外部可用文件。三组件记录使用完整NPZ或`arraylab-record.json`与`arrays.npz`组合，FMCW使用带摘要的JSON记录，脉冲便携工程使用`.prlab`。线阵与二维稀疏使用各自SQLite自动历史；其它模块不能据此推定有自动历史。既有格式标记、历史目录、CLI参数和公开Python入口未因本轮重构改名。

### 1.3 依赖方向

`application`负责具体模块组装、跨模块快照交接和GUI/CLI协调；`framework`负责启动、任务注册机制、进程生命周期和有界并行；`features`各自拥有模型/业务/记录；`shared`只放多模块复用的数值、图形、原子文件和离线场景能力。

依赖方向仍为应用层调用业务、框架与公共层，业务调用公共层和框架。基础框架不导入具体业务。模块之间通过公开入口及已完成快照协作，不从其它模块的内部变量重建隐含业务状态。本轮未增加HTTP服务、路由框架、依赖注入容器或新运行依赖。

风险优先覆盖：幅相/功率归一化及波束指标、记录数值是否完整对应、后台进程取消/异常、旧记录与新版本边界。其他模块通过已有回归检查覆盖，未宣称全部重新人工推导。

## 2. 功能点—预期行为—测试—结果清单

以下均已执行并通过。文件名指根`tests/`入口；同一文件可支撑多行，表格各行不能再相加当作独立测试数量。单元、协议/模拟故障和真实Qt/子进程证据分别注明。

| 编号 | 功能点 | 预期行为 | 测试入口/证据 | 本轮结果 |
| --- | --- | --- | --- | --- |
| F01 | 启动、任务及旧接口 | GUI/无头正确分派，旧导入与任务映射保持 | `test_architecture_contracts`、`test_framework_contracts`、`test_task_registry` | 通过；启动契约与依赖静态检查，不代表全部系统部署 |
| F02 | 线阵场与加权 | 已知阵列满足解析/显式复数和；非法输入拒绝 | `test_core`、`test_configuration`、独立复数求和探针 | 通过；4个手算/复数和场景最大场误差3.31e-14 |
| F03 | 三维与3dB指标 | 相对指标不随共同场幅度/相位缩放改变 | `test_linear3d`、`test_pattern_metrics`、`test_metric_scale_audit` | 通过；新增13项尺度/解析宽度用例 |
| F04 | 三类馈电阻抗 | 并联、串行、对称串行满足目标分配和相位 | `test_feed`、`test_electrical`、`test_pointing`、`test_power_scale_audit` | 通过；新增19项功率尺度/输入边界用例 |
| F05 | 支路电长度 | 延迟进入对应路径和输出相位，旧记录仍可读 | `test_branch_delays`、`test_branch_delay_cli`、`test_branch_compatibility` | 通过；数值、格式、CLI相关回归 |
| F06 | 线阵到馈电交接 | 使用已算快照，保持所选阵列与指向 | `test_linear_feed_import`、`test_feed_pattern_selection`、`test_series_terminal_diagram` | 通过；交接与终端连接图检查 |
| F07 | 平面阵馈电 | 规则/任意坐标及目标角正确进入电气计算 | `test_planar`、`test_pointing`、`test_electrical` | 通过；有限测试布局，不代表任意连续孔径全覆盖 |
| F08 | 二维稀疏数值综合 | 方案满足已声明计数、约束及复核规则 | `test_sparse_numeric`、`test_sparse_phase`、`test_sparse_elements` | 通过；测试场景覆盖，非全局最优证明 |
| F09 | 任意网格及资源 | 合法奇偶网格/K可用，超限明确拒绝 | `test_sparse_arbitrary_size`、`test_sparse_resources`、`test_sparse_solver_resources` | 通过；含无效、边界及资源保护 |
| F10 | 稀疏记录/显示 | 历史快照恢复、工况/曲线选择、缓存失效一致 | `test_sparse_history_store`、`test_sparse_integration`、`test_sparse_overview_plot`、`test_sparse_plot_cache` | 通过；临时历史库与图形检查 |
| F11 | 子阵级优化 | 固定模板下中心优化及编码/约束行为保持 | `test_subarray`、`test_subarray_encoding` | 通过；未新增模板或优化目标 |
| F12 | MIMO几何与波束 | Tx/Rx通道对、虚拟位置/重数及数字波束一致 | `test_mimo`、`test_mimo_io`、`test_mimo_cli` | 通过；不把虚拟通道数当硬件增益 |
| F13 | FMCW及A/B | 参数、时序、编码与所选组输出一致 | `test_waveform_core`、`test_waveform_ab`、`test_waveform_integration` | 通过；保留已有独立处理规则 |
| F14 | 两类距离解模糊 | 保存所有有效候选和失败/多解状态 | `test_waveform_ambiguity`、`test_waveform_pri` | 通过；只对已实现有限搜索域有效 |
| F15 | 雷达天线与增益 | 阵列设计、有限快拍判据和预算分项一致 | `test_waveform_antenna`、`test_waveform_gain`、`test_waveform_doa`、`test_waveform_sparse_design` | 通过；不声称硬件角分辨率保证 |
| F16 | 相控阵分析/综合 | 保留当前1.2.2修复的场、约束及可行性口径 | `test_phased_service`、`test_phased_audit_122` | 通过；没有把来源浏览器验收算作本轮GUI检查 |
| F17 | 雷达方程/信号/统计 | 独立预算、接收处理、CUT及旧校准边界一致 | `test_equation_integration`、`test_equation_sync_101`、3个`test_equation_audit_*` | 通过；固定来源反例/适配检查 |
| F18 | 脉冲处理与工程 | 已实现体制、场景/工程及数组输出兼容 | `test_pulse_integration`、`test_radar_sync` | 通过；非全部体制组合穷尽验证 |
| F19 | 共享三维场景 | 真实渲染、旋转/播放、隐藏暂停、视角记录一致 | `test_pulse_scene_view`、`test_waveform_scene_desktop`、`test_fmcw_scene_model` | 通过；使用本机Windows Qt/WebGL；不把offscreen当WebGL实测 |
| F20 | 保存/历史/坏文件 | 成功结果原样恢复，失败保留原文件与界面状态 | `test_history_io`、`test_linear_history_store`、`test_simulation_desktop`、`test_upstream_sync` | 通过；临时库、实际旧记录及异常注入 |
| F21 | 数值文件完整性 | 不接收溢出JSON、不丢失/错读数组、摘要与读取同代 | `test_simulation_io`、`test_audit_state` | 通过；本轮16项新增数据边界/竞态用例 |
| F22 | 并行与取消 | 同输入保持结果，取消/失败回收本任务进程 | `test_parallel`、`test_sparse_process`、`test_simulation_integration` | 通过；含真实子进程及模拟错误，不代表所有崩溃时序 |
| F23 | 导出与跨模块复用 | 输出对应已选结果，原子发布，旧算法标识保留 | `test_snapshot_contracts`、`test_radar_bridges`、`test_upstream_sync`、`test_simulation_integration` | 通过；版本、摘要、选择和原画面等检查 |
| F24 | 图形交互/布局/导航 | 缩放、转动、恢复及切页不导致状态混用 | `test_interactive_canvas`、`test_resize_canvas`、`test_plot_layout`、`test_footer_layout_cache`、`test_surface_lifecycle`、`test_navigation`、`test_smoke_suites` | 通过；现有窗口尺寸与图形场景，非全部DPI/显卡组合 |

## 3. 基线失败与确认缺陷

### 3.1 既有测试基线

基线收集并执行75个文件、1959个用例，5个断言失败分布于4个文件。其中`test_simulation_desktop.py`还在进程退出阶段发生访问异常。不能将这些简单统称为5个算法错误。

| 基线失败 | 定位 | 处理 |
| --- | --- | --- |
| `test_six_numerical_kernels_keep_frozen_source_hashes` | 测试仍按旧6模块清单，现来源已经包含第7个`receiver_spec` | 按现有固定源清单核对7模块，未为迎合断言修改算法 |
| 两个`test_desktop_bootstrap_preserves_order_settings_and_factories`参数化用例 | 模拟Qt枚举未包含已有共享OpenGL初始化要求 | 补齐mock与应有启动顺序期望 |
| `test_saved_record_loads_without_solver_even_if_current_inputs_are_invalid` | 保存记录已包含既有`view_state`，旧断言只比较无视图状态的结构 | 与实际记录快照合同比较，仍验证不运行求解器 |
| `test_failed_record_restore_preserves_operation_and_extended_scene_time` | 场景已由Three渲染，原故障注入不再经过真实恢复路径 | 改为实际表格恢复路径触发失败，检查完整回滚 |
| Qt退出异常 | 混用offscreen与实际WebEngine测试上下文，清理路径不足 | 采用本机Windows平台、共享GL及显式场景shutdown；60项单独复测和最终全文件回归均退出0 |

### 3.2 六类产品缺陷与修复证据

| 编号 | 根因与修前复现 | 修复 | 修后证据/兼容影响 |
| --- | --- | --- | --- |
| D01 | 球面半功率宽度先计算`abs(F)**2`。对`F(a)=A cos(a)`，A=1e-200时平方下溢，误报零场；A=1e200时溢出并使求根失败。解析结果本应均为90° | 以共同尺度稳定表示场幅度，按幅度阈值`1/sqrt(2)`求交点，保持原半功率定义 | `test_metric_scale_audit`13项通过；还覆盖复相位、1e±308及半波长双阵元60°解析宽度 |
| D02 | 端口功率直接按`A²/sum(A²)`计算，极大/极小共同尺度使等分功率变NaN，阻抗与激励无法可靠输出 | 先除共同最大幅度再平方归一化；阻抗设计和实际激励共用一处校验/功率实现 | `test_power_scale_audit`19项通过；等分2端口p=0.5、50Ω参考下变换阻抗50√2Ω；3拓扑在27°仍满足独立相位式。无法以正数表示的相对功率明确拒绝 |
| D03 | JSON仅拦截NaN/Infinity标记，`1e999`仍经默认浮点解析成为inf | 添加有限浮点解析，JSON文件和NPZ元数据使用同一严格入口 | JSON/NPZ×正/负溢出4项修前失败、修后通过；最大有限数与最小正次正规数保留 |
| D04 | 写NPZ时把数组名作为`np.savez`关键字，`allow_pickle`被当选项丢失，`file`冲突；读取`raw`和`raw.npy`时NpzFile别名规则导致两者读成同一数组 | 用标准ZIP/NPY逐成员写读，按真实成员名定位；显式禁止对象pickle | 名称碰撞4项修前失败、修后通过。合法数组名不再被库选项吞掉；标准文件格式不变 |
| D05 | 名称含NUL时ZipInfo截断名称，写入成功却不能读回；覆盖路径可能替换原有效文件 | 在创建临时写入前拒绝NUL数组名 | 新回归修前1项失败；修后拒绝并验证原文件逐字不变、无残留临时文件 |
| D06 | sidecar先哈希数组文件，关闭后重新按路径读取；原子替换可使校验与读数组属于不同代 | 哈希、ZIP预检及数组读取持有同一打开文件代 | 修前真实复现：已校验原cut值−300，返回替换值123。修后读到同一代原数组；Windows替换被文件句柄阻止，支持替换的平台也保持原打开代 |

另补齐同一序列化合同的边界：写入成员数与读取上限一致，允许当前写入器产生的NPY3中文结构化字段且先检查头/尺寸，保留Fortran顺序和调用方文件句柄所有权。对象数组仍拒绝；读入预算、重复字段及损坏数据检查没有放松。

修前证据在`tmp/audit-math-027/repro-before.log`（4个数学反例失败）、`tmp/audit-state-027/before-fix.log`（9项数据反例失败）、`tmp/audit-state-027/nul-before.log`（1项失败）。这些新反例在基线原测试之外，不混入基线5个过时断言的统计。

## 4. 修复与重构分开实施

1. 固定0.26.1基线，完成原测试和独立探针；保留失败日志及原实现副本。
2. 修复D01/D02后先执行32项新数学用例；修复D03/D04/D06后先确认9个原失败用例通过，再覆盖相关IO边界。NUL问题单独复现、增加一例并修复，未掩盖写入失败。
3. 在修复已验证之后提取职责：
   - `feed.excitation._relative_port_power`统一幅度校验与稳定相对功率，供同一业务内阻抗/激励路径使用，消除两份易漂移的公式。
   - `shared.plotting.metrics._great_circle_half_power_edge`集中交点搜索及精化，主函数保留坐标/状态组织。
   - `shared.simulation_io._npy_header`负责有界头预检；文件写读保留标准编码和事务语义。
   - `application.simulation._worker_runtime`统一计算/导出子进程命令与环境，避免冻结/源码两种启动路径配置分叉。
4. 执行受影响回归：数学相关245项通过；IO相关46项加NUL后续1项通过；worker/export相关16项通过。该阶段数量与最终2007有重叠，不重复累计。
5. 最终按文件隔离执行全部可用测试，78个文件2007项通过，随后由发布阶段执行静态/打包检查。已执行的静态边界检查为281个模块、616条导入边，无循环或反向依赖；它不是动态运行时依赖穷尽证明。

没有增加运行依赖或额外架构层级。`framework/process.py`的Job/取消管理在审查中未确认新的产品缺陷，本轮未修改它。原子保存故障测试改为实际fsync/replace失败注入，不再绑定已移除的NumPy关键字写入实现。

## 5. 性能比较

同一Windows主机、Python3.14.2、NumPy2.5.4，`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。比较原0.26.1实现与修复/重构实现，输入和采样规模相同。未固定CPU亲和性或隔离全部背景负载，不能把小幅差异解释为所有机器的性能保证。

| 路径与规模 | 修前中位耗时 | 修后中位耗时 | 变化 | 内存测量 |
| --- | --- | --- | --- | --- |
| 余弦场宽度，每侧3000次 | 0.2197 ms | 0.2057 ms | −6.37% | tracemalloc记录的Python/NumPy峰值28,356→28,025 B |
| 32元ULA宽度，每侧1000次 | 1.0936 ms | 1.1471 ms | +4.89%，约+0.054 ms | 同口径30,536→27,696 B |
| 32端口串馈，每侧50次 | 19.1835 ms | 19.7546 ms | +2.98%，约+0.571 ms | 同口径904,133→903,941 B |
| 24MiB未压缩NPZ读取，预热1次后测5次 | 10.4531 ms | 9.9592 ms | −4.72% | 进程RSS峰值增量约24.18→24.19 MiB |
| 同数据压缩NPZ读取，预热1次后测5次 | 84.3819 ms | 81.1942 ms | −3.78% | 进程RSS峰值增量约24.56→24.70 MiB |

数学路径使用tracemalloc可见分配口径，不是整进程RSS；IO采用独立新进程并在读取期间与返回时采样RSS，不能把两种内存数字横向比较。ULA与串馈出现小幅中位耗时增加，采样区间重叠；没有掩盖这项代价，也不据此宣称整体性能完全无退化。当前样本未显示数量级退化，但短样本不足以判定长期内存泄漏；长期运行、超大阵列及不同磁盘/显卡仍需专项验收。

原始测量在`tmp/audit-math-027/paired-benchmark-final.json`和`tmp/audit-state-027/performance.json`。修前还独立核对4个线阵的显式复数和，最大场差约3.31e-14；小孔径反相双源方向性与解析极限差约1.64e-10 dB。三个馈电拓扑在0°、27°、−35°的选定样例中目标与最终切面最大差不超过4.63e-11 dB。这些样例没有发现额外错误，不伪造为新的修复项。

## 6. 兼容性、假设与未覆盖事项

### 6.1 兼容性影响

- 保留任务、CLI参数、JSON/NPZ/CSV主格式、配置字段及历史存储路径。旧数值记录不被批量改写；受极端尺度缺陷影响的旧计算应重新运行。
- 改变的行为有明确缺陷依据：极端共同幅度现在得到有限结果；无法以正数表示的相对端口功率、NUL名称、浮点溢出JSON及超出读写数量预算的写入明确失败。
- 标准NPZ仍可由支持相应NPY版本的工具读取。ArrayLab按精确成员名读取碰撞名称；第三方若自行使用NumPy的关键字/别名便利接口，仍应遵循该工具自身的名称规则。NPY3中文字段是标准格式兼容，不是新的业务结果schema。
- `read_npz`增加调用方拥有的可寻址二进制流支持；原路径调用保留，外部传入流不会被关闭。哈希后并发替换不再使读取混用文件代。

### 6.2 假设与待确认事项

1. 本轮没有取得新增正式需求规格或业务规则变更，因此以仓库文档、代码合同和现有测试为基准；数学修复采用相对尺度不变量和解析参考，不调整优化目标/阈值。
2. 阵列间互耦、矢量极化、真实天线/馈线结构和硬件非线性没有新增模型；方向性、数字通道或合成统计不能自动等同硬件指标。
3. 稀疏/子阵/相控综合仍是有限搜索与有限工况复核，不证明全局最优、连续全频带/全扫描域均达标。
4. 用户历史数据库和真实工程未纳入样本；测试使用隔离目录、合成输入及明示来源的旧版fixture。没有批量打开/迁移用户数据作验收。
5. 未验证干净系统、Linux/macOS、全部DPI/显卡、断电/磁盘损坏、长期压力、每种权限/杀毒干预或所有进程崩溃时序。异常注入证明所列路径，不代表操作系统故障穷尽。
6. 最终2007指当前可收集测试集，不是完整需求覆盖率百分比。本轮10模块中，深入独立算法审查集中在馈电与球面指标；其他算法沿用已知反例和已有回归，未逐公式重新推导。
7. 本报告写入时源代码最终回归和静态检查已完成。正式二进制的完整原生界面验收、PDF逐页视觉审查、GitHub上传/下载摘要核对由发布阶段另行记载；不能从本报告预先推断其已通过。

## 7. 可复核交付

- 修改代码：`features/feed/{model,excitation}.py`、`shared/plotting/metrics.py`、`shared/simulation_io.py`、`application/simulation.py`。
- 新回归入口：`test_metric_scale_audit.py`、`test_power_scale_audit.py`、`test_audit_state.py`；其实现与各业务/公共模块就近保存。
- 本报告的机器可读伴随文件：[逐文件结果及功能矩阵](audit-0.27.0-results.json)。包含基线与最终结果、功能映射和性能摘要，未把中途结果当最终结果。
- 最终回归原始清单：本地`tmp/audit027/baseline/results.json`、`tmp/audit027/final/results.json`及相应逐文件日志；发布源代码含测试实现，可按根测试入口复验。

示例复验命令：`python -m pytest tests/test_metric_scale_audit.py tests/test_power_scale_audit.py tests/test_audit_state.py -q`。全套使用`python tools/run_regression.py audit-run`隔离子进程和历史目录；不要因单个无GUI环境无法执行桌面测试而声称其通过。推送后不重新运行软件测试，只执行发布完整性核对。

## 8. 最终发布验收补记

正式Windows10包一次通过既有all的20组1181检查点，210.34秒；另通过新雷达记录/子进程19项、FMCW/脉冲共享场景15项及无头9项。主程序314个、CLI202个模块的字节码与当前源码一致。完整桌面脚本原有根节点、任务数量和滚动/绘图缓存断言属于过时验收约定，已依据现有UI合同修正；没有据此修改业务行为。该层结果与2007项用例分开计数。

使用说明书PDF共38页，全部渲染与中文文本/嵌入字体核对通过；截图标明来源版本。PDF、源码ZIP与EXE同目录，也独立作为Release附件。软件检查均在推送前结束，公开下载只核对字节摘要；具体结果见同目录机器可读JSON的发布验收摘要。
