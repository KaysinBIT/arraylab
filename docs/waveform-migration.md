# WaveLab能力接入来源

## 0.25.0.dev2：同步1.4

本次按用户要求，使用独立固定快照`WaveLab-v1.4-local.zip`，SHA-256：

`2af9b70fee0df942fb9e3fede34b0d83b87c76652e9a9bffd281355682075509`

快照在ArrayLab的忽略目录`tmp/wavelab-v1.4-snapshot`中展开，仅用于移植与差分。没有修改原雷达工程、操作其服务或向原对话发送任务。所有新增实现留在`arraylab/features/waveform/`，遵守本项目framework/application/features/shared边界，不引入运行依赖或新架构层。

| 1.4快照文件 | ArrayLab原生实现 | 接入范围 |
| --- | --- | --- |
| antenna.js | antenna.py | Tx+Rx虚拟几何、重数、方位/俯仰切面、自适应HPBW |
| sparse-design.js | sparse_design.py | 物理Tx/Rx反向搜索、当前阵列验证、双目标/HPBW与四种算法判据、独立holdout |
| doa-superresolution.js | doa.py | 有限含噪快拍、MUSIC/MVDR/LS-ESPRIT、空间平滑、试验明细 |
| doa-dml.js | doa_dml.py | 双角联合DML投影搜索、模糊诊断 |
| gain-design.js | gain.py | ADC能量预算、功率定义、Rx合并、A/B较严组、等效口径 |
| antenna-workbench.js / app.js | settings.py、antenna_controls.py、antenna_views.py、model.py、io.py | Qt输入/图表、配置、记录和CLI；不嵌入HTML |

原先移植的engine.js、ab-waveform.js、ambiguity.js、pri-ambiguity.js在1.2与1.4中逐文件SHA相同，所以原四个Python引擎不修改，也没有重新运行它们的全部领域测试。Node只用于开发差分。

新参数与结果使用schema_version 2，读取ArrayLab旧schema 1；完整记录外层沿用schema_version 1。WaveLab schemaVersion 1/2/3都可导入，新版同时导入该雷达工作台自己的天线几何和设计条件，不自动修改独立MIMO模块。旧版本无法读取新增参数/结果，升级前导出的旧记录保持可读。

F5不运行稀疏搜索或角分辨实验。综合/验证通过独立操作启动，可以取消。GUI在综合条件不变时保留前次综合的独立快照，标明原参考频率和“未重新验证”；该快照随完整记录保存。配置文件不夹带结果，读取配置不触发搜索。

搜索路径存在一个明确的跨语言数值边界：浮点损失在约10⁻¹⁵量级的差异可以改变退火接受分支及随机流消耗，因此同种子不保证得到与JavaScript逐点相同的最终坐标。同一组坐标的指标与算法结果做差分验证；Python内固定种子及单核/双核结果保持一致。没有为了匹配某一示例而固定优化结果。

高分辨实验为已知K=2的有限统计验证，不是Pd、置信下限或硬件分辨率保证；最多48个独立位置，资源超限明确拒绝。物理孔径、等效面积只用于约束/预算，馈线与天线具体尺寸仍暂缓。

## 0.25.0.dev0：首次同步1.2（历史范围）

本轮依据用户确认，仅接入原工具已经完成的波形、FMCW A/B与距离解模糊能力。使用固定压缩包`WaveLab-v1.2-local.zip`，SHA-256：

`16cd6c2e1ad15f7ecb4ba0c15af957711d2427b972b0dee1fd2e07bc08d21ad4`

原项目及对话仍由其原任务管理。本轮未向原任务发送消息、暂停/重启任务、修改工程、操作原静态服务或使用其浏览器状态。只在ArrayLab工作区解压独立副本，并用该副本的纯数值JavaScript作差分参照。

| 快照文件 | ArrayLab独立实现 |
| --- | --- |
| `dist/engine.js` | `waveform_core.py`：FMCW/TDM/DDM指标与单目标频谱 |
| `dist/ab-waveform.js` | `waveform_ab.py`：A/B独立profile、分段/交替编排 |
| `dist/ambiguity.js` | `waveform_ambiguity.py`：ADC折叠观测与候选求解 |
| `dist/pri-ambiguity.js` | `waveform_pri.py`：跨周期时延余数与区间求交 |

界面、任务路由、记录和无头入口按ArrayLab现有Qt工作台实现，没有嵌入原HTML页面。原`antenna.js`未复制或替换ArrayLab已有MIMO；原对话正在迭代的阵列反向设计与增益预算不纳入这份固定快照。

对照验证使用局部算法场景、字段及频谱数组，另有独立物理公式、异常边界与新模块桌面/CLI检查。Node只用于开发时执行复制的参考脚本，不是ArrayLab运行依赖。原压缩包、原完整工程、网页截图、服务和开发日志不纳入公开源码包。
