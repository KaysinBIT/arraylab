# WaveLab能力接入来源

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
