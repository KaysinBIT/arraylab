# 传播模型来源与验收边界

本模块采用复基带功率表示。路径系数、自由空间扩散、传播相位、气体／雨衰和天线增益分别处理，每项只计一次。标准模型在本地计算，不需要运行时联网。

## 气体吸收：ITU-R P.676-13

- 版本：2022 年 8 月，Annex 1 的逐谱线方法，公式 (1)–(9)；均匀地面路径使用公式 (10)。
- [ITU 官方版本页](https://www.itu.int/rec/R-REC-P.676-13-202208-I/en)；[官方英文 PDF](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-13-202208-I!!PDF-E.pdf)。核验日期：2026-10-26。
- 下载 PDF 的 SHA-256：`8c09b2d2c120bdae33f60c2a7abff873374d54075ce0dc71060de8a5507bbe2f`。
- 系数直接从此版本的表 1、表 2（印刷页 8–10）提取：44 条氧气谱线、35 条水汽谱线；保留 1780 GHz 的湿连续谱伪谱线。公式页 3、4、7、10 已渲染核对。不是以旧版系数文件改名代替 13 版。
- `equationlab/data/p676_13_oxygen.json`、`p676_13_water.json` 内保留版本、表号、原文 URL、PDF 哈希及列含义。
- `pressure_hpa` 指**干空气气压** p。水汽分压 e=ρT/216.7，总气压为 p+e；不能把同一个字段又当总气压减一次 e。`temperature_k` 是传播介质气温，与接收机噪声温度不同。
- 模块支持 1–1000 GHz；频带任一点越界均报错。频率可为数组，窄带与扫频使用同一完整谱线方法，无 60 GHz 常数损耗近似。

## 雨衰：ITU-R P.838-3

- 版本：2005 年 3 月，公式 (1)–(5) 与表 1–4；结果独立对照表 5 的四舍五入数值。
- [ITU 官方版本页](https://www.itu.int/rec/R-REC-P.838-3-200503-I/en)；[官方英文 PDF](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-3-200503-I!!PDF-E.pdf)。
- 下载 PDF 的 SHA-256：`3ab7482993e51fc63c5127a72e9e8930614e73652ac614882760817e7c1469cb`。
- 输入雨率 mm/h，频率 GHz，输出 dB/km；极化倾角相对水平为 0°（水平）、90°（垂直）、45°（标准公式中的圆极化）。根据每条路径的仰角计算混合系数。
- `path_attenuation` 返回幅度因子 10^(−A/20)，输入是已包含去返的完整长度；不再自动乘二。`attenuation_for_path` 分别按发射腿和接收腿的长度与仰角累计。
- 气体与雨衰均采用沿路径均匀的用户指定气象条件；不包含大气辐射噪声、色散、折射、地球曲率、云雾衰减或年可用度预测。

## 独立数值参考

[ITU-Rpy](https://github.com/inigodelportillo/ITU-Rpy) 是测试参考，不是运行依赖。核验使用确实包含 `_ITU676_13_` 的固定提交 `f739993c4b6d34076de22249ef53d03fa5a53d73`：

- [参考实现](https://github.com/inigodelportillo/ITU-Rpy/blob/f739993c4b6d34076de22249ef53d03fa5a53d73/itur/models/itu676.py)，下载文件 SHA-256：`d4d1b9dd4c8030e1ade7d6e89c28029c4389668fcbbd4ad112c61fab72db65ac`。
- 该提交的 `v13_lines_oxygen.txt` 与 `v13_lines_water_vapour.txt` 同上述官方 PDF 提取结果逐值完全一致；前者 SHA-256 为 `fe670186e4e92426f448ed2122ee9e07013c6dd198090f3a5a6eccd515354048`，后者为 `5ad4d989295bd75d6164ba5e1487467ecbbd3d12dffff9508c60273e85642fce`。
- 在独立命名空间中执行参考文件的 `__gamma0_exact__` 与 `__gammaw_exact__`，加载其独立系数文件，生成 64 组离线参考值；没有调用 EquationLab 的计算函数生成期望值。
- 覆盖 1–1000 GHz、22/60/118/183/557 GHz 附近谱线、4 组干压/温度/湿度。保存于 `tests/fixtures/propagation_itur13_reference.json`，正式测试不联网。
- 参考软件为 MIT 许可：[固定提交许可](https://github.com/inigodelportillo/ITU-Rpy/blob/f739993c4b6d34076de22249ef53d03fa5a53d73/LICENSE.txt)。本项目不分发其源码或作为依赖打包，保留来源归属；ITU 数值表是实现所需的标准事实数据。ITU 原文保留自身版权，不随程序分发完整 PDF。

## 路径、反射与 Doppler

- 自由空间返回一个双程直达路径；平地模型返回直达–直达、反射–直达、直达–反射、反射–反射四项。两个交叉路径即使长度相同，也因出入射方向不同而分别保留。
- 阵列端 `departure`、`arrival` 都是从雷达看向真实或镜像目标的单位向量。它们用于阵列导向，并非入射波传播方向；不能把 arrival 再无条件反号。
- 路径 Doppler 采用几何总路径长度变化率／波长，正值为远离。标量径向速度对应沿直达 LOS 的三维速度；镜像目标的垂直速度反号。测试通过独立中心差分核对四项的 Doppler。
- 平整介质地面的 Fresnel 系数使用复介电常数 εr−jσ/(2πfε0)，电场时间约定 exp(+jωt)。水平为 s 极化、垂直为局部 p 极化；标准垂直系数在法向入射的符号来自局部基矢约定，不应再人为反号。默认模型是标量共极化近似，不是完整 Jones 矩阵极化传输。
- `coefficient_for_path` 在实际波形频率计算频变 Fresnel 系数；`path['coefficient']` 是载频处方便预算的系数。二者择一使用，不能相乘。载频传播相位由信号引擎单独施加一次。
- 自定义路径始终保留直达项。新增项指定相对直达的额外时延、损耗、相位和附加 Doppler；两腿各分配一半额外路径长度，方向由用户指定。它是明确的参数化模型，不能据此声称重建了真实反射几何；距离扫描时额外参数保持不变。

方法参考：[MathWorks 两径传播与方向](https://www.mathworks.com/help/radar/ref/tworaychannel-system-object.html)、[往返多径与鬼影](https://www.mathworks.com/help/radar/ug/simulate-radar-ghosts-due-to-multipath-return.html)。

## 自动验收项目

气体独立参考、雨衰官方表 5、极化与仰角、60 GHz 频带变化、功率/幅度 dB 换算、双程长度只计一次、无反射介质退化、Fresnel 法向/Brewster/掠射极限、四路径方向与相位、宽带反射、真实几何 Doppler 有限差分、自定义附加路径以及无效输入拒绝。信号引擎的多径叠加、匹配滤波、检测统计属于独立的集成验收。
