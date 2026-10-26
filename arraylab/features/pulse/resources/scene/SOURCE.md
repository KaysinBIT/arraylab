# 离线脉冲雷达场景

场景形式及主体渲染逻辑来自用户提供的 PulseLab 1.0.1 `web/src/scene.ts`。保留深蓝背景、透视相机、带阻尼的 OrbitControls、雷达平台与天线板、点目标、半透明波束、虚线航迹、光照地形和悬浮文字。

仅嵌入场景显示；不包含原网页工作台、HTTP/API服务器或计算后端。计算模型仍由 ArrayLab Python 组件负责。显示从雷达坐标 `(x前向,y左向,z向上)` 映射为 Three.js 的 `(x,z,-y)`，不修改工程坐标。

适配新增明确的离线接口、按需渲染、隐藏暂停、原地更新实体和资源释放。关键帧姿态缺省值按Python数值模型逐帧回退；离地高度轨迹加上地形高程，无效高程保留未知。DEM高度采样保留完整工程栅格，显示网格可以降低分辨率。`png()`仅导出当前图形，动画不重算信号。

依赖 Three.js 0.180.0，MIT许可证见 `THREE-LICENSE.txt`。发布包已有编译好的 `scene.bundle.js`，运行时不需要Node.js、网络或原PulseLab目录。可用安装了three 0.180.0与esbuild的node_modules重建：

```text
node tools/build_pulse_scene.mjs --deps PATH_TO_NODE_MODULES
```

`window.ArrayLabScene`接口：`ready`、`status()`/`getState()`、`setScene({project,grid,time_s,max_time_s,reset_camera,caption})`、`setTime(seconds)`、`setPlaying(bool)`、`setVisible(bool)`、`resetCamera()`、`restoreCamera({position,target,zoom})`、`png()`、`dispose()`。`status()`返回支持状态、错误、时间、播放/可见状态、相机、实体/网格数量与渲染计数；配置和数值输入不写回Python。
