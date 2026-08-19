# R2 高分辨率抽帧 Agent

## 目标

R2 判断“电压表是否并联在待测电阻两端”。Agent 只负责从当前视频生成可供模型和本地融合器使用的证据，不按视频编号、文件名、历史结果或 Excel 选择路径。

`pass` 的直接视觉证据是同一帧中：电压表可见、待测电阻可见、电压表两根线可追到待测电阻的两个不同端点。导线被遮挡、出画或路径无法连续时，证据质量下降，但仍由统一的二分类 reducer 输出 `pass` 或 `fail`，并把原因放进诊断字段。

## 周期构建

Agent 将 `measurement_1/2` 和 `recording_1/2` 统一为 `observation_recording_cycle_1/2`：

- 只检测到 measurement 时，以 measurement 为锚点，向前后扩展；
- 只检测到 recording 时，同样建立周期，标记相邻 measurement 可能未被分段器标出；
- 两个阶段都存在时合并连续范围；
- 初始窗口前后扩展 3 秒；首轮质量不足时，在当前周期内扩大到前后最多 8 秒；
- 不跨越接线、重接线、下一个周期或整理阶段；没有周期时使用统一 broad search。

## 源帧和清晰度

源视频按原生解码逐帧扫描，当前五个源文件是 3840×2160、30 fps。每个周期首轮扫描每一帧，再按时间分散、清晰度、运动模糊、曝光和动态目标候选排序，默认保留每周期 10 组图像（可由 Skill 参数控制）。

每个选中的真实帧保存：

- `native_4k/*.png`：原生全景；
- `native_4k_enhanced/*.png`：CLAHE 和轻度锐化图；
- `joint_topology_native/` 和 `joint_topology_enhanced/`：联合拓扑 ROI；
- `voltmeter_candidates/`：当前帧动态发现的电表候选；
- `resistor_candidates/`：当前帧动态发现的电阻候选。

原图始终保留，增强图只用于提高可见性，不替代原始证据。所有 ROI 都保存归一化坐标和相同的 `frame_number`/`frame_id`，防止把不同时间点拼成一张拓扑图。

## 动态定位

`detect_dynamic_object_boxes()` 每帧重新计算：

- Hough 圆和轮廓矩形产生电表候选；
- 细长矩形和轮廓填充率产生电阻候选；
- 所有候选的并集生成联合拓扑 ROI；
- 候选不存在时保留整帧，不把定位失败改写成学生操作失败。

定位器没有 `video_id` 参数，也没有按编号登记的坐标表。

## 模型输入

R2 的 Qwen 组图使用 lossless PNG，模型输入最长边上限为 4096；每组发送原生全景、增强联合拓扑、排名第一的电表候选和电阻候选。其他原始/增强候选仍全部落盘。请求按单个 `image_group` 发送，再按真实 `frame_id` 合并，避免单次请求包含过多 4K 图片。提示词要求模型只返回可见观察，不直接生成 `pass/fail`。

## Skill

正式调度注册：

- `voltmeter.parallel_endpoint_adaptive`：当前 run 检测到 measurement 或 recording 周期；
- `voltmeter.parallel_endpoint_broad_search`：没有阶段锚点时的统一 broad search。

两者都使用 `run_remaining_rubrics` 生产 R2 证据，参数实际控制周期边界、每周期组数、质量扩展阈值、ROI 模式和融合策略。旧 `stable_meter.*` 仍保留用于历史 replay/regression，不进入新版 live R2 路径。

## 运行产物

当前 run 下会生成：

```text
runs/<run-id>/remaining_rubrics/
  rubric2_live_frame_manifest.json
  rubric2_live/
    native_4k/
    native_4k_enhanced/
    joint_topology_native/
    joint_topology_enhanced/
    voltmeter_candidates/
    resistor_candidates/
```

manifest 记录周期、扫描帧数、选中帧、清晰度和动态 ROI；它只描述当前 run，不作为后续视频的历史输入。

## 限制

超过 3840×2160 的尺寸只能是插值放大，不能增加真实细节。高分辨率降低了压缩和缩放损失，但不能解决目标被完全遮挡、导线出画或原始视频失焦。此时 Agent 仍输出二分类结果，同时保存低置信度、可见性和模型原文供诊断。

## 2026-08-18 真实帧冒烟验证

使用视频 8 当前五阶段结果中的两个 observation/recording cycle 运行了不读 Excel、不请求 Qwen 的取证冒烟。周期 1 被边界裁剪为 `106–114s`，逐帧扫描 241 帧；周期 2 首轮窗口为 `126–141s`，逐帧扫描 451 帧。测试配置每周期保留 2 组，共生成 4 组 `3840×2160` 原生证据。正式默认值仍是每周期 10 组。

结果位于 `outputs/r2_actual_cycle_smoke_20260818/`。该运行只验证阶段裁剪、逐帧扫描、动态 ROI 和原生文件落盘，不是 R2 准确率评测。
