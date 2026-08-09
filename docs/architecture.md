# Architecture

## 设计原则

1. 动作分割只做时间检索先验，不直接判定 Rubric。
2. 每个 Rubric 独立选择证据，不能共用固定少量全景帧。
3. 细节图必须从原始分辨率帧裁剪，再按传输预算缩放。
4. 模型输出是结构化观察；枚举、引用帧和状态映射由本地代码验证。
5. 观察层可以记录证据不足或不确定；评分层必须按 Rubric 的预设冲突规则输出 `pass` 或 `fail`。
6. 预测工件先冻结，标签后读取；留出标签不得参与调参。

## 阶段与消费者

| 阶段 | 主要下游消费者 |
|---|---|
| `circuit_wiring` | 串联关系、并联关系、接线柱、连接时开关状态 |
| `measurement_1` | 指针状态、量程、第一次测量证据 |
| `recording_1` | 第一组纸面记录、记录时稳定拓扑 |
| `circuit_rewiring` | 换接动作、第二轮电路拓扑 |
| `measurement_2` | 指针状态、量程、第二次测量证据 |
| `recording_2` | 第二组纸面记录、记录时稳定拓扑 |
| `material_cleanup` | 拆除和整理动作 |

一个 Rubric 可以消费多个相关阶段，但必须在自己的 manifest 中记录来源阶段、原始时间戳、帧号、ROI 和本地校验结果。

## 工件链

```text
marker_filter manifest
  -> experiment boundary summary
  -> overlapping-window Map observations
  -> global Reduce and local conflict recovery
  -> monotonic seven-stage summary and terminal cleanup barrier
  -> rubric-specific evidence packet
  -> request preflight report
  -> structured model observation
  -> deterministic local decision
  -> frozen prediction manifest
  -> post-freeze offline evaluation
```

`preflight_qwen_request.py` 不调用模型。它检查图片路径、解码、尺寸、传输体积、精确重复、近重复和冗余 ROI。

`qwen_experiment_action_hierarchical_v2.py` 是当前推荐的动作分割入口。Map 只描述直接可见的接线、测量、书写和整理动作；Reduce 去重并解决跨窗口冲突；本地状态机按时间单调映射为七阶段。可信的完成态整理一旦成为终点，之后事件只进入 `ignored_noise_events`，不再回流到有效时间轴。

`run_meter_record_consistency_v1.py` 使用独立的仪表画面读取 U/I，且不把纸面值发送给模型。仪表观察完成后才在本地按量程相关容差与 U1/I1 比较，并生成二分类结果。

## 能力边界

本仓库固化的是研究型取证流水线，不包含真实训练/评测数据，也不声称对所有视频稳定自动评分。任何准确率、覆盖率或泛化结论都必须来自独立、合规且冻结后的评测。
