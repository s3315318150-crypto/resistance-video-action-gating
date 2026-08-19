# Agent Rubric 算法手册

本文档是当前 Agent 十个 Rubric 的统一技术入口。每项都按同一结构说明：评分目标、使用阶段、抽帧方式、动态 ROI、OpenCV/Qwen 分工、本地融合、输出和限制。

当前正式范围：

```text
R0, R1, R2, R3, R4, R5, R6, R7, R8, R9
```

每项主结果固定为 `pass` 或 `fail`。置信度、遮挡、冲突、候选帧和模型原文只用于诊断。

## 1. Agent 的统一数据流

```text
当前视频
  -> create_run
  -> 当前视频动作阶段定位
  -> Rubric 边界精修
  -> plan_live_skills
  -> Rubric 专用抽帧
  -> 当前帧动态 ROI
  -> OpenCV / Qwen 可见事实
  -> 本地确定性 reducer
  -> rubric_<id>.json
```

Agent 不让模型对整段视频直接打分。阶段定位只负责确定“去哪里找”，每个 Rubric 再用自己的视觉证据和规则判断。

### 1.1 阶段语义

Agent 从当前视频识别：

```text
wiring_action
measurement_action
writing_action
cleanup_action
```

再形成接线、第一轮测量/记录、改线、第二轮测量/记录和整理阶段。测量与书写可能连续或重叠，因此一轮周期可以同时保存：

- `measurement_subintervals`
- `writing_subintervals`
- `contains_measurement_evidence`
- `contains_writing_evidence`

阶段容器不能伪造未观察到的动作。阶段缺失时使用统一 `broad_search`。

### 1.2 帧和观察绑定

每张图必须绑定当前 run 的真实来源：

```json
{
  "image_group": 3,
  "frame_id": "current_f00012345",
  "frame_number": 12345,
  "timestamp_seconds": 411.5,
  "source_path": ".../frame.png",
  "roi_path": ".../frame_roi.png"
}
```

`image_group` 是一次模型请求中的证据组，`frame_id` 是当前视频真实帧身份。模型返回的编号必须属于本次请求。全景图和 ROI 只有来源帧相同时才能作为同帧证据。

### 1.3 一次取证、多项消费

| Producer | Rubric | 作用 |
|---|---|---|
| `run_switch_rubric` | R3 | 开关与接线动作重叠 |
| `run_series_rubric` | R1 | 电流表主回路拓扑 |
| `run_meter_rubrics` | R5、R6 | 表头、指针、刻度与量程 |
| `run_polarity_rubric` | R4 | 消费同一 run 的 R5 指针证据 |
| `run_record_rubrics` | R7、R9 | 分轮纸面/表盘比较和 R4/R5/R6 门控 |
| `run_remaining_rubrics` | R0、R2、R8 | 整理、并联和换电池时序 |

共享的是当前 run 已生成的原帧、ROI 和观察，不是以前运行的预测。

## 2. R0：整理归位

### 目标

判断学生是否在实验结束时执行了明确的器材收拢、移动、清理或归位。视频结束、学生离开、画面静止都不能单独证明整理完成。

### 使用阶段

优先使用 `material_cleanup`。阶段缺失时，在当前视频后段和高运动候选中统一宽搜，不读取旧结束时间。

### 抽帧与观察

```text
cleanup 候选窗口
  -> 均匀帧 + 运动峰值
  -> 保留动作前、动作中、动作后
  -> 全景和器材区域观察
  -> 相邻帧状态变化融合
```

OpenCV 负责运动强度、清晰度和候选排序。Qwen 描述手部、导线、电表和器材是否发生收拢、移动或归位。程序检查动作前后是否存在可见状态变化。

### 判定

- `pass`：出现明确整理动作，并形成稳定的归位或收拢状态。
- `fail`：当前视频内没有整理动作，或只有结束画面而没有动作证据。

定位或观察较弱时仍输出二分类，同时降低置信度并保存候选帧。

### 代码与输出

```text
agent/resistance_agent/remaining_rubrics.py
agent/runs/<run-id>/rubrics/rubric_0.json
```

## 3. R1：电流表串联

### 目标

确认电流表真正位于主回路中，而不是直接跨接电源、两端落在同一节点或存在悬空端子。

### 使用阶段

- `circuit_wiring`、`circuit_rewiring`：观察端子插拔和拓扑变化。
- 稳定测量窗口：观察最终使用状态。

接线过程中的短暂悬空不直接判错；接线期明确直接跨电源，或稳定测量期明确非串联，属于强失败证据。

### 抽帧

| 参数 | 默认值 | 作用 |
|---|---:|---|
| 粗扫间隔 | 0.5 秒 | 2 FPS 覆盖阶段 |
| 转场扫描 | 5 FPS | 捕获插拔和端点变化 |
| 稳定帧 | 每阶段至少 2 帧 | 检查最终拓扑 |
| 补证轮次 | 最多 1 轮 | 处理遮挡和冲突 |
| 补证帧数 | 最多 12 帧 | 限制成本 |

### 动态 ROI

每帧根据电流表面板、端子轮廓、彩色端子区域和导线候选生成 ROI，保存原图、原生 ROI、增强 ROI 和时间绑定。摄像机移动后重新定位，不使用固定坐标。

### Qwen 观察

Qwen 只返回结构化事实：

- `activity_context`
- `path_relation`
- `direct_across_state`
- `final_topology`
- `terminal_evidence`
- `loose_lead_endpoints`
- `direct_observations`
- `derived_observations`
- `confidence`

### 本地融合

1. 过滤阶段外观察。
2. 优先检查接线期直接跨电源。
3. 检查稳定期是否能看到电流表两端位于主回路。
4. 检查悬空端子和非串联路径。
5. 聚类相邻时间点，避免单个模糊帧成为持续违规。
6. 明确违规采用单调规则，后续正常帧不能把它覆盖成通过。

### 判定

- `pass`：稳定证据明确显示包含电流表的单一主回路，且没有强违规。
- `fail`：明确直接跨电源、双端同节点、必要端子悬空或稳定期非串联。

### 输出与限制

```text
agent/resistance_agent/series_rubric.py
agent/resistance_agent/r1_frame_sampling_agent.py
agent/runs/<run-id>/series_rubric/series_evidence_report.json
agent/runs/<run-id>/rubrics/rubric_1.json
```

已知风险是同一模型观察内部可能同时出现“直接跨接”和“完整串联”冲突字段；通过证据也可能缺少清晰的电流表两端。修改时应加强字段一致性，不能只信一个枚举值。

详细文档：[R1 动态拓扑与遮挡取证](./r1_occlusion_aware_dynamic_topology_v8/README.md)

## 4. R2：电压表并联

### 目标

确认电压表两根引线分别连接到待测电阻的两个不同端点。仅看见电压表、学生正在测量或正在记录，都不能替代并联拓扑证据。

### 周期构建

Agent 将同轮 measurement 和 recording 合并为观察周期：

- 只有 measurement：以测量为锚点扩展。
- 只有 recording：建立周期并标记测量可能漏检。
- 两者连续：合并范围。
- 初始前后扩展 3 秒，质量不足时最多扩展到 8 秒。
- 不跨接线、改线、下一轮或整理阶段。
- 无周期时使用 broad search。

### 原生帧扫描

当前视频按原始分辨率逐帧扫描，再按时间分布、清晰度、曝光、运动模糊和目标候选排序。默认每周期保留 10 组：

```text
native_4k/
native_4k_enhanced/
joint_topology_native/
joint_topology_enhanced/
voltmeter_candidates/
resistor_candidates/
```

增强图用于提高可见性，原图始终保留。

### 动态 ROI

- Hough 圆和矩形轮廓产生电表候选。
- 细长矩形、轮廓填充率产生电阻候选。
- 两类候选的并集形成联合拓扑 ROI。
- 无候选时保留整帧，不把定位失败判为学生错误。

### 模型输入

每个 `image_group` 发送原生全景、增强联合 ROI、最佳电压表候选和最佳电阻候选。使用无损 PNG，并要求 Qwen 只描述端点、导线和器件关系。

### 本地融合与判定

- `pass`：同一帧或严格相邻证据显示电压表两线分别到达电阻两个不同端点。
- `fail`：明确接到同一端、错误器件、形成非并联关系或必要端点明确悬空。

不能跨较远时间任意拼接两根导线。一侧清晰、一侧模糊时继续给二分类，但记录低可见性。

### 代码与限制

```text
agent/resistance_agent/r2_frame_sampling_agent.py
agent/resistance_agent/remaining_rubrics.py
agent/runs/<run-id>/rubrics/rubric_2.json
```

4K 只能减少缩放损失，不能恢复完全遮挡或失焦细节。当前还需要更多明确 fail 样本验证召回率。

详细文档：[R2 高分辨率抽帧](./r2_frame_sampling_agent/README.md)

## 5. R3：接线时开关断开

### 目标

判断接线或改线动作发生时开关是否保持断开。关键条件是接线动作和闭合状态在同一帧或同一连续支持段中重叠。

### 基线扫描

所有 `circuit_wiring` 和 `circuit_rewiring` 窗口按 5 FPS 扫描。OpenCV 分析：

- 刀闸桥接几何和柄位置；
- 开关端子状态；
- 插头/导线运动；
- 端子占用变化；
- 闭合状态的连续性。

### 自适应补帧

以下情况申请受限邻帧：

- 动作只出现在窗口边缘；
- 开关状态冲突；
- 端子运动明确但几何较弱；
- 手或导线遮挡 ROI；
- 时间覆盖不连续。

补帧后仍用同一共享阈值重算，不针对某个视频改变阈值。

### 动态 ROI 与误检控制

开关和插头 ROI 每帧重新定位。全画面平移不能当成插头运动；只看到刀闸柄但没有接线动作，也不能构成违规。两帧短暂闭合噪声不能自动扩展成持续闭合。

### 判定

- `fail`：当前接线阶段中，同帧或连续支持段出现 `wiring_active AND persistent_closed`。
- `pass`：接线动作期间开关持续断开，且无强冲突证据。

不同时间的闭合状态和接线动作不能拼接成一次失败。

### 代码与输出

```text
agent/resistance_agent/switch_rubric.py
agent/resistance_agent/opencv_switch_overlap.py
agent/resistance_agent/r3_frame_sampling_agent.py
agent/runs/<run-id>/rubrics/rubric_3.json
```

详细文档：[R3 自适应抽帧](./r3_frame_sampling_agent/README.md)；[R3 压力测试](./r3_stress_suite/README.md)

## 6. R4：正负接线柱

### 目标

判断电表正负接线是否正确。当前实现以同一 run 的真实通电表针方向作为直接证据，不把导线颜色、画面左右或器材颜色硬编码为正负极。

### 一次取证、多项消费

```text
run_meter_rubrics
  -> 当前 run 的原帧、动态表头 ROI、指针和刻度结果
  -> rubric_5.json

run_polarity_rubric
  -> 校验 R5 属于同一 run
  -> 生成 rubric_4.json
```

R4 不重新解码视频、不重新抽帧、不重复请求 Qwen。

### 判定逻辑

- 正常正向偏转：支持 `pass`。
- 明确反偏：支持 `fail`。
- 零位、超量程、未通电或低可见：保留真实原因并降低置信度，不能伪装成“看到接反”。

输出应保留 `source_rubric_id=5`、R5 原始 reason、指针证据路径、`wire_color_used=false` 和当前 run 审计字段。

### 语义限制

R5 fail 不总是极性错误。零位、超量程、未通电和定位失败也可能导致 R5 fail，因此新视频必须单独分析 R4/R5 标签不一致的情况。

### 代码

```text
agent/resistance_agent/polarity_rubric.py
agent/runs/<run-id>/rubrics/rubric_4.json
```

详细文档：[R4 复用电表指针证据](./r4_r5_direct_meter_pointer/README.md)

## 7. R5：正常指针偏转

### 目标

判断通电测量时电表是否出现正常正向偏转，并区分零位、反偏、满量程、超量程和假边缘。

### 当前阶段选择

优先使用当前 `measurement_subintervals`，其次使用同轮测量/记录周期内稳定窗口，最后使用当前周期 broad search。不能因存在 recording 容器就假定测量已发生。

### 动态表头定位

1. SIFT 与匿名电流表/电压表模板匹配。
2. Lowe ratio 过滤特征点。
3. RANSAC 单应性校正。
4. 归一化为 `640 x 520` 表头。
5. 通过 A/V 字样、表盘和端子结构确认表型。

每帧重新定位，不使用固定表心或视频 ROI。

### 导线屏蔽

HSV 检测红导线并膨胀，只把与红色区域接触的相邻黑边加入遮挡 mask。遮挡区域不做虚构性修复，目的是避免导线边缘被 Hough 当成指针。

### arc-to-hub 指针检测

```text
CLAHE -> Gaussian blur -> Canny -> HoughLinesP
```

候选线必须连接上方刻度弧与下方动态 hub，并排除 A/V 字样、短文字笔画、线缆平行边缘和歧义候选。hub 来自当前校正表头。

### 印刷刻度与读数

每帧检测印刷刻度端点，多帧使用 median/MAD 排除离群值，得到端点共识和中位指针位置。表盘按 30 小格换算：

```text
reading = nearest_tick * selected_range_max / 30
```

比例不提前截断：小于 0 表示 reverse 候选，大于 1 表示 overrange 候选。

### 融合

明确 CPU `reverse` 或 `overrange` 可以覆盖弱 pass；明确 `normal_rightward` 只修复特定的“未找到正常偏转”弱失败。Qwen 和 CPU 观察都保留在报告中，最终由本地 reducer 生成 R5 二分类。

### 代码与输出

```text
agent/resistance_agent/meter_rubrics.py
agent/resistance_agent/skills/cpu_tick_meter_reading.py
agent/resistance_agent/skills/r5_r6_dense_meter_state/
agent/runs/<run-id>/rubrics/rubric_5.json
```

详细文档：[R5/R6 动态表头与刻度](./r5_r6_dense_meter_state/README.md)；[文件导航](./r5_r6_dense_meter_state/AI_CONTEXT.md)

## 8. R6：电表状态与量程

### 目标

判断稳定指针状态与所选量程是否适合。R6 与 R5 共用当前测量取证，但 R6 关注状态和量程，不只是是否偏转。

### 状态分类

几何 reducer 产生诊断状态：

```text
zero
reverse
normal
full_scale
overrange
too_low
too_high
appropriate
```

这些不是第三类主结果，最终仍归并为 `pass/fail`。

### 流程

```text
当前测量帧
  -> 表头和指针定位
  -> 刻度端点多帧共识
  -> 量程读取
  -> 指针比例和稳定性
  -> closed_stable_r6_cv_v3
  -> CPU range assessment
  -> R6 二分类
```

### 判定

- `pass`：稳定状态明确合适，指针处于可读范围且量程证据适当。
- `fail`：明确反偏、超量程、`too_low`、`too_high` 或不适合的稳定状态。

直接 `appropriate` 证据可以修复特定的缺少量程观察，但不能覆盖明确异常。

### 代码与输出

```text
agent/resistance_agent/skills/closed_stable_r6_cv_v3.py
agent/resistance_agent/skills/cpu_tick_meter_reading.py
agent/runs/<run-id>/meter_rubrics/meter_evidence_report.json
agent/runs/<run-id>/rubrics/rubric_6.json
```

R5/R6 共享定位和指针计算以减少重复抽帧与请求，但分别保存结果和原因。

## 9. R7/R9：分轮记录与电表状态门控

R7 对应第一轮记录，R9 对应第二轮记录。Agent 从当前视频的 `recording_1/2` 周期动态定位记录纸和双表，分别读取纸面 U/I 与同轮电表读数，不跨轮拼接证据。

最终规则为：

```text
R7/R9 = 同轮纸面与表盘读数匹配
        AND R4 正负接线柱正确
        AND R5 指针正常偏转
        AND R6 电表量程合适
```

R4、R5、R6 任一明确为 `fail`，R7/R9 直接 `fail`。Bundle 自动先运行 R5/R6、R4，再运行 R7/R9；门控只读取当前 run 的 `rubrics/rubric_4.json`、`rubric_5.json`、`rubric_6.json`。结果诊断保存 `meter_prerequisite_gate`、失败项、同轮窗口、证据帧和数值比较。

```text
agent/resistance_agent/record_rubrics.py
agent/runs/<run-id>/record_rubrics/record_evidence_report.json
agent/runs/<run-id>/rubrics/rubric_7.json
agent/runs/<run-id>/rubrics/rubric_9.json
```

## 10. R8：换电池前断开开关

### 目标

检查有序动作链：先断开开关，再把串联电池从两节改为一节，最后重新闭合。

三抽头电池盒：

```text
T0 -- cell 1 -- T1 -- cell 2 -- T2

before: T0-T2       两节
after:  T0-T1/T1-T2 一节
```

### Skill 选择

| 当前情况 | Skill |
|---|---|
| 有改线和 recovery gap | `battery.recovery_episode` |
| 只有初次接线 | `battery.wiring_transition` |
| 阶段不可靠 | `battery.broad_transition_search` |

三种 Skill 只改变 episode 时间来源，后续使用同一动态 ROI 和 reducer。

### 抽帧

- 2 FPS：全景粗筛。
- 5 FPS：核心换线区间。
- 10 FPS：端点接触、开关变化和短转场。
- 核心区间默认前后扩展 10 秒以捕获稳定前后态。

每个 episode 独立判断，禁止跨 episode 拼接开关和端点证据。

### 动态电池 ROI

Qwen 从当前请求全景帧返回 `frame_id`、归一化 bbox、置信度和可见依据。本地校验 frame 归属、bbox 几何和统一置信度阈值，再用 ORB/affine 跟踪邻帧。跟踪失败时重新定位或保留全景。

### 两条观察路径

1. `direct-contact verifier`：手或导线是否真实触及电池端子。
2. `structured topology summary`：开关状态、换接前后端子对、换接完成时间。

端子状态冲突时，只在同一 episode 内使用 terminal-pair verifier。

### 本地时序 reducer

按顺序检查：

1. 换接前开关为 open；
2. 稳定前态为 T0-T2；
3. 存在直接端子操作；
4. 稳定后态变为一节连接；
5. 有效节数 2 -> 1；
6. closed 发生在换接完成后。

任一 episode 完整通过则视频为 `pass`；全部 episode 不满足才为 `fail`。

接口超时造成的低置信度 fail 必须标记为接口兜底，不能宣传成有效视觉命中。

### 代码与输出

```text
agent/resistance_agent/remaining_rubrics.py
agent/scripts/run_rubric8_specialized.py
agent/runs/<run-id>/rubrics/rubric_8.json
```

详细文档：[R8 动态电池 ROI 与事件链](./r8_dynamic_roi/README.md)

## 11. 统一结果结构

```json
{
  "rubric_id": 3,
  "decision": "pass",
  "predicted_score": 1,
  "confidence": 0.82,
  "reason": "...",
  "supporting_frame_ids": [],
  "supporting_timestamps_seconds": [],
  "evidence_quality": "medium",
  "diagnostics": {},
  "selection_basis": "current_video_observed_situation_only",
  "video_id_used_for_routing": false,
  "historical_artifacts_used": false,
  "fixed_video_roi_used": false
}
```

`decision=pass` 对应分数 1，`decision=fail` 对应分数 0。每个结果必须来自当前 run。

## 12. 反过拟合规则

- 不按视频 ID、文件名或学生姓名选择算法。
- 不读取过去时间窗、ROI、预测或人工复核。
- 不把 Excel 真值发送给模型。
- 不使用按视频登记的固定 ROI。
- 相同当前视觉情况选择相同 Skill 和参数。
- 阶段缺失使用统一 broad search。
- OpenCV 定位失败影响证据质量，不自动等于学生错误。
- Qwen 只输出当前图像可见事实。
- 五个开发视频结果只能叫开发集回归。

代码检查：

```powershell
rg -n "if.*video_id|video_38|fixed.*roi|historical|Excel" agent\resistance_agent agent\config.json
```

命中关联字段或审计测试不一定是问题；正式路由分支、历史回退、视频专属图工件和固定坐标必须删除。

## 13. 运行和测试

```powershell
python agent\run_agent.py `
  --scheduler deterministic `
  --mode execute `
  --video-ref data\videos\sample.mp4 `
  --run-id sample_execute

python agent\run_mcp_server.py

python -m compileall -q agent
python -m unittest discover -s agent\tests -v
python agent\run_agent.py --help
git diff --check
```

当前 Agent 测试覆盖十项 Rubric、调度约束、动态证据和发布契约。

## 14. AI 修改顺序

```text
1. 读本手册和 agent/README.md
2. 读 agent/config.json
3. 读 orchestrator.py、toolkit.py、skills/router.py
4. 读目标 Rubric 专项 README
5. 读 producer、reducer 和测试
6. 复现一个具体失败
7. 做最小根因修复
8. 增加针对性测试
9. 跑专项和 Agent 全量测试
10. 使用新 run 做 fresh 验证
```

不要为提高五个开发视频的分数新增视频专属分支、固定 ROI、历史时间窗或结果回退。
