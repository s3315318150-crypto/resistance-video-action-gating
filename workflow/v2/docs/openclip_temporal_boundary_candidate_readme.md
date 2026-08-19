# OpenCLIP 视频时间边界候选算法

## 1. 算法定位

中文名：**OpenCLIP 视频时间边界候选算法**

算法 ID：`openclip-semantic-boundaries-v1`

该算法使用 OpenCLIP 扫描实验视频，寻找画面语义发生明显变化的时间点。它的职责是生成高召回的边界候选，帮助后续 Qwen 缩小搜索范围；它不直接输出伏安法七阶段结果，也不替代成熟 v2 的 Map、Reduce、状态机和边界提示词。

推荐定位：

```text
OpenCLIP 本地扫描
    -> 无标签时间边界候选
    -> 与 v2/Rubric 候选合并
    -> 候选附近 0.5 秒密集抽帧
    -> Qwen 判断动作类别和最终边界
    -> v2 输出七阶段时间段
```

## 2. OpenCLIP 在本项目中的用途

### 2.1 全视频低成本预扫描

OpenCLIP 可以在本地运行，不消耗 Qwen API 调用。它适合先按 1 秒间隔扫描长视频，找出可能发生动作变化的区域，再让 Qwen 只查看少量候选窗口。

### 2.2 边界候选生成

算法比较相隔 1、2、3 秒的图像特征，通过余弦距离寻找视觉语义变点。这些候选可能对应：

- 从接线转为写字；
- 从写字转为重新连线；
- 从操作电路转为整理；
- 人员、器材布局或主要动作发生明显变化。

候选本身没有阶段标签，必须由 Qwen 或 v2 状态逻辑解释。

### 2.3 抽帧去冗余和代表帧排序

相邻 OpenCLIP 特征非常接近时，说明画面内容高度相似。系统可以减少这些重复帧，把有限图片预算留给特征变化较大的时刻，同时仍保留固定间隔基础帧，避免漏掉低运动的读表和书写状态。

### 2.4 补充现有精查窗口

只有当 OpenCLIP 候选落在现有 v2/Rubric 精查窗口之外时，才建议新增一个 Qwen 复核窗口。候选已经位于原精查窗口内时，不重复调用 Qwen，因为没有增加新画面证据。

## 3. 不适合直接承担的任务

当前 OpenCLIP 不适合独立完成以下工作：

- 区分连线、测量、记录、重新连线和整理的最终语义；
- 区分第一次与第二次测量或记录；
- 根据实验逻辑解释学生为什么修改电路；
- 判断一次书写属于第一组还是第二组数据；
- 直接生成最终七阶段时间段；
- 独立完成十项 Rubric 的 `pass` / `fail` 评分。

OpenCLIP 主要衡量静态画面的图文或图像特征相似性，不真正理解完整动作过程。测试中出现过以下典型错误：

- 纸张一直存在，模型便持续倾向于 `writing`；
- 器材静止或普通挪动被误认为 `cleanup`；
- 接线和测量画面外观接近，难以稳定区分；
- 单帧无法可靠区分第一轮和第二轮实验。

因此，OpenCLIP 的动作类别输出只能作为诊断信息，不能覆盖 v2 结果。

## 4. 当前实现

实验代码位于：

```text
experiments/openclip_temporal_segmentation_v1/
```

主要入口：

| 文件 | 作用 |
|---|---|
| `scripts/openclip_semantic_boundaries_v1.py` | 从视频生成单尺度无标签边界候选 |
| `scripts/rescore_multiscale_boundaries_v1.py` | 使用缓存特征生成 1、2、3 秒多尺度候选并集 |
| `scripts/openclip_temporal_segment_v1.py` | 三视图动作 Prompt 分类实验，不作为默认分段器 |
| `scripts/evaluate_rubric_openclip_union_v1.py` | 离线评估 Rubric 与 OpenCLIP 候选池的召回上限 |
| `scripts/summarize_openclip_boundary_runs_v1.py` | 汇总多视频边界覆盖结果 |

默认边界检测参数：

- 模型：`MobileCLIP-S1 / datacompdr`；
- 采样间隔：1 秒；
- 输入视图：完整画面等比补边；
- 特征跨度：1、2、3 秒；
- 候选阈值：视频内变化分数第 60 分位；
- 单尺度候选最小距离：5 秒；
- 多尺度候选合并容差：1 秒。

这些参数对所有视频一致，运行时代码不包含姓名、视频 ID 或固定黄金时间。

## 5. 五视频实验结果

参考边界来自当前确认的 v2 分段，并在 OpenCLIP 推理完成后才用于统计，不进入特征提取或候选生成。

| 方法 | 候选数 | v2 阶段边界 ±2 秒覆盖 | 最近边界 MAE |
|---|---:|---:|---:|
| 单尺度 3 秒特征差 | 195 | 17/20（85%） | 2.10 秒 |
| 1、2、3 秒多尺度并集 | 304 | 18/20（90%） | 1.06 秒 |

多尺度提高了召回率，但候选数量增加约 56%，说明 OpenCLIP 更适合生成候选池，不适合直接把所有变点当作最终边界。

三视图动作 Prompt 分类的逐秒准确率只有：

- sample_001：28.65%；
- sample_004：12.80%。

因此已停止将静态 OpenCLIP 动作标签作为七阶段输出的方向。

## 6. 与 Rubric 候选合并的结果

离线实验在每个 v2 粗边界前后 10 秒保留 OpenCLIP 候选，并与 Rubric 边界建议取并集：

| 指标 | 结果 |
|---|---:|
| Rubric 单候选 ±2 秒覆盖 | 19/20 |
| Rubric + OpenCLIP 候选池 oracle 覆盖 | 20/20 |
| 合并后候选项总数 | 87 |
| 每个边界平均候选数 | 4.35 |

这里的 `20/20` 是候选池的 oracle 召回上限，不是最终分段准确率。统计时使用黄金边界寻找候选池中的最近项；真实运行不能使用黄金答案进行选择。

唯一补中的边界位于 sample_003：

- v2/Rubric 最终建议：321.5 秒；
- 黄金参考：324.0 秒；
- OpenCLIP 候选：322.0 秒；
- 最近误差由 2.5 秒变为 2.0 秒。

但原 Rubric 已经以 0.5 秒间隔检查了 `321–327 秒`，322.0 秒对应画面本来就在 Qwen 输入中。因此当前 OpenCLIP 并没有增加新证据，也没有实际改变最终边界。

## 7. 推荐接入规则

默认流水线暂不启用 OpenCLIP。后续接入应遵循以下规则：

1. 成熟 v2、Temporal Guard、Reduce 和状态机保持不变。
2. OpenCLIP 只产生无标签候选，不决定阶段名称。
3. 对每个 v2 粗边界建立已有精查时间范围。
4. 删除已经落在精查范围内的 OpenCLIP 候选。
5. 只对范围外且变化分数较高的候选新增短窗口。
6. 新窗口按 0.5 秒抽帧，继续使用原 v2 边界提示词。
7. Qwen 没有确认对应阶段转折时，保留原 v2 边界。
8. 输出中记录候选来源、分数、是否新增画面和最终采用原因。

该设计保证 OpenCLIP 只能补充证据，不能破坏成熟 v2 的阶段序列。

## 8. 运行方式

安装实验依赖：

```powershell
pip install -r experiments/openclip_temporal_segmentation_v1/requirements.txt
```

对一个新视频生成边界候选：

```powershell
python experiments/openclip_temporal_segmentation_v1/scripts/openclip_semantic_boundaries_v1.py `
  --video "D:\path\to\video.mp4" `
  --output "outputs\openclip_temporal_segmentation_v1\new_video"
```

使用缓存特征生成多尺度候选：

```powershell
python experiments/openclip_temporal_segmentation_v1/scripts/rescore_multiscale_boundaries_v1.py `
  --run-directory "outputs\openclip_temporal_segmentation_v1\new_video" `
  --output "outputs\openclip_temporal_segmentation_v1\new_video\result_multiscale.json"
```

具体参数以各脚本的 `--help` 输出为准。

## 9. 输出解释

边界结果中的核心字段：

```json
{
  "semantic_boundary_candidates": [
    {
      "seconds": 322.0,
      "change_score": 0.1261997,
      "prominence": 0.0738993,
      "supporting_lags": [1]
    }
  ]
}
```

- `seconds`：候选相对视频起点的秒数；
- `change_score`：候选处的 OpenCLIP 特征变化强度；
- `prominence`：该候选相对邻域变化的突出程度；
- `supporting_lags`：哪些时间跨度共同支持该候选。

这些分数只用于排序和筛选，不代表动作类别概率或最终边界置信度。

## 10. 当前结论

OpenCLIP 对本项目的主要价值是：**本地、低成本地寻找可能的视觉变化位置，并为 Qwen 提供范围外的补充候选。**

当前结果不支持使用 OpenCLIP 直接完成七阶段时间分段。默认方案仍然是成熟 v2/Rubric/Qwen；OpenCLIP 保留为可选的高召回检索层，并且只有在它带来原精查窗口之外的新画面时才值得调用后续模型。
