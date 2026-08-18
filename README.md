# 伏安法实验视频理解

本仓库同时保留 Workflow 与 Agent 两条实现路线。两者共享“先定位动作、再按 Rubric 取证、最终输出 `pass` / `fail`”的基本口径，但执行方式不同。

| 项目 | 说明 |
|---|---|
| [Workflow](./workflow/) | 包含原始 V1 和成熟 V2 |
| [Agent](./agent/) | 基于成熟 V2 的实时 Skill 调度、动态抽帧与 MCP 工具编排 |

## 版本结构

```text
.
├─ workflow/
│  ├─ v1/    # 原 GitHub 第一版，完整历史基线
│  └─ v2/    # 第二版，当前推荐 Workflow
└─ agent/     # 基于 V2 复用、微调和增强的 Agent 版
```

- V1 保留原发布内容，便于复现和比较。
- V2 改善阶段容错、终态截断、边界精修与帧级补充搜索。
- Agent 复用 V2 的阶段分割和视觉能力，并改进 Skill 路由、原生帧抽取、动态 ROI、补充取证和结果融合；它不是对 V2 的简单包装。

## 数据边界

仓库不包含学生视频、姓名、Excel 标注、原始模型响应、运行输出或认证信息。视频放入本地 `data/videos/`；V2 与 Agent 的模型连接信息仅通过环境变量配置，V1 则按原发布内容保留。开发视频上的回归结果不能作为新视频泛化准确率。

具体运行方法见 [Workflow V2](./workflow/v2/README.md) 和 [Agent](./agent/README.md)。
