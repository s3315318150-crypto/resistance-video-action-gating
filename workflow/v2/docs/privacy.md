# Data and privacy boundary

## 禁止提交

- 学生或实验者的原始视频、音频、截图和 ROI；
- 姓名、学号、原始文件名或可反向识别个人的信息；
- Excel 标注、专家分数、错误说明和留出集真值；
- API Token、私有服务地址和认证日志；
- 原始模型响应、历史预测、评测输出和本机绝对路径；
- `.venv/`、缓存及临时目录。

## 允许提交

- 与个人无关的通用算法和配置；
- 合成或获得明确公开授权的匿名示例；
- 不含媒体和标签的结构示例；
- 可复现的本地测试。

## 发布前检查

```powershell
git status --short
git ls-files
git grep -n -i -E "api[_-]?key|token|authorization|bearer|https?://"
git grep -n -E "[A-Z]:\\\\|\.mp4|\.xlsx"
```

还应人工检查 Git 历史，而不只是当前工作树。若敏感内容曾被提交，删除当前文件并不能清除历史，应重建干净历史或使用专门的历史清理工具。
