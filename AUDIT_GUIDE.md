# RAGTruth 正负样本错误模式审计

`audit_negative_patterns.py` 对 RAGTruth 做两级审计：

1. 回答级：区分 `clean` 与 `hallucinated`，统计任务、模型、split、回答长度和错误字符覆盖率。
2. span 级：保留官方标签，并用确定性的启发式规则标记具体错误机制。

这里的“正负样本”统一写成 `clean`/`hallucinated`，避免不同训练代码对
positive/negative 定义相反。

## 运行

项目只需要 Python 3.10+ 标准库：

```bash
python audit_negative_patterns.py
```

指定数据和输出目录：

```bash
python audit_negative_patterns.py \
  --responses Datasets/RAGTruth/response.jsonl \
  --sources Datasets/RAGTruth/source_info.jsonl \
  --output-dir audit_outputs/ragtruth_patterns \
  --max-examples 5
```

默认输出：

| 文件 | 粒度 | 用途 |
|---|---|---|
| `audit_report.md` | 汇总 | 人工阅读任务、标签、错误机制和代表性样本 |
| `audit_report.json` | 汇总 | 程序读取完整统计和样例 |
| `sample_audit.csv` | 每个回答 | 比较 clean/hallucinated 样本分布 |
| `span_audit.csv` | 每个错误 span | 筛选具体错误模式、构造消融子集 |

## 标签体系

审计不会修改 RAGTruth 的官方标签，而是在三个正交轴上记录每个 span：

- `support_relation`：`baseless`、`conflict` 或 `unknown`；
- `severity`：`evident`、`subtle` 或 `unknown`；
- `pattern_tags`：一个 span 可以同时具有多个具体机制；
- `primary_pattern`：按固定优先级选择一个标签，仅用于互斥汇总。

具体机制包括：

| 模式 | 含义 | 对应的图建模检查 |
|---|---|---|
| `unsupported_addition` | 来源中没有的新增信息 | source-support deficit、claim 属性重构 |
| `numeric_temporal` | 数字、日期、时长、数量变化 | 数值节点和 source-copy 对齐 |
| `polarity_negation` | 否定、可用性或大小方向变化 | 带符号的关系兼容性 |
| `entity_attribute_binding` | 实体与属性/数值错绑 | entity–slot–value 异构边 |
| `source_attribution` | 错误 passage/source 引用 | typed source→claim 边 |
| `epistemic_inference` | 可能性、确定性或推断强度改变 | modality 属性重构 |
| `relation_predicate` | 证据存在但谓词关系冲突 | relation-aware edge reconstruction |

启发式规则会优先分析生成 span。本地标注说明中常见的 “not mentioned”
等审计用语不会被误当成模型生成的极性错误。对于 `conflict`，规则也会读取
`Original:` 与 `Generated:`/`AIGC:` 片段，以识别原文和生成内容之间的关系变化。

## 数据质量检查

报告同时检查：

- JSONL 解析失败；
- response 找不到 source；
- labels 字段结构错误；
- span 越界；
- `response[start:end]` 与标注文本不一致；
- 未知官方标签；
- 同一回答内错误 span 重叠。

重叠 span 在错误字符覆盖率中只计算一次。

## 研究使用边界

- 官方标签和启发式模式只用于审计、分层评价和制定消融实验，不进入无监督训练。
- `primary_pattern` 是便于统计的单标签投影；分析复合错误时使用 `pattern_tags`。
- 规则标签不是新的事实真值。正式论文实验应抽样人工复核每类 precision。
- `sample_audit.csv` 可用于检查回答级污染，`span_audit.csv` 更适合 token/claim
  级无监督异常检测。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖错误模式映射、标注说明泄漏、span 区间合并、JSONL 数据质量审计和四种
报告文件导出。
