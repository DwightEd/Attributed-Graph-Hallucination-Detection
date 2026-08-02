# 无监督 Token 属性图：先审计错误模式，再训练

本模块保留原项目的基本表示方式：每个 token 是一个节点，Transformer
attention 形成有向边，CHARM 风格 MLP 完成消息传递。输入固定拼成：

```text
Passage:
{passage_or_knowledge}

Question:
{question}

Answer:
{candidate_or_generated_answer}
```

与旧监督代码不同，新模块：

- 对最终拼接文本只 tokenize 一次，再由 offset mapping 得到 segment id；
- `0/1/2/3` 分别表示 template/special、passage、question、answer；
- 可保留 passage→question、passage→answer、question→answer 和
  answer→answer 的因果边；
- 图文件不包含 `y`、`y_token`、正确答案或候选身份；
- 使用纯 PyTorch CHARM 消息传递，不依赖 `torch_geometric`；
- 先输出错误模式审计，再由用户决定是否启动无监督重构训练。

## 研究边界

HaluEval-QA 的候选答案不是被检测开源模型自然生成的。对候选答案做
teacher-forced forward 得到的是该检测模型的 evidence-support / representation
pattern，不能直接称为原生成模型的真实思维链或内部推理过程。

无标签模式名称只是可检验假设：

| 模式 | 无标签可观测定义 |
|---|---|
| `evidence_neglect` | answer→passage attention ratio 位于参考分布低尾 |
| `question_neglect` | answer→question ratio 位于低尾 |
| `answer_self_reinforcement` | answer→previous-answer reliance 位于高尾 |
| `diffuse_attention` | answer attention entropy 位于高尾 |
| `head_disagreement` | 各 head 的 passage reliance 分歧位于高尾 |
| `late_layer_grounding_drop` | passage reliance 从中层到末层显著下降 |
| `low_confidence` | teacher-forced answer log-probability 位于低尾 |

这些模式先在无标签特征上定义。`evaluation_labels.jsonl` 只在模式和分数冻结后
由 `audit` 或 `evaluate_scores` 加载，用来判断真实错误是否确实集中在这些模式。

HaluEval-QA 是每题一正一错的 50/50 配对池，因此在同一混合池拟合 Mahalanobis 时，分数只表示
“偏离混合分布”，不能解释为“偏离多数正常样本”。若要做 one-class 主张，必须另给无标签但
可信正确的 reference features。BoolQ 的随机 label-blind test split 也可能恰好没有错误样本；
此时程序会把 AUROC 标为不可定义，而不会伪造 0 分。

## 与原图表示的关系

图结构沿用原实现：token 是节点，严格因果 `j→i` attention 边，任一层/头超过 `tau` 才保留边，
且每个层/头中不超过 `tau` 的弱权重仍置零。原始 attention diagonal 仍是节点属性的主体；新模块
另外拼接 segment one-hot、归一化位置、teacher-forced log-probability/entropy，有需要时可加入
probe hidden state。因此它是“原 token 图 + 明示属性”的扩展，不应声称完全等同旧特征。论文级
实验必须分别消融这些属性，并报告答案长度/token 数基线。

## 安装

```bash
cd /path/to/Attributed-Graph-Hallucination-Detection
python -m pip install -r requirements-unsupervised-token-graph.txt
```

无需安装 `torch_geometric`。

## HaluEval-QA：推荐先跑 300 个候选

下载官方数据后，设：

```bash
export SOURCE_DATA=/path/to/HaluEval/data/qa_data.json
export MODEL_PATH=/path/to/open-source-causal-lm
export DATASET=halueval_qa
export RUN_DIR=/path/to/output/halueval_token_graph_pilot
export PILOT_LIMIT=300
export MAX_TOKENS=1024
export MAX_ATTENTION_GIB=12
export POSTPROCESS_DEVICE=auto
export RETAIN_DENSE_ATTENTION=0
export CPU_THREADS=4
export RUN_TRAINING=0

bash ./run_unsupervised_token_graph_pilot.sh
```

`PILOT_LIMIT=300` 对应 150 个 HaluEval pair；同一问题的正确/幻觉候选相邻准备，
train/validation/test 划分始终按 `pair_id` 分组。

先阅读：

```text
${RUN_DIR}/pattern_audit/pattern_audit.md
${RUN_DIR}/pattern_audit/pattern_records.jsonl
```

重点检查：

1. 错误候选是否在同一 pair 内具有更低 passage reliance；
2. 去除答案长度、token 数等混杂后，差异是否仍存在；
3. `anomaly_score` 是否只是识别 HaluEval 合成风格；
4. passage/question/self-reliance 的方向能否在 BoolQ 模型自然错误上复现。

## BoolQ：必须先生成模型答案

BoolQ 的 `answer=false` 仅表示正确答案是 No，不是幻觉标签。脚本先让目标模型仅
根据 passage/question 生成 Yes/No，再把预测与 gold 的比较放入 evaluation sidecar：

```bash
export SOURCE_DATA=/path/to/boolq/dev.jsonl
export MODEL_PATH=/path/to/open-source-causal-lm
export DATASET=boolq
export RUN_DIR=/path/to/output/boolq_token_graph_pilot
export PILOT_LIMIT=300
export RUN_TRAINING=0

bash ./run_unsupervised_token_graph_pilot.sh
```

生成阶段的 prompt 函数没有 gold answer 参数，防止把正确布尔值输入模型。预测文件同时保存
实际生成的 token IDs；抽取时若拼接文本不能精确重分词回这些 IDs，会拒绝运行，避免把
decode→retokenize 漂移误当成原生成轨迹。

## 分步执行

### 1. 准备 HaluEval

```bash
python -m unsupervised_token_graph.prepare \
  --dataset halueval_qa \
  --input /path/to/qa_data.json \
  --output-dir outputs/halueval/prepared
```

输出严格分离：

- `examples.jsonl`：模型可见的 passage/question/answer，无标签；
- `evaluation_labels.jsonl`：只有最终审计/评价可见。

### 2. 抽取 trace 并构图

```bash
python -m unsupervised_token_graph.extract \
  --examples outputs/halueval/prepared/examples.jsonl \
  --model /path/to/model \
  --output-dir outputs/halueval/extraction \
  --max-tokens 1024 \
  --max-attention-gib 12 \
  --postprocess-device auto \
  --discard-dense-attention \
  --limit 300
```

默认规则仍是原方法的 `attention > 0.05`，但保留 prefix 内部路径。可用
`--drop-prefix-edges` 复现旧代码删除 prefix→prefix 边的行为。任何输入超过
`--max-tokens` 都会报错，不会静默截断。已存在缓存只有 extraction fingerprint
完全一致时才会复用；否则要求显式 `--overwrite`。

`--max-tokens` 是拒绝阈值，不会截断。全层全头 attention 为 `O(LHN²)`：32 层 × 32 头 ×
4096 token 的单条 float32 attention 约为 64 GiB，因此脚本默认 1024 token 并设置 12 GiB
预检上限；超过时会在 forward 前报错，而不是把机器拖入 OOM。

输出：

- `traces/*.pt`：六个 probe hidden states、log-probability、entropy、已冻结的标量特征；加 `--discard-dense-attention` 时只记录原 attention shape，不保存巨大 tensor；
- `graphs/*.pt`：无标签 token 属性图；
- `features.jsonl`：用于先看模式的标量特征；
- `extraction_manifest.json`：模型、层、构图和缓存信息。

`--postprocess-device auto` 会在显存足够时用模型 GPU 构图和归约，显存不足时自动回退 CPU；`model` 强制使用模型设备，可能 OOM，通常只用于确认显存充裕的短样本。原始 dense attention 若被丢弃，之后更换构图阈值或增加新的 attention 统计必须重新 forward。

### 3. 先做模式审计

```bash
python -m unsupervised_token_graph.audit \
  --features outputs/halueval/extraction/features.jsonl \
  --evaluation-labels outputs/halueval/prepared/evaluation_labels.jsonl \
  --examples outputs/halueval/prepared/examples.jsonl \
  --output-dir outputs/halueval/pattern_audit
```

程序先拟合无标签分位数和 robust Mahalanobis，再打开 evaluation label。报告同时给出
各特征的正确/错误中位数、AUROC、方向无关 separability，以及 HaluEval paired
ranking。标签不参与特征阈值或异常模型拟合。

人工阅读请优先看 `evaluation_casebook.jsonl`、`evaluation_error_cases.jsonl` 和
`evaluation_pair_deltas.jsonl`：它们在无标签
模式与分数冻结后才连接标签，并给出 passage/question/answer、模式名和异常分数。这里观察的是
**support-routing pattern（支持信息路由模式）**，不是模型不可见的真实思维链。

若要使用独立无标签参考集，传：

```bash
--reference-features /path/to/reference/features.jsonl
```

### 4. 审计通过后才训练

```bash
python -m unsupervised_token_graph.train \
  --graph-dir outputs/halueval/extraction/graphs \
  --output-dir outputs/halueval/training \
  --batch-size 8 \
  --epochs 50
```

训练目标是遮蔽连续 answer-token 节点属性，由 passage/question/answer causal
neighborhood 重构原属性。它没有伪造“异常图”作为负样本，也没有分类头。训练、
分组划分、早停和模型选择只使用 masked reconstruction loss。

默认同时将接触被遮蔽 token 的 attention `edge_attr` 置零，避免用该 query 自己的完整
attention 行泄漏重构目标；图拓扑和邻居节点仍保留。`--keep-masked-edge-attrs` 只用于泄漏
消融。评分默认逐 answer token leave-one-out（`--score-block-size 1`），避免 BoolQ 单 token
和长答案使用不同遮蔽比例。

最后单独评价冻结分数：

```bash
python -m unsupervised_token_graph.evaluate_scores \
  --scores outputs/halueval/training/unsupervised_scores.jsonl \
  --evaluation-labels outputs/halueval/prepared/evaluation_labels.jsonl \
  --split test \
  --output outputs/halueval/training/evaluation_only_metrics.json
```

默认只汇报按 `pair_id` 留出的 untouched test 图；validation 只用于无标签重构早停。
`--split all` 仅用于诊断，不能作为最终结果，因为其中混有训练图的分数。

## 是否继续训练的门槛

不要因为某个 attention 特征在单一数据集上 AUROC 较高就扩大实验。建议至少满足：

- HaluEval paired ranking 显著高于 0.5；
- BoolQ 自然错误上方向一致；
- answer length/token count 基线不能解释主要收益；
- 删除 passage→answer 边后性能明显下降；
- 完整图模型优于 pooled feature Mahalanobis；
- 人工 corruption 上有效但真实错误上无效时，立即停止该目标。

如果图重构不优于非图特征基线，应保留模式审计并删除 GNN 主张。
