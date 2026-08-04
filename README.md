# RAGTruth Attention Graph

本仓库当前的正式研究入口，是用 RAGTruth 已提取的逐层、逐头 attention 构造
RP/RR 有向属性图，并进行不读取幻觉标签的关系感知 masked graph autoencoder
训练。标签只在所有 test 分数冻结后用于最终评估。

## 从哪里开始看

```text
main.py                              单一 Python 入口
run_ragtruth_attention_graph.sh      单次正式实验
run_ragtruth_attention_graph_ablations.sh
                                     结构与选边消融
attention_graph/
  data.py                            formal sparse cache 与 mmap 图数据
  graph.py                           attention-only RP/RR 属性图构建
  model.py                           relation-aware MaskGAE 与四类重构目标
  train.py                           无标签训练、K=2 mixture、token/response 分数
  evaluate.py                        冻结分数后才读取标签并计算指标
  ragtruth.py                        tokenizer offset 重放与句子 mean 分数
  ablation.py                        结构、关系和 head 消融
  cli.py                             一键流程编排
docs/METHOD.md                       完整方法、假设、文献和实验边界
tests/test_attention_graph_*.py      新方法测试
```

当前方法和旧实验的边界如下：

| 范围 | 入口 | 用途 |
|---|---|---|
| 当前方法 | `main.py`、`run_ragtruth_attention_graph.sh`、`attention_graph/` | RAGTruth attention 属性图的正式开发入口 |
| 当前消融 | `run_ragtruth_attention_graph_ablations.sh`、`attention_graph/ablation.py` | 当前方法的核心结构消融 |
| 旧 token-graph 实验 | `unsupervised_token_graph/`、`run_ragtruth_typed_token_graph.sh`、`run_unsupervised_token_graph_pilot.sh` | 先前 RAGTruth token-graph pilot 的复现 |
| 旧 HaluEval/属性图实验 | `run_halueval_qa_graph_mae.sh`、`run_halueval_boolq_feature_pilots.sh`、`GNN_train.py`、`processed_graphs_attribute.py` | 先前 HaluEval 和属性图基线的复现 |

旧路径不再是当前方法的依赖。保留它们是为了不破坏已经生成的结果；开发新版方法时
不应再向其中添加文件，也不要把旧方法的结果与当前方法直接横向比较。

## HaluEval 严格 pure-attention 旧 MAE 基线

`run_halueval_qa_graph_mae.sh` 默认运行旧 `TokenGraphMaskedAutoencoder` 的严格
pure-attention 协议：节点训练属性只有逐层逐头 self-attention diagonal，边训练属性
只有逐层逐头 attention。`segment_ids` 只作为持久化元数据用于定位 response 和生成
mask；segment one-hot、position、hidden state、log-prob、entropy 和 segment edge mark
都不会进入模型。输出目录带 `graph_mae_pure_attention`，不能加载旧 `no_logits`
checkpoint。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/Attributed-Graph-Hallucination-Detection
git pull --ff-only origin main
bash ./run_halueval_qa_graph_mae.sh
```

pure 模式会跳过旧 Pattern audit，因为该报告仍包含 log-prob/entropy 等非 attention
诊断量；这不影响无标签 MAE 的训练与最终 test evaluation。

## 为什么会报 test cache 不存在

给出的 cache 根目录已经有标记为 official split identity 的 `train/`，但没有生成
`test/attention_*.pt`。这不是找不到 RAGTruth 原始 JSONL，也不要求重跑 train。
新脚本发现 test 完全缺失时，会调用
`scripts/data/prepare_ragtruth_test_attention.sh`，以 `--split test` 和
`--resume-existing` 只补提取 test。

图准备不依赖 `manifest.json` 的 complete 状态。已经原子写完的单个
`attention_*.pt` 可以直接用于显式 partial-cache smoke。这里的 `train`/`test`
只保证样本继承相应 official split identity，不表示目录已经覆盖该 split 的全部样本。
部分 cache 的状态和实际样本数会写入运行产物；正式数据集比较必须使用完整 test
inventory，不能把 partial-cache 结果称为完整 official test 结果。

## 远端直接运行

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/Attributed-Graph-Hallucination-Detection
git pull --ff-only origin main
conda activate research
python -m pip install -r requirements-unsupervised-token-graph.txt
bash ./run_ragtruth_attention_graph.sh
```

这条命令允许直接使用当前已经落盘的 partial cache，并在 `run.json` 中把实验明确标成
`partial_cache_pilot`。只有准备报告完整数据集结果时才加严格检查：

```bash
REQUIRE_COMPLETE_CACHE=1 bash ./run_ragtruth_attention_graph.sh
```

默认输入：

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/
  outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876/
```

默认输出：

```text
/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/
  ragtruth_attention_graph_<UTC时间>_seed42/
    graphs/
    training/encoder.pt
    training/history.json
    splits.json
    response_mixture.json
    token_mixture.json
    test.response_predictions.jsonl
    test.token_predictions.jsonl
    test.sentence_predictions.jsonl
    evaluation.json
    run.json
```

默认目录带 UTC 时间和 seed；CLI 会拒绝非空 `OUTPUT_DIR`，防止旧 checkpoint、预测和
评价混入新实验。`run.json` 的 `experiment_scope` 与 `cache_inventory` 分别说明这是
`smoke_test`、`partial_cache_pilot` 还是经过 manifest 精确核验的
`official_complete_cache`。

先做不读标签的小规模连通测试：

```bash
LIMIT=20 EPOCHS=2 PATIENCE=2 SKIP_EVALUATION=1 \
  OUTPUT_DIR=/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/ragtruth_attention_graph_smoke \
  bash ./run_ragtruth_attention_graph.sh
```

`LIMIT` 按文件名顺序从 official train/test 各取前 N 条，不做随机抽样，也不按
source、task 或 generator 分层，因此样本不具有代表性，小 N 时还可能不足以划分
validation source。它只能用于 smoke；正式实验必须去掉它，并使用新的 `OUTPUT_DIR`。

若 test attention 提取需要单独执行：

```bash
ATTENTION_DIR=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
  bash ./scripts/data/prepare_ragtruth_test_attention.sh
```

## 方法实际使用了什么

- 节点是 prompt 和 response token；内容属性是完整 self-attention 对角
  `node_attr[N, L*H]`，不是 hidden state，也不是六七个手工统计量。
- 边只指向 response token。`RP` 表示 prompt→response，`RR` 表示历史
  response→response。
- 每条 token-pair 边保存稀疏 `(edge, layer/head, attention)` trace；不会先做
  head/layer 平均。
- 默认 threshold 图先选择超过 cache floor 的 support，再以每个 target 最多 64 条作为
  明确的显存安全帽。这个 cap 会截断 degree 大于 64 的自然变化，因此当前默认图不是
  未截断的完整 attention 图；可显式设 `MAX_EDGES_PER_TARGET=none` 运行无 cap 版本，
  但应先用小样本检查边数和显存；`global_topk` 与 `typed_topk` 只作为对照。
- 自监督目标为 retained support、逐通道 weight、带 OTHER bucket 的 attention-row
  distribution 和节点对角 SCE 重构。
- response/token 表示在未标注 train 上拟合自由权重的 K=2 mixture；不规定异常簇更小。
  先验方向只用于解决两个簇的名称交换，不参与图编码或 mixture 拟合。
- 句子分数是成员 token posterior 的算术平均，不使用“错误 token 必须稀少”的 top-k
  假设；一键流程在所有分数冻结后，用成员 token 任一为正作为句子标签进行最终评价。

## 怎么看结果

最终的 `evaluation.json` 报告 `response`、`token`，以及启用 RAGTruth 元数据时的
`sentence` 三个粒度：

- `auroc` 衡量当前定向分数把幻觉排在正确样本之前的排序能力，0.5 约等于随机排序；
- `average_precision`（AUPR）的随机参考不是 0.5，而是同一粒度的
  `positive_fraction`；优先结合 `average_precision_lift` 解读类别不平衡下的增益；
- `orientation_free_auroc` 和 `orientation_free_average_precision` 会同时检查分数正向与
  反向，只回答“两个模式是否可分”，不能证明探索性先验把簇命名正确；
- `hallucination_probability` 是自由权重 K=2 GMM 中、被预先指定探索性先验定向为
  hallucination-like 的分量责任值，不是经过监督校准的事实错误概率；
- `test.sentence_predictions.jsonl` 是成员 token posterior 的无标签算术平均；
  `evaluation.json.sentence` 使用“任一成员 token 为幻觉则该句为正”的固定标签规则。
  这个均值仍是未校准的模式责任值，不等同于事实错误概率。

旧 HaluEval、旧 token-graph 或 partial-cache pilot 的 AUROC/AUPR 与当前方法的图构造、
split 和评分器不同，不能直接当作当前方法的基线或正式结果。

## 证明图不是装饰

运行当前已经接线的核心结构消融：

```bash
bash ./run_ragtruth_attention_graph_ablations.sh
```

核心比较包括：

- `full`：完整 model/objective；图本身仍是 threshold + 每 target 最多 64 条边；
- `no_message`：去掉消息传递，但仍保留关系重构；
- `feature_only`：零消息传递、只训练节点重构、mixture 只看 embedding；
- `source_shuffle`：保持 target、relation、degree 和 trace 边际量，破坏 source 对齐；
- `collapse_relations`：合并 RP/RR；
- `mean_heads`：每层跨 head 平均；
- `global_topk` / `typed_topk`：与默认 threshold support 比较。

这里的 `full` 是 full model/objective，不表示未截断的完整 attention 图。只有它在
多随机种子下稳定优于 feature-only 和 source-shuffle，才支持邻接和关系建模带来独立
贡献。损失权重、mask rate、阈值/top-k sweep、统计量 MLP、one-class 和 nuisance
对照仍属于后续实验计划；详细的研究限制与文献对应见 `docs/METHOD.md`。
