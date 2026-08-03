# RAGTruth 类型化 token 图：GPU 无监督实验

这条实验路径直接读取已经提取好的 RAGTruth attention cache。节点是 token；只为回答 token 建立入边，并把两种信息来源分开：

- `prefix -> response`：prompt / evidence 对当前回答 token 的支持；
- `response-history -> response`：此前回答 token 对当前 token 的内部延续。

训练、验证和打分阶段完全不读取 `hallucination_labels`。标签只在最后 `evaluate` 子命令中加入。

## 方法实际在测试什么

每个 token 的节点属性由 attention 路由统计组成：对角质量、prefix 质量、回答历史质量、归一化熵、最大集中度和严格因果质量。每条 top-k 边保存跨层/跨头均值、最大值、标准差、前半/后半层质量及相对 token 距离。正式 CSR 还记录通道覆盖率和层覆盖率（阈值保留后该边出现的比例）；只有 legacy dense cache 的后两维才是层间/头间分歧。

类型化 masked autoencoder 随机遮住回答 token，并同时重建：

1. token 自身的 attention 路由属性；
2. prefix/history 两类入邻域的加权均值与对数方差；
3. 两类路由的边数、平均权重、权重方差和相对跨度。

异常分数是三类重建残差在无标签训练集上做 median/MAD 校准后的组合。最终报告还单独给出 `node_residual`、`neighborhood_residual` 和 `route_residual`，可以直接判断“发散”究竟来自节点状态、邻域几何还是路由结构。

当前 cache 只含 attention 和 token id，不含 hidden state/logits，因此本实验不会声称使用了 hidden state 或 logits。它是 attention-token-graph 假设的独立验证。

## 工程结构

- `unsupervised_token_graph/ragtruth_graph.py`：正式 sparse-CSR / legacy dense cache 白名单读取、GPU 分块、严格因果 top-k 图及自监督目标；
- `unsupervised_token_graph/typed_model.py`：类型化消息传递、masked autoencoder、稳健损失和 token 分数；
- `unsupervised_token_graph/ragtruth_data.py`：紧凑缓存、source-group split、GPU graph store 与 batching；
- `unsupervised_token_graph/typed_experiment.py`：训练、无标签校准和逐 token 打分；
- `unsupervised_token_graph/ragtruth_evaluation.py`：冻结分数后的标签关联与指标；
- `unsupervised_token_graph/ragtruth_pipeline.py`：保持公共 API 稳定的薄 facade；
- `unsupervised_token_graph/ragtruth_cli.py`：四个前台子命令；
- `run_ragtruth_typed_token_graph.sh`：使用真实 cache 的完整前台运行脚本。

增加新方法时只需新增模型模块，并在 pipeline 的模型构造处注册；RAGTruth 读取、紧凑图、切分和最终评估不需要重写。

## 内存与速度

正式数据是 response query 的阈值保留 CSR。代码先以 CPU mmap 校验其 schema、原始 dtype、fingerprint、截断策略和 CSR 边界，再把 `row_ptr / columns / values / diagonal` 数组一次搬到 GPU，随后按 `query_block` 处理；整个路径从不构造 `N x N` attention 或邻接矩阵。阈值以下质量不可恢复，因此正式图的节点/边统计明确解释为 retained-attention lower bound。

只有 legacy 数据包含原始 `attention[L,H,N,N]`。它同样以 CPU mmap 打开，不复制整个 corpus；构图时只把

```text
[layer_chunk, heads, query_block, tokens]
```

的 layer/query tile 搬到 GPU。两种格式的最终边数都至多约为

```text
response_tokens * (prefix_top_k + history_top_k)
```

而不是 `tokens^2`。节点、边、邻域统计和训练 batch 的运算都在 GPU 上完成；没有逐 token/逐边 Python 计算循环。文件级循环和固定数量的 mask view 是必要的流式边界。

紧凑图必须写入磁盘才能跨进程保存；“GPU 常驻”指训练/打分启动后一次性将紧凑图载入显存。默认最多让常驻图使用当前空闲显存的 45%，其余空间留给 fp32 拼接视图、消息激活、梯度、优化器和 CUDA workspace。预算不足时直接报错，不会静默退化到 CPU。

## 远端运行

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/Attributed-Graph-Hallucination-Detection
conda activate research

which python
python -m pip install -r requirements-unsupervised-token-graph.txt
python -m unittest discover \
  -s tests \
  -p 'test_ragtruth_*typed*.py' \
  -v
```

也兼容按模块运行：

```bash
python -m unittest \
  tests.test_ragtruth_gpu_typed_graph \
  tests.test_ragtruth_typed_pipeline -v
```

先跑 20 条无标签 smoke test：

```bash
LIMIT=20 RUN_EVALUATION=0 bash run_ragtruth_typed_token_graph.sh
```

完整前台实验（有进度条，不使用 `nohup`/后台任务）：

```bash
unset LIMIT
RUN_EVALUATION=1 bash run_ragtruth_typed_token_graph.sh
```

脚本默认读取：

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876/train
```

如果紧凑图确实装不进显存，可明确选择逐图 GPU 流式传输（不是自动 fallback）：

```bash
RESIDENCY=stream bash run_ragtruth_typed_token_graph.sh
```

正式 CSR 的临时向量峰值过高时降低 `QUERY_BLOCK`；legacy tile 峰值过高时还可降低 `LAYER_CHUNK`。两者只影响速度/峰值内存，不改变数学结果：

```bash
QUERY_BLOCK=32 LAYER_CHUNK=1 bash run_ragtruth_typed_token_graph.sh
```

## 输出

默认输出根目录：

```text
outputs/ragtruth_typed_token_graph/fresh_attention_c8847872bedf/
```

其中：

- `compact_graphs/manifest.json`：样本/节点/边数量与唯一构图配置；
- `compact_graphs/graphs/*.pt`：不含标签的线性规模紧凑图；
- `training/best.pt`：最佳 label-free validation checkpoint；
- `training/history.json`：每轮仅保留 train/validation loss；
- `training/splits.json`：按 `source_id` 隔离的 train/validation/test；
- `test_token_scores.jsonl`：逐 token 分数和三个残差分量，不含标签；
- `test_metrics.json`：最后阶段 token/sample AUROC、AUPRC、prevalence、AUPRC lift 和分量指标。

## 标签对齐必须先核实

仓库中的旧 `get_attention.py` 一处 span 映射使用了 `add_special_tokens=False` 的 prefix 长度，而 `response_idx` 来自可能包含 BOS 的完整 tokenizer 调用。不同 tokenizer/cache 可能因此出现一位偏移。

正式 sparse-CSR cache 的 `y_token` 是全局 token 坐标，`LABEL_SHIFT` **必须为 0**；非零值会直接报错。只有 legacy cache 才允许在人工审计确认标签确实左移一位后显式运行：

```bash
LABEL_SHIFT=1 bash run_ragtruth_typed_token_graph.sh
```

正值表示把缓存标签向右移动。任何有效正标签仍落在 `response_idx` 之前时，评估会停止并要求重新核对，而不会给出貌似正常但错位的指标。

## 研究解释边界

这是 post-hoc token 检测：token `t` 的 attention query 是模型已经读入 `t` 后得到的内部状态。若要做生成时在线预警，需要重新缓存预测 `t` 时的 `t-1` 决策状态。

无监督重建有效的前提不是“错误一定偏离正常流形”，而是错误 token 的 source/history 条件关系更难由其上下文共同解释。因此主要证据应看：held-out token AUPRC lift、跨 source split 的稳定性，以及 `neighborhood_residual`/`route_residual` 是否相对纯节点残差提供增益；单看训练 loss 没有研究结论。
