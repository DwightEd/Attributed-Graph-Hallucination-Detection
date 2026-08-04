# RAGTruth 官方划分属性图无监督实验

## 数据边界

正式实验直接使用 attention cache 中的 RAGTruth `split` 字段：

- 官方 `train`：只用于无标签训练，并按 `source_id` 再切出无标签 validation；
- 官方 `test`：训练和 checkpoint 选择期间完全不可见，只在模型冻结后打分；
- `y_token`：不进入图、节点属性、损失、校准或 checkpoint 选择，只由最后的 `evaluate` 命令读取。

脚本会检查官方 train/test 都存在，并检查 `source_id` 不跨边界。只给原来的
`.../fresh_attention_.../train` 目录不再足够；`ATTENTION_DIR` 必须指向同时包含
`train/` 和 `test/` 的父目录。缓存不要求全部提取完才允许做 pilot，但两个 split
至少都要有样本，且结果必须明确理解为 partial-cache pilot。

## 图和无监督目标

每个回答构成一张有向属性多重图：token 是节点，只保留指向 response token 的严格
因果边。对每个 response token 分开选取 `prompt/evidence -> response` 和
`response-history -> response` 两类 top-k 入边，因此 prompt 与已生成回答不会混成一个
无语义的邻域。节点与边属性只来自 attention 路由统计。

类型化 masked autoencoder 只遮蔽 response 节点，利用两类邻域消息重建节点路由、邻域
均值/方差和 prompt/history 路由统计。训练集残差再做 median/MAD 无标签校准，得到每个
response token 的异常分数。

句子分数不是简单取整句最大 token。`sentences` 阶段使用提取 attention 时相同的
Llama tokenizer 与 chat template 重建字符 offset，并严格核对缓存的 `token_ids` 和
`response_idx`。每句话使用最高 20% token 分数的均值作为主分数，同时保存 max 和 mean。
这个文件仍不含标签；句子标签和 AUROC/AUPRC 只在最终评估时关联。

## 远端一条命令运行

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/Attributed-Graph-Hallucination-Detection
git pull --ff-only origin main
conda activate research
python -m pip install -r requirements-unsupervised-token-graph.txt
bash ./run_ragtruth_typed_token_graph.sh
```

默认输出到：

```text
/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/
  ragtruth_hypergraph_ssl_v2_full -> <原历史超图 full-v2 结果目录>
  ragtruth_attribute_graph_typed_mae_fresh_attention_c8847872bedf_official/
    compact_graphs/
    training/best.pt
    training/splits.json
    test_token_scores.jsonl
    test_sentence_scores.jsonl
    test_metrics.json
```

历史超图使用符号链接登记，不复制大文件，也不改变旧实验中的相对路径。可单独执行：

```bash
bash ./register_ragtruth_hypergraph_result.sh
```

如果正式 cache 的 `test/` 尚未提取，主脚本会停止并打印恢复命令。在超图项目中运行：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph
ATTENTION_CACHE_ROOT=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
RESUME_EXTRACTION=1 \
bash ./run_ragtruth_extract_validate.sh
```

显存无法容纳全部紧凑图时，显式使用流式 GPU 训练：

```bash
RESIDENCY=stream bash ./run_ragtruth_typed_token_graph.sh
```

每个 split 取 20 条的连通性 smoke test：

```bash
LIMIT=20 EPOCHS=2 PATIENCE=2 RUN_EVALUATION=1 \
bash ./run_ragtruth_typed_token_graph.sh
```

正式实验前应删除 `LIMIT`，并建议给新的 `RUN_DIR`，避免把 smoke manifest 当成全量缓存复用。
