# Original attributed-graph baseline

这个目录保存并运行上游普通属性图基线，来源固定为
[`liuzhishun/Attributed-Graph-Hallucination-Detection`](https://github.com/liuzhishun/Attributed-Graph-Hallucination-Detection)
的 commit `13e907693aa954bf070e809d8afecdf26b3b88d8`。

根目录的 `processed_graphs_attribute.py` 和 `train_charm_grid.py` 与该上游
commit 一致。这里不复制第二份源码：`ragtruth_graph.py` 只负责把已经存在的
sparse-CSR attention cache 无损适配为上游图格式，`train.py` 调用上游 `CHARM`
模型并把硬编码训练过程封装成可运行入口。

## 一条命令运行

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/Attributed-Graph-Hallucination-Detection
git pull --ff-only origin main
conda activate research
python -m pip install -r original/requirements.txt
bash ./original/run_ragtruth_original.sh
```

不需要在服务器再 clone 或推送一份上游项目。图数据使用稳定目录，不随每次
训练的时间戳改变。默认命令会审计 train/test manifest；发现未完成的 split 时
使用 `--resume-existing` 只补齐缺少的 attention，完整后才开始正式训练：

```text
/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/
  ragtruth_original_attribute_graphs/
    fresh_attention_c8847872bedf_20260731T074520Z_p876_tau0p05/
      manifest.json
      index.jsonl
      graphs/train/*.graph.pt
      graphs/test/*.graph.pt
      training/<UTC时间>/
        seed_0/checkpoint.pt
        seed_0/history.json
        ...
        summary.json
```

补齐 attention 时，辅助程序仍需要一个临时承载 replay hypergraph 的目录。它固定
使用 `RAGTruth/hypergraphs/cache_bound_sha256/<cache_tag>/`，不会复用旧的
`fresh_hypergraphs_*`。旧图没有当前校验器要求的 `attention_cache_sha256`，而
`--resume-existing` 只复用通过当前校验的图，并不负责升级旧 schema。这个隔离
只会从已有 attention cache 快速重建辅助图，不会删除 cache，也不会对已有样本
重新执行 teacher forcing。

再次运行时，来源 cache、`tau`、大小和修改时间均一致的 `.graph.pt` 会直接
复用。现有 attention cache 就是构图输入，不需要重新运行大模型；已有 v2
`.graph.pt` 也可以脱离 cache 直接加载和训练。新格式的 schema 是
`original-ragtruth-attributed-graph-v2`；旧 v1 图缺少 L/H、通道顺序和节点角色，
不会与 v2 静默混用。

## 图如何构造

记：

- `N`：拼接后的 prompt + response token 数；
- `L`：attention 层数；
- `H`：每层 attention head 数；
- `C=L×H`：逐层逐头通道数，`channel=layer×H+head`；
- `E`：最终 token-pair 边数。

节点按原序列排列：`[0,response_idx)` 是 prompt，`[response_idx,N)` 是
response。每个 token 是一个节点。节点字段如下：

| 字段 | shape / dtype | 含义 | CHARM 使用 |
|---|---|---|---|
| `token_ids` | `[N] / int64` | 每个节点的 tokenizer token ID | 否 |
| `node_role` | `[N] / int8` | `0=prompt, 1=response` | 否，仅供分析 |
| `x` | `[N,C] / float32` | `x[i,c]=attention[layer,head,i,i]`，即逐层逐头 self-attention | 是，节点属性 |
| `y_token` | `[N] / int64` | `0=非幻觉，1=幻觉`；使用全序列 token 下标，prompt 区间强制为 0 | 是，仅作为 response-token 监督标签 |

`token_ids[i]`、`node_role[i]`、`x[i]` 和 `y_token[i]` 永远描述同一个节点。
构图时若 cache 没有 `y_token`，代码会直接报错，不会生成无标签图。

边按 causal attention 的实际方向保存：

```text
edge_index[0, e] = source/key（被关注的历史 token）
edge_index[1, e] = target/query（发出 attention 的 response token）
```

只考察 `source < target` 且 target 是 response 的 token 对，因此没有
prompt→prompt 边。对每个候选 `(source,target)`：只要任意 layer/head 满足
`attention[layer,head,target,source] > tau` 就建立一条边。默认 `tau=0.05`，
比较是严格的大于，不是平均 attention，也不是 top-k。

| 字段 | shape / dtype | 含义 | CHARM 使用 |
|---|---|---|---|
| `edge_index` | `[2,E] / int64` | 每列是 `[source,target]` | 是，消息传递拓扑 |
| `edge_attr` | `[E,C] / float32` | 同一边的逐层逐头 attention；`>tau` 保留，否则为 0 | 是，边属性 |
| `edge_mark` | `[E,2] / float32` | `RP=[1,0]`，`RR=[0,1]` | 是，关系属性 |

其中 RP 是 prompt→response，RR 是历史 response→当前 response。没有跨 head
池化：`x` 和 `edge_attr` 都保留全部 `L×H` 通道。

每张图还包含以下图级字段：

| 字段 | 含义 |
|---|---|
| `schema` | 图格式版本 |
| `source_id`, `response_id`, `sample_id`, `split` | 与 RAGTruth 原样本及 train/test 的关联键 |
| `response_idx` | 第一个 response 节点的全局下标 |
| `response_label` | `any(y_token[response_idx:])`，只用于统计 |
| `tau`, `attention_floor` | 构图阈值与 sparse cache 保存下限 |
| `metadata` | L/H/C、通道顺序、边方向、RP/RR 编码、标签语义和 cache dtype |
| `upstream_commit` | 对齐的原项目 commit |
| `source_cache_path/size/mtime` | 安全 resume 使用的来源记录 |
| `attention_cache_fingerprint` | attention cache 内容身份 |

同样的机器可读字段规范也保存在数据集根目录的 `manifest.json` →
`graph_fields` 中；`index.jsonl` 只负责把样本身份、split 和相对 graph path
对应起来。

## 复用和查看图

Python 中直接安全加载一张图：

```python
from original import load_original_graph

graph = load_original_graph("/path/to/attention_xxx.graph.pt")
print(graph["x"].shape, graph["edge_index"].shape, graph["y_token"].shape)
```

查看一张图的字段、统计以及前几个节点/边：

```bash
python -m original.cli inspect \
  --graph /path/to/attention_xxx.graph.pt \
  --max-nodes 3 \
  --max-edges 3
```

输出是单个 JSON，可以直接保存或交给后续分析脚本。若稳定目录中存在此前
生成、但缺少上述自描述 metadata 的 v1 图，再次执行 build 会从现有 attention
cache 将它重建为 v2；不会重新跑模型提取 attention。训练入口同样只接受 v2，
从而保证 build、inspect 和 train 使用完全相同的数据契约。

只构图、不训练：

```bash
BUILD_ONLY=1 bash ./original/run_ragtruth_original.sh
```

只用少量样本做连通测试，输出到单独的稳定目录：

```bash
LIMIT=20 EPOCHS=2 SEEDS=0 ALLOW_PARTIAL_CACHE=1 \
  bash ./original/run_ragtruth_original.sh
```

如果只想保存当前已有的 train cache：

```bash
SPLITS=train BUILD_ONLY=1 AUTO_PREPARE_CACHE=0 ALLOW_PARTIAL_CACHE=1 \
  bash ./original/run_ragtruth_original.sh
```

`ALLOW_PARTIAL_CACHE` 默认为 `0`。只有显式设为 `1` 才允许 partial/smoke 图进入
训练；训练的 `summary.json` 会同时记录 `graph_experiment_scope` 和该显式开关。

## 方法边界

这是上游的监督逐 token 基线，不是无监督实验。训练只对 response token 使用
`y_token` 和 `BCEWithLogitsLoss`，validation 也使用标签选择 checkpoint。

上游 `train_charm_grid.py` 中计算出 `pos_weight` 后误用了未定义变量
`pos_weight_val`。这里使用它本来计算出的 response-token 类别比，并保持上游
的最大值 8；模型结构、消息传递、输入属性、标签和 BCE 目标均未改变。
