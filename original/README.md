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

再次运行时，来源 cache、`tau`、大小和修改时间均一致的 `.graph.pt` 会直接
复用。每张图包含：

```text
token_ids, response_idx, x, edge_index, edge_attr, edge_mark, y_token
```

其中 `token_ids[i]`、`x[i]` 和 `y_token[i]` 对应同一个 token；`edge_mark`
的 `[1,0]` 是 prompt→response，`[0,1]` 是 response→response。

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
