# Grounding Flow

这是一个与旧 `RelationAwareMaskGAE` 完全分开的 attention-only 无监督实验。
它不做随机节点遮蔽，不使用 hidden state、log-prob、entropy、segment/position
数值特征，也不依靠 GNN/MLP 重构一般 attention 统计。核心问题是：在控制每个
目标 token 的逐层逐头 attention 权重边际后，response 内的 source 拓扑是否仍然
表现出异常的“证据传递中断”。

## 目录

```text
grounding_flow/
├── method.py          # 证据流分解、守恒检查、条件重连 null model
├── state_model.py     # train-only PCA 与两状态 Gaussian HMM
├── experiment.py      # 轨迹数据结构、模型拟合与 token/response 打分
├── data.py            # HaluEval legacy attention cache 适配与无标签划分
├── cache.py           # 轨迹缓存、流式 PCA/HMM 拟合与逐 split 打分
├── evaluation.py      # 长度控制、预测冻结、evaluation-only 标签边界
├── artifacts.py       # 原子化 JSON/JSONL/Torch artifact I/O
├── pipeline.py        # 只编排 prepare → train → score → evaluate
├── cli.py             # Python CLI
├── run_halueval.sh    # 远端唯一运行入口
└── README.md
```

## 方法

对 response token `t` 的每个 layer/head attention 行，`method.py` 将质量守恒地
分成七类：

- 直接指向 evidence；
- 直接指向模板或 question；
- 由历史 response token 转发的 grounded / ungrounded mass；
- response self-loop 中的 grounded / ungrounded mass；
- 被旧 cache 阈值截断、无法辨认的 unknown mass。

跨层时只传播 response token 的 evidence ancestry。这里得到的是在
`head-mean attention rollout` 假设下的机制代理，不宣称复原 value/output
projection 或 residual stream 的真实因果贡献。

条件 null model 只交换 RR 边所绑定的历史 response source，同时严格保持：

- target token；
- RP/RR 类型；
- edge 上完整的 layer/head attention payload；
- target 的逐通道加权 attention mass；
- 因果方向、粗 causal-lag bin；
- 全局 source combinatorial degree。

它不声称保持每个 source 的加权 outgoing strength。只改变 prompt 内同类 token
不会改变当前证据流，因此不会被伪装成有效 relay null。无法合法重连的短回答会
明确记录为 `unswappable`。null source assignment 会去重；少于两个唯一有效 null 的
回答被标记为不可识别，不进入 PCA/HMM 或测试 AUROC，而不是用零向量代表“正常”，
也不会复制同一张重连图冒充 32 个 null。测试时只保留两个 candidate 均可识别的完整
pair；默认要求至少覆盖 90% 测试 pair，否则在读取标签前 fail closed。`run.json` 会
报告各 split 的有效覆盖率，终端同时打印 scored/total。

每个 token 最终保留完整 `[layer, head, 3]` 曲面：null-calibrated ancestry、
grounded relay、grounding debt。训练集上无标签拟合 PCA，再用两状态对角 Gaussian
HMM 学习回答中的状态转移。两个状态的 mixing weight 自由学习，所以没有“幻觉必须
是少数异常”的假设。`debt - ancestry` 只在 EM 结束后命名 detached state，不进入
PCA、HMM likelihood、transition 或 posterior。PCA 采样和 HMM 充分统计均按 response
平衡，避免长回答仅因 token 更多而主导无监督训练。

主分数是 response 内 detached-state posterior 的均值，同时固定输出：

- mean / `1-mean` / max / top-10% detached posterior；
- train-only response-length residual；
- raw `1 - ancestry`、debt、unknown mass。

这些分数都在标签读取之前落盘并写入 SHA-256 freeze manifest，`evaluation.json`
会分别报告它们的 AUROC/AUPRC/paired accuracy，便于直接判断 HMM 是否改善或破坏了
原始机制排序。

## 一键运行

在属性图项目根目录：

```bash
git pull --ff-only
bash ./grounding_flow/run_halueval.sh
```

脚本默认读取：

```text
/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/
LATEST_HALUEVAL_GRAPH_MAE_RUN.txt
```

它只复用该 run 的 `extraction/`、`prepared/examples.jsonl` 和 evaluation-only label
sidecar；不会等待旧 Graph-MAE 的 pattern audit 或训练结束，也不会跳转到
`ragtruth_cli`。新实验默认直接写入
`.../feature_extraction/halueval_grounding_flow_<timestamp>_seed<seed>/`，不会再嵌套进旧
Graph-MAE run，因此数据源和新方法的结果不会混在一起。

正式入口默认 `EXPECTED_CANDIDATES=2000` 且 `REQUIRE_COMPLETE_CACHE=1`；manifest、
无标签 examples 或图/trace 任一不完整都会直接停止，不会在小样本上静默输出“正式”
AUROC。如确实只做不完整缓存 pilot，必须显式写明：

```bash
EXPECTED_CANDIDATES=none REQUIRE_COMPLETE_CACHE=0 LIMIT_PAIRS=50 \
  bash ./grounding_flow/run_halueval.sh
```

`MIN_TEST_PAIR_COVERAGE` 默认是 `0.90`；只有明确的诊断性 pilot 才建议降低它。正式结果
不能通过降低覆盖率门槛来隐藏大量 `unswappable` 回答。

先做小规模完整链路测试：

```bash
LIMIT_PAIRS=50 NULL_SAMPLES=8 HMM_ITERATIONS=20 \
  bash ./grounding_flow/run_halueval.sh
```

正式实验：

```bash
LIMIT_PAIRS=none NULL_SAMPLES=32 HMM_ITERATIONS=50 \
  bash ./grounding_flow/run_halueval.sh
```

指定旧 extraction、GPU 或固定输出目录以断点续跑：

```bash
SOURCE_RUN=/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/halueval_qa_graph_mae_pure_attention_2000_YYYYMMDDTHHMMSSZ \
GPU_ID=1 \
OUTPUT_DIR=/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/my_grounding_flow_run \
  bash ./grounding_flow/run_halueval.sh
```

所有常用参数已经集中在 `run_halueval.sh` 顶部。`trajectory_cache/` 支持精确协议和
source identity 校验后的断点复用；不同 protocol 不会混写结果。
构图、attention flow 与 null 统计使用所选 CUDA device；离散 source swap 留在 CPU。
若后半段中断，同一 `OUTPUT_DIR` 会校验 protocol/source SHA-256 后复用昂贵轨迹并重建
可再生的 detector、score 和 evaluation artifact。

## 输出

```text
OUTPUT_DIR/
├── prepared/                         # legacy cache 的 attention-only 正式图适配
├── trajectory_cache/                 # 昂贵的逐样本 null-calibrated 轨迹
├── splits.json                       # pair/prompt-group 无泄漏划分
├── detector.json                     # PCA + HMM 参数
├── train|validation|test.*.jsonl     # response/token 预测，无标签
├── *.calibration_exclusions.jsonl    # null 不可识别及不完整 pair，显式不计入指标
├── identifiable_coverage_gate.json  # 标签读取前固定的 test-pair 覆盖率门禁
├── score_freeze.json                 # 标签读取前的不可变分数摘要
├── evaluation.json                   # test 标签仅在这里使用
├── run.json                          # 方法、训练、覆盖率、分阶段 wall time 与核心指标
└── run.log
```

训练阶段会打印每轮 `hmm_response_balanced_log_likelihood`，null 计算和逐 split 打分也会定期打印
进度。图很小且离散重连在 CPU 完成，因此 GPU 显存占用不应被当成“是否训练”的
判断依据；真正的无监督拟合是 PCA 与 HMM EM，而 attention rollout/null 统计在
CUDA 上运行。

## 当前数据限制

旧 HaluEval pure-attention cache 通常已经按 `tau=0.05` 丢弃小 attention，无法从
现有 `.pt` 恢复 dense 行。本方法绝不将缺失质量重归一化，而是放入 `unknown`。
因此当前结果应标为 `legacy_tau_censored` 实验；若后续重提 dense/floor-0.01 cache，
必须作为新的 protocol 单独报告，不能与当前数值混称。
