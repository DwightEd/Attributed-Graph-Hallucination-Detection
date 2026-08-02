# HaluEval QA / BoolQ 远端特征抽取

本入口只负责两件事：先下载并校验官方数据，再按顺序运行两个小规模 GPU pilot。它不会启动无监督图重构训练；先看错误模式是否真实存在，再决定训练方法。

## 数据与运行范围

- HaluEval QA：固定到官方仓库 commit `b7253db3cdaa0ab2c382f92b26b390109174f77e`，校验 10,000 个问答 pair。
- BoolQ：下载 Google 官方 `train.jsonl`（9,427 条）和 `dev.jsonl`（3,270 条），第一轮只在有标签的 dev 上生成目标模型自己的 Yes/No 回答。
- 下载器把 URL、版本、行数、字节数和 SHA256 写到 `${DATA_ROOT}/hallucination_datasets/dataset_manifest.json`。后续运行若哈希变化会拒绝继续。
- 默认 pilot 抽取 64 个 HaluEval candidate（32 个 pair）和 64 个 BoolQ dev 回答；同一 GPU 上严格顺序运行。

每条输入均固定拼接为 `passage/knowledge + question + candidate/generated answer`。同一次 teacher-forced forward 保存全层 attention、六个 probe hidden state、token log-probability/entropy、token 属性图和标量审计特征。evaluation label 保存在独立 sidecar，特征抽取阶段不读取它。

BoolQ 生成阶段会保存模型实际看到的完整 token IDs、attention mask 和 passage/question/answer segment IDs；抽取阶段直接回放这些 token，不以不可靠的 `decode → retokenize` 结果作为门禁。
若模型没有以 Yes/No 开头作答，该条会写入 `boolq_predictions.jsonl.invalid.jsonl` 并从本轮 pilot 跳过，不会让之前已经生成的有效样本全部作废。

## 远端完整命令

```bash
REPO=/share/home/tm902089733300000/a903202310/lys/research/Attributed-Graph-Hallucination-Detection
DATA_ROOT=/share/home/tm902089733300000/a903202310/lys/data
MODEL_PATH=/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
PYTHON_BIN=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python

cd "$REPO"
git pull --ff-only origin main
bash -n ./download_halueval_boolq.sh \
  ./run_halueval_boolq_feature_pilots.sh \
  ./run_unsupervised_token_graph_pilot.sh

# 先在前台完成下载、字段检查、条数检查与 SHA256 清单。
env DATA_ROOT="$DATA_ROOT" PYTHON_BIN="$PYTHON_BIN" FORCE_DOWNLOAD=1 \
  bash ./download_halueval_boolq.sh

# 只验证这次新增的纯代码契约；不会加载真实模型或数据到 GPU。
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_unsupervised_extraction.py'
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_unsupervised_pattern_audit.py'
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_unsupervised_token_graph.py'
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_halueval_boolq_shell_contracts.py'

nvidia-smi

RUN_ROOT="$DATA_ROOT/feature_extraction/token_graph_pilot_v3_64"
mkdir -p "$RUN_ROOT"

nohup env \
  DATA_ROOT="$DATA_ROOT" \
  MODEL_PATH="$MODEL_PATH" \
  PYTHON_BIN="$PYTHON_BIN" \
  GPU_ID=0 \
  DTYPE=float16 \
  PILOT_LIMIT=64 \
  MAX_TOKENS=1024 \
  MAX_ATTENTION_GIB=6 \
  RUN_ROOT="$RUN_ROOT" \
  DOWNLOAD_DATA=0 \
  bash ./run_halueval_boolq_feature_pilots.sh \
  >> "$RUN_ROOT/pilot.log" 2>&1 &

echo $! | tee "$RUN_ROOT/pilot.pid"
```

## 查看进度

```bash
tail -n 200 -f "$RUN_ROOT/pilot.log"
```

另开终端查看 GPU 和已保存文件：

```bash
nvidia-smi

find "$RUN_ROOT" -path '*/extraction/traces/*.pt' -type f | wc -l
find "$RUN_ROOT" -path '*/extraction/graphs/*.pt' -type f | wc -l
du -sh "$RUN_ROOT"
```

日志必须先出现一行 `"event": "GPU_WITNESS"`，之后依次出现 `Starting halueval_qa` 和 `Starting boolq`。同一 `RUN_ROOT` 重跑会按模型签名、dtype、文本、层和构图参数校验 fingerprint，然后复用已经完整保存的样本。

完成后优先阅读：

```text
${RUN_ROOT}/halueval_qa/pattern_audit/pattern_audit.md
${RUN_ROOT}/halueval_qa/pattern_audit/evaluation_error_cases.jsonl
${RUN_ROOT}/halueval_qa/pattern_audit/evaluation_pair_deltas.jsonl
${RUN_ROOT}/boolq/pattern_audit/pattern_audit.md
${RUN_ROOT}/boolq/pattern_audit/evaluation_error_cases.jsonl
```

注意：全层全头 dense attention 的时间和磁盘复杂度是 `O(layers × heads × tokens²)`。因此第一轮不要直接把 `PILOT_LIMIT` 改成 10,000。先确认 64 条的运行时间、磁盘占用、BoolQ 自然错误数量以及错误模式方向，再扩大到 300。

本轮 `pattern_audit` 是同一无标签 pilot 池上拟合分位数/Mahalanobis、随后才连接标签的 transductive 探索审计，不是 held-out 最终性能。HaluEval 候选还可能带有人工生成风格/长度捷径；BoolQ 错误首先是阅读理解错误，不能未经案例分析直接称为事实幻觉。正式结论必须另设 reference/calibration split，并冻结层、阈值和特征后再评估。
