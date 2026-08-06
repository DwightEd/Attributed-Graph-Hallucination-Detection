# Prompt-Anchored Topology Flow（PATF）

## 1. 研究假设

PATF直接编码四个待验证现象，而不是期待一个通用GNN从图级标签中自行发现它们：

1. 幻觉回答与prompt的直接连接更弱；
2. 幻觉回答更依赖已经生成的response；
3. 幻觉图的有效支持边更少且更局部；
4. 剩余attention质量集中在更少的边和source hub上。

核心对象是每个layer/head上的严格因果有向attention DAG。算法不把所有层头提前平均，也不把拓扑仅作为MLP消息传递的路由。

## 2. Prompt锚定路径分解

对response token `t`，去掉self-attention diagonal后，在全部严格因果source内归一化：

\[
\sum_{j<t} a_{tj}=1.
\]

直接prompt质量为：

\[
p_t=\sum_{j\in P}a_{tj}.
\]

对prompt节点定义grounding ancestry为1。对response节点从左到右递归：

\[
g_t=p_t+\gamma\sum_{j\in R,j<t}a_{tj}g_j,\qquad 0<\gamma<1.
\]

`gamma`使多跳response relay逐步衰减；若不衰减，严格因果attention walk最终总会回到prompt，grounding会退化为恒等于1。

response依赖进一步分成：

\[
r_t^{+}=\sum_{j\in R,j<t}a_{tj}g_j,
\]

\[
r_t^{-}=\sum_{j\in R,j<t}a_{tj}(1-g_j).
\]

其中 `r+` 是已有依据的response中继，`r-` 是缺少prompt ancestry的自反馈。两条样本即使具有相同RP/RR总质量，只要RR边指向的历史节点不同，也会得到不同的 `r-`，因此该量真正依赖source incidence和多跳路径。

稀疏缓存可能没有保留某些response行。PATF不把它们伪装成“无依据”，而是显式记录`observed_row_fraction`和`unknown_response_feedback`。对可观测且具有grounded ancestry的路径，条件期望跳数为：

\[
d_t=\frac{p_t+\gamma\sum_{j\in R,j<t}a_{tj}g_j(d_j+1)}{g_t}.
\]

## 3. 自适应Mass-Cover支持图

固定threshold会受到context长度、layer和head尺度影响。PATF对每个query独立选择最少的边，使累计attention达到 `rho`：

\[
E_t^{\rho}=\arg\min_{E}\{|E|:\sum_{j\in E}a_{tj}\geq\rho\}.
\]

在该有向支持图上计算：

- edge sparsity：`1 - |E_t^rho| / t`；
- response locality：RR边相对lag越短，值越高；
- weight concentration：归一化HHI；
- prompt-rooted reachability：是否存在由prompt或已reachable response节点到当前节点的有向路径；
- prompt source coverage；
- source hub concentration。

每层先对head取中位数和IQR，得到有序拓扑轨迹：

\[
T=[t^{(1)},t^{(2)},\ldots,t^{(L)}].
\]

## 4. 无标签反事实训练

训练阶段不读取幻觉标签。对每个真实图构造机制对齐的counterfactual：

### Incidence erosion

- 将一部分prompt质量转移至最近的response历史节点；
- 将RR source重新定位到局部窗口；
- 保持每行归一化。

### Support collapse

- 删除弱支持边；
- 对剩余权重做幂变换并重新归一化；
- 产生更少、更集中、更局部的支持。

### Composite erosion

同时执行以上两类变化，形成四个现象共同增强的伪异常。

小型GRU只编码层间拓扑轨迹，并使用pairwise ranking：

\[
\mathcal L_{rank}=\max(0,m-s(\widetilde T)+s(T)).
\]

因此神经网络不是图创新本身；创新主体是prompt锚定路径分解、mass-cover拓扑和机制保持的反事实任务。

## 5. 必须执行的判定消融

1. **Source shuffle**：保持target、relation、degree、weight和lag bucket，重连source；
2. **No ancestry**：仅保留RP/RR质量，删除递归路径量；
3. **No mass-cover**：改回固定threshold；
4. **No trajectory**：对全部层直接平均；
5. **Incidence-only / collapse-only / composite**；
6. **Feature MLP**：相同统计量但打乱layer顺序；
7. **Legacy MaskGAE/GMM**。

只有PATF在真实幻觉测试上稳定优于source-shuffle、无路径分解和静态统计量，才能声称拓扑提供独立贡献。

## 6. 当前代码入口

先验证已有缓存兼容性：

```bash
python main.py validate --input-dir /path/to/attention_cache --limit 32
```

完整流程：

```bash
python main.py extract \
  --input-dir /path/to/attention_cache \
  --output /path/to/topology_features.jsonl

python main.py train \
  --input-dir /path/to/train_attention \
  --output-dir /path/to/patf_training \
  --device cuda

python main.py score \
  --input-dir /path/to/test_attention \
  --checkpoint /path/to/patf_training/topology_flow_ranker.pt \
  --output /path/to/test.topology_scores.jsonl

python main.py evaluate \
  --scores /path/to/test.topology_scores.jsonl \
  --responses /path/to/RAGTruth/response.jsonl \
  --sources /path/to/RAGTruth/source_info.jsonl \
  --output /path/to/evaluation.json
```

原`attention_graph` CLI在迁移阶段仍可通过根入口的旧参数使用，但不再是新算法的核心依赖。


## 7. 运行约束

- train和test必须来自相同模型，层数与head数必须一致；
- 训练/验证按`source_id`分组，避免同一来源进入两侧；
- checkpoint保存mass-cover、relay discount和head reducer，score阶段默认自动复用并拒绝不一致覆盖；
- 支持dense `[L,H,N,N]`和正式`sparse-response-csr-v1`缓存；大型`.pt`文件默认mmap读取；
- `run_ragtruth_topology_flow.sh`会先预检train/test，再训练、打分，并在提供RAGTruth元数据时评价。
