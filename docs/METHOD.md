# Attention-only relation-aware graph method

## 1. Scope and status

This document specifies the current method for unsupervised, response-level
hallucination pattern discovery from RAGTruth attention caches. It describes the
implemented design and the assumptions that must be tested. It does **not**
claim that the method improves AUROC, discovers a new hallucination mechanism,
or outperforms any cited baseline; those claims require completed experiments
and ablations.

The method has two stages:

1. learn an attention-graph representation with a relation-aware masked graph
   autoencoder;
2. discover two graph-level behavior patterns with a free-weight mixture model
   and orient the two otherwise exchangeable components with a pre-specified
   exploratory structural prior.

The separation between representation learning and component orientation is
important. No hallucination label is used to construct a graph, train the graph
encoder, reconstruct a masked view, fit the mixture, or select its component
sizes.

## 2. Input contract: attention only

Each response is represented by one directed attributed graph. The learned
content input contains only self-attention values from every retained layer and
head. In particular, the method does not use hidden states, logits, token
embeddings, lexical features, or hallucination labels.

The graph also contains deterministic coordinates needed to interpret the
attention graph: normalized token position and prompt/response membership.
These coordinates define graph roles; they are not language-model hidden-state
features. Token IDs are retained for identity and auditing but are not embedded
as semantic node features by the current encoder.

Let the model have \(L\) attention layers, \(H\) heads, and
\(C=L\times H\) ordered channels. Flattening a channel does not discard its
identity: channel \(c=lH+h\) is associated with a learned channel embedding.

### 2.1 Nodes

There is one node for each prompt or response token. The attribute of token
\(i\) is its complete self-attention diagonal:

\[
x_i = [a^{1,1}_{ii},\ldots,a^{L,H}_{ii}] \in [0,1]^C.
\]

The implementation stores this tensor as `node_attr[N, C]`. It does not replace
the \(L\times H\) channels with mean, standard deviation, entropy, or any other
hand-crafted attention summary.

### 2.2 Directed RP/RR edges

Only causal history-to-response pairs are eligible for edges. For response
target \(i\) and historical source \(j<i\), the directed edge is \(j\to i\).
Its relation is

\[
r_{ji}=\begin{cases}
\mathrm{RP}, & j < b,\\
\mathrm{RR}, & b \le j < i,
\end{cases}
\]

where \(b\) is the first response-token index. RP therefore represents
prompt-to-response attention, while RR represents attention from a response
token to previously generated response tokens. Prompt tokens are still nodes
and grounding sources, but the current graph does not add prompt-target or
future-token edges.

An edge is a token pair, not one averaged attention scalar. Its attributes are
stored sparsely as triples

\[
(e,c,a^{c}_{ji}),
\]

where `trace_edge_id` identifies the pair, `trace_channel` identifies the
layer/head channel, and `trace_value` is the retained value. This avoids both a
dense `edges x C` allocation and premature head/layer averaging.

The formal cache retains values only above its extraction floor \(\tau\).
Consequently, a missing channel is **cache-censored**, not an observed zero.
This distinction constrains the interpretation of support reconstruction and
negative sampling below.

The `train` and `test` directories preserve official RAGTruth split identity,
but directory presence alone does not prove that every sample in a split has
been extracted. Partial inventories are valid for explicitly marked smoke
tests. Formal dataset comparisons require a complete test inventory and must
report the actual split counts recorded by the run.

## 3. Graph support selection

Graph construction supports three policies. They change which cached token
pairs enter the graph; they never alter the retained per-channel values.

### 3.1 `threshold`

A pair is selected when its largest observed layer/head value reaches a
threshold \(\tau_g\), with \(\tau_g\ge\tau\). Before any cap, this policy
preserves natural degree variation and directly represents support above a
declared attention floor. The current default then keeps at most 64 of the
strongest selected pairs per response target for memory safety. Degrees above
64 are therefore truncated: the default graph is not a complete untruncated
attention graph.

### 3.2 `global_topk`

For each response target, RP and RR candidates compete in one pool. The method
keeps the top \(k\) cached pairs by pair score, where the current pair score is
the sum of observed channel values divided by \(C\). This controls degree but
can allow one relation to dominate the other.

### 3.3 `typed_topk`

For each response target, the top \(k\) pairs are selected independently inside
RP and RR. This prevents an abundant relation from removing every retained pair
of the other relation. It does not invent an RP or RR edge when that relation
has no cached candidate.

Threshold and top-k graphs answer different questions. Threshold measures
support above a fixed magnitude, while top-k measures relative routing under a
degree budget. They must be compared experimentally rather than treated as
interchangeable preprocessing.

## 4. Relation-aware masked graph autoencoder

The encoder is inspired by masked graph modeling but is specialized for sparse,
typed attention traces. It learns channel identity and attention magnitude
jointly:

- node channels are projected through learned layer/head channel embeddings;
- visible traces form sparse edge-by-channel matrices. Sparse matrix
  multiplication projects both channel presence and channel-weighted attention
  into the learned channel basis, while a linear projection retains pair-level
  mean magnitude;
- the reduction never creates a `trace_nnz * latent_dim` activation;
- RP and RR have distinct learned relation embeddings used by message passing
  and the decoders;
- message passing follows the directed causal edges, and a graph embedding is
  the mean of the final response-node embeddings.

Thus the graph encoder does not receive a single precomputed mean-attention
edge feature. Its compression from \(C\) channels into the latent dimension is
learned end to end.

### 4.1 Relation-stratified masking

Edges are grouped by `(response target, relation)`. A singleton group remains
visible. For every larger group, at least one edge remains visible after
masking. This safeguard prevents augmentation from deliberately deleting an
entire available RP or RR route and turning a grounded graph into a synthetic
weak-prompt graph.

Only response-node diagonal attributes are node-masked; prompt nodes remain as
grounding context. Optional channel dropout removes the same sampled
layer/head channels from the current view while passing an explicit channel
keep mask, so a dropped channel is not interpreted as a measured zero.

The current training objective contains four label-free losses.

### 4.2 Retained-support loss

Masked selected pairs are positive support examples. For each such pair, a
causally legal unselected source with the same target and RP/RR relation is
sampled as a negative when that relation domain is not saturated. A
relation-aware decoder predicts pair support with binary cross-entropy:

\[
\mathcal L_{\mathrm{support}}
=\tfrac12\operatorname{BCE}(s^+,1)
+\tfrac12\operatorname{BCE}(s^-,0).
\]

Because the cache is censored, this objective predicts **membership in the
retained/selected support**, not whether the original dense Transformer
attention was mathematically zero. This limitation must remain explicit in any
paper or result description.

### 4.3 Per-channel attention-weight loss

For sampled retained trace entries attached to a masked pair, the decoder receives
the source embedding, target embedding, RP/RR relation, and ordered channel
embedding. It predicts the raw value with a sigmoid output and Smooth L1 loss:

\[
\mathcal L_{\mathrm{weight}}
=\operatorname{SmoothL1}(\hat a^{c}_{ji},a^{c}_{ji}).
\]

This term preserves head/layer-specific magnitudes instead of supervising only
an edge average. The default training configuration samples at most 65,536
masked traces per graph step, stratified by response target and RP/RR domain.

### 4.4 Attention-row distribution loss

For each sampled active `(response target, channel)` row, retained edge entries form
explicit categories. An additional `OTHER` category absorbs history attention
mass that was censored by the cache or excluded by graph selection. Since
\(a^c_{ii}\) is the self-attention diagonal, the available history mass is
\(1-a^c_{ii}\). The target distribution is therefore

\[
p^c_{ji}=\frac{a^c_{ji}}{1-a^c_{ii}},\qquad
p^c_{\mathrm{OTHER}}
=\frac{1-a^c_{ii}-\sum_{j\in E_i}a^c_{ji}}{1-a^c_{ii}},
\]

with numerical clamping for finite computation. A decoder predicts logits for
each retained entry and the `OTHER` bucket, and the loss is their categorical
cross-entropy:

\[
\mathcal L_{\mathrm{distribution}}
=-\sum_k p^c_{ik}\log \hat p^c_{ik}.
\]

This objective represents concentration and relative routing without claiming
that the sparse cache contains every original attention entry.

The default samples at most 512 complete rows per graph step and keeps every
retained category inside each selected row. It never samples individual
categories and renormalizes a broken row.

### 4.5 Node diagonal loss

Masked response-node self-attention vectors are reconstructed with the squared
scaled cosine error used in the GraphMAE family:

\[
\mathcal L_{\mathrm{node}}
=\mathbb E_{i\in M_V}
\left(1-\cos(\hat x_i,x_i)\right)^2.
\]

The complete training objective is

\[
\mathcal L
=\lambda_s\mathcal L_{\mathrm{support}}
+\lambda_w\mathcal L_{\mathrm{weight}}
+\lambda_d\mathcal L_{\mathrm{distribution}}
+\lambda_x\mathcal L_{\mathrm{node}}.
\]

These targets are generated from the input graph itself. Hallucination labels
are neither reconstruction targets nor model inputs.

For scalability, support positives are capped at 8,192, masked traces at
65,536, and complete distribution rows at 512 per graph step. Sampling is
stratified by target/relation where applicable, and decoder calls are chunked
at 16,384 entries. These limits are recorded in `run.json`.

## 5. Free two-component behavior mixture

The masked autoencoder learns a graph representation; it does not by itself
assign the two semantic classes. At response scoring time, the current system averages
the response-graph embedding and the four reconstruction energies over several
masked views. The mixture feature is

\[
u_G=[z_G;\log(1+e_s),\log(1+e_w),\log(1+e_d),\log(1+e_x)].
\]

Features are robustly centered by their median and scaled primarily by
`IQR / 1.349`, with a standard-deviation fallback for degenerate dimensions.
A diagonal-covariance Gaussian mixture with \(K=2\), ten initializations, and
freely learned component weights is then fitted on unlabeled graphs. There is
no fixed contamination rate, balanced-cluster constraint, or requirement that
one component contain fewer samples.

### 5.1 Why an orientation anchor is necessary

Mixture component indices are exchangeable. Without a label, prevalence
assumption, or semantic prior, it is mathematically impossible to determine
which of two discovered components should be named “hallucinated.” The current
method uses a pre-specified exploratory attention-structure prior only after
mixture assignment:

\[
d_G=-f_{RP}+f_{RR}-\rho_E-\bar\ell+C_{ret},
\]

where \(f_{RP}\) and \(f_{RR}\) are retained prompt/response mass fractions,
\(\rho_E\) is normalized mean in-degree, \(\bar\ell\) is attention-weighted
normalized causal lag, and \(C_{ret}\) is retained-row concentration. The
component with the higher mean \(d_G\) is oriented as the hallucination-like
component, reflecting the pre-specified exploratory hypothesis:

- weaker RP grounding;
- stronger RR self-reliance;
- fewer or more local retained links;
- greater retained-attention concentration.

The direction score is not used to train the graph encoder or fit the Gaussian
components. It assigns semantic orientation after the two modes have been
found. The reported `hallucination_probability` is the posterior responsibility
of the oriented component, not a supervised probability calibration.

Token scoring applies the same free-weight mixture. Every response token is
node-masked exactly once across deterministic phases. Its feature combines the
full-view token embedding with RP/RR support and weight reconstruction,
row-distribution reconstruction, and node reconstruction energies. Sentence
scores are label-free arithmetic means of member-token responsibilities after
strict tokenizer-offset replay; no top-fraction or rare-token assumption is
used. Only after all scores are frozen, the one-command evaluator also assigns
a sentence label by the fixed any-positive-member-token rule and reports
sentence-level rank metrics.

## 6. Why this is not a one-class method

Deep one-class and OOD methods generally learn a reference distribution from
normal or in-distribution training graphs and score distance or inconsistency
from that reference. That is incompatible with the intended RAGTruth setting:
the official training split contains both correct and hallucinated responses,
and the method must not silently assume that hallucinations are rare or that a
clean normal-only subset is available.

The free \(K=2\) mixture instead asks whether two behavior modes can be found
inside a mixed unlabeled training population. It assumes two sufficiently
stable, separable modes and that both are represented in the fit set. It does
not assume which mode is smaller. This is unsupervised clustering followed by
prior-based semantic orientation, not one-class anomaly detection.

For label-free fitting with post-hoc evaluation, report both the anchored score
and orientation-free discrimination. Using train labels to select the component
orientation or tune the mixture would be weak supervision and must be reported
separately.

### 6.1 Current output semantics

The current evaluator reports response-, token-, and (when the strict RAGTruth
offset adapter is enabled) sentence-level AUROC and average precision. The
random reference for average precision is the corresponding positive fraction,
so average-precision lift must be interpreted relative to that prevalence.
Orientation-free metrics take the better of the score and its reverse; they
diagnose separability but do not validate the semantic direction assigned by
the exploratory prior.

`hallucination_probability` is a posterior responsibility under the oriented
two-component Gaussian mixture. It is not a supervised, calibrated probability
of factual error. The sentence JSONL is the arithmetic mean of label-free token
responsibilities; `evaluation.json.sentence` evaluates that frozen score using
the fixed rule that a sentence is positive if any member token is positive.

## 7. Required ablations

The following experiments are necessary before claiming that graph modeling,
rather than attention marginals or dataset shortcuts, explains performance.

### 7.1 Does graph structure matter?

1. **No message passing:** set message-passing steps to zero while keeping the
   same node and trace inputs.
2. **Relation-preserving source shuffle:** permute sources and trace association
   while preserving targets, RP/RR counts, and degree. A structure-sensitive
   model should change.
3. **Node/edge statistic MLP:** give a non-graph baseline the same pooled raw
   information without adjacency.
4. **RP/RR collapse:** replace the two relations with one relation.
5. **Relation permutation:** swap or randomize relation IDs without changing
   edge endpoints.
6. **Node-only and edge-only:** remove edge reconstruction or node diagonal
   reconstruction in turn.

### 7.2 Does preserving layers and heads matter?

Compare full ordered \(L\times H\) channels against layer mean, head mean, and
one global mean. The comparison must keep encoder capacity and evaluation
splits as similar as practical.

### 7.3 How should support be selected?

Compare `threshold`, `global_topk`, and `typed_topk`; sweep \(\tau_g\) and
\(k\); report edge count, RP/RR coverage, isolated response targets, memory,
and accuracy metrics. A good result under top-k alone does not establish that a
fixed magnitude threshold is unnecessary, and vice versa.

### 7.4 Which self-supervised target matters?

Run support-only, weight-only, distribution-only, node-only, and cumulative
combinations. Compare unrestricted random edge masking with relation-stratified
guarded masking. Sweep edge, node, and channel mask rates.

### 7.5 Is the final scorer robust to a mixed fit set?

Compare the free mixture with an OCSVM/OCGTL-style one-class baseline under:

- a normal-only oracle fit set;
- the actual mixed unlabeled RAGTruth train split;
- controlled contamination or prevalence sweeps;
- free mixture weights versus forced balanced weights.

Report both anchored and orientation-free results. This isolates representation
quality from assumptions made by the final scorer.

### 7.6 Are results driven by nuisance variables?

Audit graph score and component assignment against task type, generator model,
prompt length, response length, node/edge count, and attention-cache density.
Future nuisance analysis should report within-stratum metrics and quantify the
association between mixture assignment and each nuisance variable. If
necessary, robustly center mixture features inside known task/generator strata
without using hallucination labels.

All central results should include multiple random seeds and source-ID-disjoint
train/validation partitions while holding samples with official test identity
out for evaluation. This split identity does not itself certify a complete
cache inventory. Dataset labels are joined only after scoring for evaluation.

## 8. Limitations and falsifiable boundaries

- **Censored cache:** absent trace entries are not observed zeros. Support loss
  predicts retained graph membership, and results depend on the extraction
  floor.
- **Selection-induced structure:** top-k degree is imposed by the graph builder;
  threshold degree is sensitive to scale and the default per-target cap
  truncates values above 64. Selection, cap, and edge counts must be reported.
- **Two-mode assumption:** \(K=2\) may be too simple if attention behavior is
  multimodal within correct or hallucinated responses.
- **Gaussian assumption:** a diagonal Gaussian mixture cannot model arbitrary
  curved or strongly correlated embedding distributions.
- **Semantic anchor:** the hallucination direction is a prior derived from
  earlier exploratory observations, not a theorem. If both mixture components
  have similar direction means, semantic orientation is unreliable.
- **Nuisance clustering:** the mixture may separate tasks, generators, or
  lengths rather than factuality. Stratified audits are mandatory.
- **Reconstruction is not detection:** a sufficiently expressive autoencoder
  may reconstruct both response types well. Separation must be demonstrated,
  not presumed.
- **No causal claim:** RP/RR differences are observational properties of model
  attention. They do not establish a causal mechanism for hallucination.
- **Sentence-level output:** the emitted sentence score is an uncalibrated mean
  of token-mode responsibilities. Its post-hoc metric uses an any-positive-token
  label; neither the score nor this rule makes it a calibrated probability that
  the sentence is hallucinated.

Any violated boundary should be treated as evidence to revise the model, not as
an evaluation detail to hide.

## 9. Relation to existing methods

The method combines ideas from prior work but is not a verbatim implementation
of any one paper.

| Work | Relevant idea | Deliberate difference here |
|---|---|---|
| [GraphMAE](https://arxiv.org/abs/2205.10803) | Masked node-feature reconstruction and scaled cosine error | Only the \(L\times H\) self-attention diagonal is reconstructed; graph-level anomaly scoring is added separately. |
| [MaskGAE](https://arxiv.org/abs/2205.10053) ([official code](https://github.com/EdisonLeeeee/MaskGAE)) | Mask observed edges and reconstruct missing support | Edges are directed, typed RP/RR, channel-attributed, and cache-censored; retained weights and row distributions are also reconstructed. |
| [CoLA](https://arxiv.org/abs/2103.00113) ([official code](https://github.com/TrustAGI-Lab/CoLA)) | Contrast a node with its local substructure for node anomaly detection | CoLA is most suitable as a token-level baseline on one attributed network; the present task assigns one graph to each response and learns graph-level modes. |
| [OCGTL](https://arxiv.org/abs/2205.13845) ([official code](https://github.com/boschresearch/GraphLevel-AnomalyDetection)) | End-to-end graph-level one-class transformation learning | Useful as a baseline, but its normal-training reference assumption is not adopted for mixed RAGTruth training data. |
| [SIGNET](https://arxiv.org/abs/2310.16520) | Stable original/dual-hypergraph views and self-interpretable subgraph rationales; warning that arbitrary perturbations can create anomaly-like views | The current method stays on a directed attribute graph and uses guarded RP/RR masks; dual-hypergraph rationale extraction is not claimed as implemented. |
| [GOOD-D](https://arxiv.org/abs/2211.04208) | Perturbation-free graph views and multi-granularity inconsistency | Its ID-only training and hand-designed structural view are not adopted; attention-channel reconstruction supplies the self-supervised signal. |
| [UB-GOLD](https://arxiv.org/abs/2406.15523) ([official code](https://github.com/UB-GOLD/UB-GOLD)) | Unified GLAD/GLOD evaluation and explicit training-contamination robustness analysis | Used as an evaluation and baseline reference; it is a benchmark, not the present detector, and its standard clean-ID split is not assumed. |
| [DiffGAD](https://github.com/fortunato-all/DiffGAD/tree/main) | A readable separation between entry point, method components, configuration, and experiments | Used only as a code-organization reference. No diffusion objective or reported DiffGAD result is attributed to this method. |

The main research question remains falsifiable: after preserving full
layer/head attention and controlling nuisance variables, do directed RP/RR
message passing and typed reconstruction reveal two reproducible response
behavior modes that align with factual correctness better than attention-only
non-graph baselines?
