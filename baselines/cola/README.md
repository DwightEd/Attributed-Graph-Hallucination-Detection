# CoLA baseline

This baseline adapts the original CoLA implementation to the saved RAGTruth
attention attributed graphs.

The upstream neural model is kept byte-identical in `upstream/model.py`, pinned
to `TrustAGI-Lab/CoLA@c2b5273fe6368509dfc558a485764003f5f18ca3`.
Only the data/scenario interface is adapted:

- one saved RAGTruth response graph is treated as one CoLA graph;
- target nodes are response tokens only;
- node features are exactly the original graph `x` (attention diagonal);
- `edge_index` is converted to the undirected, unweighted adjacency expected by CoLA;
- `edge_attr` and `edge_mark` are intentionally unused because upstream CoLA has no edge-feature encoder;
- the original target/context tensor layout, GCN, readout, bilinear discriminator,
  BCE contrastive loss, training-loss checkpoint rule, and anomaly score are retained;
- train graphs are used without labels; labels are opened only after test scores are frozen;
- token anomaly scores are averaged within each response for response-level evaluation.

The original CoLA helper used the removed DGL 0.4 `dgl.contrib` RWR API. The
adapter therefore reproduces the same `subgraph_size-1` local-context plus
target contract in `sampler.py` for current runtimes. The upstream neural model
and contrastive objective are not modified.

Run:

```bash
bash scripts/run_cola.sh
```
