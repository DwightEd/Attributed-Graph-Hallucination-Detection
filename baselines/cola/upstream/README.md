# Upstream CoLA source

`model.py` is copied verbatim from `TrustAGI-Lab/CoLA` at commit
`c2b5273fe6368509dfc558a485764003f5f18ca3`.

The upstream MIT license is retained in this directory. The RAGTruth adapter
outside `upstream/` changes only data loading, multi-graph batching, response
node selection, and response-level aggregation; it does not add edge features
or alter the upstream GCN/readout/discriminator.
