"""Small trajectory ranker used by Prompt-Anchored Topology Flow."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import torch
from torch import nn
from .signature import TopologySignature

@dataclass(frozen=True)
class RankerConfig:
    hidden_dim: int = 48
    epochs: int = 80
    learning_rate: float = 1e-3
    margin: float = 0.5
    origin_regularization: float = 0.02
    batch_size: int = 32
    validation_fraction: float = 0.20
    patience: int = 12
    seed: int = 42
    def __post_init__(self):
        if min(self.hidden_dim, self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("ranker dimensions, epochs, batch size and patience must be positive")
        if self.learning_rate <= 0 or self.margin <= 0:
            raise ValueError("learning_rate and margin must be positive")
        if not 0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in [0, 0.5)")

@dataclass
class RobustNormalizer:
    median: torch.Tensor
    scale: torch.Tensor
    @classmethod
    def fit(cls, xs: Iterable[torch.Tensor]):
        x = torch.cat([torch.as_tensor(v, dtype=torch.float32) for v in xs])
        return cls(x.median(0).values, (torch.quantile(x,.75,dim=0)-torch.quantile(x,.25,dim=0)).clamp_min(1e-4))
    def transform(self, x):
        return (torch.as_tensor(x,dtype=torch.float32)-self.median)/self.scale

class TopologyFlowRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(); self.input_dim=input_dim; self.hidden_dim=hidden_dim
        self.project=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim))
        self.gru=nn.GRU(hidden_dim,hidden_dim,batch_first=True)
        self.head=nn.Sequential(nn.Linear(3*hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))
    def forward(self,x):
        if x.ndim!=3: raise ValueError("trajectory must have shape [batch, layers, features]")
        seq,h=self.gru(self.project(x)); return self.head(torch.cat((seq[:,0],h[-1],seq[:,-1]-seq[:,0]),-1)).squeeze(-1)

@dataclass
class RankerCheckpoint:
    model: TopologyFlowRanker
    normalizer: RobustNormalizer
    feature_names: tuple[str,...]
    config: RankerConfig
    topology_config: dict[str,object]

def _split(groups, fraction, seed):
    unique=sorted(set(groups)); all_idx=torch.arange(len(groups))
    if fraction<=0 or len(unique)<3: return all_idx,torch.empty(0,dtype=torch.long)
    order=torch.randperm(len(unique),generator=torch.Generator().manual_seed(seed)).tolist()
    n=min(max(1,round(len(unique)*fraction)),len(unique)-1); chosen={unique[i] for i in order[:n]}
    train=torch.tensor([i for i,g in enumerate(groups) if g not in chosen]); val=torch.tensor([i for i,g in enumerate(groups) if g in chosen])
    if not len(train) or not len(val): raise ValueError("source-group split produced an empty partition")
    return train,val

def _metrics(model,a,b,idx,margin):
    if not len(idx): return float("nan"),float("nan")
    model.eval()
    with torch.no_grad():
        sa,sb=model(a[idx]),model(b[idx]); return float(torch.relu(margin-(sb-sa)).mean()),float((sb>sa).float().mean())

def fit_ranker(originals, counterfactuals, *, group_keys=None, config=None, topology_config=None, device="cpu"):
    config=config or RankerConfig()
    if not originals or len(originals)!=len(counterfactuals): raise ValueError("paired non-empty signatures are required")
    names=originals[0].feature_names; shape=originals[0].trajectory.shape
    if any(x.feature_names!=names or x.trajectory.shape!=shape for x in originals+counterfactuals):
        raise ValueError("all signatures must share feature names and trajectory shape")
    groups=group_keys or [x.source_id if x.source_id!="unknown" else x.sample_id for x in originals]
    train,val=_split(groups,config.validation_fraction,config.seed); normalizer=RobustNormalizer.fit(originals[i].trajectory for i in train.tolist())
    dev=torch.device(device); a=torch.stack([normalizer.transform(x.trajectory) for x in originals]).to(dev); b=torch.stack([normalizer.transform(x.trajectory) for x in counterfactuals]).to(dev)
    train,val=train.to(dev),val.to(dev); torch.manual_seed(config.seed)
    model=TopologyFlowRanker(len(names),config.hidden_dim).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    rng=torch.Generator().manual_seed(config.seed); best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_loss=float("inf"); stale=0; history=[]
    for epoch in range(config.epochs):
        model.train(); losses=[]; acc=[]
        order=train.cpu()[torch.randperm(len(train),generator=rng)]
        for start in range(0,len(order),config.batch_size):
            idx=order[start:start+config.batch_size].to(dev); sa,sb=model(a[idx]),model(b[idx])
            rank=torch.relu(config.margin-(sb-sa)).mean(); loss=rank+config.origin_regularization*sa.square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach())); acc.append(float((sb>sa).float().mean()))
        vr,va=_metrics(model,a,b,val,config.margin); selection=vr if len(val) else sum(losses)/len(losses)
        if selection<best_loss-1e-6: best_loss=selection; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        history.append({"epoch":float(epoch+1),"train_loss":sum(losses)/len(losses),"train_pair_accuracy":sum(acc)/len(acc),"validation_ranking_loss":vr,"validation_pair_accuracy":va})
        if len(val) and stale>=config.patience: break
    model.load_state_dict(best)
    return RankerCheckpoint(model.cpu(),normalizer,names,config,dict(topology_config or originals[0].config)),history

def score_signatures(checkpoint, signatures):
    if not signatures: return torch.empty(0)
    x=torch.stack([checkpoint.normalizer.transform(s.trajectory) for s in signatures])
    checkpoint.model.eval()
    with torch.no_grad(): return checkpoint.model(x).cpu()

def save_checkpoint(checkpoint, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save({"schema":"prompt-anchored-topology-flow-ranker-v2","state_dict":checkpoint.model.state_dict(),"input_dim":checkpoint.model.input_dim,"hidden_dim":checkpoint.model.hidden_dim,"normalizer_median":checkpoint.normalizer.median,"normalizer_scale":checkpoint.normalizer.scale,"feature_names":checkpoint.feature_names,"ranker_config":asdict(checkpoint.config),"topology_config":checkpoint.topology_config},tmp); tmp.replace(path)

def load_checkpoint(path):
    p=torch.load(Path(path),map_location="cpu",weights_only=True)
    if p.get("schema")!="prompt-anchored-topology-flow-ranker-v2": raise ValueError("unsupported topology-flow checkpoint schema")
    cfg=RankerConfig(**dict(p["ranker_config"])); model=TopologyFlowRanker(int(p["input_dim"]),int(p["hidden_dim"])); model.load_state_dict(p["state_dict"])
    return RankerCheckpoint(model,RobustNormalizer(torch.as_tensor(p["normalizer_median"]),torch.as_tensor(p["normalizer_scale"])),tuple(p["feature_names"]),cfg,dict(p["topology_config"]))
