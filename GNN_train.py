# run_hypercharm_stable.py (稳定版)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import gc
import json
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score, average_precision_score

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


def make_mlp(in_dim, hidden_dims, out_dim):
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class HyperCharm(MessagePassing):
    def __init__(self, node_dim, hedge_dim, hidden_dim, residual=True):
        super().__init__(aggr="add")
        self.residual = residual

        # same MLP used in both modes (VERY IMPORTANT for fair ablation)
        self.node2edge = make_mlp(node_dim + 2, [hidden_dim], hidden_dim)
        self.edge2node = make_mlp(hedge_dim + hidden_dim, [hidden_dim], node_dim)
        self.ln_out = nn.LayerNorm(node_dim)

    # def build_pairwise_edges(self, he_index, num_nodes):
    #     """
    #     Convert hyperedge incidence to pairwise edges when |e|=2
    #     """
    #     node_ids = he_index[0]
    #     he_ids = he_index[1]

    #     num_he = he_ids.max().item() + 1
    #     pair_nodes = [[] for _ in range(num_he)]

    #     for n, e in zip(node_ids.tolist(), he_ids.tolist()):
    #         pair_nodes[e].append(n)

    #     src, dst = [], []
    #     for u, v in pair_nodes:
    #         src.append(u); dst.append(v)
    #         src.append(v); dst.append(u)

    #     edge_index = torch.tensor([src, dst], device=node_ids.device)
    #     return edge_index

    def forward(self, x, data, graph_mode=False):
        """
        graph_mode=False : Hypergraph MP
        graph_mode=True  : Standard GNN MP (true pairwise)
        """
        # node_ids = he_index[0]
        # he_ids = he_index[1]

        # ============================
        # === Graph Mode (GNN) ======
        # ============================
        # if graph_mode:
        #     edge_index = self.build_pairwise_edges(he_index, x.size(0))

        #     src, dst = edge_index

        #     # use same MLP
        #     msg = self.node2edge(
        #         torch.cat([x[src], he_mark[he_ids[:len(src)//2]].repeat_interleave(2, dim=0)], dim=-1)
        #     )

        #     out = torch.zeros_like(x)
        #     out.index_add_(0, dst, msg)

        #     deg = torch.bincount(dst, minlength=x.size(0)).float().unsqueeze(-1)
        #     out = out / (deg + 1e-6)

        #     out = self.ln_out(out)
        #     return x + out if self.residual else out

        if graph_mode:
            src, dst = data.edge_index   # 直接用！
        
            msg = self.node2edge(
                torch.cat([x[src], data.he_mark[src]], dim=-1)
            )
        
            out = torch.zeros_like(x)
            out.index_add_(0, dst, msg)
        
            deg = torch.bincount(dst, minlength=x.size(0)).float().unsqueeze(-1)
            out = out / (deg + 1e-6)
        
            out = self.ln_out(out)
            return x + out if self.residual else out

        # ===============================
        # === Hypergraph Mode (HGNN) ===
        # ===============================
        # ----- node -> edge -----
        msg_ne = self.node2edge(torch.cat([x[node_ids], he_mark[he_ids]], dim=-1))

        agg_e = torch.zeros((he_attr.size(0), msg_ne.size(-1)), device=x.device)
        agg_e.index_add_(0, he_ids, msg_ne)
        agg_e = agg_e / (he_count.unsqueeze(-1) + 1e-6)

        # ----- edge -> node -----
        inc_msg = self.edge2node(torch.cat([he_attr[he_ids], agg_e[he_ids]], dim=-1))
        inc_msg = F.relu(inc_msg)

        out = torch.zeros_like(x)
        out.index_add_(0, node_ids, inc_msg)

        node_deg = torch.bincount(node_ids, minlength=x.size(0)).float().unsqueeze(-1)
        out = out / (node_deg + 1e-6)

        out = self.ln_out(out)
        return x + out if self.residual else out


class HyperCHARM(nn.Module):
    def __init__(self, node_dim, hedge_dim, hp):
        super().__init__()
        self.hp = hp

        self.in_proj = nn.Linear(node_dim, hp["hidden_dim"])

        self.layers = nn.ModuleList([
            HyperCharm(
                node_dim=hp["hidden_dim"],
                hedge_dim=hedge_dim,
                hidden_dim=hp["hidden_dim"],
                residual=hp["residual_mp"]
            )
            for _ in range(hp["gnn_layers"])
        ])

        self.pred = nn.Sequential(
            nn.Linear(hp["hidden_dim"], max(8, hp["hidden_dim"] // 2)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(max(8, hp["hidden_dim"] // 2), 1)
        )

    def forward(self, data):
        h = F.relu(self.in_proj(data.x))

        for layer in self.layers:
            # h = layer(
            #     h,
            #     data.he_index,
            #     data.he_attr,
            #     data.he_mark,
            #     data.he_count,
            #     graph_mode=True  # ⭐关键开关
            # )
            h = layer(h, data, graph_mode=True)

        return self.pred(h).view(-1)

class HyperGraphData(Data):
    @property
    def num_hedges(self) -> int:
        return int(self.he_attr.size(0)) if hasattr(self, "he_attr") and self.he_attr is not None else 0

    def __inc__(self, key, value, *args, **kwargs):
        if key == "he_index":
            # 必须是 (2,1) 才能和 (2,E) 正确广播相加
            return torch.tensor([[self.num_nodes], [self.num_hedges]], dtype=torch.long)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "he_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)



def he_to_edge_index_fast(he_index, he_count):
    """
    he_index: (2, E) incidence
    he_count: (num_he,)
    Assumes every hyperedge has exactly 2 nodes
    """
    node_ids = he_index[0]
    he_ids = he_index[1]

    # 找到每个 he 的两个 node（纯 tensor）
    perm = torch.argsort(he_ids)
    node_ids = node_ids[perm]
    he_ids = he_ids[perm]

    # reshape 成 (num_he, 2)
    pair_nodes = node_ids.view(-1, 2)

    src = torch.cat([pair_nodes[:, 0], pair_nodes[:, 1]], dim=0)
    dst = torch.cat([pair_nodes[:, 1], pair_nodes[:, 0]], dim=0)

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index
# ----------------- convert dict -> PyG Data -----------------
def graph_to_data(g):
  
    x = g["x"].clone().detach().to(torch.float32)
    he_index = g["he_incidence_index"].clone().detach().to(torch.long)
    he_attr = g["he_attr"].clone().detach().to(torch.float32)
    he_mark = g["he_mark"].clone().detach().to(torch.float32)
    he_count = g["he_member_counts"].clone().detach().to(torch.float32)
    y = g["y_token"].clone().detach().to(torch.float32)
    node_pos = torch.arange(x.size(0))
    response_idx = torch.tensor(g["response_idx"], dtype=torch.long)

    # # ---- clamp + normalize x ----
    # if x.numel() > 0:
    #     x = torch.clamp(x, -5.0, 5.0)
    #     x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    # # ---- clamp + normalize he_attr ----
    # if he_attr.numel() > 0:
    #     he_attr = torch.clamp(he_attr, 0.0, 1.0)
    #     he_attr = (he_attr - he_attr.mean(dim=0)) / (he_attr.std(dim=0) + 1e-6)

    # ⭐⭐⭐ 关键：一次性生成 edge_index
    edge_index = he_to_edge_index_fast(he_index, he_count)

    # ====== ⭐关键修复：node reindex ======
    unique_nodes, new_edge = torch.unique(edge_index, return_inverse=True)
    
    # new_edge 是 flatten 后的，需要 reshape 回去
    edge_index = new_edge.view(2, -1)
    
    # x / y / node_pos 全部按新编号对齐
    x = x[unique_nodes]
    y = y[unique_nodes]
    node_pos = node_pos[unique_nodes]

    return HyperGraphData(
        x=x,
        he_index=he_index,
        edge_index=edge_index,   # 加这个
        he_attr=he_attr,
        he_mark=he_mark,
        he_count=he_count,
        y=y,
        node_pos=node_pos,
        response_idx=response_idx
    )
    # return HyperGraphData(
    #     x=x,
    #     he_index=he_index,
    #     he_attr=he_attr,
    #     he_mark=he_mark,
    #     he_count=he_count,
    #     y=y,
    #     node_pos=node_pos,
    #     response_idx=response_idx
    # )


# ----------------- evaluate -----------------
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            prob = torch.sigmoid(logits)
            mask = batch.node_pos >= batch.response_idx[batch.batch]
            ys.append(batch.y[mask].cpu().numpy())
            ps.append(prob[mask].cpu().numpy())

    if len(ys) == 0:
        return {"auroc": 0.5, "aupr": 0.0}

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    try:
        auroc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
    except Exception:
        auroc = 0.5
    try:
        aupr = average_precision_score(ys, ps) if len(np.unique(ys)) > 1 else 0.0
    except Exception:
        aupr = 0.0

    return {"auroc": auroc, "aupr": aupr}


# ----------------- helpers -----------------
# 1) 重新计算 pos_weight：只考虑 response 区间的 token
def compute_pos_weight_on_response(dataset):
    ys = []
    for d in dataset:
        mask = d.node_pos >= d.response_idx
        ys.append(d.y[mask])
    y = torch.cat(ys)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return neg.float() / max(pos.float(), 1.0)


# ----------------- train_model with Focal Loss -----------------
def train_model(train_loader, val_loader,test_loader, node_dim, hedge_dim, hp):
    # 2) 训练里这么写：
    pos_weight_val = float(compute_pos_weight_on_response(train_loader.dataset))
    print(f"[Info] raw_pos_weight={pos_weight_val:.3f}")
    pos_weight_val = min(pos_weight_val, 8)  # 可以给个上限，比如 8
    print(f"[Info] clipped_pos_weight={pos_weight_val:.3f}")
    
    pos_weight_tensor = torch.tensor(pos_weight_val, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    model = HyperCHARM(node_dim, hedge_dim, hp).to(DEVICE)
    
    lr = float(hp.get("lr", 3e-4))  # 提高初始 lr
    opt = AdamW(model.parameters(), lr=lr, weight_decay=hp.get("weight_decay", 0.0))
    num_epochs = int(hp.get("epochs", 30))

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.02 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    # loss_fn = FocalLoss(alpha=pos_weight_val, gamma=2.0)  # focal loss

    best_val = -1
    best_state = None
    patience = 3
    counter = 0

    for ep in range(num_epochs):
        model.train()
        total_loss = 0.0
        # for bi, batch in enumerate(train_loader):
        for bi, batch in enumerate(tqdm(train_loader, desc=f"Epoch {ep+1}", leave=False)):
            batch = batch.to(DEVICE)
            logits = model(batch)

            mask = batch.node_pos >= batch.response_idx[batch.batch]
            if mask.sum() == 0:
                continue

            loss = loss_fn(logits[mask], batch.y[mask])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            total_loss += loss.item()

        val_m = evaluate(model, val_loader)
        
        test_m= evaluate(model, test_loader)
        val_aupr = val_m["aupr"]
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[Epoch {ep+1}] loss={avg_loss:.4f}  AUROC={val_m['auroc']:.4f} AUPR={val_aupr:.4f} test_AUROC={test_m['auroc']:.4f} test_AUPR={test_m['aupr']:.4f}")

        if val_aupr > best_val:
            best_val = val_aupr
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break
    torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------- grid search -----------------
def grid_search(train_loader, val_loader, test_loader, node_dim, hedge_dim, space,task_type,mode,model_type):
    keys = list(space.keys())
    vals = list(space.values())
    combos = list(__import__("itertools").product(*vals))
    
    test_pos_weight_val = float(compute_pos_weight_on_response(test_loader.dataset))
    print(f"Info[test_pos_weight_val]:{test_pos_weight_val}")
    best_score = -1
    best_hp = None
    best_state = None
    best_model=None
    if mode=="ablation":
        ablation_save_path = f"logger/ablation_results_{task_type}.json"
        # 如果文件已存在，先读出来；否则初始化为空列表
        if os.path.exists(ablation_save_path):
            with open(ablation_save_path, "r") as f:
                ablation_data = json.load(f)
        else:
            ablation_data = []

    for combo in combos:
        hp = dict(zip(keys, combo))

        print("\n===== Running HP:", hp)
        start_time = time.time()
        model = train_model(train_loader, val_loader, test_loader,node_dim, hedge_dim, hp)
        duration = time.time() - start_time
        val_m = evaluate(model, val_loader)
        print(f"→ VAL AUROC={val_m['auroc']:.4f} AUPR = {val_m['aupr']:.4f}, time = {duration:.1f}s")

        if mode=="ablation":
            test_m = evaluate(model, test_loader)
            record = {
                "hidden_dim": hp["hidden_dim"],
                "gnn_layers": hp["gnn_layers"],
                "test_result": test_m,
                "duration":duration
            }
            ablation_data.append(record)
            
            with open(ablation_save_path, "w") as f:
                json.dump(ablation_data, f, indent=2)
            
            print(f"Saved: hidden_dim={hp['hidden_dim']}, gnn_layers={hp['gnn_layers']}")
      
        aupr = val_m["aupr"]
        if aupr > best_score:
            best_score = aupr
            best_hp = hp
            best_model=model
            best_state = copy.deepcopy(model.state_dict())
            
    
    # state_dict = torch.load("models/hypergraph_Summary_xxxx.pkl", map_location="cpu")
    # best_model = HyperCHARM(node_dim, hedge_dim, best_hp).to(DEVICE)
    # best_model.load_state_dict(state_dict)
    
    
    if best_state is not None:
        # best_model = HyperCHARM(node_dim, hedge_dim, best_hp).to(DEVICE)
        # best_model.load_state_dict(best_state)
        test_m = evaluate(best_model, test_loader)
        if mode=="train":
            torch.save(
                best_state,
                f"models/hypergraph_{model_type}_{task_type}_{test_m['aupr']:.4f}.pkl"
            )

    else:
        test_m = {"auroc": 0.5, "aupr": 0.0}

    print("\n===== BEST TEST =====")
    print(test_m)
    torch.cuda.empty_cache()
    return best_hp, best_score, test_m


# ----------------- main -----------------
# def load_graph_dir(path):
#     files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
#     return [torch.load(os.path.join(path, f),weights_only=False) for f in files]
def load_graph_dir(path, desc=None):
    files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
    graphs = []

    for f in tqdm(files, desc=desc):
        try:
            g = torch.load(os.path.join(path, f), weights_only=False)
            graphs.append(graph_to_data(g))
        except:
            print(os.path.join(path, f))
            g = torch.load(os.path.join(path, f), weights_only=False)
            graphs.append(graph_to_data(g))

    return graphs

def load_datasets(TRAIN_DIR,TEST_DIR,val_ratio=0.2):
    train_list = load_graph_dir(TRAIN_DIR, desc="Processing train graphs")
    test_list  = load_graph_dir(TEST_DIR,  desc="Processing test graphs")

    # 统计正负样本索引
    pos_idx = [i for i, g in enumerate(train_list) if g.y.sum() > 0]  # 至少一个正样本
    neg_idx = [i for i, g in enumerate(train_list) if g.y.sum() == 0]
    
    val_size = max(1, int(len(train_list) * val_ratio))
    
    # 按比例抽取正负样本
    num_pos_val = int(len(pos_idx) / len(train_list) * val_size)
    num_neg_val = val_size - num_pos_val
    
    val_pos_idx = random.sample(pos_idx, min(num_pos_val, len(pos_idx)))
    val_neg_idx = random.sample(neg_idx, min(num_neg_val, len(neg_idx)))
    
    val_idx = val_pos_idx + val_neg_idx
    train_idx = list(set(range(len(train_list))) - set(val_idx))
    
    # 划分
    val_list = [train_list[i] for i in val_idx]
    train_list = [train_list[i] for i in train_idx]
    if len(train_list) == 0:
        raise RuntimeError("No training samples after split.")
    bs=16
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_list, batch_size=bs, shuffle=True,generator=g)
    val_loader = DataLoader(val_list, batch_size=bs,shuffle=False)
    test_loader = DataLoader(test_list, batch_size=bs,shuffle=False)
    
    sample = train_list[0]
    node_dim = sample.x.shape[1]
    hedge_dim = sample.he_attr.shape[1] if sample.he_attr.numel() > 0 else 2

    return train_loader,val_loader,test_loader,node_dim,hedge_dim
    
if __name__ == "__main__":
    mode="train" #train or ablation (ablation使用Mistral)
    model_type="Mistral-7B-Instruct-v0.3"#llama2-7b-chat-hf Mistral-7B-Instruct-v0.3 Llama-3.1-8B-Instruct Qwen3-8B
    if mode=="train":
        SEED=[1,2,3,4,5]
    else:
        SEED=[1]
    auroc_results=[]
    auprc_results=[]
    random.seed(42)#0
    #llama2-7b-chat-hf Mistral-7B-Instruct-v0.3
    task_type="Data2txt"#QA Summary Data2txt   47 475 478 480
    TRAIN_DIR = f"hypergraphs/RAGtruth/{model_type}/train_{task_type}_0.5_notopk"
    TEST_DIR  = f"hypergraphs/RAGtruth/{model_type}/test_{task_type}_0.5_notopk"
    print(f"Info[datset]:{TRAIN_DIR}")
    
    train_loader,val_loader,test_loader,node_dim,hedge_dim = load_datasets(TRAIN_DIR,TEST_DIR)
    for seed in SEED:
        print(f"[Info] SEED:{seed}")
        set_random_seeds(seed)
        
    
        SEARCH_SPACE = {
            "lr": [0.0005],#0.001 0.0005
            "scheduler": ["cosine"],
            "dropout": [0.05],
            "hidden_dim":[128],
            "gnn_layers": [1],
            "weight_decay": [0.001],
            "residual_mp": [True],
            "epochs": [20]
        }
    
        best_hp,_,test_m = grid_search(train_loader, val_loader, test_loader, node_dim, hedge_dim, SEARCH_SPACE,task_type,mode,model_type)
        auroc_results.append(test_m['auroc'])
        auprc_results.append(test_m['aupr'])

        if mode=="train":
            #不同seed结果保存
            result = {
                "seed":seed,
                "hp":best_hp,
                "auroc": test_m['auroc'],
                "auprc": test_m['aupr']
    
            }
            if "llama" in TRAIN_DIR:
                # 如果文件已存在，先读出来
                path=f"logger/llama2_{task_type}_results.json"
            elif "Llama-3" in TRAIN_DIR:
                # 如果文件已存在，先读出来
                path=f"logger/llama3_{task_type}_results.json"
            elif "Qwen3" in TRAIN_DIR:
                # 如果文件已存在，先读出来
                path=f"logger/Qwen3_{task_type}_results.json"
            else:
                path=f"logger/Mistral_{task_type}_results.json"
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_results = json.load(f)
            else:
                all_results = []
            
            # 追加
            all_results.append(result)
            
            # 写回
            with open(path, "w") as f:
                json.dump(all_results, f, indent=2)
        
        gc.collect()

    print(f"[Info]Datasets:{TRAIN_DIR}")
    
    print("All auroc:",auroc_results)
    print("AUROC mean:", np.mean(auroc_results))
    print("AUROC std:", np.std(auroc_results))

    print("All auprc:",auprc_results)
    print("AUPRC mean:", np.mean(auprc_results))
    print("AUPRC std:", np.std(auprc_results))
    

   
