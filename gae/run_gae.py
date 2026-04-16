#!/usr/bin/env python3
"""GAE (Graph Autoencoder) link prediction for Software Product Lines.

Standalone entry point that loads SPL interaction data (.interactions.txt +
dimacs), trains a GNN encoder (GCN / GraphSAGE / GAT) to produce node
embeddings, classifies links via Hadamard-product + MLP decoder, evaluates
on val/test sets, and ranks NEW/REMOVED interaction changes using a
two-universe ranking scheme:

  REMOVED: ranked against ALL interactions from the input file.
  NEW:     ranked against all 4 literal-pair combinations per feature
           pair that are NOT already in the interactions file.

Usage:
    python gae/run_gae.py --interactions data.interactions.txt --dimacs data.dimacs
    python gae/run_gae.py --interactions data.interactions.txt --dimacs data.dimacs --diff data_diff.txt --encoder sage
"""

import argparse
import copy
import csv as csv_mod
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score)


# ---------------------------------------------------------------------------
#  SPL data loading  (same logic as BUDDY / NeighborOverlap / SEAL)
# ---------------------------------------------------------------------------

def parse_dimacs(path):
    """Parse dimacs file, return dict mapping integer id -> feature name."""
    id_to_name = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('c '):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        id_to_name[int(parts[1])] = parts[2]
                    except ValueError:
                        pass
    return id_to_name


def feature_to_node(feature: int) -> int:
    """Map a signed 1-based feature to a node index.

    Positive literal = (|f|-1)*2, negative literal = (|f|-1)*2+1.
    """
    base = (abs(feature) - 1) * 2
    return base if feature > 0 else base + 1


def encode_feature_names(feature_strings, feat_dim=100):
    """Character-level encoding of feature name strings."""
    N = len(feature_strings)
    out = torch.zeros(N, feat_dim, dtype=torch.float32)
    for i, s in enumerate(feature_strings):
        chars = [ord(c) / 255.0 for c in s[:feat_dim]]
        out[i, :len(chars)] = torch.tensor(chars, dtype=torch.float32)
    return out


def load_spl_data(interactions_path, dimacs_path, feat_dim=100):
    """Load SPL data, return (Data, num_nodes, all_interactions).

    all_interactions is a (2, E) tensor of canonical (u < v) edges from the
    original file, kept on CPU for ranking universe construction.
    """
    src_nodes, tgt_nodes = [], []
    feature_set = set()

    with open(interactions_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                signed_a = int(parts[0])
                signed_b = int(parts[1])
            except ValueError:
                continue
            src_nodes.append(feature_to_node(signed_a))
            tgt_nodes.append(feature_to_node(signed_b))
            feature_set.update([abs(signed_a), abs(signed_b)])

    num_nodes = max(feature_set) * 2
    edge_index = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
    edge_index = to_undirected(edge_index)

    # Save full canonical edge set (u < v) before any subsampling
    mask_uv = edge_index[0] < edge_index[1]
    all_interactions = edge_index[:, mask_uv].cpu().clone()

    # Node features from dimacs
    id_to_name = parse_dimacs(dimacs_path)
    feature_strings = []
    for nid in range(num_nodes):
        fid = nid // 2 + 1
        pol = "+" if nid % 2 == 0 else "-"
        feature_strings.append(f"{pol}{id_to_name.get(fid, f'V{fid}')}")
    x = encode_feature_names(feature_strings, feat_dim)

    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

    n_edges = all_interactions.size(1)
    print(f"Loaded SPL data: {num_nodes} nodes, {n_edges} unique undirected edges")
    return data, num_nodes, all_interactions


def load_diff(dimacs_path, diff_path, num_nodes=None):
    """Parse a diff file with === New/Removed Interactions === sections."""
    id_to_name = parse_dimacs(dimacs_path)
    name_to_id = {v: k for k, v in id_to_name.items()}

    eval_groups = {'NEW': [], 'REMOVED': []}
    current_section = None
    skipped_dimacs = 0
    skipped_range = 0
    skipped_parse = 0

    with open(diff_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '=== New Interactions ===' in line:
                current_section = 'NEW'
                continue
            if '=== Removed Interactions ===' in line:
                current_section = 'REMOVED'
                continue
            if current_section is None:
                continue

            parts = line.split()
            if len(parts) < 2:
                skipped_parse += 1
                continue

            raw_a, raw_b = parts[0], parts[1]
            pol_a_positive = not raw_a.startswith('-')
            name_a = raw_a[1:] if raw_a.startswith('-') else raw_a
            pol_b_positive = not raw_b.startswith('-')
            name_b = raw_b[1:] if raw_b.startswith('-') else raw_b

            if name_a not in name_to_id:
                skipped_dimacs += 1
                continue
            if name_b not in name_to_id:
                skipped_dimacs += 1
                continue

            id_a = name_to_id[name_a]
            id_b = name_to_id[name_b]

            if num_nodes is not None:
                a_node = feature_to_node(id_a if pol_a_positive else -id_a)
                b_node = feature_to_node(id_b if pol_b_positive else -id_b)
                if a_node >= num_nodes or b_node >= num_nodes:
                    skipped_range += 1
                    continue

            signed_a = id_a if pol_a_positive else -id_a
            signed_b = id_b if pol_b_positive else -id_b

            pa = '+' if pol_a_positive else '-'
            pb = '+' if pol_b_positive else '-'
            label = f"{name_a}({pa}) <-> {name_b}({pb})"

            eval_groups[current_section].append((signed_a, signed_b, label))

    print(f"\nParsed {diff_path}:")
    print(f"  NEW interactions:     {len(eval_groups['NEW'])}")
    print(f"  REMOVED interactions: {len(eval_groups['REMOVED'])}")
    if skipped_dimacs:
        print(f"  Skipped (not in dimacs): {skipped_dimacs}")
    if skipped_range:
        print(f"  Skipped (out of graph range): {skipped_range}")
    if skipped_parse:
        print(f"  Skipped (malformed): {skipped_parse}")

    return eval_groups


# ---------------------------------------------------------------------------
#  GNN Encoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """GNN encoder that produces node embeddings.

    Supports GCN, GraphSAGE, and GAT via the ``encoder_type`` argument.
    """

    def __init__(self, in_channels, hidden_channels, num_layers, dropout,
                 encoder_type='gcn'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        conv_cls = {
            'gcn': GCNConv,
            'sage': SAGEConv,
            'gat': GATConv,
        }[encoder_type]

        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            self.convs.append(conv_cls(inc, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):
        # Input feature dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ---------------------------------------------------------------------------
#  MLP Decoder (Hadamard product -> scalar logit)
# ---------------------------------------------------------------------------

class MLPDecoder(nn.Module):
    """MLP link classifier.

    Takes the Hadamard product ``h[u] * h[v]`` and outputs a scalar logit.
    """

    def __init__(self, hidden_channels, mlp_layers, dropout):
        super().__init__()
        layers = []
        for i in range(mlp_layers):
            inc = hidden_channels if i == 0 else hidden_channels
            if i < mlp_layers - 1:
                layers.append(nn.Linear(inc, hidden_channels))
                layers.append(nn.BatchNorm1d(hidden_channels))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
            else:
                layers.append(nn.Linear(inc, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h_u, h_v):
        """Compute logit from node embeddings.

        Args:
            h_u: (batch, hidden_channels) embeddings of source nodes
            h_v: (batch, hidden_channels) embeddings of target nodes

        Returns:
            (batch,) logits
        """
        return self.mlp(h_u * h_v).view(-1)


# ---------------------------------------------------------------------------
#  Training and evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(encoder, decoder, data, pos_edge_index, neg_edge_index, device,
             batch_size=8192):
    """Evaluate on a set of positive and negative edges.

    Returns dict with AUC, AP, Accuracy, F1.
    """
    encoder.eval()
    decoder.eval()

    h = encoder(data.x.to(device), data.edge_index.to(device))

    def _score_edges(edge_index):
        scores = []
        n = edge_index.size(1)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            u_idx = edge_index[0, start:end]
            v_idx = edge_index[1, start:end]
            logits = decoder(h[u_idx], h[v_idx])
            scores.append(torch.sigmoid(logits).cpu())
        return torch.cat(scores) if scores else torch.tensor([])

    pos_scores = _score_edges(pos_edge_index.to(device))
    neg_scores = _score_edges(neg_edge_index.to(device))

    y_true = torch.cat([torch.ones(len(pos_scores)),
                        torch.zeros(len(neg_scores))]).numpy()
    y_scores = torch.cat([pos_scores, neg_scores]).numpy()

    if len(np.unique(y_true)) < 2:
        return {'auc': 0.0, 'ap': 0.0, 'acc': 0.0, 'f1': 0.0}

    y_pred = (y_scores > 0.5).astype(int)

    return {
        'auc': roc_auc_score(y_true, y_scores),
        'ap': average_precision_score(y_true, y_scores),
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }


@torch.no_grad()
def score_pairs(encoder, decoder, data, edge_tensor, device, batch_size=8192):
    """Score a (2, N) edge tensor, return (N,) sigmoid scores on CPU."""
    encoder.eval()
    decoder.eval()

    h = encoder(data.x.to(device), data.edge_index.to(device))

    n = edge_tensor.size(1)
    parts = []
    n_b = (n + batch_size - 1) // batch_size
    for bi in range(n_b):
        s = bi * batch_size
        e = min(s + batch_size, n)
        u_idx = edge_tensor[0, s:e].to(device)
        v_idx = edge_tensor[1, s:e].to(device)
        logits = decoder(h[u_idx], h[v_idx])
        parts.append(torch.sigmoid(logits).cpu())
        if (bi + 1) % max(1, n_b // 10) == 0 or bi == n_b - 1:
            print(f"  Scoring: {bi + 1}/{n_b} batches", end='\r')
    print()
    return torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)


# ---------------------------------------------------------------------------
#  Feature pair enumeration helper
# ---------------------------------------------------------------------------

def _feat_pair_from_lin(indices, n_feat):
    """Map linear index to feature pair (A, B) with A < B (0-based)."""
    n = n_feat
    half = (2.0 * n - 1.0) / 2.0
    A = (half - torch.sqrt(half * half - 2.0 * indices.double())).long()
    A.clamp_(0, n - 2)
    pairs_before_next = (A + 1) * n - (A + 1) * (A + 2) // 2
    A = torch.where(pairs_before_next <= indices, A + 1, A)
    pairs_before_A = A * n - A * (A + 1) // 2
    B = (A + 1 + (indices - pairs_before_A)).long()
    return A, B


# ---------------------------------------------------------------------------
#  Ranking helpers
# ---------------------------------------------------------------------------

def _compute_ranks(query, sorted_universe):
    """Compute ranks (1 = highest score) via searchsorted on sorted scores."""
    right = torch.searchsorted(sorted_universe, query, right=True)
    return len(sorted_universe) - right + 1


def _is_interaction(us, vs, int_packed_sorted, num_nodes):
    """Check if (u, v) pairs are in the interaction set."""
    packed = us.long() * num_nodes + vs.long()
    idx = torch.searchsorted(int_packed_sorted, packed)
    idx = idx.clamp(max=int_packed_sorted.shape[0] - 1)
    return int_packed_sorted[idx] == packed


def _resolve_entry(a_signed, b_signed, num_nodes):
    """Resolve signed feature pair to canonical (u, v) node pair with u < v."""
    a_node = feature_to_node(a_signed)
    b_node = feature_to_node(b_signed)
    u, v = min(a_node, b_node), max(a_node, b_node)
    return u, v


# ---------------------------------------------------------------------------
#  Report and output
# ---------------------------------------------------------------------------

MAX_PRINT = 200


def _report(gname, entries, skipped, entry_results, universe_size, prefix):
    """Print summary, save CSV and histogram for a ranking group."""
    n_orig = len(entries) + len(skipped)
    n_res = len(entry_results)
    n_skip = len(skipped)
    verbose = n_orig <= MAX_PRINT

    print(f"\n{'=' * 100}")
    print(f"Evaluation Group: {gname}  ({n_orig} interactions, "
          f"{n_res} scored, {n_skip} skipped)")
    if skipped and verbose:
        for readable, reason in skipped:
            print(f"       {readable:<55}  SKIPPED: {reason}")
        print()
    print(f"  (ranked against universe of {universe_size} scores)")
    print(f"{'=' * 100}")

    if not entry_results:
        print("  No interactions from this group could be resolved.")
        print(f"{'=' * 100}")
        return

    ranks_arr = np.array([r for _, r, _, _ in entry_results])

    if verbose:
        print(f"  {'#':>3}  {'Interaction':<55} {'Rank':>10}  {'Top%':>7}  {'Score':>8}")
        print(f"  {'-' * 90}")
        for idx, (readable, rank, pct, score) in enumerate(entry_results, 1):
            print(f"  {idx:>3}  {readable:<55} {rank:>10}  {pct:>6.2f}%  {score:>8.4f}")
        print(f"  {'-' * 90}")

    avg_r = np.mean(ranks_arr)
    med_r = np.median(ranks_arr)
    avg_p = 100.0 * avg_r / universe_size
    med_p = 100.0 * med_r / universe_size

    # AUC-ROC via Wilcoxon-Mann-Whitney from ranks
    n_eval = len(ranks_arr)
    n_other = universe_size - n_eval
    if n_other > 0 and n_eval > 0:
        auc = 1.0 - (np.sum(ranks_arr) - n_eval * (n_eval + 1) / 2) / (n_eval * n_other)
    else:
        auc = float('nan')

    print(f"\n  --- Summary: {gname} ---")
    if skipped:
        from collections import Counter
        for reason, cnt in Counter(r for _, r in skipped).most_common():
            print(f"  Skipped ({reason}): {cnt}")
    print(f"  Scored: {n_res}/{n_orig}  |  Skipped: {n_skip}")
    print(f"  AUC-ROC:      {auc:.4f}")
    print(f"  Average rank: {avg_r:.1f} / {universe_size}  (top {avg_p:.2f}%)")
    print(f"  Median rank:  {med_r:.1f} / {universe_size}  (top {med_p:.2f}%)")
    print(f"  Best rank:    {int(np.min(ranks_arr))}  |  "
          f"Worst rank: {int(np.max(ranks_arr))}")
    print(f"{'=' * 100}")

    # CSV output
    csv_path = f"{prefix}_eval_ranks_{gname.lower()}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv_mod.writer(f)
        w.writerow(['interaction', 'rank', 'rank_pct', 'score'])
        for readable, rank, pct, score in entry_results:
            w.writerow([readable, rank, f"{pct:.4f}", f"{score:.6f}"])
    print(f"  Saved rankings to {os.path.abspath(csv_path)}")

    # Histogram
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        pcts = [p for _, _, p, _ in entry_results]
        bins = np.arange(0, 102.5, 2.5)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(pcts, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Rank percentile (%)')
        ax.set_ylabel('Number of eval edges')
        ax.set_title(f'Eval edge rank distribution -- {gname} '
                     f'({len(entry_results)} edges, universe: {universe_size})')
        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_xlim(0, 100)
        hist_path = f"{prefix}_eval_ranks_{gname.lower()}_hist.png"
        fig.tight_layout()
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"  Saved histogram to {os.path.abspath(hist_path)}")
    except ImportError:
        print("  (matplotlib not available, skipping histogram)")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GAE (Graph Autoencoder) Link Prediction for SPL')

    # SPL paths
    parser.add_argument('--interactions', required=True,
                        help="path to the .interactions.txt file")
    parser.add_argument('--dimacs', required=True,
                        help="path to the .dimacs file")
    parser.add_argument('--diff', default=None,
                        help="path to diff file. If not provided, ranking is skipped.")

    # Model architecture
    parser.add_argument('--encoder', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gat'],
                        help="GNN encoder variant (default: gcn)")
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--mlp_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)

    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--feat_dim', type=int, default=100)

    # Ranking
    parser.add_argument('--train_edge_subsample', type=int, default=1,
                        help="subsample training edges: keep every N-th edge (default: 1 = all)")
    parser.add_argument('--eval_edge_subsample', type=int, default=1,
                        help="subsample ranking universes (default: 1 = all)")
    parser.add_argument('--savemod_path', type=str, default='saved_models/gae',
                        help="directory to save encoder.pt and decoder.pt "
                             "(default: saved_models/gae)")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed
    seed = int(time.time() * 1000) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed: {seed}")
    print(f"Device: {device}")
    print(args)

    # ==================== Load data ========================================
    data, num_nodes, all_interactions = load_spl_data(
        args.interactions, args.dimacs, args.feat_dim)

    # ==================== Edge subsampling =================================
    if args.train_edge_subsample > 1:
        ei = data.edge_index
        ei_undir = to_undirected(ei)
        mask_uv = ei_undir[0] < ei_undir[1]
        ei_unique = ei_undir[:, mask_uv]
        n_orig = ei_unique.shape[1]
        n_keep = n_orig // args.train_edge_subsample
        perm = torch.randperm(n_orig)[:n_keep]
        ei_sub = ei_unique[:, perm]
        data.edge_index = to_undirected(ei_sub)
        print(f"Edge subsampling: kept {n_keep} / {n_orig} unique edges "
              f"(1/{args.train_edge_subsample}, {100.0 * n_keep / n_orig:.1f}%)")

    # ==================== Train/val/test split =============================
    transform = RandomLinkSplit(
        is_undirected=True,
        num_val=0.05,
        num_test=0.05,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = transform(data)

    # Extract positive and negative edges from split data
    def _get_edges(split_data):
        """Return (pos_edge_index, neg_edge_index) each as (2, N) tensors."""
        el = split_data.edge_label_index
        labels = split_data.edge_label
        pos_mask = labels == 1
        neg_mask = labels == 0
        return el[:, pos_mask], el[:, neg_mask]

    train_pos, train_neg = _get_edges(train_data)
    val_pos, val_neg = _get_edges(val_data)
    test_pos, test_neg = _get_edges(test_data)

    print(f"\nTrain: {train_pos.size(1)} pos, {train_neg.size(1)} neg")
    print(f"Val:   {val_pos.size(1)} pos, {val_neg.size(1)} neg")
    print(f"Test:  {test_pos.size(1)} pos, {test_neg.size(1)} neg")

    # ==================== Build model ======================================
    encoder = GNNEncoder(
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        encoder_type=args.encoder,
    ).to(device)

    decoder = MLPDecoder(
        hidden_channels=args.hidden_channels,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nEncoder: {args.encoder.upper()}, {args.num_layers} layers, "
          f"{args.hidden_channels} hidden dim")
    print(f"Decoder: {args.mlp_layers}-layer MLP")
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Move data to device for training
    train_edge_index = train_data.edge_index.to(device)
    train_x = train_data.x.to(device)

    # ==================== Training =========================================
    best_val_auc = 0.0
    best_encoder_sd = None
    best_decoder_sd = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        encoder.train()
        decoder.train()

        h = encoder(train_x, train_edge_index)

        # Shuffle training edges and process in batches
        n_train = train_pos.size(1)
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            batch_idx = perm[start:end]

            pos_u = train_pos[0, batch_idx].to(device)
            pos_v = train_pos[1, batch_idx].to(device)
            neg_u = train_neg[0, batch_idx].to(device)
            neg_v = train_neg[1, batch_idx].to(device)

            pos_logits = decoder(h[pos_u], h[pos_v])
            neg_logits = decoder(h[neg_u], h[neg_v])

            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([torch.ones_like(pos_logits),
                                torch.zeros_like(neg_logits)])

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Recompute embeddings after parameter update
            h = encoder(train_x, train_edge_index)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        # For validation, use the training graph message-passing edges
        val_eval_data = Data(x=train_data.x, edge_index=train_data.edge_index,
                             num_nodes=num_nodes)
        metrics = evaluate(encoder, decoder, val_eval_data, val_pos, val_neg,
                           device, args.batch_size)
        val_auc = metrics['auc']

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_encoder_sd = copy.deepcopy(encoder.state_dict())
            best_decoder_sd = copy.deepcopy(decoder.state_dict())
            best_epoch = epoch
            marker = " ** new best **"
        else:
            marker = ""

        print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}  "
              f"val_auc={val_auc:.4f}  val_ap={metrics['ap']:.4f}  "
              f"({time.time()-t0:.1f}s){marker}")

    # Restore best model
    if best_encoder_sd is not None:
        encoder.load_state_dict(best_encoder_sd)
        decoder.load_state_dict(best_decoder_sd)
        print(f"\nRestored best model from epoch {best_epoch} "
              f"(val AUC = {best_val_auc:.4f})")

    # ==================== Test =============================================
    # For test evaluation, use all training + validation edges for message passing
    test_mp_ei = to_undirected(
        torch.cat([train_data.edge_index,
                   val_pos], dim=1))
    test_eval_data = Data(x=data.x, edge_index=test_mp_ei, num_nodes=num_nodes)

    test_metrics = evaluate(encoder, decoder, test_eval_data,
                            test_pos, test_neg, device, args.batch_size)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"  AUC:      {test_metrics['auc']:.4f}")
    print(f"  AP:       {test_metrics['ap']:.4f}")
    print(f"  Accuracy: {test_metrics['acc']:.4f}")
    print(f"  F1:       {test_metrics['f1']:.4f}")
    print(f"{'='*60}")

    # ==================== ALL edges combined ===============================
    all_pos = torch.cat([train_pos, val_pos, test_pos], dim=1)
    # Sample equal-count random negatives for combined evaluation
    n_all_pos = all_pos.size(1)
    neg_src = torch.randint(0, num_nodes, (n_all_pos,))
    neg_dst = torch.randint(0, num_nodes, (n_all_pos,))
    all_neg_combined = torch.stack([neg_src, neg_dst], dim=0)
    all_eval_data = Data(x=data.x, edge_index=test_mp_ei, num_nodes=num_nodes)
    all_metrics = evaluate(encoder, decoder, all_eval_data,
                           all_pos, all_neg_combined, device, args.batch_size)

    print(f"\n{'='*60}")
    print(f"ALL EDGES COMBINED (train+val+test pos vs random neg)")
    print(f"  AUC:      {all_metrics['auc']:.4f}")
    print(f"  AP:       {all_metrics['ap']:.4f}")
    print(f"  Accuracy: {all_metrics['acc']:.4f}")
    print(f"  F1:       {all_metrics['f1']:.4f}")
    print(f"{'='*60}")

    # ==================== Save model ======================================
    save_dir = args.savemod_path
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = os.path.join(save_dir, 'encoder.pt')
    decoder_path = os.path.join(save_dir, 'decoder.pt')
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"\nModel saved to {os.path.abspath(save_dir)}/")
    print(f"  encoder: {os.path.abspath(encoder_path)}")
    print(f"  decoder: {os.path.abspath(decoder_path)}")

    # ==================== Ranking evaluation ===============================
    if not args.diff or not os.path.exists(args.diff):
        return

    eval_groups = load_diff(args.dimacs, args.diff, num_nodes)

    # For ranking, use all known edges for message passing
    full_ei = to_undirected(all_interactions)
    rank_data = Data(x=data.x, edge_index=full_ei, num_nodes=num_nodes)

    # Build interaction lookup (on CPU)
    all_interactions_cpu = all_interactions.cpu()
    n_all_int = all_interactions_cpu.size(1)
    int_packed = all_interactions_cpu[0].long() * num_nodes + all_interactions_cpu[1].long()
    int_packed_sorted, _ = torch.sort(int_packed)

    num_features = num_nodes // 2

    # --- Resolve eval entries ---
    removed_entries = []
    removed_skipped = []
    for entry in eval_groups.get('REMOVED', []):
        a_s, b_s, readable = entry
        u, v = _resolve_entry(a_s, b_s, num_nodes)
        if u >= num_nodes or v >= num_nodes:
            removed_skipped.append((readable, "OUT OF RANGE"))
        elif not _is_interaction(torch.tensor([u]), torch.tensor([v]),
                                 int_packed_sorted, num_nodes)[0]:
            removed_skipped.append((readable, "NOT IN INTERACTIONS FILE"))
        else:
            removed_entries.append((u, v, readable))

    new_entries = []
    new_skipped = []
    for entry in eval_groups.get('NEW', []):
        a_s, b_s, readable = entry
        u, v = _resolve_entry(a_s, b_s, num_nodes)
        if u >= num_nodes or v >= num_nodes:
            new_skipped.append((readable, "OUT OF RANGE"))
        elif _is_interaction(torch.tensor([u]), torch.tensor([v]),
                             int_packed_sorted, num_nodes)[0]:
            new_skipped.append((readable, "ALREADY IN INTERACTIONS FILE"))
        else:
            new_entries.append((u, v, readable))

    # ==================================================================
    #  REMOVED RANKING
    #  Universe: all interactions from file (subsampled, force-include)
    # ==================================================================
    if removed_entries:
        removed_uv_packed = set(u * num_nodes + v for u, v, _ in removed_entries)

        if args.eval_edge_subsample > 1:
            n_keep = max(1, n_all_int // args.eval_edge_subsample)
            perm = torch.randperm(n_all_int)[:n_keep]
            sampled = all_interactions_cpu[:, perm]
            sampled_packed = set(
                (sampled[0, i].item() * num_nodes + sampled[1, i].item())
                for i in range(sampled.shape[1]))
            extra_u, extra_v = [], []
            for u, v, _ in removed_entries:
                if u * num_nodes + v not in sampled_packed:
                    extra_u.append(u)
                    extra_v.append(v)
            if extra_u:
                extra_t = torch.tensor([extra_u, extra_v], dtype=torch.long)
                univ = torch.cat([sampled, extra_t], dim=1)
            else:
                univ = sampled
            print(f"\nREMOVED universe: {univ.shape[1]} interactions "
                  f"(subsampled {n_keep}/{n_all_int}, "
                  f"+{len(extra_u)} force-included)")
        else:
            univ = all_interactions_cpu
            print(f"\nREMOVED universe: {n_all_int} interactions (all)")

        n_univ = univ.shape[1]
        shuf = torch.randperm(n_univ)
        t0 = time.time()
        print(f"Scoring {n_univ} interactions...")
        scores_shuf = score_pairs(encoder, decoder, rank_data,
                                  univ[:, shuf], device, args.batch_size)
        unshuf = torch.empty_like(shuf)
        unshuf[shuf] = torch.arange(n_univ)
        univ_scores = scores_shuf[unshuf]
        print(f"  Done in {time.time() - t0:.1f}s")

        # Build packed -> index for eval lookup
        univ_packed = univ[0].long() * num_nodes + univ[1].long()
        pk2idx = {}
        for i in range(n_univ):
            pk2idx[univ_packed[i].item()] = i

        sorted_scores, _ = torch.sort(univ_scores)
        results = []
        for u, v, readable in removed_entries:
            idx = pk2idx.get(u * num_nodes + v)
            if idx is None:
                continue
            s = univ_scores[idx].item()
            r = _compute_ranks(torch.tensor([s]), sorted_scores).item()
            pct = 100.0 * r / n_univ
            results.append((readable, int(r), pct, s))
        _report('REMOVED', removed_entries, removed_skipped, results,
                n_univ, 'gae')

    elif len(eval_groups.get('REMOVED', [])) > 0:
        print(f"\nREMOVED: {len(eval_groups['REMOVED'])} entries, "
              f"all skipped -- nothing to rank.")

    # ==================================================================
    #  NEW RANKING
    #  Universe: all 4 literal combos per feature pair NOT in interactions
    # ==================================================================
    if new_entries:
        new_packed_list = sorted(u * num_nodes + v for u, v, _ in new_entries)
        new_packed_sorted_t = torch.tensor(new_packed_list, dtype=torch.long)
        new_packed_set = set(new_packed_list)

        n_feat_pairs = num_features * (num_features - 1) // 2
        subsample_step = max(args.eval_edge_subsample, 1)

        print(f"\nNEW universe: enumerating non-interaction pairs "
              f"({num_features} features, {n_feat_pairs} feature pairs)...")

        FEAT_BATCH = 50000
        chunks_u = []
        chunks_v = []
        global_counter = 0
        new_eval_found = set()
        t0 = time.time()

        n_fb = (n_feat_pairs + FEAT_BATCH - 1) // FEAT_BATCH
        for fb in range(n_fb):
            s = fb * FEAT_BATCH
            e = min(s + FEAT_BATCH, n_feat_pairs)
            fp = torch.arange(s, e, dtype=torch.long)
            A0, B0 = _feat_pair_from_lin(fp, num_features)
            A1 = A0 + 1
            B1 = B0 + 1
            nAp = 2 * (A1 - 1)
            nAn = nAp + 1
            nBp = 2 * (B1 - 1)
            nBn = nBp + 1

            us4 = torch.cat([nAp, nAp, nAn, nAn])
            vs4 = torch.cat([nBp, nBn, nBp, nBn])
            u_min = torch.min(us4, vs4)
            v_max = torch.max(us4, vs4)

            is_int = _is_interaction(u_min, v_max, int_packed_sorted, num_nodes)
            ni_mask = ~is_int
            u_ni = u_min[ni_mask]
            v_ni = v_max[ni_mask]
            n_ni = u_ni.shape[0]
            if n_ni == 0:
                continue

            packed = u_ni.long() * num_nodes + v_ni.long()

            # Check NEW eval membership
            if new_packed_sorted_t.shape[0] > 0:
                si = torch.searchsorted(new_packed_sorted_t, packed)
                si = si.clamp(max=new_packed_sorted_t.shape[0] - 1)
                is_new = new_packed_sorted_t[si] == packed
            else:
                is_new = torch.zeros(n_ni, dtype=torch.bool)

            # Subsampling
            local_idx = torch.arange(n_ni, dtype=torch.long)
            keep_sample = ((global_counter + local_idx) % subsample_step) == 0
            keep = keep_sample | is_new
            global_counter += n_ni

            u_k = u_ni[keep]
            v_k = v_ni[keep]
            if u_k.shape[0] > 0:
                chunks_u.append(u_k)
                chunks_v.append(v_k)
                # Track found NEW entries
                pk_k = packed[keep]
                for pk_val in pk_k.tolist():
                    if pk_val in new_packed_set:
                        new_eval_found.add(pk_val)

            if (fb + 1) % max(1, n_fb // 20) == 0 or fb == n_fb - 1:
                elapsed = time.time() - t0
                n_collected = sum(c.shape[0] for c in chunks_u)
                print(f"  Enumerating: {fb + 1}/{n_fb} batches, "
                      f"{n_collected} pairs  ({elapsed:.1f}s)", end='\r')
        print()

        # Force-include any NEW entries missed during enumeration
        missing_u, missing_v = [], []
        for u, v, readable in new_entries:
            pk = u * num_nodes + v
            if pk not in new_eval_found:
                missing_u.append(u)
                missing_v.append(v)
        if missing_u:
            chunks_u.append(torch.tensor(missing_u, dtype=torch.long))
            chunks_v.append(torch.tensor(missing_v, dtype=torch.long))
            print(f"  Force-included {len(missing_u)} NEW entries "
                  f"not found during enumeration")

        if not chunks_u:
            print("  No non-interaction pairs found.")
            _report('NEW', new_entries, new_skipped, [], 0, 'gae')
        else:
            univ_u = torch.cat(chunks_u)
            univ_v = torch.cat(chunks_v)
            univ_new = torch.stack([univ_u, univ_v], dim=0)
            n_new_univ = univ_new.shape[1]

            elapsed = time.time() - t0
            print(f"  NEW universe: {n_new_univ} non-interaction pairs "
                  f"({elapsed:.1f}s)")
            if args.eval_edge_subsample > 1:
                print(f"  (subsampled ~1/{args.eval_edge_subsample})")

            shuf = torch.randperm(n_new_univ)
            t0 = time.time()
            print(f"Scoring {n_new_univ} non-interaction pairs...")
            scores_shuf = score_pairs(encoder, decoder, rank_data,
                                      univ_new[:, shuf], device, args.batch_size)
            unshuf = torch.empty_like(shuf)
            unshuf[shuf] = torch.arange(n_new_univ)
            new_univ_scores = scores_shuf[unshuf]
            print(f"  Done in {time.time() - t0:.1f}s")

            new_pk = univ_u.long() * num_nodes + univ_v.long()
            pk2idx_new = {}
            for i in range(n_new_univ):
                pk2idx_new[new_pk[i].item()] = i

            sorted_new, _ = torch.sort(new_univ_scores)
            results_new = []
            for u, v, readable in new_entries:
                idx = pk2idx_new.get(u * num_nodes + v)
                if idx is None:
                    continue
                s = new_univ_scores[idx].item()
                r = _compute_ranks(torch.tensor([s]), sorted_new).item()
                pct = 100.0 * r / n_new_univ
                results_new.append((readable, int(r), pct, s))
            _report('NEW', new_entries, new_skipped, results_new,
                    n_new_univ, 'gae')

            # Save NEW universe to file, sorted by score descending
            new_eval_packed_set = set(u * num_nodes + v for u, v, _ in new_entries)
            score_order = torch.argsort(new_univ_scores, descending=True)
            universe_path = "gae_new_universe.txt"
            with open(universe_path, 'w') as f_out:
                for idx in score_order:
                    i = idx.item()
                    u_node = univ_u[i].item()
                    v_node = univ_v[i].item()
                    sa = (u_node // 2 + 1) if (u_node % 2 == 0) else -(u_node // 2 + 1)
                    sb = (v_node // 2 + 1) if (v_node % 2 == 0) else -(v_node // 2 + 1)
                    pk = u_node * num_nodes + v_node
                    if pk in new_eval_packed_set:
                        f_out.write(f"{sa} {sb}\t<= variability bug\n")
                    else:
                        f_out.write(f"{sa} {sb}\n")
            print(f"  Saved NEW universe ({n_new_univ} pairs, sorted by score) to {os.path.abspath(universe_path)}")

    elif len(eval_groups.get('NEW', [])) > 0:
        print(f"\nNEW: {len(eval_groups['NEW'])} entries, "
              f"all skipped -- nothing to rank.")


if __name__ == '__main__':
    main()
