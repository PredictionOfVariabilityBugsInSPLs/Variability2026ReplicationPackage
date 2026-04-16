#!/usr/bin/env python3
"""Heuristic MLP link prediction for Software Product Lines.

Computes 5 classical graph heuristic scores (Common Neighbors, Adamic-Adar,
Resource Allocation, Jaccard Coefficient, Preferential Attachment) for each
candidate node pair and feeds them into a small MLP for binary link
classification.

Usage:
    python heuristic/run_heuristic.py --interactions data.interactions.txt \
        --dimacs data.dimacs --epochs 100

    python heuristic/run_heuristic.py --interactions data.interactions.txt \
        --dimacs data.dimacs --diff changes.txt --eval_edge_subsample 10
"""

import argparse
import copy
import csv as csv_mod
import os
import random
import time

import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score)
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  Heuristic computation
# ---------------------------------------------------------------------------

def compute_heuristics(A, src, dst):
    """Compute 5 heuristics for each (src[i], dst[i]) pair.

    A: scipy sparse adjacency matrix (symmetric)
    Returns: numpy array of shape (n_pairs, 5)
    """
    A_csr = A.tocsr()
    n = len(src)
    degrees = np.array(A_csr.sum(axis=1)).flatten()

    features = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        u, v = int(src[i]), int(dst[i])
        nu = set(A_csr[u].indices)
        nv = set(A_csr[v].indices)
        cn = nu & nv
        union = nu | nv

        features[i, 0] = len(cn)  # CN
        features[i, 1] = sum(1.0 / np.log(degrees[w]) for w in cn if degrees[w] > 1)  # AA
        features[i, 2] = sum(1.0 / degrees[w] for w in cn if degrees[w] > 0)  # RA
        features[i, 3] = len(cn) / len(union) if union else 0  # JC
        features[i, 4] = degrees[u] * degrees[v]  # PA

    return features


def compute_heuristics_batched(A, src, dst, batch_size=10000):
    """Same as compute_heuristics but with progress bar."""
    n = len(src)
    all_features = np.zeros((n, 5), dtype=np.float32)
    A_csr = A.tocsr()
    degrees = np.array(A_csr.sum(axis=1)).flatten()

    for start in tqdm(range(0, n, batch_size), desc="Computing heuristics"):
        end = min(start + batch_size, n)
        for i in range(start, end):
            u, v = int(src[i]), int(dst[i])
            nu = set(A_csr[u].indices)
            nv = set(A_csr[v].indices)
            cn = nu & nv
            union = nu | nv
            all_features[i, 0] = len(cn)
            all_features[i, 1] = sum(1.0 / np.log(degrees[w]) for w in cn if degrees[w] > 1)
            all_features[i, 2] = sum(1.0 / degrees[w] for w in cn if degrees[w] > 0)
            all_features[i, 3] = len(cn) / len(union) if union else 0
            all_features[i, 4] = degrees[u] * degrees[v]
    return all_features


# ---------------------------------------------------------------------------
#  MLP classifier
# ---------------------------------------------------------------------------

class HeuristicMLP(nn.Module):
    """MLP that takes a 5-dim heuristic feature vector and outputs a logit."""

    def __init__(self, input_dim=5, hidden_channels=64, num_layers=3,
                 dropout=0.3):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_channels))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_channels
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_dropout(x)
        return self.net(x).view(-1)


# ---------------------------------------------------------------------------
#  SPL data loading
# ---------------------------------------------------------------------------

def parse_dimacs(path):
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
    base = (abs(feature) - 1) * 2
    return base if feature > 0 else base + 1


def load_spl_data(interactions_path, dimacs_path):
    """Load SPL data, return (edge_set, num_nodes, adj_matrix).

    edge_set: set of (u, v) tuples with u < v (undirected unique edges)
    num_nodes: total number of literal nodes
    adj_matrix: scipy CSR adjacency matrix (symmetric, no self-loops)
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
            u = feature_to_node(signed_a)
            v = feature_to_node(signed_b)
            src_nodes.append(u)
            tgt_nodes.append(v)
            feature_set.update([abs(signed_a), abs(signed_b)])

    num_nodes = max(feature_set) * 2

    # Build undirected edge set (u < v)
    edge_set = set()
    for u, v in zip(src_nodes, tgt_nodes):
        a, b = min(u, v), max(u, v)
        if a != b:
            edge_set.add((a, b))

    # Build scipy sparse adjacency (symmetric)
    rows, cols = [], []
    for u, v in edge_set:
        rows.extend([u, v])
        cols.extend([v, u])
    adj = ssp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(num_nodes, num_nodes)
    )

    print(f"Loaded SPL data: {num_nodes} nodes, {len(edge_set)} unique "
          f"undirected edges")
    return edge_set, num_nodes, adj


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
#  Edge splitting utilities
# ---------------------------------------------------------------------------

def split_edges(edge_set, num_nodes, val_ratio=0.05, test_ratio=0.05):
    """Randomly split edges into train/val/test and sample negatives.

    Returns dict with keys: train_pos, train_neg, val_pos, val_neg,
    test_pos, test_neg.  Each is an (N, 2) numpy array.
    Also returns the training adjacency matrix (scipy CSR).
    """
    edges = np.array(list(edge_set))
    n = len(edges)
    perm = np.random.permutation(n)

    n_test = max(int(n * test_ratio), 1)
    n_val = max(int(n * val_ratio), 1)
    n_train = n - n_val - n_test

    train_edges = edges[perm[:n_train]]
    val_edges = edges[perm[n_train:n_train + n_val]]
    test_edges = edges[perm[n_train + n_val:]]

    # Build training adjacency
    rows, cols = [], []
    for u, v in train_edges:
        rows.extend([u, v])
        cols.extend([v, u])
    train_adj = ssp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(num_nodes, num_nodes)
    )

    # Sample negatives (same count as positives for each split)
    all_pos = set(map(tuple, edges))

    def sample_neg(num_neg, exclude):
        negs = []
        while len(negs) < num_neg:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u == v:
                continue
            a, b = min(u, v), max(u, v)
            if (a, b) in exclude:
                continue
            negs.append((a, b))
            exclude.add((a, b))
        return np.array(negs)

    neg_exclude = set(all_pos)
    train_neg = sample_neg(len(train_edges), neg_exclude)
    val_neg = sample_neg(len(val_edges), neg_exclude)
    test_neg = sample_neg(len(test_edges), neg_exclude)

    print(f"\nEdge split: {len(train_edges)} train, {len(val_edges)} val, "
          f"{len(test_edges)} test")
    print(f"Negatives:  {len(train_neg)} train, {len(val_neg)} val, "
          f"{len(test_neg)} test")

    return {
        'train_pos': train_edges,
        'train_neg': train_neg,
        'val_pos': val_edges,
        'val_neg': val_neg,
        'test_pos': test_edges,
        'test_neg': test_neg,
    }, train_adj


# ---------------------------------------------------------------------------
#  Feature pair enumeration helper for NEW universe
# ---------------------------------------------------------------------------

def _feat_pair_from_lin(indices, n_feat):
    n = n_feat
    half = (2.0 * n - 1.0) / 2.0
    A_arr = (half - np.sqrt(half * half - 2.0 * indices.astype(np.float64))).astype(np.int64)
    A_arr = np.clip(A_arr, 0, n - 2)
    pairs_before_next = (A_arr + 1) * n - (A_arr + 1) * (A_arr + 2) // 2
    A_arr = np.where(pairs_before_next <= indices, A_arr + 1, A_arr)
    pairs_before_A = A_arr * n - A_arr * (A_arr + 1) // 2
    B_arr = (A_arr + 1 + (indices - pairs_before_A)).astype(np.int64)
    return A_arr, B_arr


def _ranks(query, sorted_u):
    right = np.searchsorted(sorted_u, query, side='right')
    return len(sorted_u) - right + 1


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Heuristic MLP Link Prediction for SPL')

    # SPL paths
    parser.add_argument('--interactions', required=True,
                        help="path to the .interactions.txt file")
    parser.add_argument('--dimacs', required=True,
                        help="path to the .dimacs file")
    parser.add_argument('--diff', default=None,
                        help="path to diff file. If not provided, ranking "
                             "is skipped.")

    # MLP architecture
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.001)

    # Model saving
    parser.add_argument('--savemod_path', type=str,
                        default='saved_models/heuristic',
                        help="directory to save trained model and feature "
                             "normalization stats")

    # Ranking
    parser.add_argument('--train_edge_subsample', type=int, default=1,
                        help="subsample training edges: keep every N-th "
                             "edge (default: 1 = all)")
    parser.add_argument('--eval_edge_subsample', type=int, default=1,
                        help="subsample ranking universes "
                             "(default: 1 = all)")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed
    seed = int(time.time() * 1000) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed: {seed}")
    print(args)

    # ==================== Load data ========================================
    edge_set, num_nodes, full_adj = load_spl_data(
        args.interactions, args.dimacs)

    # ==================== Edge subsampling =================================
    if args.train_edge_subsample > 1:
        edges_list = list(edge_set)
        n_orig = len(edges_list)
        n_keep = max(n_orig // args.train_edge_subsample, 1)
        perm = np.random.permutation(n_orig)[:n_keep]
        edge_set = set(edges_list[i] for i in perm)
        print(f"Edge subsampling: kept {n_keep} / {n_orig} unique edges "
              f"(1/{args.train_edge_subsample}, "
              f"{100.0 * n_keep / n_orig:.1f}%)")

    # ==================== Train/val/test split =============================
    splits, train_adj = split_edges(edge_set, num_nodes)

    # ==================== Compute heuristic features =======================
    print("\nComputing heuristic features from training adjacency...")

    def make_features_labels(pos_pairs, neg_pairs, adj):
        """Compute heuristic features + labels for pos and neg pairs."""
        all_pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
        labels = np.concatenate([
            np.ones(len(pos_pairs), dtype=np.float32),
            np.zeros(len(neg_pairs), dtype=np.float32)
        ])
        feats = compute_heuristics(adj, all_pairs[:, 0], all_pairs[:, 1])
        return feats, labels

    train_feats, train_labels = make_features_labels(
        splits['train_pos'], splits['train_neg'], train_adj)
    val_feats, val_labels = make_features_labels(
        splits['val_pos'], splits['val_neg'], train_adj)
    test_feats, test_labels = make_features_labels(
        splits['test_pos'], splits['test_neg'], train_adj)

    # Normalize features (fit on train)
    feat_mean = train_feats.mean(axis=0)
    feat_std = train_feats.std(axis=0) + 1e-8
    train_feats = (train_feats - feat_mean) / feat_std
    val_feats = (val_feats - feat_mean) / feat_std
    test_feats = (test_feats - feat_mean) / feat_std

    print(f"  Train features shape: {train_feats.shape}")
    print(f"  Val features shape:   {val_feats.shape}")
    print(f"  Test features shape:  {test_feats.shape}")

    # Convert to tensors
    train_X = torch.tensor(train_feats, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.float32)
    val_X = torch.tensor(val_feats, dtype=torch.float32)
    val_y = torch.tensor(val_labels, dtype=torch.float32)
    test_X = torch.tensor(test_feats, dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.float32)

    # ==================== Build model ======================================
    model = HeuristicMLP(
        input_dim=5,
        hidden_channels=args.hidden_channels,
        num_layers=args.mlp_layers,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(model)

    # ==================== Training =========================================
    best_val_auc = 0.0
    best_model_sd = None
    best_epoch = -1

    n_train = len(train_X)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        # Shuffle training data
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            idx = perm[start:end]
            batch_x = train_X[idx].to(device)
            batch_y = train_y[idx].to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X.to(device)).cpu()
            val_preds = torch.sigmoid(val_logits).numpy()
            val_loss = criterion(val_logits, val_y).item()

        val_auc = (roc_auc_score(val_y.numpy(), val_preds)
                   if len(np.unique(val_y.numpy())) > 1 else 0.0)

        marker = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_sd = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            marker = " ** new best **"

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"Epoch {epoch:3d}  train_loss={avg_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  "
                  f"({time.time()-t0:.1f}s){marker}")

    # Restore best
    if best_model_sd is not None:
        model.load_state_dict(best_model_sd)
        print(f"\nRestored best model from epoch {best_epoch} "
              f"(val AUC={best_val_auc:.4f})")

    # ==================== Test =============================================
    model.eval()
    with torch.no_grad():
        test_logits = model(test_X.to(device)).cpu()
        test_preds = torch.sigmoid(test_logits).numpy()

    test_labels_np = test_y.numpy()
    test_binary = (test_preds > 0.5).astype(int)

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"  AUC:      {roc_auc_score(test_labels_np, test_preds):.4f}")
    print(f"  AP:       {average_precision_score(test_labels_np, test_preds):.4f}")
    print(f"  Accuracy: {accuracy_score(test_labels_np, test_binary):.4f}")
    print(f"  F1:       {f1_score(test_labels_np, test_binary):.4f}")
    print(f"{'='*60}")

    # ==================== Save model ======================================
    os.makedirs(args.savemod_path, exist_ok=True)
    model_save_path = os.path.join(args.savemod_path, 'heuristic_model.pt')
    stats_save_path = os.path.join(args.savemod_path, 'feature_stats.pt')
    torch.save(model.state_dict(), model_save_path)
    torch.save({'mean': feat_mean, 'std': feat_std}, stats_save_path)
    print(f"\nModel saved to {os.path.abspath(model_save_path)}")
    print(f"Feature stats saved to {os.path.abspath(stats_save_path)}")

    # ==================== Ranking evaluation ===============================
    if not args.diff or not os.path.exists(args.diff):
        return

    eval_groups = load_diff(args.dimacs, args.diff, num_nodes)

    # Build full interaction set (original data, before any subsampling)
    all_pos_set_orig, _, _ = load_spl_data(args.interactions, args.dimacs)

    n_feat = num_nodes // 2

    # Helper: score pairs with the trained MLP
    def score_pairs(adj, src, dst):
        """Compute heuristics and score with MLP. Returns numpy scores."""
        if len(src) == 0:
            return np.array([])
        feats = compute_heuristics_batched(adj, src, dst, batch_size=10000)
        feats = (feats - feat_mean) / feat_std
        feats_t = torch.tensor(feats, dtype=torch.float32)
        model.eval()
        scores = []
        bs = args.batch_size
        with torch.no_grad():
            for i in range(0, len(feats_t), bs):
                batch = feats_t[i:i+bs].to(device)
                logits = model(batch).cpu()
                scores.append(torch.sigmoid(logits).numpy())
        return np.concatenate(scores)

    # --- Resolve eval entries with validation ---
    eval_resolved = {}
    eval_skipped = {}

    for gname, entries in eval_groups.items():
        resolved, skipped = [], []
        for sa, sb, readable in entries:
            u = feature_to_node(sa)
            v = feature_to_node(sb)
            u, v = min(u, v), max(u, v)
            if u >= num_nodes or v >= num_nodes:
                skipped.append((readable, "OUT OF RANGE"))
            elif gname == 'REMOVED' and (u, v) not in all_pos_set_orig:
                skipped.append((readable, "NOT IN INTERACTIONS FILE"))
            elif gname == 'NEW' and (u, v) in all_pos_set_orig:
                skipped.append((readable, "ALREADY IN INTERACTIONS FILE"))
            else:
                resolved.append((u, v, readable))
        eval_resolved[gname] = resolved
        eval_skipped[gname] = skipped

    # Collect eval pair sets per group for force-inclusion during subsampling
    eval_pair_sets = {}
    for gname in eval_resolved:
        s = set()
        for u, v, _ in eval_resolved[gname]:
            s.add((u, v))
        eval_pair_sets[gname] = s

    # Use the full adjacency for heuristic computation during ranking
    # (train_adj excludes val/test edges; full_adj has all edges)
    rank_adj = full_adj

    # =================================================================
    #  REMOVED ranking: universe = ALL interactions from the input file
    # =================================================================
    gname = 'REMOVED'
    resolved = eval_resolved[gname]
    skipped = eval_skipped[gname]
    n_orig = len(eval_groups.get(gname, []))
    eval_force_set = eval_pair_sets[gname]

    if resolved:
        all_pos_list = list(all_pos_set_orig)
        if args.eval_edge_subsample > 1:
            rng = np.random.default_rng()
            n_total = len(all_pos_list)
            n_keep = max(n_total // args.eval_edge_subsample, 1)
            force_pairs = [p for p in all_pos_list if p in eval_force_set]
            other_pairs = [p for p in all_pos_list if p not in eval_force_set]
            n_sample_other = max(n_keep - len(force_pairs), 0)
            if n_sample_other < len(other_pairs):
                idx = rng.choice(len(other_pairs), size=n_sample_other,
                                 replace=False)
                sampled_other = [other_pairs[i] for i in idx]
            else:
                sampled_other = other_pairs
            removed_universe_pairs = force_pairs + sampled_other
            print(f"\nREMOVED universe: subsampled "
                  f"{len(removed_universe_pairs)} / {n_total} interactions "
                  f"(1/{args.eval_edge_subsample}), force-included "
                  f"{len(force_pairs)} eval entries")
        else:
            removed_universe_pairs = all_pos_list
            print(f"\nREMOVED universe: {len(removed_universe_pairs)} "
                  f"interactions (full set)")

        # Score REMOVED universe
        rem_src = np.array([u for u, v in removed_universe_pairs])
        rem_dst = np.array([v for u, v in removed_universe_pairs])
        print(f"Scoring {len(removed_universe_pairs)} REMOVED universe "
              f"pairs...")
        removed_scores = score_pairs(rank_adj, rem_src, rem_dst)

        # Build pair -> index map
        rem_pair_to_idx = {}
        for i, (u, v) in enumerate(removed_universe_pairs):
            rem_pair_to_idx[(u, v)] = i

        removed_sorted = np.sort(removed_scores)
        rem_u_size = len(removed_scores)
        rem_u_label = f"all interactions ({rem_u_size})"

        rem_eval_scores = (
            np.array([removed_scores[rem_pair_to_idx[(u, v)]]
                      for u, v, _ in resolved])
            if resolved else np.array([])
        )
        rem_ranks = (_ranks(rem_eval_scores, removed_sorted)
                     if len(rem_eval_scores) > 0 else np.array([]))
        rem_pcts = (100.0 * rem_ranks / rem_u_size
                    if len(rem_ranks) > 0 else np.array([]))

        # Print REMOVED results
        verbose = n_orig <= 200
        print(f"\n{'='*100}")
        n_not_in = sum(1 for _, r in skipped
                       if r == "NOT IN INTERACTIONS FILE")
        if n_not_in == 0 and len(resolved) > 0:
            print(f"Evaluation Group: {gname}  ({n_orig} interactions, "
                  f"all confirmed in interactions file)")
        elif n_not_in > 0:
            print(f"Evaluation Group: {gname}  ({n_orig} interactions, "
                  f"{n_not_in} NOT IN INTERACTIONS FILE)")
            if verbose:
                for readable, reason in skipped:
                    if reason == "NOT IN INTERACTIONS FILE":
                        print(f"       {readable:<55}  {reason}")
        else:
            print(f"Evaluation Group: {gname}  ({n_orig} interactions)")
        print(f"  (ranked against {rem_u_label})")
        print(f"{'='*100}")

        if verbose and len(resolved) > 0:
            print(f"  {'#':>3}  {'Interaction':<55} {'Nodes':>10}  "
                  f"{'Rank':>10}  {'Top%':>7}  {'Score':>8}")
            print(f"  {'-'*100}")
            for idx_i, ((u, v, readable), r, pct, s) in enumerate(
                    zip(resolved, rem_ranks, rem_pcts, rem_eval_scores), 1):
                print(f"  {idx_i:>3}  {readable:<55} {u}-{v:>8}  "
                      f"{int(r):>10}  {pct:>6.2f}%  {s:>8.4f}")
            print(f"  {'-'*100}")

        print(f"\n  --- Summary: {gname} ---")
        if skipped:
            from collections import Counter
            for reason, cnt in Counter(r for _, r in skipped).most_common():
                print(f"  Skipped ({reason}): {cnt}")
        if len(rem_ranks) > 0:
            avg_rank = np.mean(rem_ranks)
            med_rank = np.median(rem_ranks)
            avg_pct = 100.0 * avg_rank / rem_u_size
            med_pct = 100.0 * med_rank / rem_u_size
            n_eval = len(rem_ranks)
            n_other = rem_u_size - n_eval
            auc = (1.0 - (np.sum(rem_ranks) - n_eval * (n_eval + 1) / 2)
                   / (n_eval * n_other)) if n_other > 0 else float('nan')
            print(f"  Scored: {len(resolved)}/{n_orig}  |  "
                  f"Skipped: {len(skipped)}  |  Missing/Error: 0")
            print(f"  AUC-ROC:      {auc:.4f}")
            print(f"  Average rank: {avg_rank:.1f} / {rem_u_size}  "
                  f"(top {avg_pct:.2f}%)")
            print(f"  Median rank:  {med_rank:.1f} / {rem_u_size}  "
                  f"(top {med_pct:.2f}%)")
            print(f"  Best rank:    {int(np.min(rem_ranks))}  |  "
                  f"Worst rank: {int(np.max(rem_ranks))}")
        else:
            print("  No interactions from this group could be resolved.")
        print(f"{'='*100}")

        # REMOVED CSV + Histogram
        if len(rem_ranks) > 0:
            rank_records = [
                (readable, int(r), float(pct), float(s))
                for (u, v, readable), r, pct, s
                in zip(resolved, rem_ranks, rem_pcts, rem_eval_scores)
            ]

            csv_path = f"heuristic_eval_ranks_{gname.lower()}.csv"
            with open(csv_path, 'w', newline='') as f:
                w = csv_mod.writer(f)
                w.writerow(['interaction', 'rank', 'rank_pct', 'score'])
                for readable, rank, pct, score in rank_records:
                    w.writerow([readable, rank, f"{pct:.4f}",
                                f"{score:.6f}"])
            print(f"  Saved rankings to {os.path.abspath(csv_path)}")

            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                hist_pcts = [rec[2] for rec in rank_records]
                bins = np.arange(0, 102.5, 2.5)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(hist_pcts, bins=bins, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Rank percentile (%)')
                ax.set_ylabel('Number of eval edges')
                ax.set_title(
                    f'Eval edge rank distribution -- {gname} '
                    f'({len(rank_records)} edges, '
                    f'universe: {rem_u_label})')
                ax.set_xticks(np.arange(0, 110, 10))
                ax.set_xlim(0, 100)
                hist_path = f"heuristic_eval_ranks_{gname.lower()}_hist.png"
                fig.tight_layout()
                fig.savefig(hist_path, dpi=150)
                plt.close(fig)
                print(f"  Saved histogram to {os.path.abspath(hist_path)}")
            except ImportError:
                print("  (matplotlib not available, skipping histogram)")
    elif n_orig > 0:
        print(f"\nREMOVED: {n_orig} entries, all skipped -- nothing to rank.")

    # =================================================================
    #  NEW ranking: universe = all 4 literal-pair combos per feature
    #               pair NOT in the interactions file
    # =================================================================
    gname = 'NEW'
    resolved = eval_resolved[gname]
    skipped = eval_skipped[gname]
    n_orig = len(eval_groups.get(gname, []))
    eval_force_set = eval_pair_sets[gname]

    if resolved:
        total_feat_pairs = n_feat * (n_feat - 1) // 2

        print(f"\nNEW universe: enumerating literal pairs for "
              f"{total_feat_pairs} feature pairs (n_feat={n_feat}), "
              f"excluding interactions...")

        new_universe_rows = []
        new_universe_cols = []
        BATCH = 500_000
        n_excluded = 0

        for start in range(0, total_feat_pairs, BATCH):
            end = min(start + BATCH, total_feat_pairs)
            lin = np.arange(start, end)
            A_arr, B_arr = _feat_pair_from_lin(lin, n_feat)
            # 4 combos: (+A,+B), (+A,-B), (-A,+B), (-A,-B)
            posA = A_arr * 2
            negA = A_arr * 2 + 1
            posB = B_arr * 2
            negB = B_arr * 2 + 1

            for u_arr, v_arr in [(posA, posB), (posA, negB),
                                 (negA, posB), (negA, negB)]:
                lo = np.minimum(u_arr, v_arr)
                hi = np.maximum(u_arr, v_arr)
                for u, v in zip(lo, hi):
                    u_int, v_int = int(u), int(v)
                    if (u_int, v_int) not in all_pos_set_orig:
                        new_universe_rows.append(u_int)
                        new_universe_cols.append(v_int)
                    else:
                        n_excluded += 1

        print(f"  Total candidate literal pairs: {total_feat_pairs * 4}")
        print(f"  Excluded (in interactions file): {n_excluded}")
        print(f"  NEW universe before subsampling: {len(new_universe_rows)}")

        # Subsampling for NEW universe
        if args.eval_edge_subsample > 1:
            rng = np.random.default_rng()
            n_total_new = len(new_universe_rows)
            force_indices = []
            other_indices = []
            for i in range(n_total_new):
                pair = (new_universe_rows[i], new_universe_cols[i])
                if pair in eval_force_set:
                    force_indices.append(i)
                else:
                    other_indices.append(i)
            other_indices = np.array(other_indices)
            keep_other = other_indices[::args.eval_edge_subsample]
            keep_indices = sorted(force_indices + keep_other.tolist())
            new_universe_rows = [new_universe_rows[i] for i in keep_indices]
            new_universe_cols = [new_universe_cols[i] for i in keep_indices]
            print(f"  Subsampled NEW universe: {len(new_universe_rows)} "
                  f"(every {args.eval_edge_subsample}-th + "
                  f"{len(force_indices)} force-included eval entries)")
        else:
            print(f"  NEW universe (full): {len(new_universe_rows)}")

        # Score NEW universe
        n_new_universe = len(new_universe_rows)
        new_src = np.array(new_universe_rows)
        new_dst = np.array(new_universe_cols)
        print(f"Scoring {n_new_universe} NEW universe pairs...")
        new_scores = score_pairs(rank_adj, new_src, new_dst)

        # Build pair -> index map
        new_pair_to_idx = {}
        for i in range(n_new_universe):
            pair = (new_universe_rows[i], new_universe_cols[i])
            new_pair_to_idx[pair] = i

        new_sorted = np.sort(new_scores)
        new_u_size = len(new_scores)
        new_u_label = f"non-interaction literal pairs ({new_u_size})"

        new_eval_scores = (
            np.array([new_scores[new_pair_to_idx[(u, v)]]
                      for u, v, _ in resolved])
            if resolved else np.array([])
        )
        new_ranks = (_ranks(new_eval_scores, new_sorted)
                     if len(new_eval_scores) > 0 else np.array([]))
        new_pcts = (100.0 * new_ranks / new_u_size
                    if len(new_ranks) > 0 else np.array([]))

        # Print NEW results
        verbose = n_orig <= 200
        print(f"\n{'='*100}")
        print(f"Evaluation Group: {gname}  ({n_orig} interactions, "
              f"{len(resolved)} scored, {len(skipped)} skipped)")
        if skipped and verbose:
            for readable, reason in skipped:
                print(f"       {readable:<55}  SKIPPED: {reason}")
            print()
        print(f"  (ranked against {new_u_label})")
        print(f"{'='*100}")

        if verbose and len(resolved) > 0:
            print(f"  {'#':>3}  {'Interaction':<55} {'Nodes':>10}  "
                  f"{'Rank':>10}  {'Top%':>7}  {'Score':>8}")
            print(f"  {'-'*100}")
            for idx_i, ((u, v, readable), r, pct, s) in enumerate(
                    zip(resolved, new_ranks, new_pcts, new_eval_scores), 1):
                print(f"  {idx_i:>3}  {readable:<55} {u}-{v:>8}  "
                      f"{int(r):>10}  {pct:>6.2f}%  {s:>8.4f}")
            print(f"  {'-'*100}")

        print(f"\n  --- Summary: {gname} ---")
        if skipped:
            from collections import Counter
            for reason, cnt in Counter(r for _, r in skipped).most_common():
                print(f"  Skipped ({reason}): {cnt}")
        if len(new_ranks) > 0:
            avg_rank = np.mean(new_ranks)
            med_rank = np.median(new_ranks)
            avg_pct = 100.0 * avg_rank / new_u_size
            med_pct = 100.0 * med_rank / new_u_size
            n_eval = len(new_ranks)
            n_other = new_u_size - n_eval
            auc = (1.0 - (np.sum(new_ranks) - n_eval * (n_eval + 1) / 2)
                   / (n_eval * n_other)) if n_other > 0 else float('nan')
            print(f"  Scored: {len(resolved)}/{n_orig}  |  "
                  f"Skipped: {len(skipped)}  |  Missing/Error: 0")
            print(f"  AUC-ROC:      {auc:.4f}")
            print(f"  Average rank: {avg_rank:.1f} / {new_u_size}  "
                  f"(top {avg_pct:.2f}%)")
            print(f"  Median rank:  {med_rank:.1f} / {new_u_size}  "
                  f"(top {med_pct:.2f}%)")
            print(f"  Best rank:    {int(np.min(new_ranks))}  |  "
                  f"Worst rank: {int(np.max(new_ranks))}")
        else:
            print("  No interactions from this group could be resolved.")
        print(f"{'='*100}")

        # NEW CSV + Histogram
        if len(new_ranks) > 0:
            rank_records = [
                (readable, int(r), float(pct), float(s))
                for (u, v, readable), r, pct, s
                in zip(resolved, new_ranks, new_pcts, new_eval_scores)
            ]

            csv_path = f"heuristic_eval_ranks_{gname.lower()}.csv"
            with open(csv_path, 'w', newline='') as f:
                w = csv_mod.writer(f)
                w.writerow(['interaction', 'rank', 'rank_pct', 'score'])
                for readable, rank, pct, score in rank_records:
                    w.writerow([readable, rank, f"{pct:.4f}",
                                f"{score:.6f}"])
            print(f"  Saved rankings to {os.path.abspath(csv_path)}")

            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                hist_pcts = [rec[2] for rec in rank_records]
                bins = np.arange(0, 102.5, 2.5)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(hist_pcts, bins=bins, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Rank percentile (%)')
                ax.set_ylabel('Number of eval edges')
                ax.set_title(
                    f'Eval edge rank distribution -- {gname} '
                    f'({len(rank_records)} edges, '
                    f'universe: {new_u_label})')
                ax.set_xticks(np.arange(0, 110, 10))
                ax.set_xlim(0, 100)
                hist_path = f"heuristic_eval_ranks_{gname.lower()}_hist.png"
                fig.tight_layout()
                fig.savefig(hist_path, dpi=150)
                plt.close(fig)
                print(f"  Saved histogram to {os.path.abspath(hist_path)}")
            except ImportError:
                print("  (matplotlib not available, skipping histogram)")

        # Save NEW universe to file, sorted by score descending
        new_eval_set = set((u, v) for u, v, _ in resolved)
        score_order = np.argsort(new_scores)[::-1]
        universe_path = "heuristic_new_universe.txt"
        with open(universe_path, 'w') as f_out:
            for idx in score_order:
                i = int(idx)
                u_node = new_universe_rows[i]
                v_node = new_universe_cols[i]
                sa = (u_node // 2 + 1) if (u_node % 2 == 0) else -(u_node // 2 + 1)
                sb = (v_node // 2 + 1) if (v_node % 2 == 0) else -(v_node // 2 + 1)
                if (u_node, v_node) in new_eval_set:
                    f_out.write(f"{sa} {sb}\t<= variability bug\n")
                else:
                    f_out.write(f"{sa} {sb}\n")
        print(f"  Saved NEW universe ({n_new_universe} pairs, sorted by score) to {os.path.abspath(universe_path)}")

    elif n_orig > 0:
        print(f"\nNEW: {n_orig} entries, all skipped -- nothing to rank.")


if __name__ == '__main__':
    main()
