"""SEAL inference script for Software Product Lines.

Loads a trained SEAL (DGCNN) model and runs two-universe ranking
on a diff file, using the FULL adjacency matrix (no train/test split).

Usage:
    python seal/inference.py \
        --model_path saved_models/seal \
        --interactions data.interactions.txt \
        --dimacs data.dimacs \
        --diff changes.diff
"""

import sys
import os

# Ensure the seal package directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import random
import math
import time
import argparse
import scipy.sparse as ssp
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_sort_pool
from torch.nn import Linear, Conv1d, MaxPool1d, BatchNorm1d

from util_functions import (
    load_spl_data, load_diff, feature_to_node, score_links,
    subgraph_extraction_labeling
)


# ---------------------------------------------------------------------------
#  DGCNN model (must match training architecture in Main.py)
# ---------------------------------------------------------------------------

class DGCNN(torch.nn.Module):
    """DGCNN for graph classification (link prediction via subgraph)."""

    def __init__(self, num_features, hidden_channels=32, num_layers=3,
                 k=30, conv1d_channels=16, conv1d_kernel=5):
        super().__init__()
        self.k = k
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        total_dim = hidden_channels * num_layers

        self.conv1d_1 = Conv1d(1, conv1d_channels, kernel_size=total_dim,
                               stride=total_dim)
        self.conv1d_2 = Conv1d(conv1d_channels, conv1d_channels * 2,
                               kernel_size=conv1d_kernel, stride=1)
        self.pool = MaxPool1d(2, 2)

        dense_input = conv1d_channels * 2 * ((k - conv1d_kernel + 1) // 2)
        dense_input = max(dense_input, 1)
        self.lin1 = Linear(dense_input, 128)
        self.lin2 = Linear(128, 1)

        self.bn1 = BatchNorm1d(conv1d_channels)
        self.bn2 = BatchNorm1d(conv1d_channels * 2)

    def forward(self, data):
        x, edge_index, batch = data.z, data.edge_index, data.batch

        if hasattr(data, 'max_z'):
            max_z = data.max_z
            if isinstance(max_z, torch.Tensor):
                max_z = max_z.max().item()
        else:
            max_z = x.max().item()
        x = x.clamp(max=max_z)
        x = F.one_hot(x, num_classes=max_z + 1).float()

        if hasattr(data, 'node_features') and data.node_features is not None:
            x = torch.cat([x, data.node_features], dim=-1)

        xs = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            xs.append(x)
        x = torch.cat(xs, dim=-1)

        x = global_sort_pool(x, batch, self.k)

        x = x.unsqueeze(1)
        x = self.bn1(F.relu(self.conv1d_1(x)))
        x = self.bn2(F.relu(self.conv1d_2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x.view(-1)


# ---------------------------------------------------------------------------
#  Helper functions (same as Main.py ranking helpers)
# ---------------------------------------------------------------------------

def _feat_pair_from_lin(indices, n_feat):
    """Convert linear upper-triangle index to 0-based (A, B) feature pair."""
    n = n_feat
    half = (2.0 * n - 1.0) / 2.0
    A_arr = (half - np.sqrt(half * half - 2.0 * indices.astype(np.float64))).astype(np.int64)
    A_arr = np.clip(A_arr, 0, n - 2)
    pairs_before_next = (A_arr + 1) * n - (A_arr + 1) * (A_arr + 2) // 2
    A_arr = np.where(pairs_before_next <= indices, A_arr + 1, A_arr)
    pairs_before_A = A_arr * n - A_arr * (A_arr + 1) // 2
    B_arr = (A_arr + 1 + (indices - pairs_before_A)).astype(np.int64)
    return A_arr, B_arr


def _compute_ranks(query_scores, sorted_universe):
    """Rank = (# entries with strictly higher score) + 1."""
    right_idx = np.searchsorted(sorted_universe, query_scores, side='right')
    return len(sorted_universe) - right_idx + 1


def determine_max_n_label(A, node_features, hop, max_nodes_per_hop,
                          n_samples=200):
    """Sample edges from A and extract subgraphs to find max DRNL label."""
    triu = ssp.triu(A, k=1)
    row, col, _ = ssp.find(triu)
    n_edges = len(row)
    if n_edges == 0:
        return 0

    n_samples = min(n_samples, n_edges)
    sample_idx = np.random.choice(n_edges, size=n_samples, replace=False)

    max_label = 0
    for idx in sample_idx:
        data = subgraph_extraction_labeling(
            (int(row[idx]), int(col[idx])), A, hop, max_nodes_per_hop,
            node_features)
        max_label = max(max_label, data.z.max().item())

    return max_label


# ---------------------------------------------------------------------------
#  Main inference
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SEAL Inference for SPL link prediction')

    parser.add_argument('--model_path', type=str, required=True,
                        help='directory containing seal_model.pt')
    parser.add_argument('--interactions', type=str, required=True,
                        help='path to the .interactions.txt file')
    parser.add_argument('--dimacs', type=str, required=True,
                        help='path to the .dimacs file')
    parser.add_argument('--diff', type=str, required=True,
                        help='path to diff file')

    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--max-nodes-per-hop', type=int, default=100)
    parser.add_argument('--feat-dim', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--sortpooling-k', type=float, default=0.6)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--train-edge-subsample', type=int, default=1)
    parser.add_argument('--eval-edge-subsample', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    args = parser.parse_args()
    device = torch.device(
        'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(args)

    # ==================== Load SPL data (full graph) ==========================
    A, node_features, num_nodes = load_spl_data(
        args.interactions, args.dimacs, args.feat_dim)

    # Optionally subsample the adjacency for training-edge subsampling
    if args.train_edge_subsample > 1:
        triu = ssp.triu(A, k=1)
        row, col, _ = ssp.find(triu)
        n_edges = len(row)
        n_keep = max(n_edges // args.train_edge_subsample, 1)
        perm = np.random.permutation(n_edges)[:n_keep]
        row_sub, col_sub = row[perm], col[perm]
        all_rows = np.concatenate([row_sub, col_sub])
        all_cols = np.concatenate([col_sub, row_sub])
        A = ssp.csc_matrix(
            (np.ones(len(all_rows)), (all_rows, all_cols)),
            shape=A.shape)
        A.setdiag(0)
        A[A > 1] = 1
        A.eliminate_zeros()
        print(f"Subsampled adjacency: {ssp.triu(A, k=1).nnz} edges "
              f"(1/{args.train_edge_subsample} of original)")

    # ==================== Determine max_n_label from sample ===================
    print("\nSampling subgraphs to determine max DRNL label...")
    max_n_label = determine_max_n_label(
        A, node_features, args.hop, args.max_nodes_per_hop)
    print(f"Max DRNL label (from sample): {max_n_label}")

    # ==================== Determine sortpooling k =============================
    # Sample some subgraphs to estimate node count distribution
    triu = ssp.triu(A, k=1)
    row, col, _ = ssp.find(triu)
    n_edges = len(row)
    n_sample_k = min(500, n_edges)
    sample_idx = np.random.choice(n_edges, size=n_sample_k, replace=False)
    node_counts = []
    for idx in sample_idx:
        data = subgraph_extraction_labeling(
            (int(row[idx]), int(col[idx])), A, args.hop,
            args.max_nodes_per_hop, node_features)
        node_counts.append(data.num_nodes)
    node_counts.sort()

    if args.sortpooling_k <= 1:
        k_idx = int(math.ceil(args.sortpooling_k * len(node_counts))) - 1
        k_idx = max(0, min(k_idx, len(node_counts) - 1))
        k = max(10, node_counts[k_idx])
    else:
        k = int(args.sortpooling_k)
    print(f"SortPooling k: {k}")

    # ==================== Build and load model ================================
    num_features = max_n_label + 1
    if node_features is not None:
        num_features += node_features.shape[1]

    model = DGCNN(num_features, hidden_channels=args.hidden,
                  num_layers=args.num_layers, k=k).to(device)

    model_file = os.path.join(args.model_path, 'seal_model.pt')
    if not os.path.isfile(model_file):
        print(f"ERROR: Model file not found: {model_file}")
        sys.exit(1)

    state_dict = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {model_file}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== Parse diff file =====================================
    eval_groups = load_diff(args.dimacs, args.diff, num_nodes=num_nodes)

    # Build canonical edge set from the FULL adjacency matrix A
    interaction_set = set()
    full_triu = ssp.triu(A, k=1)
    full_row, full_col, _ = ssp.find(full_triu)
    for r, c in zip(full_row, full_col):
        interaction_set.add((min(int(r), int(c)), max(int(r), int(c))))

    num_features_graph = num_nodes // 2

    # ==================== Resolve eval entries =================================
    eval_resolved = {}
    eval_skipped = {}

    for gname, entries in eval_groups.items():
        if not entries:
            eval_resolved[gname] = []
            eval_skipped[gname] = []
            continue

        resolved = []
        skipped = []
        for entry in entries:
            a_signed, b_signed, readable = entry
            a_node = feature_to_node(a_signed)
            b_node = feature_to_node(b_signed)
            u, v = min(a_node, b_node), max(a_node, b_node)

            if u >= num_nodes or v >= num_nodes:
                skipped.append((readable, "OUT OF RANGE"))
            elif gname == 'REMOVED' and (u, v) not in interaction_set:
                skipped.append((readable, "NOT IN INTERACTIONS FILE"))
            elif gname == 'NEW' and (u, v) in interaction_set:
                skipped.append((readable, "ALREADY IN INTERACTIONS FILE"))
            else:
                resolved.append((u, v, readable))

        eval_resolved[gname] = resolved
        eval_skipped[gname] = skipped

    eval_pair_sets = {}
    for gname in eval_resolved:
        eval_pair_sets[gname] = set((u, v) for u, v, _ in eval_resolved[gname])

    subsample = max(args.eval_edge_subsample, 1)
    MAX_PRINT = 200

    # ======================================================================
    # REMOVED ranking: universe = all interactions from A
    # ======================================================================
    gname = 'REMOVED'
    resolved = eval_resolved.get(gname, [])
    skipped = eval_skipped.get(gname, [])
    n_original = len(eval_groups.get(gname, []))
    eval_pair_set_rem = eval_pair_sets.get(gname, set())

    if resolved:
        all_rows = full_row.copy()
        all_cols = full_col.copy()

        if subsample > 1 and len(all_rows) > 0:
            n_target = max(len(all_rows) // subsample, 1)
            perm = np.random.permutation(len(all_rows))
            keep_mask = np.zeros(len(all_rows), dtype=bool)
            for idx in range(len(all_rows)):
                pair = (min(int(all_rows[idx]), int(all_cols[idx])),
                        max(int(all_rows[idx]), int(all_cols[idx])))
                if pair in eval_pair_set_rem:
                    keep_mask[idx] = True
            n_forced = int(np.sum(keep_mask))
            n_remaining = max(n_target - n_forced, 0)
            count = 0
            for idx in perm:
                if count >= n_remaining:
                    break
                if not keep_mask[idx]:
                    keep_mask[idx] = True
                    count += 1
            all_rows = all_rows[keep_mask]
            all_cols = all_cols[keep_mask]

        n_removed_universe = len(all_rows)
        print(f"\nREMOVED universe: {n_removed_universe} interactions"
              f"{' (subsampled from ' + str(len(full_row)) + ')' if subsample > 1 else ''}")

        if n_removed_universe > 0:
            print(f"Scoring {n_removed_universe} interaction pairs "
                  f"for REMOVED ranking...")
            # SEAL inference with full adjacency: subgraphs from full A
            removed_uni_scores = score_links(
                model, A, (all_rows, all_cols),
                args.hop, args.max_nodes_per_hop, node_features,
                args.batch_size, device, max_n_label)

            removed_pair_to_idx = {}
            for idx in range(len(all_rows)):
                pair = (min(int(all_rows[idx]), int(all_cols[idx])),
                        max(int(all_rows[idx]), int(all_cols[idx])))
                removed_pair_to_idx[pair] = idx

            removed_sorted = np.sort(removed_uni_scores)

            eval_scores_rem = np.array([
                removed_uni_scores[removed_pair_to_idx[(u, v)]]
                for u, v, _ in resolved
            ]) if resolved else np.array([])

            if len(eval_scores_rem) > 0:
                ranks_rem = _compute_ranks(eval_scores_rem, removed_sorted)
                pcts_rem = 100.0 * ranks_rem / n_removed_universe
            else:
                ranks_rem = np.array([])
                pcts_rem = np.array([])
        else:
            eval_scores_rem = np.array([])
            ranks_rem = np.array([])
            pcts_rem = np.array([])
            removed_sorted = np.array([])

        verbose = n_original <= MAX_PRINT
        print(f"\n{'='*100}")
        n_not_in = sum(1 for _, r in skipped if r == "NOT IN INTERACTIONS FILE")
        if n_not_in == 0 and len(resolved) > 0:
            print(f"Evaluation Group: {gname}  ({n_original} interactions, "
                  f"all confirmed in interactions file)")
        elif n_not_in > 0:
            print(f"Evaluation Group: {gname}  ({n_original} interactions, "
                  f"{n_not_in} NOT in interactions file)")
            if verbose:
                for readable, reason in skipped:
                    if reason == "NOT IN INTERACTIONS FILE":
                        print(f"       {readable:<55}  {reason}")
        else:
            print(f"Evaluation Group: {gname}  ({n_original} interactions)")
        print(f"  (ranked against all interactions: "
              f"{n_removed_universe} scores)")
        if subsample > 1:
            print(f"  (subsampled ~1/{subsample} of "
                  f"{len(full_row)} total interactions)")
        print(f"{'='*100}")

        if verbose and len(resolved) > 0 and len(ranks_rem) > 0:
            print(f"  {'#':>3}  {'Interaction':<55} {'Nodes':>10}  "
                  f"{'Rank':>10}  {'Top%':>7}  {'Score':>8}")
            print(f"  {'-'*100}")
            for idx, ((u, v, readable), r, pct, s) in enumerate(
                    zip(resolved, ranks_rem, pcts_rem, eval_scores_rem), 1):
                print(f"  {idx:>3}  {readable:<55} {u}-{v:>8}  "
                      f"{int(r):>10}  {pct:>6.2f}%  {s:>8.4f}")
            print(f"  {'-'*100}")

        print(f"\n  --- Summary: {gname} ---")
        if skipped:
            from collections import Counter
            for reason, cnt in Counter(r for _, r in skipped).most_common():
                print(f"  Skipped ({reason}): {cnt}")
        if len(ranks_rem) > 0:
            avg_rank = np.mean(ranks_rem)
            med_rank = np.median(ranks_rem)
            avg_pct = 100.0 * avg_rank / n_removed_universe
            med_pct = 100.0 * med_rank / n_removed_universe
            n_eval = len(ranks_rem)
            n_other = n_removed_universe - n_eval
            auc = (1.0 - (np.sum(ranks_rem) - n_eval * (n_eval + 1) / 2)
                   / (n_eval * n_other)) if n_other > 0 else float('nan')
            print(f"  Scored: {len(resolved)}/{n_original}  |  "
                  f"Skipped: {len(skipped)}  |  Missing/Error: 0")
            print(f"  AUC-ROC:      {auc:.4f}")
            print(f"  Average rank: {avg_rank:.1f} / {n_removed_universe}  "
                  f"(top {avg_pct:.2f}%)")
            print(f"  Median rank:  {med_rank:.1f} / {n_removed_universe}  "
                  f"(top {med_pct:.2f}%)")
            print(f"  Best rank:    {int(np.min(ranks_rem))}  |  "
                  f"Worst rank: {int(np.max(ranks_rem))}")
        else:
            print(f"  No interactions from this group could be resolved.")
        print(f"{'='*100}")

        # CSV + Histogram for REMOVED
        if len(ranks_rem) > 0:
            import csv as csv_mod
            rank_records = [
                (readable, int(r), float(pct), float(s))
                for (u, v, readable), r, pct, s
                in zip(resolved, ranks_rem, pcts_rem, eval_scores_rem)
            ]
            csv_path = f"seal_eval_ranks_{gname.lower()}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv_mod.writer(f)
                writer.writerow(['interaction', 'rank', 'rank_pct', 'score'])
                for readable, rank, pct, score in rank_records:
                    writer.writerow([readable, rank, f"{pct:.4f}",
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
                ax.set_title(f'Eval edge rank distribution -- {gname} '
                             f'({len(rank_records)} edges, '
                             f'universe: all interactions)')
                ax.set_xticks(np.arange(0, 110, 10))
                ax.set_xlim(0, 100)
                hist_path = f"seal_eval_ranks_{gname.lower()}_hist.png"
                fig.tight_layout()
                fig.savefig(hist_path, dpi=150)
                plt.close(fig)
                print(f"  Saved histogram to {os.path.abspath(hist_path)}")
            except ImportError:
                print("  (matplotlib not available, skipping histogram)")
    elif n_original > 0:
        print(f"\nREMOVED: {n_original} entries, all skipped "
              "-- nothing to rank.")

    # ======================================================================
    # NEW ranking: universe = all non-interaction literal-pair combos
    # ======================================================================
    gname = 'NEW'
    resolved = eval_resolved.get(gname, [])
    skipped = eval_skipped.get(gname, [])
    n_original = len(eval_groups.get(gname, []))
    eval_pair_set_new = eval_pair_sets.get(gname, set())

    if resolved:
        n_feat = num_features_graph
        total_feat_pairs = n_feat * (n_feat - 1) // 2
        print(f"\nNEW universe: enumerating non-interaction literal pairs "
              f"across {total_feat_pairs} feature pairs ({n_feat} features)...")

        CHUNK = 500_000
        new_uni_rows = []
        new_uni_cols = []

        for start in range(0, total_feat_pairs, CHUNK):
            end = min(start + CHUNK, total_feat_pairs)
            lin_idx = np.arange(start, end)
            fA, fB = _feat_pair_from_lin(lin_idx, n_feat)

            pos_A = fA * 2
            neg_A = fA * 2 + 1
            pos_B = fB * 2
            neg_B = fB * 2 + 1

            combo_rows = np.concatenate([pos_A, pos_A, neg_A, neg_A])
            combo_cols = np.concatenate([pos_B, neg_B, pos_B, neg_B])

            lo = np.minimum(combo_rows, combo_cols)
            hi = np.maximum(combo_rows, combo_cols)

            keep = np.array([
                (int(lo[i]), int(hi[i])) not in interaction_set
                for i in range(len(lo))
            ], dtype=bool)
            lo = lo[keep]
            hi = hi[keep]

            if subsample > 1 and len(lo) > 0:
                is_eval = np.array([
                    (int(lo[i]), int(hi[i])) in eval_pair_set_new
                    for i in range(len(lo))
                ], dtype=bool)
                subsample_mask = np.zeros(len(lo), dtype=bool)
                subsample_mask[::subsample] = True
                subsample_mask |= is_eval
                lo = lo[subsample_mask]
                hi = hi[subsample_mask]

            new_uni_rows.append(lo)
            new_uni_cols.append(hi)

        new_uni_rows = (np.concatenate(new_uni_rows)
                        if new_uni_rows
                        else np.array([], dtype=np.int64))
        new_uni_cols = (np.concatenate(new_uni_cols)
                        if new_uni_cols
                        else np.array([], dtype=np.int64))

        n_new_universe = len(new_uni_rows)
        print(f"NEW universe: {n_new_universe} non-interaction literal pairs"
              f"{' (subsampled ~1/' + str(subsample) + ')' if subsample > 1 else ''}")

        if n_new_universe > 0:
            print(f"Scoring {n_new_universe} non-interaction pairs "
                  f"for NEW ranking...")
            # SEAL inference with full adjacency: subgraphs from full A
            new_uni_scores = score_links(
                model, A, (new_uni_rows, new_uni_cols),
                args.hop, args.max_nodes_per_hop, node_features,
                args.batch_size, device, max_n_label)

            new_pair_to_idx = {}
            for idx in range(len(new_uni_rows)):
                pair = (min(int(new_uni_rows[idx]), int(new_uni_cols[idx])),
                        max(int(new_uni_rows[idx]), int(new_uni_cols[idx])))
                if pair not in new_pair_to_idx:
                    new_pair_to_idx[pair] = idx

            new_sorted = np.sort(new_uni_scores)

            eval_scores_new = np.array([
                new_uni_scores[new_pair_to_idx[(u, v)]]
                for u, v, _ in resolved
            ]) if resolved else np.array([])

            if len(eval_scores_new) > 0:
                ranks_new = _compute_ranks(eval_scores_new, new_sorted)
                pcts_new = 100.0 * ranks_new / n_new_universe
            else:
                ranks_new = np.array([])
                pcts_new = np.array([])
        else:
            eval_scores_new = np.array([])
            ranks_new = np.array([])
            pcts_new = np.array([])
            new_sorted = np.array([])

        verbose = n_original <= MAX_PRINT
        print(f"\n{'='*100}")
        print(f"Evaluation Group: {gname}  ({n_original} interactions, "
              f"{len(resolved)} scored, {len(skipped)} skipped)")
        if skipped and verbose:
            for readable, reason in skipped:
                print(f"       {readable:<55}  SKIPPED: {reason}")
            print()
        print(f"  (ranked against non-interaction literal pairs: "
              f"{n_new_universe} scores)")
        if subsample > 1:
            print(f"  (subsampled ~1/{subsample})")
        print(f"{'='*100}")

        if verbose and len(resolved) > 0 and len(ranks_new) > 0:
            print(f"  {'#':>3}  {'Interaction':<55} {'Nodes':>10}  "
                  f"{'Rank':>10}  {'Top%':>7}  {'Score':>8}")
            print(f"  {'-'*100}")
            for idx, ((u, v, readable), r, pct, s) in enumerate(
                    zip(resolved, ranks_new, pcts_new, eval_scores_new), 1):
                print(f"  {idx:>3}  {readable:<55} {u}-{v:>8}  "
                      f"{int(r):>10}  {pct:>6.2f}%  {s:>8.4f}")
            print(f"  {'-'*100}")

        print(f"\n  --- Summary: {gname} ---")
        if skipped:
            from collections import Counter
            for reason, cnt in Counter(r for _, r in skipped).most_common():
                print(f"  Skipped ({reason}): {cnt}")
        if len(ranks_new) > 0:
            avg_rank = np.mean(ranks_new)
            med_rank = np.median(ranks_new)
            avg_pct = 100.0 * avg_rank / n_new_universe
            med_pct = 100.0 * med_rank / n_new_universe
            n_eval = len(ranks_new)
            n_other = n_new_universe - n_eval
            auc = (1.0 - (np.sum(ranks_new) - n_eval * (n_eval + 1) / 2)
                   / (n_eval * n_other)) if n_other > 0 else float('nan')
            print(f"  Scored: {len(resolved)}/{n_original}  |  "
                  f"Skipped: {len(skipped)}  |  Missing/Error: 0")
            print(f"  AUC-ROC:      {auc:.4f}")
            print(f"  Average rank: {avg_rank:.1f} / {n_new_universe}  "
                  f"(top {avg_pct:.2f}%)")
            print(f"  Median rank:  {med_rank:.1f} / {n_new_universe}  "
                  f"(top {med_pct:.2f}%)")
            print(f"  Best rank:    {int(np.min(ranks_new))}  |  "
                  f"Worst rank: {int(np.max(ranks_new))}")
        else:
            print(f"  No interactions from this group could be resolved.")
        print(f"{'='*100}")

        # CSV + Histogram for NEW
        if len(ranks_new) > 0:
            import csv as csv_mod
            rank_records = [
                (readable, int(r), float(pct), float(s))
                for (u, v, readable), r, pct, s
                in zip(resolved, ranks_new, pcts_new, eval_scores_new)
            ]
            csv_path = f"seal_eval_ranks_{gname.lower()}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv_mod.writer(f)
                writer.writerow(['interaction', 'rank', 'rank_pct', 'score'])
                for readable, rank, pct, score in rank_records:
                    writer.writerow([readable, rank, f"{pct:.4f}",
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
                ax.set_title(f'Eval edge rank distribution -- {gname} '
                             f'({len(rank_records)} edges, '
                             f'universe: non-interaction pairs)')
                ax.set_xticks(np.arange(0, 110, 10))
                ax.set_xlim(0, 100)
                hist_path = f"seal_eval_ranks_{gname.lower()}_hist.png"
                fig.tight_layout()
                fig.savefig(hist_path, dpi=150)
                plt.close(fig)
                print(f"  Saved histogram to {os.path.abspath(hist_path)}")
            except ImportError:
                print("  (matplotlib not available, skipping histogram)")

        # Save NEW universe to file, sorted by score descending
        new_eval_set = set((u, v) for u, v, _ in resolved)
        score_order = np.argsort(new_uni_scores)[::-1]
        universe_path = "seal_inference_new_universe.txt"
        with open(universe_path, 'w') as f_out:
            for idx in score_order:
                i = int(idx)
                u_node = int(new_uni_rows[i])
                v_node = int(new_uni_cols[i])
                sa = (u_node // 2 + 1) if (u_node % 2 == 0) else -(u_node // 2 + 1)
                sb = (v_node // 2 + 1) if (v_node % 2 == 0) else -(v_node // 2 + 1)
                if (min(u_node, v_node), max(u_node, v_node)) in new_eval_set:
                    f_out.write(f"{sa} {sb}\t<= variability bug\n")
                else:
                    f_out.write(f"{sa} {sb}\n")
        print(f"  Saved NEW universe ({n_new_universe} pairs, sorted by score) to {os.path.abspath(universe_path)}")

    elif n_original > 0:
        print(f"\nNEW: {n_original} entries, all skipped "
              "-- nothing to rank.")


if __name__ == '__main__':
    main()
