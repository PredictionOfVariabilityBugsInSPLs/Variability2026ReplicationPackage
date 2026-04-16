"""SEAL link prediction for Software Product Lines.

Trains a DGCNN-style graph classifier on enclosing subgraphs extracted
around candidate links, then evaluates on NEW/REMOVED interaction changes.

Usage:
    python seal/Main.py --interactions data.interactions.txt --dimacs data.dimacs --epochs 50 --batch-size 50 --hop 1
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import math
import time
import os
import argparse
import copy
import scipy.sparse as ssp
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import degree
from torch.nn import Linear, Conv1d, MaxPool1d, BatchNorm1d

from util_functions import (
    load_spl_data, load_diff, sample_neg, links2subgraphs,
    feature_to_node, score_links
)


# ---------------------------------------------------------------------------
#  DGCNN model (PyG-based, replaces external pytorch_DGCNN dependency)
# ---------------------------------------------------------------------------

class DGCNN(torch.nn.Module):
    """DGCNN for graph classification (link prediction via subgraph).

    Architecture:
      1. Multiple GCN layers produce node embeddings
      2. SortPooling selects top-k nodes by last-channel value
      3. 1D convolution + dense layers for classification
    """

    def __init__(self, num_features, hidden_channels=32, num_layers=3,
                 k=30, conv1d_channels=16, conv1d_kernel=5):
        super().__init__()
        self.k = k
        self.num_layers = num_layers

        # GCN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # After concat of all GCN layer outputs
        total_dim = hidden_channels * num_layers

        # 1D conv layers after sort pooling
        self.conv1d_1 = Conv1d(1, conv1d_channels, kernel_size=total_dim,
                               stride=total_dim)
        self.conv1d_2 = Conv1d(conv1d_channels, conv1d_channels * 2,
                               kernel_size=conv1d_kernel, stride=1)
        self.pool = MaxPool1d(2, 2)

        # Dense layers
        dense_input = conv1d_channels * 2 * ((k - conv1d_kernel + 1) // 2)
        # Clamp to avoid zero or negative
        dense_input = max(dense_input, 1)
        self.lin1 = Linear(dense_input, 128)
        self.lin2 = Linear(128, 1)

        self.bn1 = BatchNorm1d(conv1d_channels)
        self.bn2 = BatchNorm1d(conv1d_channels * 2)

    def forward(self, data):
        x, edge_index, batch = data.z, data.edge_index, data.batch

        # One-hot encode DRNL labels
        if hasattr(data, 'max_z'):
            max_z = data.max_z
            if isinstance(max_z, torch.Tensor):
                max_z = max_z.max().item()
        else:
            max_z = x.max().item()
        x = x.clamp(max=max_z)
        x = F.one_hot(x, num_classes=max_z + 1).float()

        # Concatenate node features if available
        if hasattr(data, 'node_features') and data.node_features is not None:
            x = torch.cat([x, data.node_features], dim=-1)

        # GCN layers with concatenation
        xs = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            xs.append(x)
        x = torch.cat(xs, dim=-1)

        # Sort pooling
        x = global_sort_pool(x, batch, self.k)

        # 1D convolutions
        x = x.unsqueeze(1)  # (batch, 1, k * total_dim)
        x = self.bn1(F.relu(self.conv1d_1(x)))
        x = self.bn2(F.relu(self.conv1d_2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Dense layers
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x.view(-1)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SEAL Link Prediction for SPL')

    # SPL file paths
    parser.add_argument('--interactions', type=str, required=True,
                        help="path to the .interactions.txt file")
    parser.add_argument('--dimacs', type=str, required=True,
                        help="path to the .dimacs file")
    parser.add_argument('--diff', type=str, default=None,
                        help="path to diff file. If not provided, ranking is skipped.")

    # SEAL settings
    parser.add_argument('--hop', type=int, default=1, help='enclosing subgraph hop number')
    parser.add_argument('--max-nodes-per-hop', type=int, default=100,
                        help='upper bound on neighbors per hop (for scalability)')
    parser.add_argument('--feat-dim', type=int, default=100,
                        help='character-level feature encoding dimension')

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test-ratio', type=float, default=0.05)
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--max-train-num', type=int, default=100000)
    parser.add_argument('--no-parallel', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    # DGCNN model
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--sortpooling-k', type=float, default=0.6,
                        help='fraction of nodes for sort pooling (if <= 1)')

    # Evaluation
    parser.add_argument('--train-edge-subsample', type=int, default=1,
                        help="subsample training edges (default: 1 = all)")
    parser.add_argument('--eval-edge-subsample', type=int, default=1,
                        help="subsample ranking universes (default: 1 = all)")

    # Model save/load
    parser.add_argument('--savemod-path', type=str, default='saved_models/seal')

    args = parser.parse_args()
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available()
                          else 'cpu')

    # Seed
    seed = int(time.time() * 1000) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    print(f"Random seed: {seed}")
    print(args)

    # ==================== Load SPL data ====================================
    A, node_features, num_nodes = load_spl_data(
        args.interactions, args.dimacs, args.feat_dim)

    # ==================== Train/val/test split =============================
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        A, args.test_ratio, max_train_num=args.max_train_num)

    print(f"Train pos: {len(train_pos[0])}, Train neg: {len(train_neg[0])}")
    print(f"Test pos: {len(test_pos[0])}, Test neg: {len(test_neg[0])}")

    # Build observed network (mask test edges)
    A_train = A.copy()
    A_train[test_pos[0], test_pos[1]] = 0
    A_train[test_pos[1], test_pos[0]] = 0
    A_train.eliminate_zeros()

    # ==================== Extract subgraphs ================================
    print("\nExtracting training subgraphs...")
    train_pos_graphs, max_n1 = links2subgraphs(
        A_train, train_pos, 1, args.hop, args.max_nodes_per_hop,
        node_features, args.no_parallel)
    train_neg_graphs, max_n2 = links2subgraphs(
        A_train, train_neg, 0, args.hop, args.max_nodes_per_hop,
        node_features, args.no_parallel)
    train_graphs = train_pos_graphs + train_neg_graphs
    max_n_label = max(max_n1, max_n2)

    print("\nExtracting test subgraphs...")
    test_pos_graphs, max_n3 = links2subgraphs(
        A_train, test_pos, 1, args.hop, args.max_nodes_per_hop,
        node_features, args.no_parallel)
    test_neg_graphs, max_n4 = links2subgraphs(
        A_train, test_neg, 0, args.hop, args.max_nodes_per_hop,
        node_features, args.no_parallel)
    test_graphs = test_pos_graphs + test_neg_graphs
    max_n_label = max(max_n_label, max_n3, max_n4)

    print(f"\n# train: {len(train_graphs)}, # test: {len(test_graphs)}")
    print(f"Max DRNL label: {max_n_label}")

    # Set max_z on all graphs for consistent one-hot encoding
    for g in train_graphs + test_graphs:
        g.max_z = max_n_label

    # ==================== Determine sortpooling_k ==========================
    if args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        k_idx = int(math.ceil(args.sortpooling_k * len(num_nodes_list))) - 1
        k_idx = max(0, min(k_idx, len(num_nodes_list) - 1))
        k = max(10, num_nodes_list[k_idx])
    else:
        k = int(args.sortpooling_k)
    print(f"SortPooling k: {k}")

    # ==================== Build model ======================================
    # Input features: one-hot DRNL labels + optional node features
    num_features = max_n_label + 1
    if node_features is not None:
        num_features += node_features.shape[1]

    model = DGCNN(num_features, hidden_channels=args.hidden,
                  num_layers=args.num_layers, k=k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== Train/val split ==================================
    random.shuffle(train_graphs)
    val_num = int(args.val_ratio * len(train_graphs))
    val_graphs = train_graphs[:val_num]
    train_graphs = train_graphs[val_num:]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    # ==================== Training loop ====================================
    best_val_loss = None
    best_model_sd = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                out, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(train_graphs)

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = F.binary_cross_entropy_with_logits(
                    out, batch.y.float())
                val_loss += loss.item() * batch.num_graphs
                val_preds.append(torch.sigmoid(out).cpu())
                val_labels.append(batch.y.cpu())
        val_loss /= len(val_graphs)
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_preds) if len(np.unique(val_labels)) > 1 else 0

        # Best model checkpointing
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_sd = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            marker = " ** new best **"
        else:
            marker = ""

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  "
              f"({elapsed:.1f}s){marker}")

    # Restore best model
    if best_model_sd is not None:
        model.load_state_dict(best_model_sd)
        print(f"\nRestored best model from epoch {best_epoch} "
              f"(val_loss={best_val_loss:.4f})")

    # ==================== Test evaluation ==================================
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            test_preds.append(torch.sigmoid(out).cpu())
            test_labels.append(batch.y.cpu())
    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()

    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, f1_score)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_ap = average_precision_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
    test_f1 = f1_score(test_labels, (test_preds > 0.5).astype(int))

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"  AUC:      {test_auc:.4f}")
    print(f"  AP:       {test_ap:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1:       {test_f1:.4f}")
    print(f"{'='*60}")

    # ==================== Save model =======================================
    if args.savemod_path:
        os.makedirs(args.savemod_path, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(args.savemod_path, 'seal_model.pt'))
        print(f"Model saved to {args.savemod_path}/seal_model.pt")

    # ==================== Rank links + diff evaluation ======================
    if args.diff and os.path.exists(args.diff):
        eval_groups = load_diff(args.dimacs, args.diff,
                                num_nodes=num_nodes)

        # Build canonical edge set from the FULL adjacency matrix A
        interaction_set = set()
        full_triu = ssp.triu(A, k=1)
        full_row, full_col, _ = ssp.find(full_triu)
        for r, c in zip(full_row, full_col):
            interaction_set.add((min(int(r), int(c)), max(int(r), int(c))))

        num_features_graph = num_nodes // 2

        # ---- Helper: convert linear upper-triangle index to feature pair ----
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

        # ---- Resolve eval entries with new validation rules ----
        eval_resolved = {}   # gname -> [(u, v, readable), ...]
        eval_skipped = {}    # gname -> [(readable, reason), ...]

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

        # Build eval pair sets per group for force-inclusion during subsampling
        eval_pair_sets = {}
        for gname in eval_resolved:
            eval_pair_sets[gname] = set((u, v) for u, v, _ in eval_resolved[gname])

        subsample = max(args.eval_edge_subsample, 1)
        MAX_PRINT = 200

        # ==================================================================
        # REMOVED ranking: universe = all interactions from A
        # ==================================================================
        gname = 'REMOVED'
        resolved = eval_resolved.get(gname, [])
        skipped = eval_skipped.get(gname, [])
        n_original = len(eval_groups.get(gname, []))
        eval_pair_set_rem = eval_pair_sets.get(gname, set())

        if resolved:
            # Build universe: all edges from the full adjacency matrix
            all_rows = full_row.copy()
            all_cols = full_col.copy()

            if subsample > 1 and len(all_rows) > 0:
                # Subsample to ~1/N, but force-include all REMOVED eval entries
                n_target = max(len(all_rows) // subsample, 1)
                perm = np.random.permutation(len(all_rows))
                keep_mask = np.zeros(len(all_rows), dtype=bool)
                # Force-include eval entries
                for idx in range(len(all_rows)):
                    pair = (min(int(all_rows[idx]), int(all_cols[idx])),
                            max(int(all_rows[idx]), int(all_cols[idx])))
                    if pair in eval_pair_set_rem:
                        keep_mask[idx] = True
                # Fill remaining slots from shuffled order
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

            # Score the full REMOVED universe
            if n_removed_universe > 0:
                print(f"Scoring {n_removed_universe} interaction pairs for REMOVED ranking...")
                removed_uni_scores = score_links(
                    model, A_train, (all_rows, all_cols),
                    args.hop, args.max_nodes_per_hop, node_features,
                    args.batch_size, device, max_n_label)

                # Build mapping from (u,v)->index for eval lookups
                removed_pair_to_idx = {}
                for idx in range(len(all_rows)):
                    pair = (min(int(all_rows[idx]), int(all_cols[idx])),
                            max(int(all_rows[idx]), int(all_cols[idx])))
                    removed_pair_to_idx[pair] = idx

                removed_sorted = np.sort(removed_uni_scores)

                # Extract scores for REMOVED eval entries
                eval_scores_rem = np.array([removed_uni_scores[removed_pair_to_idx[(u, v)]]
                                            for u, v, _ in resolved]) if resolved else np.array([])
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

            # Print REMOVED results
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
            print(f"  (ranked against all interactions: {n_removed_universe} scores)")
            if subsample > 1:
                print(f"  (subsampled ~1/{subsample} of {len(full_row)} total interactions)")
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
                auc = 1.0 - (np.sum(ranks_rem) - n_eval * (n_eval + 1) / 2) / (n_eval * n_other) if n_other > 0 else float('nan')
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
                rank_records = [(readable, int(r), float(pct), float(s))
                                for (u, v, readable), r, pct, s
                                in zip(resolved, ranks_rem, pcts_rem, eval_scores_rem)]
                csv_path = f"seal_eval_ranks_{gname.lower()}.csv"
                with open(csv_path, 'w', newline='') as f:
                    writer = csv_mod.writer(f)
                    writer.writerow(['interaction', 'rank', 'rank_pct', 'score'])
                    for readable, rank, pct, score in rank_records:
                        writer.writerow([readable, rank, f"{pct:.4f}", f"{score:.6f}"])
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
            print(f"\nREMOVED: {n_original} entries, all skipped — nothing to rank.")

        # ==================================================================
        # NEW ranking: universe = all non-interaction literal-pair combos
        # ==================================================================
        gname = 'NEW'
        resolved = eval_resolved.get(gname, [])
        skipped = eval_skipped.get(gname, [])
        n_original = len(eval_groups.get(gname, []))
        eval_pair_set_new = eval_pair_sets.get(gname, set())

        if resolved:
            # Enumerate all 4 literal-pair combos for each feature pair (A < B),
            # keeping only those NOT in the interaction set.
            n_feat = num_features_graph
            total_feat_pairs = n_feat * (n_feat - 1) // 2
            print(f"\nNEW universe: enumerating non-interaction literal pairs "
                  f"across {total_feat_pairs} feature pairs ({n_feat} features)...")

            # Process in chunks to avoid huge memory spikes
            CHUNK = 500_000
            new_uni_rows = []
            new_uni_cols = []
            new_eval_indices = []  # track which universe entries are eval pairs

            n_processed_feat_pairs = 0
            for start in range(0, total_feat_pairs, CHUNK):
                end = min(start + CHUNK, total_feat_pairs)
                lin_idx = np.arange(start, end)
                fA, fB = _feat_pair_from_lin(lin_idx, n_feat)

                # 4 combos per feature pair: (+A,+B), (+A,-B), (-A,+B), (-A,-B)
                pos_A = fA * 2        # positive node for feature A
                neg_A = fA * 2 + 1    # negative node for feature A
                pos_B = fB * 2
                neg_B = fB * 2 + 1

                combo_rows = np.concatenate([pos_A, pos_A, neg_A, neg_A])
                combo_cols = np.concatenate([pos_B, neg_B, pos_B, neg_B])

                # Canonical order (min, max)
                lo = np.minimum(combo_rows, combo_cols)
                hi = np.maximum(combo_rows, combo_cols)

                # Filter out pairs that ARE in the interaction set
                keep = np.array([
                    (int(lo[i]), int(hi[i])) not in interaction_set
                    for i in range(len(lo))
                ], dtype=bool)
                lo = lo[keep]
                hi = hi[keep]

                if subsample > 1 and len(lo) > 0:
                    # Keep every N-th pair, but force-include NEW eval entries
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
                n_processed_feat_pairs = end

            new_uni_rows = np.concatenate(new_uni_rows) if new_uni_rows else np.array([], dtype=np.int64)
            new_uni_cols = np.concatenate(new_uni_cols) if new_uni_cols else np.array([], dtype=np.int64)

            n_new_universe = len(new_uni_rows)
            print(f"NEW universe: {n_new_universe} non-interaction literal pairs"
                  f"{' (subsampled ~1/' + str(subsample) + ')' if subsample > 1 else ''}")

            if n_new_universe > 0:
                print(f"Scoring {n_new_universe} non-interaction pairs for NEW ranking...")
                new_uni_scores = score_links(
                    model, A_train, (new_uni_rows, new_uni_cols),
                    args.hop, args.max_nodes_per_hop, node_features,
                    args.batch_size, device, max_n_label)

                # Build mapping from (u,v)->index for eval lookups
                new_pair_to_idx = {}
                for idx in range(len(new_uni_rows)):
                    pair = (min(int(new_uni_rows[idx]), int(new_uni_cols[idx])),
                            max(int(new_uni_rows[idx]), int(new_uni_cols[idx])))
                    # Keep first occurrence (eval entries are force-included)
                    if pair not in new_pair_to_idx:
                        new_pair_to_idx[pair] = idx

                new_sorted = np.sort(new_uni_scores)

                # Extract scores for NEW eval entries
                eval_scores_new = np.array([new_uni_scores[new_pair_to_idx[(u, v)]]
                                            for u, v, _ in resolved]) if resolved else np.array([])
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

            # Print NEW results
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
                auc = 1.0 - (np.sum(ranks_new) - n_eval * (n_eval + 1) / 2) / (n_eval * n_other) if n_other > 0 else float('nan')
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
                rank_records = [(readable, int(r), float(pct), float(s))
                                for (u, v, readable), r, pct, s
                                in zip(resolved, ranks_new, pcts_new, eval_scores_new)]
                csv_path = f"seal_eval_ranks_{gname.lower()}.csv"
                with open(csv_path, 'w', newline='') as f:
                    writer = csv_mod.writer(f)
                    writer.writerow(['interaction', 'rank', 'rank_pct', 'score'])
                    for readable, rank, pct, score in rank_records:
                        writer.writerow([readable, rank, f"{pct:.4f}", f"{score:.6f}"])
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
            universe_path = "seal_new_universe.txt"
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
            print(f"\nNEW: {n_original} entries, all skipped — nothing to rank.")


if __name__ == '__main__':
    main()
