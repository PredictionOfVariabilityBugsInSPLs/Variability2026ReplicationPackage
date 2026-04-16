#!/usr/bin/env python3
"""BUDDY inference script for Software Product Lines.

Standalone script that loads a trained BUDDY model and performs two-universe
ranking on a diff file, using the same scoring approach as run_spl.py.

Usage:
    python buddy/inference.py \
        --model_path saved_models/buddy \
        --interactions data.interactions.txt \
        --dimacs data.dimacs \
        --diff data.diff
"""

import argparse
import csv as csv_mod
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Add buddy/src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from models.elph import BUDDY
from hashing import ElphHashes
from utils import str2bool

import scipy.sparse as ssp


# ---------------------------------------------------------------------------
#  SPL data loading  (same as run_spl.py)
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


def encode_feature_names(feature_strings, feat_dim=100):
    N = len(feature_strings)
    out = torch.zeros(N, feat_dim, dtype=torch.float32)
    for i, s in enumerate(feature_strings):
        chars = [ord(c) / 255.0 for c in s[:feat_dim]]
        out[i, :len(chars)] = torch.tensor(chars, dtype=torch.float32)
    return out


def load_spl_data(interactions_path, dimacs_path, feat_dim=100):
    """Load SPL data, return (Data, num_nodes)."""
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

    # Node features from dimacs
    id_to_name = parse_dimacs(dimacs_path)
    feature_strings = []
    for nid in range(num_nodes):
        fid = nid // 2 + 1
        pol = "+" if nid % 2 == 0 else "-"
        feature_strings.append(f"{pol}{id_to_name.get(fid, f'V{fid}')}")
    x = encode_feature_names(feature_strings, feat_dim)

    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    data.edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    n_edges = edge_index[:, edge_index[0] < edge_index[1]].size(1)
    print(f"Loaded SPL data: {num_nodes} nodes, {n_edges} unique undirected edges")
    return data, num_nodes


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
#  Score links with trained BUDDY model
# ---------------------------------------------------------------------------

def score_buddy_links(model, hash_dataset, links, args, device):
    """Score a set of (u, v) pairs using the trained BUDDY model.

    Builds hash features for the given links using the hash tables
    already computed in hash_dataset, then runs inference.

    Args:
        links: Tensor [n, 2]
    Returns:
        numpy array of sigmoid scores
    """
    model.eval()
    data = hash_dataset

    # Build subgraph features for these links
    sf = data['elph_hashes'].get_subgraph_features(
        links, data['hashes'], data['cards'],
        args.subgraph_feature_batch_size)
    if args.floor_sf:
        sf[sf < 0] = 0

    scores = []
    loader = DataLoader(range(len(links)), args.eval_batch_size, shuffle=False)
    with torch.no_grad():
        for indices in loader:
            curr_links = links[indices]
            subgraph_features = sf[indices].to(device)
            node_features = data['x'][curr_links].to(device)
            degrees = data['degrees'][curr_links].to(device)
            logits = model(subgraph_features, node_features,
                           degrees[:, 0], degrees[:, 1], None, None)
            scores.append(torch.sigmoid(logits.view(-1)).cpu())

    return torch.cat(scores).numpy()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='BUDDY Inference for SPL')

    # Required paths
    parser.add_argument('--model_path', required=True,
                        help="directory containing buddy_model.pt")
    parser.add_argument('--interactions', required=True,
                        help="path to the .interactions.txt file")
    parser.add_argument('--dimacs', required=True,
                        help="path to the .dimacs file")
    parser.add_argument('--diff', required=True,
                        help="path to diff file")

    # Data
    parser.add_argument('--feat_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=100000)
    parser.add_argument('--train_edge_subsample', type=int, default=1,
                        help="subsample edges: keep every N-th edge (default: 1 = all)")
    parser.add_argument('--eval_edge_subsample', type=int, default=1,
                        help="subsample ranking universes (default: 1 = all)")

    # BUDDY model architecture (must match training)
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--feature_dropout', type=float, default=0.5)
    parser.add_argument('--sign_dropout', type=float, default=0.5)
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--use_feature', type=str2bool, default=True)
    parser.add_argument('--use_struct_feature', type=str2bool, default=True)
    parser.add_argument('--add_normed_features', type=str2bool, default=False)
    parser.add_argument('--feature_prop', type=str, default='gcn')

    # Hash
    parser.add_argument('--max_hash_hops', type=int, default=2)
    parser.add_argument('--hll_p', type=int, default=8)
    parser.add_argument('--minhash_num_perm', type=int, default=128)
    parser.add_argument('--use_zero_one', type=str2bool, default=True)
    parser.add_argument('--floor_sf', type=str2bool, default=False)
    parser.add_argument('--subgraph_feature_batch_size', type=int, default=11000000)

    # Misc
    parser.add_argument('--use_RA', type=str2bool, default=False)

    # Compat stubs (not used but needed by BUDDY/HashDataset/ElphHashes internals)
    parser.add_argument('--model', default='BUDDY')
    parser.add_argument('--dataset_name', default='SPL')
    parser.add_argument('--train_node_embedding', action='store_true')
    parser.add_argument('--propagate_embeddings', action='store_true')
    parser.add_argument('--pretrained_node_embedding', default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--log_features', action='store_true')
    parser.add_argument('--dynamic_train', action='store_true')
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--num_negs', type=int, default=1)
    parser.add_argument('--train_samples', type=float, default=math.inf)
    parser.add_argument('--val_samples', type=float, default=math.inf)
    parser.add_argument('--test_samples', type=float, default=math.inf)
    parser.add_argument('--year', type=int, default=0)
    parser.add_argument('--use_edge_weight', action='store_true')
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_features', action='store_true')
    parser.add_argument('--load_hashes', action='store_true')
    parser.add_argument('--cache_subgraph_features', action='store_true')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args)

    # ==================== Load data ========================================
    data, num_nodes = load_spl_data(args.interactions, args.dimacs, args.feat_dim)

    # Save full interaction set BEFORE any subsampling (for ranking later)
    ei_full = to_undirected(data.edge_index)
    mask_full = ei_full[0] < ei_full[1]
    ei_full_unique = ei_full[:, mask_full]
    all_pos_set_orig = set(
        (ei_full_unique[0, i].item(), ei_full_unique[1, i].item())
        for i in range(ei_full_unique.shape[1]))

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
        data.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float)
        print(f"Edge subsampling: kept {n_keep} / {n_orig} unique edges "
              f"(1/{args.train_edge_subsample}, {100.0 * n_keep / n_orig:.1f}%)")

    # ==================== Build hash tables from full edge set =============
    print("\nBuilding hash tables for scoring...")
    elph = ElphHashes(args)
    hashes, cards = elph.build_hash_tables(num_nodes, data.edge_index)

    # Build adjacency for degrees
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float)
    A = ssp.csr_matrix(
        (edge_weight.numpy(), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
        shape=(num_nodes, num_nodes)
    )
    degrees = torch.tensor(A.sum(axis=0, dtype=float), dtype=torch.float).flatten()

    # Preprocess node features (one step of GCN propagation, same as HashDataset)
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    import torch_sparse
    gcn_ei, gcn_ew = gcn_norm(data.edge_index, edge_weight.float(), num_nodes)
    x = torch_sparse.spmm(gcn_ei, gcn_ew, data.x.shape[0], data.x.shape[0], data.x)

    # Package hash data for scoring
    hash_data = {
        'elph_hashes': elph,
        'hashes': hashes,
        'cards': cards,
        'x': x,
        'degrees': degrees,
    }

    # ==================== Load model ======================================
    model_file = os.path.join(args.model_path, 'buddy_model.pt')
    if not os.path.exists(model_file):
        print(f"ERROR: Model file not found at {model_file}")
        sys.exit(1)

    num_features = x.shape[1]
    model = BUDDY(args, num_features).to(device)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLoaded model from {os.path.abspath(model_file)}")
    print(f"Model parameters: {total_params:,}")

    # ==================== Parse diff ======================================
    eval_groups = load_diff(args.dimacs, args.diff, num_nodes)

    # Use the full interaction set saved before subsampling
    all_pos_set = all_pos_set_orig
    print(f"\nFull interaction set: {len(all_pos_set)} unique undirected edges")

    n_feat = num_nodes // 2

    # Helper: map linear index to feature pair (A, B) with A < B
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
            elif gname == 'REMOVED' and (u, v) not in all_pos_set:
                skipped.append((readable, "NOT IN INTERACTIONS FILE"))
            elif gname == 'NEW' and (u, v) in all_pos_set:
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

    # =====================================================================
    #  REMOVED ranking: universe = ALL interactions from the input file
    # =====================================================================
    gname = 'REMOVED'
    resolved = eval_resolved[gname]
    skipped = eval_skipped[gname]
    n_orig = len(eval_groups.get(gname, []))
    eval_force_set = eval_pair_sets[gname]

    if resolved:
        all_pos_list = list(all_pos_set)
        if args.eval_edge_subsample > 1:
            rng = np.random.default_rng()
            n_total = len(all_pos_list)
            n_keep = max(n_total // args.eval_edge_subsample, 1)
            force_pairs = [p for p in all_pos_list if p in eval_force_set]
            other_pairs = [p for p in all_pos_list if p not in eval_force_set]
            n_sample_other = max(n_keep - len(force_pairs), 0)
            if n_sample_other < len(other_pairs):
                idx = rng.choice(len(other_pairs), size=n_sample_other, replace=False)
                sampled_other = [other_pairs[i] for i in idx]
            else:
                sampled_other = other_pairs
            removed_universe_pairs = force_pairs + sampled_other
            print(f"\nREMOVED universe: subsampled {len(removed_universe_pairs)} / "
                  f"{n_total} interactions (1/{args.eval_edge_subsample}), "
                  f"force-included {len(force_pairs)} eval entries")
        else:
            removed_universe_pairs = all_pos_list
            print(f"\nREMOVED universe: {len(removed_universe_pairs)} interactions (full set)")

        # Score REMOVED universe
        print(f"Scoring {len(removed_universe_pairs)} REMOVED universe pairs...")
        rem_links = torch.tensor(removed_universe_pairs, dtype=torch.long)
        removed_scores = score_buddy_links(model, hash_data, rem_links, args, device)

        rem_pair_to_idx = {}
        for i, (u, v) in enumerate(removed_universe_pairs):
            rem_pair_to_idx[(u, v)] = i

        removed_sorted = np.sort(removed_scores)
        rem_u_size = len(removed_scores)
        rem_u_label = f"all interactions ({rem_u_size})"

        rem_eval_scores = np.array([removed_scores[rem_pair_to_idx[(u, v)]]
                                    for u, v, _ in resolved]) if resolved else np.array([])
        rem_ranks = _ranks(rem_eval_scores, removed_sorted) if len(rem_eval_scores) > 0 else np.array([])
        rem_pcts = 100.0 * rem_ranks / rem_u_size if len(rem_ranks) > 0 else np.array([])

        # Print REMOVED results
        verbose = n_orig <= 200
        print(f"\n{'='*100}")
        n_not_in = sum(1 for _, r in skipped if r == "NOT IN INTERACTIONS FILE")
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
            for idx, ((u, v, readable), r, pct, s) in enumerate(
                    zip(resolved, rem_ranks, rem_pcts, rem_eval_scores), 1):
                print(f"  {idx:>3}  {readable:<55} {u}-{v:>8}  "
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
            auc = 1.0 - (np.sum(rem_ranks) - n_eval * (n_eval + 1) / 2) / (n_eval * n_other) if n_other > 0 else float('nan')
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
            print(f"  No interactions from this group could be resolved.")
        print(f"{'='*100}")

        # REMOVED CSV + Histogram
        if len(rem_ranks) > 0:
            rank_records = [(readable, int(r), float(pct), float(s))
                            for (u, v, readable), r, pct, s
                            in zip(resolved, rem_ranks, rem_pcts, rem_eval_scores)]

            csv_path = f"buddy_eval_ranks_{gname.lower()}.csv"
            with open(csv_path, 'w', newline='') as f:
                w = csv_mod.writer(f)
                w.writerow(['interaction', 'rank', 'rank_pct', 'score'])
                for readable, rank, pct, score in rank_records:
                    w.writerow([readable, rank, f"{pct:.4f}", f"{score:.6f}"])
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
                             f'universe: {rem_u_label})')
                ax.set_xticks(np.arange(0, 110, 10))
                ax.set_xlim(0, 100)
                hist_path = f"buddy_eval_ranks_{gname.lower()}_hist.png"
                fig.tight_layout()
                fig.savefig(hist_path, dpi=150)
                plt.close(fig)
                print(f"  Saved histogram to {os.path.abspath(hist_path)}")
            except ImportError:
                print("  (matplotlib not available, skipping histogram)")
    elif n_orig > 0:
        print(f"\nREMOVED: {n_orig} entries, all skipped -- nothing to rank.")

    # =====================================================================
    #  NEW ranking: universe = all 4 literal-pair combos per feature pair
    #               NOT in the interactions file
    # =====================================================================
    gname = 'NEW'
    resolved = eval_resolved[gname]
    skipped = eval_skipped[gname]
    n_orig = len(eval_groups.get(gname, []))
    eval_force_set = eval_pair_sets[gname]

    if resolved:
        total_feat_pairs = n_feat * (n_feat - 1) // 2

        print(f"\nNEW universe: enumerating literal pairs for {total_feat_pairs} "
              f"feature pairs (n_feat={n_feat}), excluding interactions...")

        new_universe_rows = []
        new_universe_cols = []
        BATCH = 500_000
        n_excluded = 0

        for start in range(0, total_feat_pairs, BATCH):
            end = min(start + BATCH, total_feat_pairs)
            lin = np.arange(start, end)
            A_arr, B_arr = _feat_pair_from_lin(lin, n_feat)
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
                    if (u_int, v_int) not in all_pos_set:
                        new_universe_rows.append(u_int)
                        new_universe_cols.append(v_int)
                    else:
                        n_excluded += 1

        print(f"  Total candidate literal pairs: {total_feat_pairs * 4}")
        print(f"  Excluded (in interactions file): {n_excluded}")
        print(f"  NEW universe before subsampling: {len(new_universe_rows)}")

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
                  f"(every {args.eval_edge_subsample}-th + {len(force_indices)} "
                  f"force-included eval entries)")
        else:
            print(f"  NEW universe (full): {len(new_universe_rows)}")

        # Score NEW universe
        n_new_universe = len(new_universe_rows)
        print(f"Scoring {n_new_universe} NEW universe pairs...")
        new_links = torch.tensor(list(zip(new_universe_rows, new_universe_cols)),
                                 dtype=torch.long)
        new_scores = score_buddy_links(model, hash_data, new_links, args, device)

        new_pair_to_idx = {}
        for i in range(n_new_universe):
            pair = (new_universe_rows[i], new_universe_cols[i])
            new_pair_to_idx[pair] = i

        new_sorted = np.sort(new_scores)
        new_u_size = len(new_scores)
        new_u_label = f"non-interaction literal pairs ({new_u_size})"

        new_eval_scores = np.array([new_scores[new_pair_to_idx[(u, v)]]
                                    for u, v, _ in resolved]) if resolved else np.array([])
        new_ranks = _ranks(new_eval_scores, new_sorted) if len(new_eval_scores) > 0 else np.array([])
        new_pcts = 100.0 * new_ranks / new_u_size if len(new_ranks) > 0 else np.array([])

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
            for idx, ((u, v, readable), r, pct, s) in enumerate(
                    zip(resolved, new_ranks, new_pcts, new_eval_scores), 1):
                print(f"  {idx:>3}  {readable:<55} {u}-{v:>8}  "
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
            auc = 1.0 - (np.sum(new_ranks) - n_eval * (n_eval + 1) / 2) / (n_eval * n_other) if n_other > 0 else float('nan')
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
            print(f"  No interactions from this group could be resolved.")
        print(f"{'='*100}")

        # NEW CSV + Histogram
        if len(new_ranks) > 0:
            rank_records = [(readable, int(r), float(pct), float(s))
                            for (u, v, readable), r, pct, s
                            in zip(resolved, new_ranks, new_pcts, new_eval_scores)]

            csv_path = f"buddy_eval_ranks_{gname.lower()}.csv"
            with open(csv_path, 'w', newline='') as f:
                w = csv_mod.writer(f)
                w.writerow(['interaction', 'rank', 'rank_pct', 'score'])
                for readable, rank, pct, score in rank_records:
                    w.writerow([readable, rank, f"{pct:.4f}", f"{score:.6f}"])
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
                             f'universe: {new_u_label})')
                ax.set_xticks(np.arange(0, 110, 10))
                ax.set_xlim(0, 100)
                hist_path = f"buddy_eval_ranks_{gname.lower()}_hist.png"
                fig.tight_layout()
                fig.savefig(hist_path, dpi=150)
                plt.close(fig)
                print(f"  Saved histogram to {os.path.abspath(hist_path)}")
            except ImportError:
                print("  (matplotlib not available, skipping histogram)")

        # Save NEW universe to file, sorted by score descending
        new_eval_set = set((u, v) for u, v, _ in resolved)
        score_order = np.argsort(new_scores)[::-1]
        universe_path = "buddy_inference_new_universe.txt"
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
