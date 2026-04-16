#!/usr/bin/env python3
"""GAE inference script for Software Product Lines.

Loads a saved GAE model (encoder.pt + decoder.pt) and runs two-universe
ranking on a diff file, producing the same output format as run_gae.py
(AUC-ROC, avg/median rank, CSV, histogram).

Usage:
    python gae/inference.py \
        --model_path saved_models/gae \
        --interactions data.interactions.txt \
        --dimacs data.dimacs \
        --diff data_diff.txt
"""

import argparse
import csv as csv_mod
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_undirected

# Import helpers from run_gae
from gae.run_gae import (
    parse_dimacs,
    feature_to_node,
    encode_feature_names,
    load_diff,
    _feat_pair_from_lin,
    GNNEncoder,
    MLPDecoder,
    score_pairs,
    _compute_ranks,
    _is_interaction,
    _resolve_entry,
    _report,
)


def load_spl_data_full(interactions_path, dimacs_path, feat_dim=100):
    """Load SPL data with ALL edges (no train/val/test split).

    Returns (Data, num_nodes, all_interactions).
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

    # Save full canonical edge set (u < v)
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


def main():
    parser = argparse.ArgumentParser(
        description='GAE Inference -- rank diff interactions using a saved model')

    parser.add_argument('--model_path', required=True,
                        help="directory containing encoder.pt and decoder.pt")
    parser.add_argument('--interactions', required=True,
                        help="path to the .interactions.txt file")
    parser.add_argument('--dimacs', required=True,
                        help="path to the .dimacs file")
    parser.add_argument('--diff', required=True,
                        help="path to diff file")

    # Model architecture (must match training)
    parser.add_argument('--encoder', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gat'],
                        help="GNN encoder variant -- must match training (default: gcn)")
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--mlp_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="dropout rate (default: 0.0, no dropout during inference)")
    parser.add_argument('--feat_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8192)

    # Edge subsampling
    parser.add_argument('--train_edge_subsample', type=int, default=1,
                        help="subsample edges for adjacency (default: 1 = all)")
    parser.add_argument('--eval_edge_subsample', type=int, default=1,
                        help="subsample ranking universes (default: 1 = all)")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(args)

    # ==================== Load data ========================================
    data, num_nodes, all_interactions = load_spl_data_full(
        args.interactions, args.dimacs, args.feat_dim)

    # ==================== Edge subsampling for adjacency ===================
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

    # ==================== Build model ======================================
    encoder = GNNEncoder(
        in_channels=args.feat_dim,
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

    # ==================== Load weights =====================================
    encoder_path = os.path.join(args.model_path, 'encoder.pt')
    decoder_path = os.path.join(args.model_path, 'decoder.pt')
    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f"encoder.pt not found at {encoder_path}")
    if not os.path.isfile(decoder_path):
        raise FileNotFoundError(f"decoder.pt not found at {decoder_path}")

    encoder.load_state_dict(torch.load(encoder_path, map_location=device,
                                       weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device,
                                       weights_only=True))
    print(f"\nLoaded model from {os.path.abspath(args.model_path)}/")

    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in decoder.parameters())
    print(f"Encoder: {args.encoder.upper()}, {args.num_layers} layers, "
          f"{args.hidden_channels} hidden dim")
    print(f"Decoder: {args.mlp_layers}-layer MLP")
    print(f"Total parameters: {total_params:,}")

    # ==================== Forward pass =====================================
    encoder.eval()
    decoder.eval()

    # ==================== Parse diff =======================================
    eval_groups = load_diff(args.dimacs, args.diff, num_nodes)

    # For ranking, use all edges for message passing
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
                n_univ, 'gae_inference')

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
            _report('NEW', new_entries, new_skipped, [], 0, 'gae_inference')
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
                    n_new_univ, 'gae_inference')

            # Save NEW universe to file, sorted by score descending
            new_eval_packed_set = set(u * num_nodes + v for u, v, _ in new_entries)
            score_order = torch.argsort(new_univ_scores, descending=True)
            universe_path = "gae_inference_new_universe.txt"
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
