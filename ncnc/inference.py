"""Standalone inference script for NCNC link prediction.

Loads a pretrained model and runs score_and_rank_spl on the full
interaction graph (no train/val/test split).

Usage:
    python ncnc/inference.py \
        --model_path saved_models/ncnc \
        --interactions data.interactions.txt \
        --dimacs data.dimacs \
        --diff diff.txt \
        --predictor cn1 --model puregcn --hiddim 256 --mplayers 1 \
        --ln --lnnn --jk --use_xlin --tailact
"""

import argparse
import os
import sys

import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from functools import partial

# Ensure the ncnc package directory is on sys.path so that sibling
# modules (model, ogbdataset, etc.) can be imported when running this
# script directly (e.g. ``python ncnc/inference.py ...``).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from model import predictor_dict, convdict, GCN  # noqa: E402
from ogbdataset import MyOwnDataset  # noqa: E402
from NeighborOverlap import (  # noqa: E402
    load_diff,
    score_and_rank_spl,
    metrics_all_edges_spl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NCNC inference: load a pretrained model and rank interactions")

    # --- required paths ---
    parser.add_argument('--model_path', type=str, required=True,
                        help="path to directory with model.pt and predictor.pt")
    parser.add_argument('--interactions', type=str, required=True,
                        help="path to .interactions.txt file")
    parser.add_argument('--dimacs', type=str, required=True,
                        help="path to .dimacs file")
    parser.add_argument('--diff', type=str, required=True,
                        help="path to diff.txt file for ranking")

    # --- subsampling / batching ---
    parser.add_argument('--train_edge_subsample', type=int, default=1,
                        help="subsample edges for adjacency (computational filter, default: 1 = all)")
    parser.add_argument('--eval_edge_subsample', type=int, default=1,
                        help="subsample ranking universes (default: 1 = all)")
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="batch size for scoring (default: 1024)")
    parser.add_argument('--feat_dim', type=int, default=100,
                        help="fixed dimension of character-level feature encoding (default: 100)")

    # --- model architecture args (must match the trained model) ---
    parser.add_argument('--predictor', type=str, default='cn1',
                        choices=predictor_dict.keys(),
                        help="predictor type (default: cn1)")
    parser.add_argument('--model', type=str, default='puregcn',
                        choices=convdict.keys(),
                        help="GNN convolution type (default: puregcn)")
    parser.add_argument('--hiddim', type=int, default=256,
                        help="hidden dimension (default: 256)")
    parser.add_argument('--mplayers', type=int, default=1,
                        help="number of message passing layers (default: 1)")
    parser.add_argument('--nnlayers', type=int, default=3,
                        help="number of MLP layers (default: 3)")
    parser.add_argument('--ln', action='store_true',
                        help="use LayerNorm in MPNN")
    parser.add_argument('--lnnn', action='store_true',
                        help="use LayerNorm in MLP")
    parser.add_argument('--jk', action='store_true',
                        help="use JumpingKnowledge connection")
    parser.add_argument('--use_xlin', action='store_true',
                        help="use xlin in predictor")
    parser.add_argument('--tailact', action='store_true',
                        help="use tail activation in predictor")
    parser.add_argument('--twolayerlin', action='store_true',
                        help="use two-layer linear in predictor")
    parser.add_argument('--cndeg', type=int, default=-1,
                        help="common-neighbor degree sampling (default: -1 = no sampling)")
    parser.add_argument('--predp', type=float, default=0.05,
                        help="dropout ratio of predictor (default: 0.05)")
    parser.add_argument('--preedp', type=float, default=0.4,
                        help="edge dropout ratio of predictor (default: 0.4)")
    parser.add_argument('--gnndp', type=float, default=0.05,
                        help="dropout ratio of GNN (default: 0.05)")
    parser.add_argument('--xdp', type=float, default=0.7,
                        help="input dropout ratio (default: 0.7)")
    parser.add_argument('--tdp', type=float, default=0.3,
                        help="tail dropout ratio (default: 0.3)")
    parser.add_argument('--gnnedp', type=float, default=0.0,
                        help="edge dropout ratio of GNN (default: 0.0)")
    parser.add_argument('--beta', type=float, default=1.0,
                        help="beta parameter for predictor (default: 1.0)")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="alpha parameter (default: 1.0)")
    parser.add_argument('--res', action='store_true',
                        help="use residual connection in GNN")
    parser.add_argument('--depth', type=int, default=1,
                        help="number of completion steps in NCNC (default: 1)")
    parser.add_argument('--splitsize', type=int, default=-1,
                        help="split size for NCNC operations (default: -1)")
    parser.add_argument('--probscale', type=float, default=4.3,
                        help="probability scale for NCNC (default: 4.3)")
    parser.add_argument('--proboffset', type=float, default=2.8,
                        help="probability offset for NCNC (default: 2.8)")
    parser.add_argument('--pt', type=float, default=0.75,
                        help="pt parameter for NCNC (default: 0.75)")
    parser.add_argument('--learnpt', action='store_true',
                        help="use learnable pt in NCNC")
    parser.add_argument('--trndeg', type=int, default=-1,
                        help="max sampled neighbors during training (default: -1)")
    parser.add_argument('--tstdeg', type=int, default=-1,
                        help="max sampled neighbors during test (default: -1)")

    return parser.parse_args()


def main():
    args = parse_args()
    print(args, flush=True)

    # --- Validate paths ---
    model_pt = os.path.join(args.model_path, 'model.pt')
    pred_pt = os.path.join(args.model_path, 'predictor.pt')
    if not os.path.exists(model_pt) or not os.path.exists(pred_pt):
        raise FileNotFoundError(
            f"Expected model.pt and predictor.pt in {args.model_path}")

    if not os.path.exists(args.diff):
        raise FileNotFoundError(f"Diff file not found: {args.diff}")

    # --- Load dataset (full, no split) ---
    spl_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'dataset', 'spl')
    spl_raw = os.path.join(spl_root, 'raw')

    interactions_file = args.interactions
    if os.path.dirname(os.path.abspath(interactions_file)) == os.path.abspath(spl_raw):
        interactions_for_dataset = os.path.basename(interactions_file)
    else:
        interactions_for_dataset = interactions_file

    dataset = MyOwnDataset(
        root=spl_root,
        file_name=interactions_for_dataset,
        dimacs_path=args.dimacs,
        feat_dim=args.feat_dim)
    data = dataset[0]
    data.num_nodes = data.x.shape[0]

    # --- Save full interaction set (canonical u < v) ---
    ei_full = data.edge_index
    ei_full_undir = to_undirected(ei_full)
    ei_full_undir = ei_full_undir[:, ei_full_undir[0] < ei_full_undir[1]]
    data.all_interactions = ei_full_undir  # (2, E_full) canonical (u < v)

    # --- Optionally subsample edges for adjacency ---
    if args.train_edge_subsample > 1:
        n_orig = ei_full_undir.shape[1]
        n_keep = max(1, n_orig // args.train_edge_subsample)
        perm = torch.randperm(n_orig)[:n_keep]
        ei_sub = ei_full_undir[:, perm]
        edge_index = to_undirected(ei_sub)
        print(f"Edge subsampling: kept {n_keep} / {n_orig} unique edges "
              f"(every {args.train_edge_subsample}-th, "
              f"{100.0 * n_keep / n_orig:.1f}%)")
    else:
        edge_index = data.edge_index

    data.edge_index = edge_index

    # --- Build adjacency from ALL (possibly subsampled) edges ---
    data.adj_t = SparseTensor.from_edge_index(
        edge_index,
        sparse_sizes=(data.num_nodes, data.num_nodes)
    ).to_symmetric().coalesce()

    data.max_x = -1
    data.edge_weight = None

    # Build a dummy split_edge with all edges as 'train' for compatibility
    # with score_and_rank_spl (it concatenates train+valid+test).
    all_edges_t = ei_full_undir.t()  # (E, 2)
    split_edge = {
        'train': {'edge': all_edges_t},
        'valid': {'edge': torch.zeros(0, 2, dtype=torch.long),
                  'edge_neg': torch.zeros(0, 2, dtype=torch.long)},
        'test':  {'edge': torch.zeros(0, 2, dtype=torch.long),
                  'edge_neg': torch.zeros(0, 2, dtype=torch.long)},
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # --- Build predictor factory ---
    predfn = predictor_dict[args.predictor]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact,
                         twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor in ["cn0", "cn1", "incn1cn1"]:
        predfn = partial(predfn, sigm=False)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize,
                         scale=args.probscale, offset=args.proboffset,
                         trainresdeg=args.trndeg, testresdeg=args.tstdeg,
                         pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)

    # --- Build model & predictor ---
    model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                args.gnndp, args.ln, args.res, data.max_x,
                args.model, args.jk, args.gnnedp, xdropout=args.xdp,
                taildropout=args.tdp, noinputlin=False).to(device)
    predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                       args.predp, args.preedp, args.lnnn).to(device)

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"Trainable Parameters (GNN): {count_parameters(model)}")
    print(f"Trainable Parameters (Predictor): {count_parameters(predictor)}")
    print(f"Total Trainable Parameters: "
          f"{count_parameters(model) + count_parameters(predictor)}")
    print("------------------------------------", flush=True)

    # --- Load pretrained weights ---
    print(f"\n*** Loading pretrained model from {args.model_path} ***")
    keys = model.load_state_dict(
        torch.load(model_pt, map_location="cpu", weights_only=True),
        strict=False)
    print(f"  GNN unmatched keys: {keys}")
    keys = predictor.load_state_dict(
        torch.load(pred_pt, map_location="cpu", weights_only=True),
        strict=False)
    print(f"  Predictor unmatched keys: {keys}")

    # --- Metrics on all interaction edges ---
    metrics_all_edges_spl(model, predictor, data, split_edge, args.batch_size)

    # --- Ranking with diff file ---
    eval_groups = load_diff(args.dimacs, args.diff, num_nodes=data.num_nodes)
    score_and_rank_spl(model, predictor, data, split_edge,
                       args.batch_size, eval_groups=eval_groups,
                       eval_edge_subsample=args.eval_edge_subsample,
                       method_prefix='ncnc_inference')


if __name__ == "__main__":
    main()
