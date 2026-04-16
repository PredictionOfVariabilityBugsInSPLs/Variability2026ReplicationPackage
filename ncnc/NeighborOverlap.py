import argparse
import numpy as np
import torch
import copy as _copy
import os
import time

from torch_sparse import SparseTensor
from model import predictor_dict, convdict, GCN
from functools import partial
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             accuracy_score)
from torch_geometric.utils import negative_sampling
from utils import PermIterator
from ogbdataset import loaddataset_spl, MyOwnDataset
from typing import Iterable


def set_seed():
    seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"Random seed: {seed}")



class FocalLoss(torch.nn.Module):
    """Symmetric focal loss for binary classification with logits.

    Focal loss down-weights easy examples and focuses training on hard
    ones near the decision boundary.  Applied symmetrically to both
    positives and negatives:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logit) for positives, 1-sigmoid(logit) for
    negatives, and alpha_t balances the classes.

    With gamma=0 this reduces to standard BCE (optionally weighted by alpha).

    Args:
        gamma: focusing parameter (default 2.0). Higher values increase
            the focus on hard examples.
        alpha: class balance weight for positives (default 0.5 = equal).
            Negatives receive weight (1 - alpha).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        eps = 1e-7
        p = p.clamp(eps, 1 - eps)

        # p_t = probability of the true class
        p_t = labels * p + (1 - labels) * (1 - p)
        # alpha_t = class weight for the true class
        alpha_t = labels * self.alpha + (1 - labels) * (1 - self.alpha)

        focal_weight = (1 - p_t) ** self.gamma
        loss = -alpha_t * focal_weight * torch.log(p_t)
        return loss.mean()


def train_spl(model,
              predictor,
              data,
              split_edge,
              optimizer,
              batch_size,
              maskinput: bool = True,
              cnprobs: Iterable[float] = [],
              alpha: float = None,
              focal_gamma: float = 0.0):
    if alpha is not None:
        predictor.setalpha(alpha)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device).t()
    num_pos = pos_train_edge.shape[1]

    # Clamp batch_size so that at least one batch is produced
    # (PermIterator with training=True yields 0 batches when bs > size)
    batch_size = min(batch_size, num_pos)

    if focal_gamma > 0:
        criterion = FocalLoss(gamma=focal_gamma)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)

    # Sample one negative edge per positive edge
    negedge = negative_sampling(
        data.edge_index.to(pos_train_edge.device),
        data.adj_t.sizes()[0],
        num_neg_samples=num_pos
    )
    # negative_sampling may return fewer than requested; pad by repeating
    if negedge.shape[1] < num_pos:
        repeats = (num_pos + negedge.shape[1] - 1) // negedge.shape[1]
        negedge = negedge.repeat(1, repeats)[:, :num_pos]

    perm_iter = PermIterator(adjmask.device, adjmask.shape[0], batch_size)
    n_batches = len(perm_iter)
    for b_idx, perm in enumerate(perm_iter, 1):

        optimizer.zero_grad()

        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(
                tei,
                sparse_sizes=(data.num_nodes, data.num_nodes)
            ).to_device(pos_train_edge.device)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t

        h = model(data.x, adj)

        pos_edge = pos_train_edge[:, perm]
        neg_edge = negedge[:, perm]

        pos_out = predictor(h, adj, pos_edge).view(-1)
        neg_out = predictor(h, adj, neg_edge).view(-1)

        logits = torch.cat([pos_out, neg_out])
        labels = torch.cat([
            torch.ones_like(pos_out),
            torch.zeros_like(neg_out)
        ])

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        print(f"  Training: {b_idx} / {n_batches} batches  loss {loss.item():.4f}", end='\r')

    print()
    return np.mean(total_loss)


@torch.no_grad()
def eval_val_spl(model, predictor, data, split_edge, batch_size):
    """Evaluate on the validation set only (used during the training loop)."""

    model.eval()
    predictor.eval()

    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    def predict_edges(edge_tensor):
        return torch.cat([
            predictor(h, adj, edge_tensor[perm].t()).view(-1).cpu()
            for perm in PermIterator(
                edge_tensor.device,
                edge_tensor.shape[0],
                batch_size,
                False)
        ])

    pos_valid_logits = predict_edges(pos_valid_edge)
    neg_valid_logits = predict_edges(neg_valid_edge)

    pos_valid_scores = torch.sigmoid(pos_valid_logits)
    neg_valid_scores = torch.sigmoid(neg_valid_logits)

    y_val_true = torch.cat([
        torch.ones(len(pos_valid_scores)),
        torch.zeros(len(neg_valid_scores))
    ]).numpy()

    y_val_scores = torch.cat([
        pos_valid_scores,
        neg_valid_scores
    ]).numpy()

    auc = roc_auc_score(y_val_true, y_val_scores)
    ap = average_precision_score(y_val_true, y_val_scores)
    y_pred = (y_val_scores > 0.5).astype(int)

    print(f"\n--- VALIDATION (threshold=0.500) ---")
    print("AUC:", auc)
    print("AP:", ap)
    print("Accuracy:", accuracy_score(y_val_true, y_pred))
    print("Precision:", precision_score(y_val_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_val_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_val_true, y_pred, zero_division=0))


@torch.no_grad()
def test_spl(model, predictor, data, split_edge, batch_size,
             use_valedges_as_input):
    """Full evaluation on both validation and test sets."""

    model.eval()
    predictor.eval()

    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    def predict_edges(edge_tensor):
        return torch.cat([
            predictor(h, adj, edge_tensor[perm].t()).view(-1).cpu()
            for perm in PermIterator(
                edge_tensor.device,
                edge_tensor.shape[0],
                batch_size,
                False)
        ])

    # ----- get logits -----
    pos_valid_logits = predict_edges(pos_valid_edge)
    neg_valid_logits = predict_edges(neg_valid_edge)

    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)

    pos_test_logits = predict_edges(pos_test_edge)
    neg_test_logits = predict_edges(neg_test_edge)

    # ----- sigmoid -----
    pos_valid_scores = torch.sigmoid(pos_valid_logits)
    neg_valid_scores = torch.sigmoid(neg_valid_logits)
    pos_test_scores = torch.sigmoid(pos_test_logits)
    neg_test_scores = torch.sigmoid(neg_test_logits)

    # ----- threshold tuning on validation set -----
    y_val_true = torch.cat([
        torch.ones(len(pos_valid_scores)),
        torch.zeros(len(neg_valid_scores))
    ]).numpy()

    y_val_scores = torch.cat([
        pos_valid_scores,
        neg_valid_scores
    ]).numpy()

    thresholds = np.linspace(0, 1, 200)
    best_t, best_f1 = 0.5, 0

    for t in thresholds:
        pred = (y_val_scores > t).astype(int)
        f1 = f1_score(y_val_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"\nBest threshold from validation: {best_t:.3f}")

    # ----- metrics helper -----
    def report(pos_scores, neg_scores, stage, threshold):
        y_true = torch.cat([
            torch.ones(len(pos_scores)),
            torch.zeros(len(neg_scores))
        ]).numpy()

        y_scores = torch.cat([pos_scores, neg_scores]).numpy()

        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        y_pred = (y_scores > threshold).astype(int)

        print(f"\n--- {stage} (threshold={threshold:.3f}) ---")
        print("AUC:", auc)
        print("AP:", ap)
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred, zero_division=0))
        print("Recall:", recall_score(y_true, y_pred, zero_division=0))
        print("F1:", f1_score(y_true, y_pred, zero_division=0))

    # Validation: use default threshold (0.5) to avoid optimistic bias from
    # tuning the threshold on the same data we evaluate on.
    # AUC and AP are threshold-independent and remain valid either way.
    report(pos_valid_scores, neg_valid_scores, "VALIDATION", threshold=0.5)
    # Test: use the threshold tuned on validation — this is unbiased.
    report(pos_test_scores, neg_test_scores, "TEST", threshold=best_t)


def load_diff(dimacs_path, diff_path, num_nodes: int = None):
    """Parse a dimacs file and a diff file to build NEW and REMOVED groups.

    The diff file has two sections delimited by
    ``=== New Interactions ===`` and ``=== Removed Interactions ===``.
    Each entry is two space-separated feature names (as they appear in
    the dimacs ``c`` lines).  A leading ``-`` on a name means negative
    polarity; otherwise positive.

    Returns:
        eval_groups: dict with keys ``'NEW'`` and ``'REMOVED'``, each
            mapping to a list of ``(signed_a, signed_b, readable_label)``
            tuples.
    """
    # 1. Parse dimacs
    name_to_id = {}
    with open(dimacs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('c '):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        name_to_id[parts[2]] = int(parts[1])
                    except ValueError:
                        continue
    print(f"Loaded {len(name_to_id)} name -> variable ID mappings from dimacs")

    # 2. Parse diff file
    eval_groups = {'NEW': [], 'REMOVED': []}
    current_section = None
    skipped_parse = 0
    skipped_dimacs = 0
    skipped_range = 0

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

            if raw_a.startswith('-'):
                pol_a_positive = False
                name_a = raw_a[1:]
            else:
                pol_a_positive = True
                name_a = raw_a

            if raw_b.startswith('-'):
                pol_b_positive = False
                name_b = raw_b[1:]
            else:
                pol_b_positive = True
                name_b = raw_b

            if name_a not in name_to_id:
                skipped_dimacs += 1
                continue
            if name_b not in name_to_id:
                skipped_dimacs += 1
                continue

            id_a = name_to_id[name_a]
            id_b = name_to_id[name_b]

            # Check node range
            if num_nodes is not None:
                a_node = (abs(id_a) - 1) * 2 + (0 if pol_a_positive else 1)
                b_node = (abs(id_b) - 1) * 2 + (0 if pol_b_positive else 1)
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


@torch.no_grad()
def score_and_rank_spl(model, predictor, data, split_edge, batch_size,
                       eval_groups=None, eval_edge_subsample: int = 1,
                       method_prefix: str = 'ncnc'):
    """Score and rank interactions for REMOVED/NEW evaluation.

    Two ranking universes:

      REMOVED universe -- all interactions from the input file.  When
      *eval_edge_subsample* > 1 the universe is randomly subsampled to
      ~1/N of its full size, but every REMOVED eval entry is
      force-included so its rank is always available.

      NEW universe -- every literal-pair combination (4 per feature
      pair) that is NOT in the interactions file.  Subsampled and
      force-included analogously.

    Args:
        eval_groups: dict mapping group name (``'NEW'``, ``'REMOVED'``)
            to a list of ``(signed_a, signed_b, readable_label)`` tuples.
        eval_edge_subsample: keep every *N*-th pair in each ranking
            universe (default 1 = all pairs).
    """
    import time as _time

    model.eval()
    predictor.eval()

    device = data.x.device
    num_nodes = data.num_nodes
    num_features = num_nodes // 2

    # ---- adjacency for GNN forward pass (all known edges) ----
    all_pos_list = []
    for split in ['train', 'valid', 'test']:
        all_pos_list.append(split_edge[split]['edge'])
    all_pos_tensor = torch.cat(all_pos_list, dim=0)
    u_all = all_pos_tensor[:, 0]
    v_all = all_pos_tensor[:, 1]
    lo = torch.min(u_all, v_all)
    hi = torch.max(u_all, v_all)
    canon = torch.stack([lo, hi], dim=1)
    canon_unique = torch.unique(canon, dim=0)

    all_ei = canon_unique.t().contiguous()
    all_ei_undirected = torch.cat([all_ei, all_ei.flip(0)], dim=-1)
    full_adj = SparseTensor.from_edge_index(
        all_ei_undirected.to(device),
        sparse_sizes=(num_nodes, num_nodes)
    ).coalesce()

    h = model(data.x, full_adj)

    # ---- full interaction set from original file (kept on CPU) ----
    all_interactions = data.all_interactions.cpu()  # (2, E_full), canonical u < v
    n_all_int = all_interactions.shape[1]

    int_packed = all_interactions[0].long() * num_nodes + all_interactions[1].long()
    int_packed_sorted, _ = torch.sort(int_packed)

    def _is_interaction(us, vs):
        packed = us.long() * num_nodes + vs.long()
        idx = torch.searchsorted(int_packed_sorted, packed)
        idx = idx.clamp(max=int_packed_sorted.shape[0] - 1)
        return int_packed_sorted[idx] == packed

    if not eval_groups:
        print("No eval groups provided, nothing to rank.")
        return

    # ---- resolve eval entries to (u, v) node pairs ----
    def _resolve(a_signed, b_signed):
        a_node = (abs(a_signed) - 1) * 2 + (0 if a_signed > 0 else 1)
        b_node = (abs(b_signed) - 1) * 2 + (0 if b_signed > 0 else 1)
        return min(a_node, b_node), max(a_node, b_node)

    removed_entries = []
    removed_skipped = []
    for entry in eval_groups.get('REMOVED', []):
        a_s, b_s, readable = entry
        u, v = _resolve(a_s, b_s)
        if u >= num_nodes or v >= num_nodes:
            removed_skipped.append((readable, "OUT OF RANGE"))
        elif not _is_interaction(torch.tensor([u]), torch.tensor([v]))[0]:
            removed_skipped.append((readable, "NOT IN INTERACTIONS FILE"))
        else:
            removed_entries.append((u, v, readable))

    new_entries = []
    new_skipped = []
    for entry in eval_groups.get('NEW', []):
        a_s, b_s, readable = entry
        u, v = _resolve(a_s, b_s)
        if u >= num_nodes or v >= num_nodes:
            new_skipped.append((readable, "OUT OF RANGE"))
        elif _is_interaction(torch.tensor([u]), torch.tensor([v]))[0]:
            new_skipped.append((readable, "ALREADY IN INTERACTIONS FILE"))
        else:
            new_entries.append((u, v, readable))

    # ---- helpers ----
    def _score_pairs(edge_t):
        """Score a (2, N) edge tensor, return (N,) sigmoid scores on CPU."""
        n = edge_t.shape[1]
        parts = []
        n_b = (n + batch_size - 1) // batch_size
        for bi in range(n_b):
            s = bi * batch_size
            e = min(s + batch_size, n)
            logits = predictor(h, full_adj, edge_t[:, s:e].to(device)).view(-1).cpu()
            parts.append(torch.sigmoid(logits))
            print(f"  Scoring: {bi + 1}/{n_b} batches", end='\r')
        print()
        return torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)

    def _compute_ranks(query, sorted_u):
        right = torch.searchsorted(sorted_u, query, right=True)
        return len(sorted_u) - right + 1

    def _feat_pair_from_lin(indices, n_feat):
        n = n_feat
        half = (2.0 * n - 1.0) / 2.0
        A = (half - torch.sqrt(half * half - 2.0 * indices.double())).long()
        A.clamp_(0, n - 2)
        pairs_before_next = (A + 1) * n - (A + 1) * (A + 2) // 2
        A = torch.where(pairs_before_next <= indices, A + 1, A)
        pairs_before_A = A * n - A * (A + 1) // 2
        B = (A + 1 + (indices - pairs_before_A)).long()
        return A, B

    MAX_PRINT = 200

    def _report(gname, entries, skipped, entry_results, universe_size, prefix):
        n_orig = len(eval_groups.get(gname, []))
        n_res = len(entries)
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

        import csv
        csv_path = f"{prefix}_eval_ranks_{gname.lower()}.csv"
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['interaction', 'rank', 'rank_pct', 'score'])
            for readable, rank, pct, score in entry_results:
                w.writerow([readable, rank, f"{pct:.4f}", f"{score:.6f}"])
        print(f"  Saved rankings to {os.path.abspath(csv_path)}")

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

    # ==================================================================
    #  REMOVED RANKING
    #  Universe: all interactions from file (subsampled, force-include)
    # ==================================================================
    if removed_entries:
        removed_uv_packed = set(u * num_nodes + v for u, v, _ in removed_entries)

        if eval_edge_subsample > 1:
            n_keep = max(1, n_all_int // eval_edge_subsample)
            perm = torch.randperm(n_all_int)[:n_keep]
            sampled = all_interactions[:, perm]
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
            univ = all_interactions
            print(f"\nREMOVED universe: {n_all_int} interactions (all)")

        n_univ = univ.shape[1]
        shuf = torch.randperm(n_univ)
        t0 = _time.time()
        print(f"Scoring {n_univ} interactions...")
        scores_shuf = _score_pairs(univ[:, shuf])
        unshuf = torch.empty_like(shuf)
        unshuf[shuf] = torch.arange(n_univ)
        univ_scores = scores_shuf[unshuf]
        print(f"  Done in {_time.time() - t0:.1f}s")

        # build packed->index for eval lookup
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
        _report('REMOVED', removed_entries, removed_skipped, results, n_univ, method_prefix)

    # ==================================================================
    #  NEW RANKING
    #  Universe: all non-interaction literal pairs (4 per feature pair)
    #  subsampled, force-include NEW entries
    # ==================================================================
    if new_entries:
        new_packed_list = sorted(u * num_nodes + v for u, v, _ in new_entries)
        new_packed_sorted_t = torch.tensor(new_packed_list, dtype=torch.long)
        new_packed_set = set(new_packed_list)

        n_feat_pairs = num_features * (num_features - 1) // 2
        subsample_step = max(eval_edge_subsample, 1)

        print(f"\nNEW universe: enumerating non-interaction pairs "
              f"({num_features} features, {n_feat_pairs} feature pairs)...")

        FEAT_BATCH = 50000
        chunks_u = []
        chunks_v = []
        global_counter = 0
        new_eval_found = set()
        t0 = _time.time()

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

            is_int = _is_interaction(u_min, v_max)
            ni_mask = ~is_int
            u_ni = u_min[ni_mask]
            v_ni = v_max[ni_mask]
            n_ni = u_ni.shape[0]
            if n_ni == 0:
                continue

            packed = u_ni.long() * num_nodes + v_ni.long()

            # check NEW eval membership
            if new_packed_sorted_t.shape[0] > 0:
                si = torch.searchsorted(new_packed_sorted_t, packed)
                si = si.clamp(max=new_packed_sorted_t.shape[0] - 1)
                is_new = new_packed_sorted_t[si] == packed
            else:
                is_new = torch.zeros(n_ni, dtype=torch.bool)

            # subsampling
            local_idx = torch.arange(n_ni, dtype=torch.long)
            keep_sample = ((global_counter + local_idx) % subsample_step) == 0
            keep = keep_sample | is_new
            global_counter += n_ni

            u_k = u_ni[keep]
            v_k = v_ni[keep]
            if u_k.shape[0] > 0:
                chunks_u.append(u_k)
                chunks_v.append(v_k)
                # track found NEW entries
                pk_k = packed[keep]
                for pk_val in pk_k.tolist():
                    if pk_val in new_packed_set:
                        new_eval_found.add(pk_val)

            if (fb + 1) % max(1, n_fb // 20) == 0 or fb == n_fb - 1:
                elapsed = _time.time() - t0
                n_collected = sum(c.shape[0] for c in chunks_u)
                print(f"  Enumerating: {fb + 1}/{n_fb} batches, "
                      f"{n_collected} pairs  ({elapsed:.1f}s)", end='\r')
        print()

        # force-include any NEW entries missed during enumeration
        missing_u, missing_v = [], []
        for u, v, readable in new_entries:
            pk = u * num_nodes + v
            if pk not in new_eval_found:
                missing_u.append(u)
                missing_v.append(v)
        if missing_u:
            chunks_u.append(torch.tensor(missing_u, dtype=torch.long))
            chunks_v.append(torch.tensor(missing_v, dtype=torch.long))
            print(f"  Force-included {len(missing_u)} NEW entries not found during enumeration")

        if not chunks_u:
            print("  No non-interaction pairs found.")
            _report('NEW', new_entries, new_skipped, [], 0, 'ncnc')
        else:
            univ_u = torch.cat(chunks_u)
            univ_v = torch.cat(chunks_v)
            univ_new = torch.stack([univ_u, univ_v], dim=0)
            n_new_univ = univ_new.shape[1]

            elapsed = _time.time() - t0
            print(f"  NEW universe: {n_new_univ} non-interaction pairs ({elapsed:.1f}s)")
            if eval_edge_subsample > 1:
                print(f"  (subsampled ~1/{eval_edge_subsample})")

            shuf = torch.randperm(n_new_univ)
            t0 = _time.time()
            print(f"Scoring {n_new_univ} non-interaction pairs...")
            scores_shuf = _score_pairs(univ_new[:, shuf])
            unshuf = torch.empty_like(shuf)
            unshuf[shuf] = torch.arange(n_new_univ)
            new_univ_scores = scores_shuf[unshuf]
            print(f"  Done in {_time.time() - t0:.1f}s")

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
            _report('NEW', new_entries, new_skipped, results_new, n_new_univ, method_prefix)

            # Save NEW universe to file, sorted by score descending
            new_eval_packed_set = set(u * num_nodes + v for u, v, _ in new_entries)
            score_order = torch.argsort(new_univ_scores, descending=True)
            universe_path = f"{method_prefix}_new_universe.txt"
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


@torch.no_grad()
def metrics_all_edges_spl(model, predictor, data, split_edge, batch_size):
    """Score ALL edges (train + val + test) as positives, sample
    an equal number of random non-edges as negatives, and compute
    binary classification metrics (AUC, AP, Accuracy, Precision, Recall, F1).

    This evaluates how well the model can distinguish real interactions
    from random non-interactions across the entire interaction list.
    """
    model.eval()
    predictor.eval()

    device = data.x.device
    num_nodes = data.num_nodes

    # Build adjacency from training edges
    adj = data.adj_t

    # Get node embeddings
    h = model(data.x, adj)

    # Collect ALL unique undirected positive edges from train+val+test (u < v)
    all_pos_list = []
    for split in ['train', 'valid', 'test']:
        edges = split_edge[split]['edge']
        all_pos_list.append(edges)
    all_pos_tensor = torch.cat(all_pos_list, dim=0)
    u_all = all_pos_tensor[:, 0]
    v_all = all_pos_tensor[:, 1]
    lo = torch.min(u_all, v_all)
    hi = torch.max(u_all, v_all)
    canon = torch.stack([lo, hi], dim=1)
    pos_edges_2d = torch.unique(canon, dim=0)    # (E, 2) unique u<v
    pos_edges = pos_edges_2d.t().contiguous()     # (2, E)
    num_pos = pos_edges.shape[1]

    # Build full edge_index (undirected) for negative sampling exclusion
    full_ei = torch.cat([pos_edges, pos_edges.flip(0)], dim=1)

    # Sample num_pos random negative edges (balanced 1:1)
    neg_edges = negative_sampling(
        full_ei.to(device), num_nodes,
        num_neg_samples=num_pos
    )
    # negative_sampling may return fewer than requested; pad by repeating
    if neg_edges.shape[1] < num_pos:
        extra = negative_sampling(
            full_ei.to(device), num_nodes,
            num_neg_samples=num_pos - neg_edges.shape[1]
        )
        neg_edges = torch.cat([neg_edges, extra], dim=1)
    neg_edges = neg_edges[:, :num_pos]
    num_neg = neg_edges.shape[1]

    # Score positive edges in batches
    pos_scores_list = []
    for bs_start in range(0, num_pos, batch_size):
        bs_end = min(bs_start + batch_size, num_pos)
        edge_t = pos_edges[:, bs_start:bs_end].to(device)
        logits = predictor(h, adj, edge_t).view(-1).cpu()
        pos_scores_list.append(torch.sigmoid(logits))
    pos_scores = torch.cat(pos_scores_list)

    # Score negative edges in batches
    neg_scores_list = []
    for bs_start in range(0, num_neg, batch_size):
        bs_end = min(bs_start + batch_size, num_neg)
        edge_t = neg_edges[:, bs_start:bs_end].to(device)
        logits = predictor(h, adj, edge_t).view(-1).cpu()
        neg_scores_list.append(torch.sigmoid(logits))
    neg_scores = torch.cat(neg_scores_list)

    # Compute metrics
    y_true = torch.cat([
        torch.ones(len(pos_scores)),
        torch.zeros(len(neg_scores))
    ]).numpy()
    y_scores = torch.cat([pos_scores, neg_scores]).numpy()

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # Tune threshold on this data
    thresholds = np.linspace(0, 1, 200)
    best_t, best_f1 = 0.5, 0
    for t in thresholds:
        pred = (y_scores > t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred = (y_scores > best_t).astype(int)

    n_train = split_edge['train']['edge'].shape[0]
    n_val = split_edge['valid']['edge'].shape[0]
    n_test = split_edge['test']['edge'].shape[0]

    print(f"\n{'='*80}")
    print(f"METRICS ON ALL INTERACTION EDGES (train+val+test)")
    print(f"  Positive edges: {num_pos} unique "
          f"(train={n_train}, val={n_val}, test={n_test})")
    print(f"  Negative edges (random non-edges): {num_neg}")
    print(f"  Best threshold: {best_t:.3f}")
    print(f"{'='*80}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  AP:        {ap:.4f}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"{'='*80}")


def parseargs():
    parser = argparse.ArgumentParser(
        description="GNN-based link prediction for Software Product Lines (SPL)")
    parser.add_argument('--use_valedges_as_input', action='store_true',
                        help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs")

    parser.add_argument('--batch_size', type=int, default=16384, help="batch size")
    parser.add_argument('--testbs', type=int, default=16384, help="batch size for test")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")

    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hiddim', type=int, default=256, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0.05, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.7, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.0, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.05, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.4, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.0043, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.0024, help="learning rate of predictor")
    # detailed hyperparameters
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--increasealpha", action="store_true")

    parser.add_argument('--splitsize', type=int, default=-1,
                        help="split some operations inner the model. Only speed and GPU memory consumption are affected.")

    # parameters used to calibrate the edge existence probability in NCNC
    parser.add_argument('--probscale', type=float, default=4.3)
    parser.add_argument('--proboffset', type=float, default=2.8)
    parser.add_argument('--pt', type=float, default=0.75)
    parser.add_argument("--learnpt", action="store_true")

    # For scalability, NCNC samples neighbors to complete common neighbor.
    parser.add_argument('--trndeg', type=int, default=-1,
                        help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--tstdeg', type=int, default=-1,
                        help="maximum number of sampled neighbors during the test process")
    # NCN can sample common neighbors for scalability. Generally not used.
    parser.add_argument('--cndeg', type=int, default=-1)

    # predictor used, such as NCN, NCNC
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps in NCNC")
    # gnn used, such as gin, gcn.
    parser.add_argument('--model', choices=convdict.keys())

    # not used in experiments
    parser.add_argument('--cnprob', type=float, default=0)

    # --- SPL input file paths ---
    parser.add_argument('--dimacs', type=str, required=True,
                        help="path to the .dimacs file")
    parser.add_argument('--interactions', type=str, required=True,
                        help="path to the .interactions.txt file (space-separated signed integer pairs)")
    parser.add_argument('--diff', type=str, default=None,
                        help="path to the diff file (section-based text with feature names). "
                             "If not provided, ranking is skipped.")

    parser.add_argument('--savemod_path', type=str, default='saved_models/ncnc',
                        help="directory to save model.pt and predictor.pt after training "
                             "(default: saved_models/ncnc)")

    parser.add_argument('--feat_dim', type=int, default=100,
                        help="fixed dimension of the character-level feature encoding "
                             "(strings are truncated or zero-padded to this length, default: 100)")

    parser.add_argument('--train_edge_subsample', type=int, default=1,
                        help="subsample training edges: keep every N-th edge from the "
                             "interactions file before train/val/test split (default: 1 = all)")
    parser.add_argument('--eval_edge_subsample', type=int, default=1,
                        help="subsample ranking universes: keep every N-th pair in each "
                             "ranking universe during evaluation (default: 1 = all)")

    parser.add_argument('--focal_gamma', type=float, default=0.0,
                        help="focal loss gamma parameter (default: 0.0 = standard "
                             "BCE). Higher values down-weight easy examples and "
                             "focus on hard ones near the decision boundary. "
                             "Applied symmetrically to both positives and negatives.")

    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    print(args, flush=True)

    # --- Resolve file paths ---------------------------------------------------
    spl_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'spl')
    spl_raw = os.path.join(spl_root, 'raw')

    interactions_file = args.interactions
    dimacs_file = args.dimacs
    diff_file = args.diff

    # If the interactions file lives inside spl_raw, pass just the basename;
    # otherwise pass the full path so MyOwnDataset can find it.
    if os.path.dirname(os.path.abspath(interactions_file)) == os.path.abspath(spl_raw):
        interactions_for_dataset = os.path.basename(interactions_file)
    else:
        interactions_for_dataset = interactions_file

    dataset = MyOwnDataset(
        root=spl_root,
        file_name=interactions_for_dataset,
        dimacs_path=dimacs_file,
        feat_dim=args.feat_dim)
    data, split_edge = loaddataset_spl(dataset, args.use_valedges_as_input,
                                       train_edge_subsample=args.train_edge_subsample)

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # --- Build predictor factory --------------------------------------------------
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

    # --- Build model & predictor --------------------------------------------------
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
    print(f"Total Trainable Parameters: {count_parameters(model) + count_parameters(predictor)}")
    print("------------------------------------", flush=True)

    # ==================== TRAINING MODE ===========================================
    set_seed()

    optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr},
                                   {'params': predictor.parameters(), 'lr': args.prelr}])

    if args.focal_gamma > 0:
        print(f"\nUsing FocalLoss (gamma={args.focal_gamma:.2f})")
    else:
        print("\nUsing standard BCEWithLogitsLoss")

    # Best-model checkpointing (track best training loss)
    best_loss = None
    best_model_sd = None
    best_predictor_sd = None
    best_epoch = -1

    for epoch in range(1, 1 + args.epochs):
        alpha = max(0, min((epoch - 5) * 0.1, 1)) if args.increasealpha else None
        t1 = time.time()
        loss = train_spl(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha,
                         focal_gamma=args.focal_gamma)
        # Best-model checkpointing (track lowest training loss)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_model_sd = _copy.deepcopy(model.state_dict())
            best_predictor_sd = _copy.deepcopy(predictor.state_dict())
            best_epoch = epoch
            print(f"Epoch: {epoch}  Loss: {loss:.6f}  ** new best **")
        else:
            print(f"Epoch: {epoch}  Loss: {loss:.6f}  "
                  f"(best: {best_loss:.6f} @ epoch {best_epoch})")
        print("====================================")
        print(f"trn time {time.time() - t1:.2f} s", flush=True)

        t1 = time.time()
        if (epoch + 1) % 10 == 0:
            eval_val_spl(model, predictor, data, split_edge, args.testbs)
            print(f"val time {time.time() - t1:.2f} s")

        print('---', flush=True)

    # ==================== POST-TRAINING ===================================
    # Restore best model weights (lowest training loss)
    if best_model_sd is not None:
        model.load_state_dict(best_model_sd)
        predictor.load_state_dict(best_predictor_sd)
        print(f"\nRestored best model from epoch {best_epoch} "
              f"(loss {best_loss:.6f})")
    else:
        print("\nWARNING: no best checkpoint recorded, using last-epoch weights")

    # Full evaluation (val + test) with the best model
    test_spl(model, predictor, data, split_edge,
             args.testbs, args.use_valedges_as_input)

    # Save model for later inference
    save_dir = args.savemod_path
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    torch.save(predictor.state_dict(), os.path.join(save_dir, 'predictor.pt'))
    print(f"\nModel saved to {save_dir}/")

    # Metrics on all interaction edges
    metrics_all_edges_spl(model, predictor, data, split_edge,
                          args.testbs)

    # Ranking + diff comparison
    if diff_file and os.path.exists(diff_file):
        eval_groups = load_diff(dimacs_file, diff_file,
                                num_nodes=data.num_nodes)
        score_and_rank_spl(model, predictor, data, split_edge,
                           args.testbs, eval_groups=eval_groups,
                           eval_edge_subsample=args.eval_edge_subsample)


if __name__ == "__main__":
    main()

    '''
    # Training:
    python ncnc/NeighborOverlap.py --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.00025 --prelr 0.00025  --batch_size 1024  --ln --lnnn --predictor cn1  --epochs 500 --model puregcn --hiddim 256 --mplayers 1  --testbs 1024  --maskinput  --jk  --use_xlin  --tailact  --savemod_path saved_models/ncnc  --dimacs <path>.dimacs  --interactions <path>.interactions.txt

    # Inference (see ncnc/inference.py):
    python ncnc/inference.py --model_path saved_models/ncnc  --interactions <path>.interactions.txt  --dimacs <path>.dimacs  --diff <path>_diff.txt  --predictor cn1 --model puregcn --hiddim 256 --mplayers 1  --ln --lnnn --jk --use_xlin --tailact
    '''
