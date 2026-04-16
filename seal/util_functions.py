"""SEAL utility functions: subgraph extraction, DRNL labeling, SPL data loading."""

import numpy as np
import random
import os
import math
import time
from tqdm import tqdm
import scipy.sparse as ssp
from sklearn import metrics
import torch
from torch_geometric.data import Data
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)


# ---------------------------------------------------------------------------
#  SPL data loading
# ---------------------------------------------------------------------------

def parse_dimacs(path):
    """Parse a .dimacs file and return {var_id: feature_name}."""
    id_to_name = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('c '):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        var_id = int(parts[1])
                        name = parts[2]
                        id_to_name[var_id] = name
                    except ValueError:
                        continue
    return id_to_name


def feature_to_node(feature: int) -> int:
    """Map a 1-based signed feature variable to a graph node index.
    Positive literal: (|feature| - 1) * 2
    Negative literal: (|feature| - 1) * 2 + 1
    """
    base = (abs(feature) - 1) * 2
    return base if feature > 0 else base + 1


def encode_feature_names(feature_strings, feat_dim=100):
    """Character-level encoding for feature name strings.
    Returns a (N, feat_dim) float32 numpy array.
    """
    N = len(feature_strings)
    out = np.zeros((N, feat_dim), dtype=np.float32)
    for i, s in enumerate(feature_strings):
        chars = [ord(c) / 255.0 for c in s[:feat_dim]]
        out[i, :len(chars)] = chars
    return out


def load_spl_data(interactions_path, dimacs_path, feat_dim=100):
    """Load an SPL interactions CSV and dimacs file, returning:
      - A: scipy sparse adjacency matrix (symmetric, no self-loops)
      - node_features: (num_nodes, feat_dim) numpy array
      - num_nodes: int
    """
    edges = []
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
            node_a = feature_to_node(signed_a)
            node_b = feature_to_node(signed_b)
            feature_set.update([abs(signed_a), abs(signed_b)])
            edges.append((node_a, node_b))

    max_feature = max(feature_set)
    num_nodes = max_feature * 2

    # Build symmetric adjacency matrix
    if edges:
        rows, cols = zip(*edges)
        rows, cols = np.array(rows), np.array(cols)
        # Add both directions
        all_rows = np.concatenate([rows, cols])
        all_cols = np.concatenate([cols, rows])
        A = ssp.csc_matrix(
            (np.ones(len(all_rows)), (all_rows, all_cols)),
            shape=(num_nodes, num_nodes)
        )
        # Remove self-loops and binarize
        A.setdiag(0)
        A[A > 1] = 1
        A.eliminate_zeros()
    else:
        A = ssp.csc_matrix((num_nodes, num_nodes))

    # Build node features from dimacs
    id_to_name = parse_dimacs(dimacs_path)
    feature_strings = []
    for node_id in range(num_nodes):
        feat_id = node_id // 2 + 1
        is_positive = (node_id % 2 == 0)
        name = id_to_name.get(feat_id, f"V{feat_id}")
        prefix = "+" if is_positive else "-"
        feature_strings.append(f"{prefix}{name}")
    node_features = encode_feature_names(feature_strings, feat_dim)

    print(f"Loaded SPL data: {num_nodes} nodes, "
          f"{len(edges)} directed edges, "
          f"{ssp.triu(A, k=1).nnz} unique undirected edges")

    return A, node_features, num_nodes


def load_diff(dimacs_path, diff_path, num_nodes=None):
    """Parse a diff file with === New/Removed Interactions === sections.
    Each entry: space-separated feature names with optional - prefix.
    Returns dict with 'NEW' and 'REMOVED' keys.
    """
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
#  SEAL subgraph extraction and DRNL labeling
# ---------------------------------------------------------------------------

def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None,
               max_train_num=None):
    """Sample positive and negative train/test links."""
    net_triu = ssp.triu(net, k=1)
    row, col, _ = ssp.find(net_triu)

    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])

    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])

    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    total_neg = train_num + test_num
    n = net.shape[0]
    print(f'Sampling {total_neg} negative links for train and test...')

    # Vectorized negative sampling: generate random pairs in batches
    # and filter out existing edges. Much faster than one-at-a-time.
    neg_rows, neg_cols = [], []
    # Build a set of existing edges for O(1) lookup
    net_triu = ssp.triu(net, k=1)
    pos_row, pos_col, _ = ssp.find(net_triu)
    pos_set = set(zip(pos_row.tolist(), pos_col.tolist()))
    batch_mult = 4  # oversample to account for collisions
    while len(neg_rows) < total_neg:
        needed = total_neg - len(neg_rows)
        # Generate random pairs (i < j)
        ii = np.random.randint(0, n, size=needed * batch_mult)
        jj = np.random.randint(0, n, size=needed * batch_mult)
        # Ensure i < j
        lo = np.minimum(ii, jj)
        hi = np.maximum(ii, jj)
        mask = lo < hi  # remove self-loops
        lo, hi = lo[mask], hi[mask]
        # Filter out existing edges
        for a, b in zip(lo, hi):
            if (a, b) not in pos_set:
                neg_rows.append(a)
                neg_cols.append(b)
                if len(neg_rows) >= total_neg:
                    break

    train_neg = (neg_rows[:train_num], neg_cols[:train_num])
    test_neg = (neg_rows[train_num:total_neg], neg_cols[train_num:total_neg])
    return train_pos, train_neg, test_pos, test_neg


def neighbors(fringe, A):
    """Find all 1-hop neighbors of nodes in fringe from adjacency A."""
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        res = res.union(set(nei))
    return res


def node_label(subgraph):
    """Double-Radius Node Labeling (DRNL)."""
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + \
        d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0
    labels[labels < -1e6] = 0
    return labels


def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None,
                                 node_information=None):
    """Extract the h-hop enclosing subgraph around link ind=(i,j).
    Returns a PyG Data object.
    """
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = set(random.sample(sorted(fringe), max_nodes_per_hop))
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)

    # Move target nodes to positions 0 and 1
    nodes.discard(ind[0])
    nodes.discard(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]

    # DRNL labeling
    labels = node_label(subgraph)

    # Get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]

    # Remove link between target nodes
    subgraph = subgraph.tolil()
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    subgraph = subgraph.tocsc()

    # Convert to PyG Data
    edge_index = torch.tensor(
        np.array(subgraph.nonzero()), dtype=torch.long
    )

    # Node features: DRNL label one-hot + optional node features
    z = torch.tensor(labels, dtype=torch.long)

    data = Data(edge_index=edge_index, num_nodes=len(nodes))
    data.z = z
    if features is not None:
        data.node_features = torch.tensor(features, dtype=torch.float)

    return data



def links2subgraphs(A, links, y, h=1, max_nodes_per_hop=None,
                    node_information=None, no_parallel=False):
    """Extract enclosing subgraphs for a list of links.

    Uses serial extraction with a progress bar (multiprocessing on
    Windows/WSL can deadlock when pickling large sparse matrices).

    Args:
        A: scipy sparse adjacency matrix
        links: tuple (row_indices, col_indices)
        y: label for these links (1 for positive, 0 for negative)
        h: hop count
        max_nodes_per_hop: upper bound on neighbors per hop
        node_information: optional (N, d) node feature array
        no_parallel: unused (kept for API compat), always serial

    Returns:
        list of PyG Data objects with .y set, max DRNL label
    """
    max_n_label = 0
    g_list = []

    pairs = list(zip(links[0], links[1]))
    for i, j in tqdm(pairs, desc=f"Subgraphs (y={y})"):
        data = subgraph_extraction_labeling(
            (i, j), A, h, max_nodes_per_hop, node_information)
        data.y = torch.tensor([y], dtype=torch.long)
        max_n_label = max(max_n_label, data.z.max().item())
        g_list.append(data)

    return g_list, max_n_label


# ---------------------------------------------------------------------------
#  Scoring helpers
# ---------------------------------------------------------------------------

def score_links(model, A, links, h, max_nodes_per_hop, node_information,
                batch_size, device, max_z):
    """Score a set of links using the trained model.

    Extracts subgraphs and scores in streaming chunks to bound memory
    usage (only *batch_size* subgraph Data objects are alive at a time).

    Returns numpy array of scores (probabilities).
    """
    from torch_geometric.loader import DataLoader

    n_links = len(links[0])
    model.eval()
    all_scores = []

    for chunk_start in tqdm(range(0, n_links, batch_size),
                            desc="Scoring links",
                            total=(n_links + batch_size - 1) // batch_size):
        chunk_end = min(chunk_start + batch_size, n_links)
        graphs = []
        for idx in range(chunk_start, chunk_end):
            i, j = links[0][idx], links[1][idx]
            data = subgraph_extraction_labeling(
                (i, j), A, h, max_nodes_per_hop, node_information)
            data.y = torch.tensor([0], dtype=torch.long)
            data.max_z = max_z
            graphs.append(data)

        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                prob = torch.sigmoid(out).cpu().numpy().flatten()
                all_scores.append(prob)

    return np.concatenate(all_scores)
