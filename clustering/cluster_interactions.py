"""
Cluster nodes from SPL feature-model interaction data with Bayesian
hyperparameter optimization.

Supports two CSV formats (auto-detected):

  1. "interactions" format (e.g. clean2017-1_new.interactions.csv)
     Header: varA_id,varA_name,varB_id,varB_name,type
     Each row's type field like mutex(T,T) encodes polarities.

  2. "signed-pair" format (e.g. automotive-1.csv, ecos.csv)
     No header.  Each row is two signed integers: nodeA,nodeB
     The sign directly encodes polarity (+id = T, -id = F).

Usage:
    python cluster_interactions.py <filepath> [--validate <valid_invalid.txt> --dimacs <file.dimacs>]

The Louvain algorithm clusters the graph.  Optuna (TPE Bayesian optimisation)
tunes the resolution parameter to maximise:

    score = intra_fraction
            - alpha * (num_clusters / num_nodes)
            - beta  * (max_cluster_size / num_nodes)

When --validate and --dimacs are provided, the script also:
  - Reads a name->ID mapping from the DIMACS comment lines
  - Parses the valid_invalid.txt file into two lists:
      List 1 (NEW/overlap): interactions expected to be valid
      List 2 (REMOVED): interactions expected to be invalid
  - Reports what percentage of List 1 links fall within the same cluster
  - Reports what percentage of List 2 links fall between different clusters
"""

import argparse
import os
import re
import sys
import networkx as nx
import community as community_louvain
import numpy as np
import optuna
import matplotlib.pyplot as plt
from collections import defaultdict


# =============================================================================
# Format detection
# =============================================================================

def detect_format(filepath):
    """
    Sniff the first non-empty line of the file to decide the format.

    Returns
    -------
    'interactions' - if the first line looks like a header with 'varA_id'
    'signed_pair'  - if the first line consists of exactly two signed integers
    """
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for interactions header
            if 'varA_id' in line or 'varB_id' in line:
                return 'interactions'

            # Check if line is two comma-separated signed integers
            parts = line.split(',')
            if len(parts) == 2:
                try:
                    int(parts[0])
                    int(parts[1])
                    return 'signed_pair'
                except ValueError:
                    pass

            # Fallback: if line has 5+ fields and contains (T or (F, likely
            # interactions data without the expected header keywords
            if len(parts) >= 5 and re.search(r'\([TF]', line):
                return 'interactions'

            break

    raise ValueError(
        f"Cannot auto-detect format of '{filepath}'. "
        "Expected either an interactions CSV (with header containing "
        "'varA_id') or a signed-pair CSV (two integers per line)."
    )


# =============================================================================
# Parsers
# =============================================================================

def parse_interactions_line(line):
    """
    Parse a line from the interactions format:
        1,v1,5,v5,mutex(T,T)
    The type field contains a comma inside parentheses.
    Returns (node_a, node_b) as signed ints, or None.
    """
    parts = line.strip().split(',')
    if len(parts) < 6:
        return None
    var_a_id = parts[0]
    var_b_id = parts[2]
    type_str = parts[4] + ',' + parts[5]
    match = re.search(r'\(([TF]),([TF])\)', type_str)
    if not match:
        return None
    pol_a, pol_b = match.group(1), match.group(2)
    node_a = int(var_a_id) if pol_a == 'T' else -int(var_a_id)
    node_b = int(var_b_id) if pol_b == 'T' else -int(var_b_id)
    return node_a, node_b


def parse_signed_pair_line(line):
    """
    Parse a line from the signed-pair format:
        3,-1
    Returns (node_a, node_b) as signed ints, or None.
    """
    parts = line.strip().split(',')
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


# =============================================================================
# Graph construction
# =============================================================================

def build_graph(filepath):
    """
    Build the undirected graph from a CSV file.
    Auto-detects the file format and dispatches to the right parser.
    """
    fmt = detect_format(filepath)
    print(f"Detected format: {fmt}")

    if fmt == 'interactions':
        parser = parse_interactions_line
        skip_header = True
    else:
        parser = parse_signed_pair_line
        skip_header = False

    G = nx.Graph()
    edges_seen = set()
    all_var_ids = set()  # absolute variable ids

    with open(filepath, 'r') as f:
        if skip_header:
            f.readline()
        for line in f:
            result = parser(line)
            if result is None:
                continue
            node_a, node_b = result
            all_var_ids.add(abs(node_a))
            all_var_ids.add(abs(node_b))

            if node_a == node_b:
                continue
            edge_key = frozenset((node_a, node_b))
            if edge_key not in edges_seen:
                edges_seen.add(edge_key)
                G.add_edge(node_a, node_b)

    # Ensure every variable has both +/- nodes even if isolated
    for vid in all_var_ids:
        if vid not in G:
            G.add_node(vid)
        if -vid not in G:
            G.add_node(-vid)

    return G


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_partition(G, partition):
    """
    Given a graph and a node->cluster partition, compute clustering statistics
    including the density matrix.
    """
    num_clusters = max(partition.values()) + 1
    clusters = defaultdict(set)
    for node, cid in partition.items():
        clusters[cid].add(node)

    cluster_sizes = {cid: len(clusters[cid]) for cid in range(num_clusters)}

    actual_links = np.zeros((num_clusters, num_clusters), dtype=int)
    max_possible = np.zeros((num_clusters, num_clusters), dtype=int)

    for i in range(num_clusters):
        ni = cluster_sizes[i]
        max_possible[i][i] = ni * (ni - 1) // 2
        for j in range(i + 1, num_clusters):
            nj = cluster_sizes[j]
            max_possible[i][j] = ni * nj
            max_possible[j][i] = ni * nj

    for u, v in G.edges():
        cu, cv = partition[u], partition[v]
        if cu == cv:
            actual_links[cu][cv] += 1
        else:
            actual_links[cu][cv] += 1
            actual_links[cv][cu] += 1

    density_matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            if max_possible[i][j] > 0:
                density_matrix[i][j] = actual_links[i][j] / max_possible[i][j] * 100.0

    total_intra = sum(actual_links[i][i] for i in range(num_clusters))
    total_edges = G.number_of_edges()
    intra_fraction = total_intra / total_edges if total_edges > 0 else 0.0

    return {
        'intra_fraction': intra_fraction,
        'num_clusters': num_clusters,
        'cluster_sizes': cluster_sizes,
        'clusters': clusters,
        'actual_links': actual_links,
        'max_possible': max_possible,
        'density_matrix': density_matrix,
        'total_intra': total_intra,
        'total_edges': total_edges,
    }


# =============================================================================
# Objective function for Optuna
# =============================================================================

def make_objective(G, alpha, beta):
    """
    Create the Optuna objective function.

    score = intra_fraction
            - alpha * (num_clusters / num_nodes)    # penalize fragmentation
            - beta  * (max_cluster_size / num_nodes) # penalize dominance
    """
    num_nodes = G.number_of_nodes()

    def objective(trial):
        resolution = trial.suggest_float('resolution', 0.1, 20.0, log=True)

        partition = community_louvain.best_partition(
            G, random_state=42, resolution=resolution
        )

        stats = evaluate_partition(G, partition)
        nc = stats['num_clusters']
        max_size = max(stats['cluster_sizes'].values())

        intra_frac = stats['intra_fraction']
        fragmentation_penalty = alpha * (nc / num_nodes)
        dominance_penalty = beta * (max_size / num_nodes)

        score = intra_frac - fragmentation_penalty - dominance_penalty

        trial.set_user_attr('num_clusters', nc)
        trial.set_user_attr('intra_pct', intra_frac * 100)
        trial.set_user_attr('max_cluster_size', max_size)
        trial.set_user_attr('min_cluster_size',
                            min(stats['cluster_sizes'].values()))

        return score

    return objective


# =============================================================================
# Display helpers
# =============================================================================

def print_density_matrix(stats):
    """Print the full density matrix and raw counts."""
    nc = stats['num_clusters']
    cs = stats['cluster_sizes']
    dm = stats['density_matrix']
    al = stats['actual_links']
    mp = stats['max_possible']

    print("\n" + "=" * 80)
    print("CLUSTER DENSITY MATRIX (percentages: actual links / max possible links)")
    print("  Diagonal = intra-cluster density, Off-diagonal = inter-cluster density")
    print("=" * 80)

    # Adaptive column width based on max cluster size digits
    max_size = max(cs.values())
    size_width = len(str(max_size))
    col_fmt = f"  Cl{{:>2d}}({{:>{size_width}d}})"

    '''
    header = f"{'':>12s}"
    for j in range(nc):
        header += col_fmt.format(j, cs[j])
    print(header)
    print("-" * len(header))

    for i in range(nc):
        row_str = f"Cl{i:>2d}({cs[i]:>{size_width}d})"
        for j in range(nc):
            row_str += f"  {dm[i][j]:>9.2f}%"
        print(row_str)

    print("\n" + "=" * 80)
    print("RAW LINK COUNTS (actual / max possible)")
    print("=" * 80)

    header = f"{'':>12s}"
    for j in range(nc):
        header += f"  {'Cl'+str(j):>14s}"
    print(header)
    print("-" * len(header))

    for i in range(nc):
        row_str = f"{'Cl'+str(i):>12s}"
        for j in range(nc):
            row_str += f"  {al[i][j]:>6d}/{mp[i][j]:<6d}"
        print(row_str)
    '''

    total_inter = sum(al[i][j] for i in range(nc) for j in range(i + 1, nc))
    print(f"\nTotal intra-cluster links: {stats['total_intra']}")
    print(f"Total inter-cluster links: {total_inter}")
    print(f"Total links: {stats['total_edges']}")
    print(f"Fraction of links that are intra-cluster: "
          f"{stats['intra_fraction'] * 100:.1f}%")

    # ------------------------------------------------------------------
    # Search space reduction metric
    # ------------------------------------------------------------------
    num_nodes = sum(cs[i] for i in range(nc))
    total_possible_no_clusters = num_nodes * (num_nodes - 1) // 2
    total_possible_within_clusters = sum(mp[i][i] for i in range(nc))

    if total_possible_no_clusters > 0:
        reduction_pct = (total_possible_within_clusters
                         / total_possible_no_clusters * 100.0)
    else:
        reduction_pct = 0.0

    print(f"\n--- Search space reduction ---")
    print(f"  Max possible links (no clusters):     {total_possible_no_clusters:>10d}")
    print(f"  Max possible links (within clusters):  {total_possible_within_clusters:>10d}")
    print(f"  Remaining search space:                {reduction_pct:>9.2f}%")


def print_cluster_summary(stats):
    """Print a short summary of each cluster's membership."""
    clusters = stats['clusters']

    for cid in sorted(clusters.keys()):
        members = sorted(clusters[cid], key=lambda x: (abs(x), -x))
        pos = [n for n in members if n > 0]
        neg = [n for n in members if n < 0]
        display = members[:20]
        suffix = f" ... (+{len(members)-20} more)" if len(members) > 20 else ""
        print(f"  Cluster {cid}: {len(members)} nodes "
              f"({len(pos)} pos, {len(neg)} neg)")
        print(f"    Members: {display}{suffix}")


# =============================================================================
# Validation: valid_invalid.txt + DIMACS name mapping
# =============================================================================

def load_dimacs_name_map(dimacs_path):
    """
    Parse comment lines from a DIMACS file to build name -> integer ID mapping.
    Lines look like:  c 36 CONFIG_UCLIBC_HAS_FENV
    Returns dict: {'CONFIG_UCLIBC_HAS_FENV': 36, ...}
    """
    name_to_id = {}
    with open(dimacs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('c '):
                continue
            parts = line.split(None, 2)  # split on whitespace, max 3 parts
            if len(parts) < 3:
                continue
            try:
                var_id = int(parts[1])
            except ValueError:
                continue
            name = parts[2]
            name_to_id[name] = var_id
    return name_to_id


def parse_valid_invalid(txt_path, name_to_id):
    """
    Parse valid_invalid.txt into two lists of (node_a, node_b) signed-int edges.

    The file has two sections separated by headers starting with '==='.
    Section 1 (NEW interactions): lines starting with '  +'
    Section 2 (REMOVED interactions): lines starting with '  -'

    Each interaction line looks like:
      + CONFIG_FOO <-> CONFIG_BAR  [mutex(T,T)]

    The interaction type in brackets determines polarities.

    Returns (new_edges, removed_edges) as lists of (node_a, node_b) tuples.
    Names not found in name_to_id are skipped with a warning count.
    """
    new_edges = []
    removed_edges = []
    current_section = None
    skipped = 0

    interaction_re = re.compile(
        r'^\s*[+-]\s+'
        r'(\S+)\s+<->\s+(\S+)\s+'
        r'\[(\w+)\(([TF]),([TF])\)\]'
    )

    with open(txt_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            # Detect section headers
            if stripped.startswith('=== NEW'):
                current_section = 'new'
                continue
            elif stripped.startswith('=== REMOVED'):
                current_section = 'removed'
                continue
            elif stripped.startswith('==='):
                current_section = None
                continue

            if current_section is None:
                continue

            match = interaction_re.match(line)
            if not match:
                continue

            name_a = match.group(1)
            name_b = match.group(2)
            pol_a = match.group(4)
            pol_b = match.group(5)

            if name_a not in name_to_id or name_b not in name_to_id:
                skipped += 1
                continue

            id_a = name_to_id[name_a]
            id_b = name_to_id[name_b]
            node_a = id_a if pol_a == 'T' else -id_a
            node_b = id_b if pol_b == 'T' else -id_b

            edge = (node_a, node_b)
            if current_section == 'new':
                new_edges.append(edge)
            else:
                removed_edges.append(edge)

    if skipped > 0:
        print(f"  Warning: {skipped} interactions skipped "
              f"(variable name not in DIMACS mapping)")

    return new_edges, removed_edges


def validate_against_clusters(partition, new_edges, removed_edges):
    """
    Check how well the clustering predicts validity of interactions.

    For NEW (valid) interactions: report % that fall within the same cluster.
    For REMOVED (invalid) interactions: report % that fall between clusters.
    """
    print("\n" + "=" * 80)
    print("VALIDATION: valid_invalid.txt against clustering")
    print("=" * 80)

    # --- List 1: NEW interactions -> expect intra-cluster ---
    intra_count = 0
    inter_count = 0
    unknown_count = 0
    for node_a, node_b in new_edges:
        if node_a not in partition or node_b not in partition:
            unknown_count += 1
            continue
        if partition[node_a] == partition[node_b]:
            intra_count += 1
        else:
            inter_count += 1

    total_new = intra_count + inter_count
    if total_new > 0:
        intra_pct = intra_count / total_new * 100
    else:
        intra_pct = 0.0

    print(f"\n  LIST 1 - NEW interactions (overlap, expected valid)")
    print(f"    Total interactions:          {len(new_edges)}")
    if unknown_count > 0:
        print(f"    Skipped (node not in graph): {unknown_count}")
    print(f"    Within same cluster:         {intra_count:>5d}  ({intra_pct:.1f}%)")
    print(f"    Between diff. clusters:      {inter_count:>5d}  ({100-intra_pct:.1f}%)")

    # --- List 2: REMOVED interactions -> expect inter-cluster ---
    intra_count2 = 0
    inter_count2 = 0
    unknown_count2 = 0
    for node_a, node_b in removed_edges:
        if node_a not in partition or node_b not in partition:
            unknown_count2 += 1
            continue
        if partition[node_a] == partition[node_b]:
            intra_count2 += 1
        else:
            inter_count2 += 1

    total_removed = intra_count2 + inter_count2
    if total_removed > 0:
        inter_pct = inter_count2 / total_removed * 100
    else:
        inter_pct = 0.0

    print(f"\n  LIST 2 - REMOVED interactions (expected invalid)")
    print(f"    Total interactions:          {len(removed_edges)}")
    if unknown_count2 > 0:
        print(f"    Skipped (node not in graph): {unknown_count2}")
    print(f"    Between diff. clusters:      {inter_count2:>5d}  ({inter_pct:.1f}%)")
    print(f"    Within same cluster:         {intra_count2:>5d}  ({100-inter_pct:.1f}%)")


# =============================================================================
# Configuration
# =============================================================================

FILEPATH = 'dataset/spl/raw/automotive-1.csv'
VALIDATE_PATH = None  # Disabled validation
DIMACS_PATH = None    # Disabled validation
ALPHA = 0.3
BETA = 0.2
N_TRIALS = 10
EDGE_THICKNESS = 0.1  # User setting for edge thickness

# =============================================================================


# =============================================================================
# Plotting
# =============================================================================

def plot_clusters(G, partition, output_file='cluster_plot.png'):
    """
    Plot the graph where cluster nodes have the same color and are close to each other,
    while nodes from different clusters are as far apart as possible.
    Links are drawn as gray edges with configurable thickness.
    """
    print("\n" + "=" * 80)
    print("PLOTTING GRAPH")
    print("=" * 80)
    
    # 1. Compute layout to separate clusters
    print("Computing layout...")
    
    # Create a coarse-grained graph where nodes are clusters
    # This helps position the cluster centers far apart first
    cluster_graph = nx.Graph()
    clusters = defaultdict(list)
    for node, cid in partition.items():
        clusters[cid].append(node)
    
    cluster_ids = list(clusters.keys())
    cluster_graph.add_nodes_from(cluster_ids)
    
    # Add weighted edges between clusters
    # (weight = count of inter-cluster edges)
    for u, v in G.edges():
        cu, cv = partition[u], partition[v]
        if cu != cv:
            if cluster_graph.has_edge(cu, cv):
                cluster_graph[cu][cv]['weight'] += 1
            else:
                cluster_graph.add_edge(cu, cv, weight=1)
                
    # Layout the clusters (super-nodes)
    # Using a large scale to ensure separation
    # k parameter controls ideal distance between nodes.
    # We want clusters far apart, so we can use a generous k or scale.
    pos_clusters = nx.spring_layout(cluster_graph, weight='weight', k=3.0, iterations=50, seed=42)
    
    # Scale up the cluster positions to make room for nodes within clusters
    scale_factor = 20.0
    for cid in pos_clusters:
        pos_clusters[cid] *= scale_factor
        
    # Layout nodes within each cluster
    pos = {}
    for cid, nodes in clusters.items():
        subG = G.subgraph(nodes)
        if len(nodes) == 0:
            continue
            
        # Center the subgraph layout around the cluster position
        center = pos_clusters[cid]
        
        # Layout the subgraph
        # k controls node spacing within cluster
        sub_pos = nx.spring_layout(subG, center=center, k=0.5, iterations=30, seed=42)
        pos.update(sub_pos)
        
    # 2. Draw the graph
    print("Drawing graph elements...")
    plt.figure(figsize=(15, 15))
    
    # Draw edges
    # Use global EDGE_THICKNESS
    nx.draw_networkx_edges(
        G, pos, 
        alpha=0.2, 
        edge_color='gray', 
        width=EDGE_THICKNESS
    )
    
    # Draw nodes
    # Color by cluster ID
    # Use a colormap
    num_clusters = len(clusters)
    # Generate colors
    cmap = plt.get_cmap('tab20')
    
    node_colors = []
    for node in G.nodes():
        cid = partition[node]
        # Cycle through colors if more clusters than colors
        color = cmap(cid % 20)
        node_colors.append(color)
        
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=30, 
        node_color=node_colors, 
        alpha=0.8,
        linewidths=0
    )
    
    plt.title(f"Cluster Visualization (Clusters: {num_clusters})")
    plt.axis('off')
    
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("Done.")


# =============================================================================
# Main
# =============================================================================

def main():
    filepath = FILEPATH
    if not os.path.isfile(filepath):
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    print(f"File: {filepath}")

    # Build graph once
    G = build_graph(filepath)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Bayesian optimization with Optuna (TPE sampler)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"BAYESIAN OPTIMIZATION (Optuna TPE, {N_TRIALS} trials)")
    print(f"  Objective = intra_fraction "
          f"- {ALPHA}*(num_clusters/num_nodes) "
          f"- {BETA}*(max_cluster_size/num_nodes)")
    print(f"  Search range: resolution in [0.1, 20.0] (log scale)")
    print("=" * 80)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    objective = make_objective(G, alpha=ALPHA, beta=BETA)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ------------------------------------------------------------------
    # Print optimization results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS (top 15 trials by score)")
    print("=" * 80)
    print(f"{'Trial':>6s}  {'Resolution':>11s}  {'Score':>8s}  "
          f"{'Intra%':>7s}  {'#Clust':>7s}  {'MaxSize':>8s}  {'MinSize':>8s}")
    print("-" * 75)

    trials_sorted = sorted(study.trials, key=lambda t: t.value or -999,
                           reverse=True)
    for t in trials_sorted[:15]:
        res = t.params['resolution']
        attrs = t.user_attrs
        print(f"{t.number:>6d}  {res:>11.4f}  {t.value:>8.4f}  "
              f"{attrs['intra_pct']:>6.1f}%  "
              f"{attrs['num_clusters']:>7d}  "
              f"{attrs['max_cluster_size']:>8d}  "
              f"{attrs['min_cluster_size']:>8d}")

    # ------------------------------------------------------------------
    # Use best resolution for final clustering
    # ------------------------------------------------------------------
    best_resolution = study.best_params['resolution']
    best_score = study.best_value
    print(f"\nBest resolution: {best_resolution:.6f}")
    print(f"Best score:      {best_score:.6f}")

    partition = community_louvain.best_partition(
        G, random_state=42, resolution=best_resolution
    )
    stats = evaluate_partition(G, partition)

    print(f"\n--- Final clustering with resolution={best_resolution:.6f} ---")
    print(f"Number of clusters: {stats['num_clusters']}")
    print_cluster_summary(stats)
    print_density_matrix(stats)

    # ------------------------------------------------------------------
    # Validation against valid_invalid.txt
    # ------------------------------------------------------------------
    if VALIDATE_PATH and DIMACS_PATH:
        if not os.path.isfile(VALIDATE_PATH):
            print(f"\nWarning: validation file not found: {VALIDATE_PATH}",
                  file=sys.stderr)
        elif not os.path.isfile(DIMACS_PATH):
            print(f"\nWarning: DIMACS file not found: {DIMACS_PATH}",
                  file=sys.stderr)
        else:
            print(f"\nLoading DIMACS name mapping from: {DIMACS_PATH}")
            name_to_id = load_dimacs_name_map(DIMACS_PATH)
            print(f"  Loaded {len(name_to_id)} variable name -> ID mappings")

            print(f"Parsing validation file: {VALIDATE_PATH}")
            new_edges, removed_edges = parse_valid_invalid(
                VALIDATE_PATH, name_to_id
            )
            print(f"  Parsed {len(new_edges)} NEW interactions, "
                  f"{len(removed_edges)} REMOVED interactions")

            validate_against_clusters(partition, new_edges, removed_edges)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plot_clusters(G, partition)


if __name__ == '__main__':
    main()
