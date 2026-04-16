import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import RandomLinkSplit

import os.path as osp

from torch_geometric.data import Data, Dataset


class MyOwnDataset(Dataset):
    """
    Dataset loader for ``.interactions.txt`` files containing
    space-separated signed integer pairs (one pair per line).

    Each line: ``signed_A signed_B``
    Positive value = positive literal, negative = negative literal.

    If a ``.dimacs`` file is supplied via *dimacs_path*, the feature
    names found in the ``c <id> <name>`` comment lines are used to
    create character-level node features of fixed dimension *feat_dim*.
    Otherwise, identity embeddings (one-hot style) are used as features.
    """

    def __init__(self, root, file_name, dimacs_path=None, feat_dim=100,
                 transform=None, pre_transform=None, pre_filter=None):
        self.file_name = file_name
        self.dimacs_path = dimacs_path
        self.feat_dim = feat_dim

        # Determine whether file_name is an absolute path or a basename
        if osp.isabs(file_name) or osp.exists(file_name):
            # It's an external path — store it directly
            self._abs_csv_path = osp.abspath(file_name)
        else:
            # It lives inside the dataset raw/ directory
            self._abs_csv_path = None  # resolved later via raw_paths

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.file_name]

    @property
    def raw_paths(self):
        if self._abs_csv_path is not None:
            return [self._abs_csv_path]
        return [osp.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self):
        # Include the input filename in the cache key so different
        # interactions files get separate processed caches.
        import hashlib
        base = osp.basename(self.file_name)
        name = osp.splitext(base)[0]
        # Also hash the dimacs path to invalidate if it changes
        dimacs_tag = ""
        if self.dimacs_path:
            dimacs_tag = "_" + hashlib.md5(
                osp.abspath(self.dimacs_path).encode()).hexdigest()[:8]
        return [f'data_{name}{dimacs_tag}.pt']

    @staticmethod
    def _feature_to_node(feature: int) -> int:
        """Map a 1-based signed feature variable to two graph nodes:
        node for positive literal:  (|feature| - 1) * 2
        node for negative literal: (|feature| - 1) * 2 + 1
        """
        base = (abs(feature) - 1) * 2
        return base if feature > 0 else base + 1

    def _parse_interactions(self):
        """Parse a space-separated signed-integer-pair interactions file.

        Each line: ``signed_A signed_B``
        Positive value = positive literal, negative = negative literal.

        Returns (edges, feature_set) where edges are (signed_a, signed_b).
        """
        path = self.raw_paths[0]
        edges = []
        feature_set = set()
        skipped = 0

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    skipped += 1
                    continue
                try:
                    signed_a = int(parts[0])
                    signed_b = int(parts[1])
                except ValueError:
                    skipped += 1
                    continue
                feature_set.update([abs(signed_a), abs(signed_b)])
                edges.append((signed_a, signed_b))

        print(f"  Parsed {len(edges)} interactions"
              f"{f' (skipped {skipped})' if skipped else ''}")
        return edges, feature_set

    @staticmethod
    def _parse_dimacs(dimacs_path):
        """Extract variable-name mapping from dimacs comment lines."""
        var_names = {}  # id -> name
        with open(dimacs_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('c '):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            var_id = int(parts[1])
                            var_name = parts[2]
                            var_names[var_id] = var_name
                        except ValueError:
                            continue
        return var_names

    @staticmethod
    def _encode_feature_names(feature_strings, feat_dim=100):
        """Create a simple character-level encoding for each string.

        Each character is mapped to its ordinal value (0-255) normalised
        to [0, 1].  Strings shorter than *feat_dim* are zero-padded;
        longer strings are truncated.

        Returns a (N, feat_dim) float tensor where N = len(feature_strings).
        """
        N = len(feature_strings)
        out = torch.zeros(N, feat_dim, dtype=torch.float32)
        for i, s in enumerate(feature_strings):
            chars = [ord(c) / 255.0 for c in s[:feat_dim]]
            out[i, :len(chars)] = torch.tensor(chars, dtype=torch.float32)
        return out

    def process(self):
        edges, feature_set = self._parse_interactions()

        # Map each (signed) feature pair to graph nodes
        src_nodes = []
        tgt_nodes = []
        for src, tgt in edges:
            src_node = self._feature_to_node(src)
            tgt_node = self._feature_to_node(tgt)
            src_nodes.append(src_node)
            tgt_nodes.append(tgt_node)

        edge_index = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
        # Make undirected
        edge_index = to_undirected(edge_index)

        # Determine number of nodes: 2 nodes per feature (positive + negative literal)
        max_feature = max(feature_set)
        num_nodes = max_feature * 2

        # Build node features
        if self.dimacs_path is not None:
            var_names = self._parse_dimacs(self.dimacs_path)
            # Build feature strings for each node (pos/neg literal)
            feature_strings = []
            for node_id in range(num_nodes):
                feat_id = node_id // 2 + 1  # 1-based feature id
                is_positive = (node_id % 2 == 0)
                name = var_names.get(feat_id, f"V{feat_id}")
                prefix = "+" if is_positive else "-"
                feature_strings.append(f"{prefix}{name}")
            x = self._encode_feature_names(feature_strings,
                                           feat_dim=self.feat_dim)
        else:
            x = torch.zeros((num_nodes, 1), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        if idx != 0:
            raise IndexError("This dataset has only 1 element.")
        return self.data


def randomsplit_spl(dataset, val_ratio: float = 0.05, test_ratio: float = 0.05,
                    train_edge_subsample: int = 1):
    def removerepeated(ei):
        """Deduplicate undirected positive edges: make symmetric then keep u < v."""
        ei = to_undirected(ei)
        ei = ei[:, ei[0] < ei[1]]
        return ei

    # Load the single data object from your dataset
    data = dataset[0]
    data.num_nodes = data.x.shape[0]

    # Subsample edges if requested (keep every N-th edge)
    if train_edge_subsample > 1:
        ei = data.edge_index
        # Deduplicate to undirected (u < v) before subsampling to ensure
        # consistent selection regardless of edge ordering.
        ei_undir = to_undirected(ei)
        ei_undir = ei_undir[:, ei_undir[0] < ei_undir[1]]  # unique (u < v)
        n_orig = ei_undir.shape[1]
        # Randomly keep 1/N of the edges
        perm = torch.randperm(n_orig)[:n_orig // train_edge_subsample]
        ei_sub = ei_undir[:, perm]
        n_kept = ei_sub.shape[1]
        # Restore to undirected (both directions) for RandomLinkSplit
        data.edge_index = to_undirected(ei_sub)
        print(f"Edge subsampling: kept {n_kept} / {n_orig} unique edges "
              f"(every {train_edge_subsample}-th, {100.0 * n_kept / n_orig:.1f}%)")

    # Use PyG's RandomLinkSplit transform to split into train/val/test
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = transform(data)

    val_pos_mask = val_data.edge_label == 1
    val_neg_mask = val_data.edge_label == 0

    test_pos_mask = test_data.edge_label == 1
    test_neg_mask = test_data.edge_label == 0

    # Construct split_edge dictionary in the same format as OGB.
    # Only deduplicate *positive* edges (which are undirected and stored both ways).
    # Negative edges are directional samples from RandomLinkSplit -- do NOT pass
    # through removerepeated(), as that would discard valid negatives and bias
    # evaluation metrics upward.
    split_edge = {
        'train': {'edge': removerepeated(train_data.edge_index).t()},
        'valid': {
            'edge': removerepeated(val_data.edge_label_index[:, val_pos_mask]).t(),
            'edge_neg': val_data.edge_label_index[:, val_neg_mask].t(),
        },
        'test': {
            'edge': removerepeated(test_data.edge_label_index[:, test_pos_mask]).t(),
            'edge_neg': test_data.edge_label_index[:, test_neg_mask].t(),
        },
    }

    return split_edge


def loaddataset_spl(dataset: MyOwnDataset, use_valedges_as_input: bool,
                    train_edge_subsample: int = 1):
    # dataset is already an instance of MyOwnDataset
    data = dataset[0]

    # Save the FULL (pre-subsample) canonical edge set so that
    # score_and_rank_spl can check whether an eval entry is already
    # in interactions.csv (even if it was subsampled out of the split).
    ei_full = data.edge_index
    ei_full_undir = to_undirected(ei_full)
    ei_full_undir = ei_full_undir[:, ei_full_undir[0] < ei_full_undir[1]]
    data.all_interactions = ei_full_undir  # (2, E_full)  canonical (u < v)

    split_edge = randomsplit_spl(dataset, train_edge_subsample=train_edge_subsample)

    # Build the edge_index from the training edges
    data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
    data.edge_weight = None

    print(f"Number of nodes: {data.num_nodes}, max node index: {edge_index.max().item()}")

    # Build sparse adjacency matrix
    data.adj_t = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)
    ).to_symmetric().coalesce()

    data.max_x = -1

    print("\nDataset split summary:")
    for key1 in split_edge:
        for key2 in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])
    print()

    # Optionally include validation edges as input for inference
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(
            full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)
        ).to_symmetric().coalesce()
    else:
        data.full_adj_t = data.adj_t

    return data, split_edge
