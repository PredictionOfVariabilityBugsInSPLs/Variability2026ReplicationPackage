# SPL Link Prediction Benchmark

A link prediction system for **Software Product Line (SPL)** feature-model interaction data. It compares five graph-based approaches to predict which feature interactions will appear (**NEW**) or disappear (**REMOVED**) between versions of configurable software systems.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ ([install guide](https://pytorch.org/get-started/locally/))
- PyTorch Geometric 2.4+ ([install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html))
- `torch_sparse` (installed alongside PyG, must match PyTorch + CUDA version)

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

All five approaches auto-detect CUDA GPUs and fall back to CPU when unavailable.

## Approaches

| Module | Method | Description |
|--------|--------|-------------|
| `ncnc/` | **NCN** | Neural Common Neighbors — GCN backbone + learned common-neighbor overlap predictor |
| `seal/` | **SEAL** | Subgraph Enclosing And Learning — DGCNN on h-hop enclosing subgraphs with DRNL labeling |
| `buddy/` | **BUDDY** | Hash-based link prediction using MinHash / HyperLogLog++ sketches (ELPH framework) |
| `gae/` | **GAE** | Graph Autoencoder — GNN encoder + Hadamard product MLP decoder |
| `heuristic/` | **Heuristic MLP** | Classical graph metrics (CN, Adamic-Adar, RA, Jaccard, PA) fed into an MLP |

## Input Data Format

Each SPL system requires three files:

**`.dimacs`** — feature variable definitions
```
c 1 CONFIG_KILL
c 2 CONFIG_FEATURE_LESS_ASK_TERMINAL
p cnf 949 796
180 0
```
Only `c <id> <name>` lines are used. CNF clauses are ignored.

**`.interactions.txt`** — space-separated signed integer pairs (no header)
```
4 -11
-4 11
6 -45
-1 -74
```
Positive = positive literal, negative = negative literal.

**`_diff.txt`** — evaluation file with NEW and REMOVED interaction changes
```
=== New Interactions ===
-CONFIG_LONG_OPTS CONFIG_ADDGROUP
CONFIG_FEATURE_2_4_MODULES CONFIG_MODPROBE_SMALL

=== Removed Interactions ===
-CONFIG_NC CONFIG_NC_110_COMPAT
```
Leading `-` on a name = negative polarity. Names must match the dimacs file exactly.

Example data is in `spls/Busybox/`, `spls/Linux/`, etc.

---

## Training

All five predictors share the same CLI pattern:

```
--interactions PATH    (required) .interactions.txt file
--dimacs PATH          (required) .dimacs file
--diff PATH            (optional) _diff.txt — if provided, ranking is performed after training
--train_edge_subsample N   subsample training edges (keep every N-th, default: 1 = all)
--eval_edge_subsample N    subsample ranking universes (default: 1 = all)
--savemod_path PATH        directory to save trained model (has per-method defaults)
```

### NCN (Neural Common Neighbors)

```bash
python3 ncnc/NeighborOverlap.py \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt \
    --predictor cn1 --model puregcn --hiddim 256 --mplayers 1 \
    --epochs 100 --batch_size 1024 --testbs 1024 \
    --xdp 0.7 --tdp 0.3 --gnndp 0.05 --predp 0.05 --preedp 0.4 --gnnedp 0.0 \
    --pt 0.75 --probscale 4.3 --proboffset 2.8 --alpha 1.0 \
    --gnnlr 0.00025 --prelr 0.00025 \
    --ln --lnnn --maskinput --jk --use_xlin --tailact \
    --savemod_path saved_models/ncnc
```

Key NCN-specific args: `--predictor` (cn1, incn1cn1, scn1, ...), `--model` (puregcn, gcn, sage, gin, ...), `--maskinput` (target link removal during training).

### SEAL

```bash
python3 seal/Main.py \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt \
    --epochs 50 --batch-size 1024 --hop 1 --hidden 256
```

Key SEAL-specific args: `--hop` (subgraph depth), `--max-nodes-per-hop`, `--sortpooling-k`. SEAL is inherently slower than other methods due to per-link subgraph extraction. Use `--eval-edge-subsample` for large datasets.

### BUDDY

```bash
python3 buddy/run_spl.py \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt \
    --epochs 50 --batch_size 1024 --lr 0.0001
```

Key BUDDY-specific args: `--max_hash_hops` (2 or 3), `--hidden_channels` (default 1024), `--minhash_num_perm`, `--hll_p`.

### GAE (Graph Autoencoder)

```bash
python3 gae/run_gae.py \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt \
    --encoder gcn --hidden_channels 256 --num_layers 2 --mlp_layers 3 \
    --epochs 100 --lr 0.001
```

Key GAE-specific args: `--encoder {gcn,sage,gat}`, `--num_layers`, `--mlp_layers`.

### Heuristic MLP

```bash
python3 heuristic/run_heuristic.py \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt \
    --hidden_channels 256 --mlp_layers 3 --epochs 10
```

No GNN — purely structural. Computes Common Neighbors, Adamic-Adar, Resource Allocation, Jaccard, and Preferential Attachment per edge pair.

---

## Inference (using a saved model)

Each module has a separate `inference.py` script. Inference uses **all edges** from the interactions file (no train/val/test split) and requires a diff file for ranking.

```
--model_path PATH          (required) directory with saved model files
--interactions PATH        (required) .interactions.txt file
--dimacs PATH              (required) .dimacs file
--diff PATH                (required) _diff.txt file for ranking
--train_edge_subsample N   subsample edges for adjacency (computational filter)
--eval_edge_subsample N    subsample ranking universes
```

### NCN Inference

```bash
python3 ncnc/inference.py \
    --model_path saved_models/ncnc \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt \
    --predictor cn1 --model puregcn --hiddim 256 --mplayers 1 \
    --ln --lnnn --jk --use_xlin --tailact
```

Model architecture flags must match training.

### SEAL Inference

```bash
python3 seal/inference.py \
    --model_path saved_models/seal \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt \
    --hop 1 --hidden 64
```

### BUDDY Inference

```bash
python3 buddy/inference.py \
    --model_path saved_models/buddy \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt
```

### GAE Inference

```bash
python3 gae/inference.py \
    --model_path saved_models/gae \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt \
    --encoder gcn --hidden_channels 256 --num_layers 2 --mlp_layers 3
```

### Heuristic MLP Inference

```bash
python3 heuristic/inference.py \
    --model_path saved_models/heuristic \
    --interactions spls/Busybox/busybox2017_01.interactions.txt \
    --dimacs spls/Busybox/busybox2017_01.dimacs \
    --diff spls/Busybox/busybox2017_01_busybox2020_12_diff.txt \
    --hidden_channels 256 --mlp_layers 3
```

---

## Evaluation: Ranking Scheme

When a `_diff.txt` file is provided, all five methods produce the same ranking output:

| Universe | Contents | Purpose |
|----------|----------|---------|
| **REMOVED** | All interactions from the input file | Rank REMOVED entries among all known interactions |
| **NEW** | All 4 literal-pair combos per feature pair NOT in interactions | Rank NEW entries among all non-interactions |

Each feature pair (A, B) generates 4 literal combinations: (+A,+B), (+A,-B), (-A,+B), (-A,-B).

### Metrics

- **AUC-ROC** — probability that a random eval entry scores higher than a random non-eval entry in the universe (computed from ranks via Wilcoxon-Mann-Whitney statistic)
- **Average/Median rank** — position in the universe (rank 1 = highest score), shown as absolute and percentile
- **Best/Worst rank**

### Subsampling

For large SPL systems, ranking universes can be very large. Use `--eval_edge_subsample N` to keep every N-th pair. Eval entries (NEW/REMOVED) are always force-included regardless of subsampling.

`--train_edge_subsample N` independently subsamples edges used for training / the forward-pass adjacency.

### Output Files

Each run produces (when a diff file is provided):
- `{method}_eval_ranks_new.csv` — per-entry rankings for NEW interactions
- `{method}_eval_ranks_removed.csv` — per-entry rankings for REMOVED interactions
- `{method}_eval_ranks_new_hist.png` — rank distribution histogram
- `{method}_eval_ranks_removed_hist.png` — rank distribution histogram

### Skipped Entries

Eval entries are skipped (with reported reasons) when:
- Feature names are not in the `.dimacs` file
- Node indices exceed the graph size
- A NEW interaction is already in the interactions file
- A REMOVED interaction is not in the interactions file

## Project Structure

```
.
├── ncnc/
│   ├── NeighborOverlap.py    # Training
│   ├── inference.py           # Inference
│   ├── model.py               # GCN backbone + predictor variants
│   ├── ogbdataset.py          # Dataset loader
│   └── utils.py               # Sparse tensor utilities
├── seal/
│   ├── Main.py                # Training
│   ├── inference.py           # Inference
│   └── util_functions.py      # Subgraph extraction, DRNL labeling
├── buddy/
│   ├── run_spl.py             # Training
│   ├── inference.py           # Inference
│   └── src/                   # ELPH model, hashing, datasets
├── gae/
│   ├── run_gae.py             # Training
│   └── inference.py           # Inference
├── heuristic/
│   ├── run_heuristic.py       # Training
│   └── inference.py           # Inference
├── spls/                      # Example SPL datasets
│   ├── Busybox/
│   ├── Linux/
│   └── ...
├── saved_models/              # Trained model checkpoints (auto-created)
├── requirements.txt
└── README.md
```
