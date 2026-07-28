#!/usr/bin/env python3
"""aider-style repo map: rank files with per-message personalized PageRank.

Builds a synthetic 'repo dependency graph' with STRING external ids —
file nodes ("file:src/pkg_3/mod_07.py") and symbol nodes
("sym:pkg_3.mod_07.func_2") — with definition edges (symbol -> defining
file) and reference edges (referencing file -> symbol). Each simulated
chat message mentions a few files/symbols; one mg.ppr_topk batch (B =
number of messages) ranks the whole repo per message, and the top-20
files per message are printed — exactly the aider repo-map loop.

Run from the repo root after building:
    PYTHONPATH=python python3 examples/repo_map.py
"""

import sys

import numpy as np

try:
    import metal_graph as mg
except ImportError:
    mg = None

N_PACKAGES = 12
MODS_PER_PKG = 25          # 300 files
SYMS_PER_MOD = 4           # 1200 symbols
REFS_PER_FILE = (3, 12)    # references to random symbols elsewhere
K = 400                    # rank the whole graph, filter to files, keep 20
TOP_FILES = 20


def build_repo_edges(seed=11):
    rng = np.random.default_rng(seed)
    files, symbols = [], []
    sym_owner = {}
    for p in range(N_PACKAGES):
        for m in range(MODS_PER_PKG):
            f = f"file:src/pkg_{p}/mod_{m:02d}.py"
            files.append(f)
            for s in range(SYMS_PER_MOD):
                sym = f"sym:pkg_{p}.mod_{m:02d}.func_{s}"
                symbols.append(sym)
                sym_owner[sym] = f

    src, dst, w = [], [], []
    # definition edges: symbol -> defining file (weight 1.0)
    for sym, f in sym_owner.items():
        src.append(sym)
        dst.append(f)
        w.append(1.0)
    # reference edges: referencing file -> symbol (weight 0.5),
    # biased toward "popular" low-index packages (power-law-ish)
    n_sym = len(symbols)
    pop = 1.0 / (np.arange(n_sym) + 1.0) ** 0.7
    pop /= pop.sum()
    for f in files:
        n_ref = int(rng.integers(*REFS_PER_FILE))
        for sym_idx in rng.choice(n_sym, size=n_ref, replace=False, p=pop):
            sym = symbols[sym_idx]
            if sym_owner[sym] == f:
                continue  # skip self-references
            src.append(f)
            dst.append(sym)
            w.append(0.5)
    return (np.asarray(src), np.asarray(dst),
            np.asarray(w, dtype=np.float32), files, symbols)


def chat_messages(files, symbols, seed=5):
    """Each message mentions a few files/symbols (the PPR seeds)."""
    rng = np.random.default_rng(seed)
    msgs = []
    for i in range(3):
        mentions = ([files[int(j)] for j in
                     rng.choice(len(files), size=2, replace=False)] +
                    [symbols[int(j)] for j in
                     rng.choice(len(symbols), size=2, replace=False)])
        msgs.append((f"chat message {i + 1}", mentions))
    return msgs


def main():
    if mg is None:
        print("metal_graph not importable — build first, then run from the "
              "repo root with PYTHONPATH=python")
        return 1

    src, dst, w, files, symbols = build_repo_edges()
    g = mg.Graph.from_edges(src, dst, weights=w, directed=True)
    print(f"repo graph: {g.num_vertices:,} nodes "
          f"({len(files)} files, {len(symbols)} symbols), "
          f"{g.num_edges:,} edges")

    msgs = chat_messages(files, symbols)
    seed_ids, seed_w, offsets = [], [], [0]
    for _, mentions in msgs:
        idx = np.asarray(g.index_of(np.asarray(mentions)))
        seed_ids.append(idx.astype(np.uint32))
        seed_w.append(np.ones(len(idx), dtype=np.float32))
        offsets.append(offsets[-1] + len(idx))

    ids, scores = mg.ppr_topk(
        g,
        np.concatenate(seed_ids),
        np.concatenate(seed_w),
        np.asarray(offsets, dtype=np.uint64),
        k=K, alpha=0.85, tol=1e-6, max_iter=50)
    ids = np.asarray(ids).reshape(len(msgs), K)
    scores = np.asarray(scores).reshape(len(msgs), K)
    ext = np.asarray(g.external_ids)
    info = mg.last_run_info()

    for q, (title, mentions) in enumerate(msgs):
        print(f"\n--- {title} ---")
        print("mentions:", ", ".join(m.split(":", 1)[1] for m in mentions))
        print(f"top {TOP_FILES} files for the repo map:")
        shown = 0
        for rank_id, score in zip(ids[q], scores[q]):
            if rank_id < 0:
                break
            name = str(ext[rank_id])
            if not name.startswith("file:"):
                continue  # rank flows file -> symbol -> file; keep files
            shown += 1
            print(f"  {shown:2d}. {name.split(':', 1)[1]:<34s} {score:.5f}")
            if shown >= TOP_FILES:
                break

    print(f"\nbatch of {len(msgs)} messages ranked in {info['ms']:.2f} ms "
          f"engine time on the {info['path']} path "
          f"({info['iterations']} iterations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
