# src/stage5_hybrid_retrieval.py
# Stage 5: Hybrid Retrieval — FAISS vector search + GraphRAG matching
#
# Input:
#   data/processed/jd_features.json         — Stage 1 output (JD schema)
#   data/processed/evidence/                — Stage 3 output (100 evidence files)
#   data/processed/knowledge_graph.gexf     — Stage 4 output (graph)
#
# Output:
#   data/processed/retrieval_results.json   — merged ranked candidate pool
#
# No LLM calls. No network calls. Pure local computation.
# Uses:
#   utils/embedding_client.py  (SKILLS.md pattern — do not duplicate)
#   src/stage4_graph_builder.py find_graph_matches() (SKILLS.md pattern)
#   FAISS IndexFlatIP pattern from SKILLS.md
#
# Usage:
#   python src/stage5_hybrid_retrieval.py
#   python src/stage5_hybrid_retrieval.py \
#       --jd   data/processed/jd_features.json \
#       --ev   data/processed/evidence \
#       --graph data/processed/knowledge_graph.gexf \
#       --out  data/processed/retrieval_results.json

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# IMPORTANT: sentence_transformers must be imported (and model loaded) BEFORE faiss
# on macOS ARM (M1/M2). We import faiss lazily inside build_faiss_index().
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.embedding_client import get_embedding, get_batch_embeddings
from src.stage4_graph_builder import (
    find_graph_matches,
    build_full_graph,
    extract_jd_entities,
)

ROOT            = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_JD      = os.path.join(ROOT, "data", "processed", "jd_features.json")
DEFAULT_EV_DIR  = os.path.join(ROOT, "data", "processed", "evidence")
DEFAULT_GRAPH   = os.path.join(ROOT, "data", "processed", "knowledge_graph.gexf")
DEFAULT_OUT     = os.path.join(ROOT, "data", "processed", "retrieval_results.json")

FAISS_TOP_K  = 30   # retrieve top 30 from vector search
GRAPH_TOP_K  = 30   # retrieve top 30 from graph search

# Hybrid merge weights (from task spec)
FAISS_WEIGHT = 0.6
GRAPH_WEIGHT = 0.4


# ---------------------------------------------------------------------------
# 1. Build JD embedding text
#    Concatenate: role_title + all explicit_requirements skills +
#                 all implicit_requirements + ideal_candidate_summary
# ---------------------------------------------------------------------------

def build_jd_text(jd: dict) -> str:
    """
    Produce a single text string representing the JD for embedding.
    Concatenates role_title + must/nice-to-have skill names +
    implicit requirement texts + ideal_candidate_summary.
    """
    parts = []

    role = jd.get("role_title") or ""
    if role:
        parts.append(role)

    for entry in jd.get("must_have_skills") or []:
        skill = entry.get("skill") if isinstance(entry, dict) else str(entry)
        if skill:
            parts.append(skill)
        ctx = entry.get("context") if isinstance(entry, dict) else ""
        if ctx:
            parts.append(ctx)

    for entry in jd.get("nice_to_have_skills") or []:
        skill = entry.get("skill") if isinstance(entry, dict) else str(entry)
        if skill:
            parts.append(skill)

    for req in jd.get("implicit_requirements") or []:
        txt = req.get("requirement") if isinstance(req, dict) else str(req)
        if txt:
            parts.append(txt)

    summary = jd.get("ideal_candidate_summary") or ""
    if summary:
        parts.append(summary)

    return ". ".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# 2. Build candidate embedding text from evidence file
#    Concatenate: candidate_id, evidence item claims, entity list
# ---------------------------------------------------------------------------

def build_candidate_text(ev_data: dict) -> str:
    """
    Produce a single text string representing a candidate for embedding.
    Uses: all evidence claims + entity strings from Stage 3 output.
    """
    parts = [ev_data.get("candidate_id", "")]

    for item in ev_data.get("evidence", []):
        claim = item.get("claim", "").strip()
        if claim:
            parts.append(claim)
        src = item.get("source_text") or ""
        if src and src != claim:
            parts.append(src)

    for entity in ev_data.get("entities", []):
        if entity.strip():
            parts.append(entity)

    return ". ".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# 3. FAISS index build and search — exactly from SKILLS.md
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS flat inner-product index. Embeddings must be L2-normalised."""
    import faiss   # lazy import — must come after sentence_transformers model load
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def search_faiss(
    index,
    query_embedding: np.ndarray,
    top_k: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, indices) for top_k nearest neighbours."""
    scores, indices = index.search(
        query_embedding.reshape(1, -1).astype(np.float32),
        top_k,
    )
    return scores[0], indices[0]


# ---------------------------------------------------------------------------
# 4. Load all evidence files
# ---------------------------------------------------------------------------

def load_evidence_files(ev_dir: str) -> list[dict]:
    """
    Load all *_evidence.json files from ev_dir.
    Skips error sentinels (files with 'error' key and empty evidence).
    """
    records = []
    for fname in sorted(os.listdir(ev_dir)):
        if not fname.endswith("_evidence.json"):
            continue
        path = os.path.join(ev_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Skip error sentinels
        if d.get("error") and not d.get("evidence"):
            continue
        records.append(d)
    return records


# ---------------------------------------------------------------------------
# 5. Main hybrid retrieval logic
# ---------------------------------------------------------------------------

def run_hybrid_retrieval(
    jd_path: str,
    ev_dir: str,
    out_path: str,
) -> dict:
    """
    Full hybrid retrieval pipeline.
    Returns the output dict (also saves to out_path).
    """

    # -- Load JD --
    with open(jd_path, "r", encoding="utf-8") as f:
        jd = json.load(f)

    jd_role = jd.get("role_title", "unknown")
    print(f"[Stage 5] JD role: {jd_role}")

    # -- Build JD embedding --
    jd_text = build_jd_text(jd)
    print(f"[Stage 5] JD embedding text ({len(jd_text)} chars) ...")
    jd_embedding = get_embedding(jd_text)   # 768-dim float32, L2-normalised

    # -- Load evidence files --
    ev_records = load_evidence_files(ev_dir)
    print(f"[Stage 5] Loaded {len(ev_records)} candidate evidence files")

    if not ev_records:
        print("[Stage 5 ERROR] No evidence files found.")
        sys.exit(1)

    candidate_ids = [d["candidate_id"] for d in ev_records]

    # -- Build candidate embeddings --
    print(f"[Stage 5] Building candidate embeddings ...")
    candidate_texts = [build_candidate_text(d) for d in ev_records]
    candidate_embeddings = get_batch_embeddings(candidate_texts)
    # candidate_embeddings shape: (N, 768), already L2-normalised

    # -- FAISS path --
    print(f"[Stage 5] Building FAISS index ({len(candidate_ids)} candidates) ...")
    faiss_index = build_faiss_index(candidate_embeddings)

    print(f"[Stage 5] Searching FAISS top {FAISS_TOP_K} ...")
    faiss_scores_raw, faiss_indices = search_faiss(faiss_index, jd_embedding, top_k=FAISS_TOP_K)

    # Map: candidate_id → (faiss_rank, faiss_similarity)
    # faiss_scores_raw are cosine similarities in [-1, 1] (inner product of normalised vecs)
    faiss_results: dict[str, tuple[int, float]] = {}
    for rank_0based, (idx, sim) in enumerate(zip(faiss_indices, faiss_scores_raw)):
        if idx < 0 or idx >= len(candidate_ids):
            continue
        cid = candidate_ids[int(idx)]
        faiss_results[cid] = (rank_0based + 1, float(sim))  # 1-indexed rank

    # Normalise FAISS similarities to [0, 1]
    # All sims are already cosine sims after L2-norm → range [-1, 1]
    # Map: 0.0 → 0.0, max_sim → 1.0  (linear rescale using actual range)
    if faiss_results:
        all_sims = [v[1] for v in faiss_results.values()]
        sim_min  = min(all_sims)
        sim_max  = max(all_sims)
        sim_range = sim_max - sim_min if sim_max > sim_min else 1.0

        def normalise_faiss(sim: float) -> float:
            return (sim - sim_min) / sim_range
    else:
        def normalise_faiss(sim: float) -> float:
            return 0.0

    print(f"[Stage 5] FAISS top-3 similarities: "
          f"{[round(s, 4) for s in sorted([v[1] for v in faiss_results.values()], reverse=True)[:3]]}")

    # -- Graph path --
    print(f"[Stage 5] Running graph-based matching ...")
    jd_entities = extract_jd_entities(jd)

    # Rebuild candidate graphs from evidence (Stage 4 pattern)
    # We need candidate_graphs dict — rebuild in-memory from evidence files
    # (cheaper than re-loading the full GEXF which only has merged nodes)
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.stage4_graph_builder import build_candidate_graph
    candidate_graphs: dict = {}
    for ev in ev_records:
        cid = ev["candidate_id"]
        cg  = build_candidate_graph(cid, ev.get("evidence", []))
        candidate_graphs[cid] = cg

    graph_matches = find_graph_matches(jd_entities, candidate_graphs, top_k=GRAPH_TOP_K)

    # Map: candidate_id → graph_score
    graph_results: dict[str, float] = {cid: score for cid, score in graph_matches}

    # Normalise graph scores to [0, 1]
    if graph_results:
        g_max = max(graph_results.values())
        g_min = min(graph_results.values())
        g_range = g_max - g_min if g_max > g_min else 1.0

        def normalise_graph(score: float) -> float:
            return (score - g_min) / g_range
    else:
        def normalise_graph(score: float) -> float:
            return 0.0

    print(f"[Stage 5] Graph top-3 matches: "
          f"{[(cid, round(s, 4)) for cid, s in graph_matches[:3]]}")

    # -- Merge: union of FAISS top-30 and graph top-30 --
    all_candidate_ids = set(faiss_results.keys()) | set(graph_results.keys())
    # Also include all candidates with a 0 score for completeness
    all_candidate_ids |= set(candidate_ids)

    print(f"[Stage 5] Merging results — union pool: {len(all_candidate_ids)} candidates")

    merged: list[dict] = []
    for cid in all_candidate_ids:
        faiss_rank    = None
        faiss_sim_raw = None   # None = not retrieved by FAISS
        g_score       = 0.0

        if cid in faiss_results:
            faiss_rank    = faiss_results[cid][0]
            faiss_sim_raw = faiss_results[cid][1]

        if cid in graph_results:
            g_score = graph_results[cid]

        # Normalise each component independently:
        # - FAISS: 0.0 for non-retrieved; normalised value for retrieved
        # - Graph: 0.0 for non-retrieved; normalised value for retrieved
        faiss_norm = normalise_faiss(faiss_sim_raw) if faiss_sim_raw is not None else 0.0
        graph_norm = normalise_graph(g_score) if g_score > 0 else 0.0

        hybrid = round(
            (faiss_norm * FAISS_WEIGHT) + (graph_norm * GRAPH_WEIGHT),
            6,
        )

        merged.append({
            "candidate_id":     cid,
            "faiss_rank":       faiss_rank,
            "faiss_similarity": round(faiss_sim_raw, 6) if faiss_sim_raw is not None else 0.0,
            "graph_score":      round(g_score, 6),
            "hybrid_score":     hybrid,
            "hybrid_rank":      None,   # assigned below after sort
        })

    # Sort by hybrid_score descending, then candidate_id ascending for ties
    merged.sort(key=lambda x: (-x["hybrid_score"], x["candidate_id"]))

    # Assign hybrid_rank (1-indexed)
    for rank_idx, row in enumerate(merged, start=1):
        row["hybrid_rank"] = rank_idx

    # -- Build output --
    output = {
        "retrieval_timestamp": datetime.now().isoformat() + "Z",
        "jd_role":             jd_role,
        "faiss_top_k":         FAISS_TOP_K,
        "graph_top_k":         GRAPH_TOP_K,
        "hybrid_weights":      {"faiss": FAISS_WEIGHT, "graph": GRAPH_WEIGHT},
        "candidates":          merged,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Hybrid retrieval — FAISS + GraphRAG."
    )
    parser.add_argument("--jd",    default=DEFAULT_JD,
                        help="JD features JSON (default: data/processed/jd_features.json)")
    parser.add_argument("--ev",    default=DEFAULT_EV_DIR,
                        help="Evidence directory (default: data/processed/evidence/)")
    parser.add_argument("--graph", default=DEFAULT_GRAPH,
                        help="Knowledge graph GEXF (default: data/processed/knowledge_graph.gexf)")
    parser.add_argument("--out",   default=DEFAULT_OUT,
                        help="Output JSON (default: data/processed/retrieval_results.json)")
    args = parser.parse_args()

    for path, label in [
        (args.jd,    "jd_features.json"),
        (args.ev,    "evidence dir"),
        (args.graph, "knowledge_graph.gexf"),
    ]:
        if not os.path.exists(path):
            print(f"[Stage 5 ERROR] {label} not found: {path}")
            sys.exit(1)

    print("[Stage 5] Starting hybrid retrieval ...")
    output = run_hybrid_retrieval(
        jd_path  = args.jd,
        ev_dir   = args.ev,
        out_path = args.out,
    )

    candidates  = output["candidates"]
    top1        = candidates[0] if candidates else {}
    top_cid     = top1.get("candidate_id", "?")
    top_score   = top1.get("hybrid_score", 0.0)
    total_pool  = len(candidates)

    print(
        f"\n[Stage 5 Complete] Hybrid retrieval done. "
        f"Top candidate: {top_cid} (hybrid_score: {top_score:.4f}). "
        f"Total pool for scoring: {total_pool} candidates."
    )
    print(f"[Stage 5] Results saved → {args.out}")


if __name__ == "__main__":
    main()
