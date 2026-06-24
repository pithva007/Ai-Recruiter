# src/stage4_graph_builder.py
# Stage 4: GraphRAG Knowledge Graph Builder
#
# Input:
#   data/processed/evidence/           — {candidate_id}_evidence.json per candidate
#   data/processed/jd_features.json    — structured JD requirements from Stage 1
#
# Output:
#   data/processed/knowledge_graph.gexf  — full graph (networkx GEXF format)
#   data/processed/graph_summary.json    — node/edge/candidate/entity counts
#
# Uses ONLY networkx — no graph databases.
# Graph builder pattern follows SKILLS.md exactly.
#
# Usage:
#   python src/stage4_graph_builder.py
#   python src/stage4_graph_builder.py \
#       --evidence data/processed/evidence \
#       --jd data/processed/jd_features.json \
#       --out-graph data/processed/knowledge_graph.gexf \
#       --out-summary data/processed/graph_summary.json

import argparse
import json
import os
import sys

import networkx as nx

ROOT             = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_EV_DIR   = os.path.join(ROOT, "data", "processed", "evidence")
DEFAULT_JD       = os.path.join(ROOT, "data", "processed", "jd_features.json")
DEFAULT_GRAPH    = os.path.join(ROOT, "data", "processed", "knowledge_graph.gexf")
DEFAULT_SUMMARY  = os.path.join(ROOT, "data", "processed", "graph_summary.json")

# ---------------------------------------------------------------------------
# Node type constants — exactly as specified in the task
# ---------------------------------------------------------------------------
NT_CANDIDATE      = "CANDIDATE"
NT_SKILL          = "SKILL"
NT_TOOL           = "TOOL"
NT_DOMAIN         = "DOMAIN"
NT_IMPACT_KEYWORD = "IMPACT_KEYWORD"
NT_JD_REQUIREMENT = "JD_REQUIREMENT"

# ---------------------------------------------------------------------------
# Entity classification helpers
# ---------------------------------------------------------------------------

# Known tools (specific software, platforms, APIs)
KNOWN_TOOLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "mlflow", "weights & biases", "wandb", "bentoml",
    "triton", "onnx", "hugging face", "huggingface", "airflow", "kafka",
    "spark", "pyspark", "dbt", "snowflake", "databricks", "docker",
    "kubernetes", "k8s", "aws", "gcp", "azure", "fastapi", "flask",
    "postgresql", "redis", "cassandra", "pytorch", "tensorflow",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "pgvector", "lancedb", "chroma", "bm25", "solr", "lucene",
    "feast", "tecton", "hopsworks", "ray", "dask",
    "dream11", "redrob", "swiggy", "zomato", "flipkart",  # company names as tools
}

# Domain / industry verticals
KNOWN_DOMAINS = {
    "healthcare", "fintech", "finance", "e-commerce", "ecommerce",
    "edtech", "education", "retail", "logistics", "manufacturing",
    "gaming", "media", "entertainment", "telecommunications", "telecom",
    "automotive", "real estate", "travel", "hospitality", "insurance",
    "hr tech", "hr-tech", "hrtech", "recruiting", "talent",
    "ad tech", "adtech", "marketing tech", "martech",
    "saas", "b2b", "b2c", "marketplace", "platform",
    "it services", "consulting", "paper products",
}

# Impact keywords — measurable outcome terms
IMPACT_KEYWORDS = {
    "reduced", "increased", "improved", "built", "led", "launched",
    "deployed", "shipped", "optimized", "accelerated", "scaled",
    "achieved", "drove", "delivered", "generated", "saved",
    "reduced latency", "increased throughput", "improved precision",
    "improved recall", "reduced cost", "increased revenue",
    "million", "billion", "users", "requests", "daily", "monthly",
    "%", "percent", "x improvement", "x faster",
}


def classify_entity(entity: str) -> str:
    """
    Classify an entity string into one of the 4 non-candidate node types.
    Priority: TOOL > DOMAIN > IMPACT_KEYWORD > SKILL (default)
    """
    el = entity.lower().strip()

    # Check TOOL first — specific software/platform names
    if any(tool in el for tool in KNOWN_TOOLS):
        return NT_TOOL

    # Check DOMAIN — industry verticals
    if any(dom in el for dom in KNOWN_DOMAINS):
        return NT_DOMAIN

    # Check IMPACT_KEYWORD — measurable outcome terms or numbers
    if any(kw in el for kw in IMPACT_KEYWORDS):
        return NT_IMPACT_KEYWORD
    # Pure numeric or percentage strings are impact keywords
    if any(c.isdigit() for c in el) and (
        "%" in el or "x" in el or any(
            word in el for word in ["million", "billion", "k users", "m users",
                                    "request", "latency", "throughput", "ms"]
        )
    ):
        return NT_IMPACT_KEYWORD

    # Default: SKILL — programming languages, frameworks, methodologies
    return NT_SKILL


def make_node_id(entity: str, node_type: str) -> str:
    """Create a deterministic node ID from entity name + type."""
    clean = entity.strip().lower().replace(" ", "_").replace("/", "_")
    return f"{node_type}::{clean}"


# ---------------------------------------------------------------------------
# Graph builder — from SKILLS.md pattern
# ---------------------------------------------------------------------------

def build_candidate_graph(candidate_id: str, evidence_items: list) -> nx.Graph:
    """
    Build a per-candidate subgraph.
    Matches the pattern defined in SKILLS.md exactly:
      G.add_node(candidate_id, node_type="CANDIDATE")
      for each evidence item → node + edge
    Returns nx.Graph.
    """
    G = nx.Graph()
    G.add_node(candidate_id, node_type=NT_CANDIDATE)

    for item in evidence_items:
        entity     = item.get("claim", "").strip()
        if not entity:
            continue
        node_type  = classify_entity(entity)
        node_id    = make_node_id(entity, node_type)

        G.add_node(node_id, node_type=node_type, label=entity)
        G.add_edge(
            candidate_id,
            node_id,
            confidence    = item.get("confidence", "medium"),
            weight        = float(item.get("weight", 1.0)),
            evidence_type = item.get("evidence_type", "technical"),
        )

    return G


def find_graph_matches(jd_entities: list, candidate_graphs: dict, top_k: int = 20) -> list:
    """
    Matches the find_graph_matches() signature from SKILLS.md exactly.

    Args:
        jd_entities:      list of entity strings from the JD
        candidate_graphs: {candidate_id: nx.Graph}
        top_k:            number of top results to return

    Returns:
        Sorted list of (candidate_id, graph_score) tuples, descending by score.

    Scoring formula from SKILLS.md:
        shared = count of jd_entities that exist as nodes in candidate graph
        neighbor_overlap = for each jd_entity, count neighbors of candidate node
                           whose label contains the jd_entity string
        graph_score = (shared * 2 + neighbor_overlap) / (len(jd_entities) + 1)
    """
    scores = []
    for cid, G in candidate_graphs.items():
        shared = sum(1 for e in jd_entities if any(
            e.lower() in G.nodes[n].get("label", "").lower()
            for n in G.nodes
        ))
        neighbor_overlap = sum(
            1 for e in jd_entities
            for n in G.neighbors(cid)
            if e.lower() in G.nodes[n].get("label", "").lower()
        )
        graph_score = (shared * 2 + neighbor_overlap) / (len(jd_entities) + 1)
        scores.append((cid, graph_score))

    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# JD entity extraction
# ---------------------------------------------------------------------------

def extract_jd_entities(jd_features: dict) -> list[str]:
    """
    Extract all skill/requirement strings from jd_features.json.
    Returns a flat list of entity strings.
    """
    entities = []

    for skill_entry in jd_features.get("must_have_skills", []) or []:
        if isinstance(skill_entry, dict):
            s = skill_entry.get("skill", "")
        else:
            s = str(skill_entry)
        if s:
            entities.append(s)

    for skill_entry in jd_features.get("nice_to_have_skills", []) or []:
        if isinstance(skill_entry, dict):
            s = skill_entry.get("skill", "")
        else:
            s = str(skill_entry)
        if s:
            entities.append(s)

    for req in jd_features.get("implicit_requirements", []) or []:
        if isinstance(req, dict):
            r = req.get("requirement", "")
        else:
            r = str(req)
        if r:
            entities.append(r)

    return entities


def add_jd_nodes(G: nx.Graph, jd_features: dict) -> list[str]:
    """
    Add JD_REQUIREMENT nodes to the graph for every skill/requirement in jd_features.
    Returns the list of JD entity strings (used by find_graph_matches).
    """
    jd_entities = extract_jd_entities(jd_features)

    for entity in jd_entities:
        node_id = make_node_id(entity, NT_JD_REQUIREMENT)
        G.add_node(node_id, node_type=NT_JD_REQUIREMENT, label=entity)

    return jd_entities


# ---------------------------------------------------------------------------
# Main graph assembly
# ---------------------------------------------------------------------------

def build_full_graph(evidence_dir: str, jd_features: dict) -> tuple[nx.Graph, dict, list]:
    """
    Build the complete knowledge graph from all candidate evidence files + JD.

    Returns:
        (G, candidate_graphs, jd_entities)
        G:                 the full merged networkx Graph
        candidate_graphs:  {candidate_id: per-candidate nx.Graph}
        jd_entities:       list of JD entity strings
    """
    G = nx.Graph()

    # Add JD requirement nodes
    jd_entities = add_jd_nodes(G, jd_features)

    candidate_graphs: dict[str, nx.Graph] = {}
    candidate_ids = []

    # Load all evidence files
    evidence_files = sorted(
        f for f in os.listdir(evidence_dir)
        if f.endswith("_evidence.json")
    )

    for filename in evidence_files:
        filepath = os.path.join(evidence_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            ev_data = json.load(f)

        cid            = ev_data.get("candidate_id", "")
        evidence_items = ev_data.get("evidence", [])
        entity_list    = ev_data.get("entities", [])

        if not cid or ev_data.get("error"):
            continue   # skip error sentinel files

        candidate_ids.append(cid)

        # Build per-candidate subgraph (from SKILLS.md pattern)
        cg = build_candidate_graph(cid, evidence_items)
        candidate_graphs[cid] = cg

        # Merge into the full graph
        G.add_nodes_from(cg.nodes(data=True))
        G.add_edges_from(cg.edges(data=True))

        # Also add entity strings as additional SKILL/TOOL/DOMAIN nodes
        # (entities list from Stage 3 — richer than just claims)
        for entity in entity_list:
            if not entity.strip():
                continue
            node_type = classify_entity(entity)
            node_id   = make_node_id(entity, node_type)
            if not G.has_node(node_id):
                G.add_node(node_id, node_type=node_type, label=entity)
            # Link entity node to candidate if not already connected
            if not G.has_edge(cid, node_id):
                G.add_edge(
                    cid,
                    node_id,
                    confidence    = "medium",
                    weight        = 1.0,
                    evidence_type = "technical",
                )

    return G, candidate_graphs, jd_entities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Build GraphRAG knowledge graph from candidate evidence."
    )
    parser.add_argument("--evidence",    default=DEFAULT_EV_DIR,
                        help="Evidence directory (default: data/processed/evidence/)")
    parser.add_argument("--jd",          default=DEFAULT_JD,
                        help="JD features JSON (default: data/processed/jd_features.json)")
    parser.add_argument("--out-graph",   default=DEFAULT_GRAPH,
                        help="Output GEXF path (default: data/processed/knowledge_graph.gexf)")
    parser.add_argument("--out-summary", default=DEFAULT_SUMMARY,
                        help="Output summary JSON (default: data/processed/graph_summary.json)")
    args = parser.parse_args()

    # Validate inputs
    for path, label in [(args.evidence, "evidence dir"), (args.jd, "jd_features.json")]:
        if not os.path.exists(path):
            print(f"[Stage 4 ERROR] {label} not found: {path}")
            sys.exit(1)

    # Load JD features
    with open(args.jd, "r", encoding="utf-8") as f:
        jd_features = json.load(f)

    print(f"[Stage 4] Building knowledge graph ...")
    print(f"[Stage 4] Evidence dir:  {args.evidence}")
    print(f"[Stage 4] JD features:   {jd_features.get('role_title', 'unknown')}")

    # Build graph
    G, candidate_graphs, jd_entities = build_full_graph(args.evidence, jd_features)

    n_candidates = len(candidate_graphs)
    nodes        = G.number_of_nodes()
    edges        = G.number_of_edges()

    # Count unique non-candidate, non-JD entities
    unique_entities = sum(
        1 for _, d in G.nodes(data=True)
        if d.get("node_type") not in (NT_CANDIDATE, NT_JD_REQUIREMENT)
    )

    # Node type breakdown
    type_counts: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        nt = d.get("node_type", "UNKNOWN")
        type_counts[nt] = type_counts.get(nt, 0) + 1

    # Run find_graph_matches as a smoke test
    top_matches = find_graph_matches(jd_entities, candidate_graphs, top_k=10)

    # Save GEXF
    os.makedirs(os.path.dirname(args.out_graph), exist_ok=True)
    nx.write_gexf(G, args.out_graph)

    # Save JSON summary
    summary = {
        "nodes":             nodes,
        "edges":             edges,
        "candidates":        n_candidates,
        "unique_entities":   unique_entities,
        "jd_requirements":   type_counts.get(NT_JD_REQUIREMENT, 0),
        "node_type_counts":  type_counts,
        "top10_graph_matches": [
            {"candidate_id": cid, "graph_score": round(score, 4)}
            for cid, score in top_matches
        ],
    }
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[Stage 4 Complete] Graph built: {nodes} nodes, {edges} edges "
        f"across {n_candidates} candidates."
    )
    print(f"[Stage 4] Node type breakdown:")
    for nt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {nt:<20}: {count:>5}")
    print(f"\n[Stage 4] Top 5 graph matches (JD entity overlap):")
    for cid, score in top_matches[:5]:
        print(f"  {cid}  graph_score={score:.4f}")
    print(f"\n[Stage 4] Graph saved  → {args.out_graph}")
    print(f"[Stage 4] Summary saved → {args.out_summary}")


if __name__ == "__main__":
    main()
