# run_pipeline.py
# Full pipeline runner — executes all 8 stages in sequence with timing.
#
# Usage:
#   python run_pipeline.py
#
# Stages:
#   1. JD Analysis           (src/stage1_jd_analysis.py)
#   2. Precompute features   (src/precompute.py)
#   3. Evidence extraction   (src/stage3_evidence_extraction.py)
#   4. Graph builder         (src/stage4_graph_builder.py)
#   5. Hybrid retrieval      (src/stage5_hybrid_retrieval.py)
#   6. Scoring engine        (src/stage6_scoring_engine.py)
#   7. Explainable ranking   (src/stage7_ranking.py)
#   8. PDF report            (utils/report_generator.py)

import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Required files guard — checked BEFORE any stage runs
# ---------------------------------------------------------------------------
REQUIRED_FILES = {
    "AGENT.md":   "AGENT.md",
    "CLAUDE.md":  "CLAUDE.md",
    "SKILLS.md":  "SKILLS.md",
    "JD docx":    os.path.join("data", "raw", "job_description.docx"),
    "JD txt":     os.path.join("data", "raw", "job_description.txt"),
    "Candidates JSONL": os.path.join("data", "raw", "candidates.jsonl"),
}

# At least one of JD docx/txt and one candidates file must exist
JD_FILES  = [
    os.path.join(ROOT, "data", "raw", "job_description.docx"),
    os.path.join(ROOT, "data", "raw", "job_description.txt"),
]
CAND_FILES = [
    os.path.join(ROOT, "data", "raw", "candidates.jsonl"),
    os.path.join(ROOT, "data", "raw", "candidates.csv"),
    os.path.join(ROOT, "data", "raw", "candidates.json"),
]
ALWAYS_REQUIRED = [
    os.path.join(ROOT, "AGENT.md"),
    os.path.join(ROOT, "CLAUDE.md"),
    os.path.join(ROOT, "SKILLS.md"),
]


def check_prerequisites() -> None:
    """Verify all required input files exist before running any stage."""
    errors = []

    for path in ALWAYS_REQUIRED:
        if not os.path.exists(path):
            errors.append(f"  MISSING: {os.path.relpath(path)}")

    if not any(os.path.exists(p) for p in JD_FILES):
        errors.append(
            "  MISSING: job description file. Expected one of:\n"
            + "\n".join(f"    {os.path.relpath(p)}" for p in JD_FILES)
        )

    if not any(os.path.exists(p) for p in CAND_FILES):
        errors.append(
            "  MISSING: candidates file. Expected one of:\n"
            + "\n".join(f"    {os.path.relpath(p)}" for p in CAND_FILES)
        )

    if errors:
        print("\n[Pipeline ERROR] Required files are missing. Cannot start pipeline.\n")
        for e in errors:
            print(e)
        print(
            "\nEnsure the challenge bundle is in India_runs_data_and_ai_challenge/ "
            "and data/raw/ symlinks are in place."
        )
        sys.exit(1)

    print("[Pipeline] All prerequisite files found.")


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------
def run_stage(stage_num: int, stage_name: str, fn, *args, **kwargs):
    """
    Run one stage function, print timing, handle exceptions.
    Exits with code 1 if the stage raises.
    """
    print()
    print("=" * 62)
    print(f"Running Stage {stage_num}: {stage_name}")
    print("=" * 62)

    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"[Stage {stage_num}] Completed in {elapsed:.1f}s")
        return result, elapsed
    except SystemExit as e:
        # Stage called sys.exit() — treat non-zero as failure
        if e.code and e.code != 0:
            elapsed = time.time() - t0
            print(f"\n[Pipeline FAILED] Stage {stage_num} exited with code {e.code}.")
            print(
                f"Pipeline failed at Stage {stage_num}. "
                f"Check error above. "
                f"Stages 1-{stage_num - 1} outputs are saved and valid."
            )
            sys.exit(1)
        elapsed = time.time() - t0
        print(f"[Stage {stage_num}] Completed in {elapsed:.1f}s")
        return None, elapsed
    except Exception:
        elapsed = time.time() - t0
        print(f"\n[Pipeline ERROR] Stage {stage_num} raised an exception after {elapsed:.1f}s:\n")
        traceback.print_exc()
        print(
            f"\nPipeline failed at Stage {stage_num}. "
            f"Check error above. "
            f"Stages 1-{stage_num - 1} outputs are saved and valid."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Stage wrappers — call each stage's main() function
# ---------------------------------------------------------------------------
def stage1_jd_analysis():
    from src.stage1_jd_analysis import main
    main()

def stage2_precompute():
    from src.precompute import main
    main()

def stage3_evidence():
    from src.stage3_evidence_extraction import main
    main()

def stage4_graph():
    from src.stage4_graph_builder import main
    main()

def stage5_retrieval():
    from src.stage5_hybrid_retrieval import main
    main()

def stage6_scoring():
    from src.stage6_scoring_engine import main
    main()

def stage7_ranking():
    from src.stage7_ranking import main
    main()

def stage8_report():
    from utils.report_generator import generate_report
    page_count = generate_report()
    print(f"[Report] PDF report saved: outputs/shortlist_report.pdf ({page_count} pages)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    print("=" * 62)
    print("  AI Recruiter Pipeline — Redrob Hackathon")
    print("  Intelligent Candidate Discovery & Ranking")
    print("=" * 62)

    # Prerequisites check
    check_prerequisites()

    pipeline_start = time.time()
    stage_times = {}

    # Stage 1 — JD Analysis
    _, t = run_stage(1, "JD Analysis", stage1_jd_analysis)
    stage_times[1] = t

    # Stage 2 — Precompute features for all 100K candidates
    _, t = run_stage(2, "Precompute Features (100K candidates)", stage2_precompute)
    stage_times[2] = t

    # Stage 3 — Evidence extraction for top-30 candidates
    _, t = run_stage(3, "Evidence Extraction (top-30)", stage3_evidence)
    stage_times[3] = t

    # Stage 4 — GraphRAG knowledge graph
    _, t = run_stage(4, "GraphRAG Knowledge Graph Builder", stage4_graph)
    stage_times[4] = t

    # Stage 5 — Hybrid retrieval (FAISS + Graph)
    _, t = run_stage(5, "Hybrid Retrieval (FAISS + Graph)", stage5_retrieval)
    stage_times[5] = t

    # Stage 6 — LLM Scoring Engine
    _, t = run_stage(6, "LLM Hiring Intelligence Engine (Scoring)", stage6_scoring)
    stage_times[6] = t

    # Stage 7 — Explainable Ranking + Dark Horse Discovery
    _, t = run_stage(7, "Explainable Ranking + Dark Horse Discovery", stage7_ranking)
    stage_times[7] = t

    # Stage 8 — PDF Report
    _, t = run_stage(8, "PDF Report Generation", stage8_report)
    stage_times[8] = t

    # Also run rank.py + reason.py to produce the final submission CSV
    print()
    print("=" * 62)
    print("Generating final submission CSV (rank + reason)")
    print("=" * 62)
    from src.rank import main as rank_main
    from src.reason import main as reason_main
    rank_main()
    reason_main()

    total_elapsed = time.time() - pipeline_start

    # Summary
    print()
    print("=" * 62)
    print("Pipeline Complete")
    print(f"Total time: {total_elapsed:.1f}s")
    print()
    print("Stage timings:")
    stage_names = {
        1: "JD Analysis",
        2: "Precompute (100K)",
        3: "Evidence Extraction",
        4: "Graph Builder",
        5: "Hybrid Retrieval",
        6: "Scoring Engine",
        7: "Ranking + Dark Horse",
        8: "PDF Report",
    }
    for n, t in stage_times.items():
        print(f"  Stage {n} ({stage_names[n]:<22}): {t:>6.1f}s")
    print()
    print("Outputs:")
    print("  Ranked output:   outputs/ranked_candidates.csv")
    print("  Submission CSV:  outputs/submission.csv")
    print("  PDF report:      outputs/shortlist_report.pdf")
    print()
    print("Launch dashboard: streamlit run app.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
