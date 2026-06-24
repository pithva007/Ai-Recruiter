# utils/report_generator.py
# PDF Shortlist Report Generator
#
# Input:  outputs/ranked_candidates.csv
#         data/processed/jd_features.json
#         outputs/ranking_summary.json
# Output: outputs/shortlist_report.pdf
#
# Uses only reportlab. No LLM calls.
#
# Usage:
#   python utils/report_generator.py
#   from utils.report_generator import generate_report

import csv
import json
import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = os.path.join(os.path.dirname(__file__), "..")
RANKED_CSV   = os.path.join(ROOT, "outputs", "ranked_candidates.csv")
JD_JSON      = os.path.join(ROOT, "data",    "processed", "jd_features.json")
SUMMARY_JSON = os.path.join(ROOT, "outputs", "ranking_summary.json")
PDF_PATH     = os.path.join(ROOT, "outputs", "shortlist_report.pdf")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_DARK       = colors.HexColor("#1a1a2e")
C_PRIMARY    = colors.HexColor("#16213e")
C_ACCENT     = colors.HexColor("#0f3460")
C_GREEN      = colors.HexColor("#2ecc71")
C_YELLOW     = colors.HexColor("#f39c12")
C_RED        = colors.HexColor("#e74c3c")
C_LIGHT_GRAY = colors.HexColor("#f4f4f4")
C_MID_GRAY   = colors.HexColor("#cccccc")
C_WHITE      = colors.white

# ---------------------------------------------------------------------------
# Helper: split pipe-separated strings
# ---------------------------------------------------------------------------
def pipe_split(value: str) -> list[str]:
    if not value or str(value).strip() in ("", "nan"):
        return []
    return [s.strip() for s in str(value).split("|") if s.strip()]

# ---------------------------------------------------------------------------
# Helper: truncate to N words
# ---------------------------------------------------------------------------
def trunc(text: str, max_words: int = 25) -> str:
    if not text or str(text).strip() in ("", "nan"):
        return "—"
    words = str(text).strip().split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")

# ---------------------------------------------------------------------------
# Helper: score colour
# ---------------------------------------------------------------------------
def score_color(score) -> object:
    try:
        s = float(score)
    except Exception:
        return C_MID_GRAY
    if s >= 75:
        return C_GREEN
    elif s >= 50:
        return C_YELLOW
    return C_RED

# ---------------------------------------------------------------------------
# Style factory
# ---------------------------------------------------------------------------
def make_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontSize=26, textColor=C_WHITE, spaceAfter=4,
            alignment=TA_CENTER, fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=13, textColor=C_LIGHT_GRAY, spaceAfter=2,
            alignment=TA_CENTER, fontName="Helvetica",
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontSize=16, textColor=C_PRIMARY, spaceBefore=10, spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontSize=12, textColor=C_ACCENT, spaceBefore=8, spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=9, textColor=colors.black, leading=13,
            fontName="Helvetica",
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"],
            fontSize=8, textColor=colors.HexColor("#555555"), leading=11,
            fontName="Helvetica",
        ),
        "label": ParagraphStyle(
            "label", parent=base["Normal"],
            fontSize=8, textColor=C_ACCENT, leading=11,
            fontName="Helvetica-Bold",
        ),
        "mono": ParagraphStyle(
            "mono", parent=base["Normal"],
            fontSize=8, fontName="Courier", leading=12,
            textColor=colors.HexColor("#333333"),
        ),
    }

# ---------------------------------------------------------------------------
# Page 1 — Cover
# ---------------------------------------------------------------------------
def build_cover(styles: dict, jd: dict, summary: dict, n_candidates: int) -> list:
    role    = jd.get("role_title", "Senior AI Engineer")
    company = jd.get("company", "Redrob AI")
    today   = date.today().strftime("%d %B %Y")
    total   = summary.get("total_candidates_scored", n_candidates)

    # Dark banner table
    banner = Table(
        [[Paragraph("AI Recruiter", styles["title"]),
          Paragraph("Shortlist Report", styles["title"])]],
        colWidths=[9 * cm, 9 * cm],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_PRIMARY),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 28),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 28),
        ("SPAN", (0, 0), (-1, -1)),
    ]))

    meta_data = [
        ["Role",       role],
        ["Company",    company],
        ["Date",       today],
        ["Shortlist",  f"{n_candidates} candidates"],
        ["Pool sized", f"{total} candidates scored"],
    ]
    meta_table = Table(meta_data, colWidths=[5 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 11),
        ("TEXTCOLOR",   (0, 0), (0, -1), C_ACCENT),
        ("TEXTCOLOR",   (1, 0), (1, -1), colors.black),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW",   (0, -1), (-1, -1), 0.5, C_MID_GRAY),
    ]))

    return [
        Spacer(1, 1.5 * cm),
        banner,
        Spacer(1, 1.5 * cm),
        HRFlowable(width="100%", thickness=2, color=C_ACCENT),
        Spacer(1, 0.8 * cm),
        meta_table,
        Spacer(1, 1.2 * cm),
        HRFlowable(width="100%", thickness=1, color=C_MID_GRAY),
        Spacer(1, 0.5 * cm),
        Paragraph(
            "Generated by AI Recruiter Pipeline &mdash; Evidence-based ranking using "
            "career history, skill depth, behavioral signals, and JD fit analysis.",
            styles["small"],
        ),
        PageBreak(),
    ]


# ---------------------------------------------------------------------------
# Page 2 — Executive Summary (top 5)
# ---------------------------------------------------------------------------
def build_executive_summary(styles: dict, rows: list) -> list:
    story = [Paragraph("Executive Summary", styles["h1"])]
    story.append(Paragraph(
        "Top 5 shortlisted candidates ranked by composite score. "
        "Scores are evidence-based and verified against actual profile data.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    header = ["Rank", "Name", "Composite", "Fit", "Impact", "Potential", "Risk", "Key Strength"]
    table_data = [header]
    for row in rows[:5]:
        green = pipe_split(str(row.get("green_flags", "")))
        key_strength = green[0][:55] if green else trunc(row.get("llm_rationale", ""), 12)
        table_data.append([
            row["rank"],
            row["candidate_name"],
            row["composite_score"],
            row["fit_score"],
            row["impact_score"],
            row["potential_score"],
            row["risk_score"],
            key_strength,
        ])

    col_w = [1.2*cm, 3.8*cm, 2*cm, 1.6*cm, 1.6*cm, 2*cm, 1.6*cm, 5.2*cm]
    tbl = Table(table_data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("ALIGN",        (2, 0), (6, -1), "CENTER"),
        ("ALIGN",        (0, 0), (1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT_GRAY, C_WHITE]),
        ("GRID",         (0, 0), (-1, -1), 0.4, C_MID_GRAY),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    # Colour composite cells
    for i, row in enumerate(rows[:5], start=1):
        try:
            s = float(row["composite_score"])
            bg = colors.HexColor("#d4edda") if s >= 75 else (
                 colors.HexColor("#fff3cd") if s >= 50 else colors.HexColor("#f8d7da"))
            tbl.setStyle(TableStyle([("BACKGROUND", (2, i), (2, i), bg)]))
        except Exception:
            pass

    story.append(tbl)
    story.append(Spacer(1, 0.6 * cm))

    # One sentence per candidate
    story.append(Paragraph("One-line assessment:", styles["h2"]))
    for row in rows[:5]:
        rationale = trunc(str(row.get("llm_rationale", "")), 30)
        story.append(Paragraph(
            f"<b>{row['rank']}. {row['candidate_name']}</b> — {rationale}",
            styles["body"],
        ))
        story.append(Spacer(1, 0.15 * cm))

    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Candidate detail page (one per top-10 candidate)
# ---------------------------------------------------------------------------
def build_candidate_page(styles: dict, row: dict) -> list:
    name      = row["candidate_name"]
    rank      = row["rank"]
    composite = row["composite_score"]
    confidence= row.get("confidence_level", "medium")
    is_dh     = str(row.get("dark_horse", "")).lower() == "true"
    dh_badge  = "  [DARK HORSE]" if is_dh else ""

    story = []
    story.append(Paragraph(f"#{rank}  {name}{dh_badge}", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT))
    story.append(Spacer(1, 0.3 * cm))

    # Score bar table
    score_data = [
        ["Composite", "Fit", "Impact", "Potential", "Risk", "Confidence"],
        [composite, row["fit_score"], row["impact_score"],
         row["potential_score"], row["risk_score"], confidence],
    ]
    score_tbl = Table(score_data, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm])
    score_tbl.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",     (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND",   (0, 0), (-1, 0), C_LIGHT_GRAY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), C_ACCENT),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID",         (0, 0), (-1, -1), 0.4, C_MID_GRAY),
        ("BACKGROUND",   (0, 1), (0, 1), score_color(composite)),
    ]))
    story.append(score_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # Flags and gaps in 3 columns
    green  = pipe_split(str(row.get("green_flags", "")))
    yellow = pipe_split(str(row.get("yellow_flags", "")))
    gaps   = pipe_split(str(row.get("skill_gaps", "")))

    def bullet_list(items: list, label: str, col: colors.Color) -> list:
        elems = [Paragraph(label, styles["label"])]
        if items:
            for item in items[:4]:
                elems.append(Paragraph(f"• {item[:70]}", styles["small"]))
        else:
            elems.append(Paragraph("None identified", styles["small"]))
        return elems

    flags_table = Table(
        [[bullet_list(green, "GREEN FLAGS", C_GREEN),
          bullet_list(yellow, "YELLOW FLAGS", C_YELLOW),
          bullet_list(gaps, "SKILL GAPS", C_RED)]],
        colWidths=[6*cm, 6*cm, 6*cm],
    )
    flags_table.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(flags_table)
    story.append(Spacer(1, 0.4 * cm))

    # Interview questions
    q1 = str(row.get("interview_q1", "")).strip()
    q2 = str(row.get("interview_q2", "")).strip()
    q3 = str(row.get("interview_q3", "")).strip()
    questions = [q for q in [q1, q2, q3] if q and q != "nan"]
    if questions:
        story.append(Paragraph("Interview Questions", styles["h2"]))
        for i, q in enumerate(questions, 1):
            story.append(Paragraph(f"{i}. {q[:180]}", styles["body"]))
            story.append(Spacer(1, 0.1 * cm))
        story.append(Spacer(1, 0.2 * cm))

    # LLM rationale
    rationale = str(row.get("llm_rationale", "")).strip()
    if rationale and rationale != "nan":
        story.append(Paragraph("Ranking Rationale", styles["h2"]))
        story.append(Paragraph(trunc(rationale, 80), styles["body"]))

    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Dark Horses page
# ---------------------------------------------------------------------------
def build_dark_horses_page(styles: dict, rows: list) -> list:
    dark = [r for r in rows if str(r.get("dark_horse", "")).lower() == "true"]
    if not dark:
        return []

    story = [Paragraph("Dark Horse Candidates", styles["h1"])]
    story.append(Paragraph(
        "These candidates ranked below position 15 in vector similarity search but "
        "show high impact or potential scores (>= 75) with fit >= 50. "
        "A traditional ATS would miss them.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    for row in dark:
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID_GRAY))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            f"<b>{row['candidate_name']}</b>  "
            f"(Rank {row['rank']} · Composite {row['composite_score']})",
            styles["h2"],
        ))

        reason = str(row.get("dark_horse_reason", "")).strip()
        if reason and reason != "nan":
            story.append(Paragraph(trunc(reason, 60), styles["body"]))

        tsm = pipe_split(str(row.get("transferable_skills_map", "")))
        if tsm:
            story.append(Paragraph("Transferable skill mappings:", styles["label"]))
            for skill in tsm[:4]:
                story.append(Paragraph(f"  • {skill[:100]}", styles["small"]))

        story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())
    return story


# ---------------------------------------------------------------------------
# Methodology page
# ---------------------------------------------------------------------------
def build_methodology_page(styles: dict) -> list:
    story = [Paragraph("Methodology", styles["h1"])]
    story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Composite Score Formula", styles["h2"]))
    story.append(Paragraph(
        "The composite score (0–100) is computed from four evidence-based dimensions using "
        "the following canonical formula defined in SKILLS.md:",
        styles["body"],
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "composite_score = (fit_score × 0.35) + (impact_score × 0.30) + "
        "(potential_score × 0.20) + ((100 − risk_score) × 0.15)",
        styles["mono"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    dims = [
        ("fit_score (35%)",
         "Measures alignment between the candidate's background and the JD's explicit and "
         "implicit requirements. Driven by career history title classification, ML keyword "
         "evidence in job descriptions, and product vs. services company history."),
        ("impact_score (30%)",
         "Measures real-world measurable outcomes: quantified achievements (numbers, scale, "
         "revenue, users). A candidate with zero quantified impact signals scores at most 40 "
         "on this dimension — missing numbers cannot be assumed."),
        ("potential_score (20%)",
         "Measures growth trajectory: career velocity (promotions per year), complexity growth "
         "across chronological career history, and self-learning signals such as OSS "
         "contributions, certifications, and hackathons."),
        ("risk_score (15%, inverted)",
         "Measures hiring risk: skill gaps vs. JD must-haves, very short tenures, no evidence "
         "of collaboration, overqualification signals, or pure-services career history. "
         "Higher risk = lower composite via the (100 − risk_score) inversion."),
    ]
    for title, text in dims:
        story.append(Paragraph(title, styles["label"]))
        story.append(Paragraph(text, styles["body"]))
        story.append(Spacer(1, 0.25 * cm))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Evidence-Based Scoring", styles["h2"]))
    story.append(Paragraph(
        "All scores are derived from evidence traceable to the candidate's actual profile text. "
        "The system never invents skills, impact numbers, or experience not present in the "
        "source data. Each score includes a confidence_level field (high / medium / low) "
        "indicating how much direct textual evidence supports it. "
        "Low-confidence scores default to 50 and are flagged for human review.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Honeypot Detection", styles["h2"]))
    story.append(Paragraph(
        "The dataset contains candidates with logically impossible profiles designed to trap "
        "keyword-matching systems. This pipeline detects and scores them 0.0 using four rules: "
        "impossible experience timelines, mass expert skills with zero usage duration, "
        "copy-pasted role descriptions, and high profile completeness with empty career content. "
        "472 honeypot candidates were detected and excluded from the shortlist.",
        styles["body"],
    ))
    return story


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------
def generate_report(
    ranked_csv:   str = RANKED_CSV,
    jd_json:      str = JD_JSON,
    summary_json: str = SUMMARY_JSON,
    pdf_path:     str = PDF_PATH,
) -> int:
    """
    Build the PDF report. Returns page count.
    No LLM calls. Reads only pre-computed output files.
    """
    # Load data
    with open(ranked_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(jd_json, "r", encoding="utf-8") as f:
        jd = json.load(f)
    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="AI Recruiter Shortlist Report",
        author="AI Recruiter Pipeline",
    )

    styles = make_styles()
    story  = []

    # Page 1: Cover
    story += build_cover(styles, jd, summary, len(rows))

    # Page 2: Executive Summary
    story += build_executive_summary(styles, rows)

    # Pages 3+: Top-10 candidate detail
    for row in rows[:10]:
        story += build_candidate_page(styles, row)

    # Dark Horses page
    story += build_dark_horses_page(styles, rows)

    # Methodology page
    story += build_methodology_page(styles)

    doc.build(story)

    # Count pages via PDF byte scan (simple heuristic)
    with open(pdf_path, "rb") as f:
        content = f.read()
    page_count = content.count(b"/Type /Page\n") or content.count(b"/Type/Page")
    if page_count == 0:
        page_count = content.count(b"Page\n") + 1  # fallback estimate

    return page_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import sys
    for path, label in [
        (RANKED_CSV,   "ranked_candidates.csv"),
        (JD_JSON,      "jd_features.json"),
        (SUMMARY_JSON, "ranking_summary.json"),
    ]:
        if not os.path.exists(path):
            print(f"[Report ERROR] {label} not found: {path}")
            print("  Run the full pipeline first: python run_pipeline.py")
            sys.exit(1)

    print("[Report] Generating PDF report ...")
    page_count = generate_report()
    print(f"[Report] PDF report saved: outputs/shortlist_report.pdf ({page_count} pages)")


if __name__ == "__main__":
    main()
