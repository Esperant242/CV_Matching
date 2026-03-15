"""
Generate a professional architecture diagram for the CV Matcher RAG pipeline.
Output: docs/architecture.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withStroke

os.makedirs("docs", exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#0F0C29"
PANEL       = "#1a1040"
ACCENT      = "#818CF8"
VIOLET      = "#7C3AED"
PINK        = "#EC4899"
GREEN       = "#34D399"
TEAL        = "#2DD4BF"
AMBER       = "#FCD34D"
WHITE       = "#F1F0FF"
MUTED       = "#8B89B8"
BORDER      = "#2E2B5A"

fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")

# ── Helper functions ───────────────────────────────────────────────────────────

def box(ax, x, y, w, h, color, alpha=0.18, border=None, radius=0.35):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color,
        alpha=alpha,
        edgecolor=border or color,
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(rect)

def label(ax, x, y, text, size=9, color=WHITE, weight="bold", ha="center", va="center", zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            path_effects=[withStroke(linewidth=2, foreground=BG)])

def sublabel(ax, x, y, text, size=7.5, color=MUTED, ha="center", va="center"):
    ax.text(x, y, text, fontsize=size, color=color, fontweight="normal",
            ha=ha, va=va, zorder=5,
            path_effects=[withStroke(linewidth=1.5, foreground=BG)])

def arrow(ax, x1, y1, x2, y2, color=ACCENT, lw=1.8, style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=4,
    )

def arrow_label(ax, x, y, text, color=ACCENT):
    ax.text(x, y, text, fontsize=6.5, color=color, ha="center", va="center",
            fontweight="bold", zorder=6,
            path_effects=[withStroke(linewidth=2, foreground=BG)])

def section_title(ax, x, y, text, color=MUTED):
    ax.text(x, y, text.upper(), fontsize=7, color=color, fontweight="bold",
            ha="left", va="center", letter_spacing=0.1, zorder=5,
            path_effects=[withStroke(linewidth=1.5, foreground=BG)])

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(9, 10.5, "CV Matcher — RAG Pipeline Architecture",
        fontsize=16, color=WHITE, fontweight="bold", ha="center", va="center",
        path_effects=[withStroke(linewidth=3, foreground=BG)])
ax.text(9, 10.1, "Retrieval-Augmented Generation for resume matching",
        fontsize=9, color=MUTED, ha="center", va="center")

# Gradient line under title
from matplotlib.lines import Line2D
ax.add_line(Line2D([2, 16], [9.82, 9.82], color=BORDER, linewidth=1, zorder=3))

# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE INDEXING PIPELINE (top row)
# ══════════════════════════════════════════════════════════════════════════════
# Background section
box(ax, 0.4, 7.2, 17.2, 2.3, VIOLET, alpha=0.07, border=BORDER, radius=0.5)
ax.text(0.85, 9.35, "OFFLINE  ·  Run once to build the vector index",
        fontsize=7.5, color=VIOLET, fontweight="bold", alpha=0.85)

# Step 1 — Dataset
box(ax, 0.7, 7.45, 2.6, 1.6, VIOLET, alpha=0.22, border=VIOLET)
label(ax, 2.0, 8.55, "CSV Dataset", size=9.5, color=WHITE)
sublabel(ax, 2.0, 8.25, "Preprocessed_Data.txt", size=7.5)
sublabel(ax, 2.0, 7.95, "13,389 CVs · 43 categories", size=7)
sublabel(ax, 2.0, 7.65, "2 columns: Category · Text", size=7)

# Step 2 — Document builder
box(ax, 4.1, 7.45, 2.6, 1.6, ACCENT, alpha=0.22, border=ACCENT)
label(ax, 5.4, 8.55, "Document Builder", size=9.5, color=WHITE)
sublabel(ax, 5.4, 8.25, "build_documents.py", size=7.5)
sublabel(ax, 5.4, 7.95, "LangChain Document objects", size=7)
sublabel(ax, 5.4, 7.65, "metadata: resume_id · category", size=7)

# Step 3 — Chunking
box(ax, 7.5, 7.45, 2.6, 1.6, TEAL, alpha=0.22, border=TEAL)
label(ax, 8.8, 8.55, "Chunker", size=9.5, color=WHITE)
sublabel(ax, 8.8, 8.25, "RecursiveCharacterTextSplitter", size=7.5)
sublabel(ax, 8.8, 7.95, "chunk_size=1000 · overlap=150", size=7)
sublabel(ax, 8.8, 7.65, "13,389 docs → 67,101 chunks", size=7)

# Step 4 — Embeddings
box(ax, 10.9, 7.45, 2.6, 1.6, GREEN, alpha=0.22, border=GREEN)
label(ax, 12.2, 8.55, "Embeddings", size=9.5, color=WHITE)
sublabel(ax, 12.2, 8.25, "OpenAI Embeddings", size=7.5)
sublabel(ax, 12.2, 7.95, "text-embedding-3-small", size=7)
sublabel(ax, 12.2, 7.65, "1,536-dim dense vectors", size=7)

# Step 5 — FAISS Index
box(ax, 14.3, 7.45, 2.9, 1.6, AMBER, alpha=0.22, border=AMBER)
label(ax, 15.75, 8.55, "FAISS Index", size=9.5, color=WHITE)
sublabel(ax, 15.75, 8.25, "faiss-cpu · IndexFlatL2", size=7.5)
sublabel(ax, 15.75, 7.95, "67,101 vectors saved to disk", size=7)
sublabel(ax, 15.75, 7.65, "D:/RAG_vectorstore/", size=7)

# Arrows — offline pipeline
for x1, x2, col, lbl in [
    (3.3, 4.1, VIOLET, "load"),
    (6.7, 7.5, ACCENT, "split"),
    (10.1, 10.9, TEAL, "embed"),
    (13.5, 14.3, GREEN, "index"),
]:
    arrow(ax, x1, 8.25, x2, 8.25, color=col)
    arrow_label(ax, (x1+x2)/2, 8.42, lbl, color=col)

# ══════════════════════════════════════════════════════════════════════════════
# ONLINE QUERY PIPELINE (bottom rows)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.4, 1.0, 17.2, 5.9, ACCENT, alpha=0.05, border=BORDER, radius=0.5)
ax.text(0.85, 6.75, "ONLINE  ·  Real-time query pipeline",
        fontsize=7.5, color=ACCENT, fontweight="bold", alpha=0.85)

# ── Row 1: Input → Embed → Retrieve ──────────────────────────────────────────

# Job Description input
box(ax, 0.7, 5.2, 2.6, 1.6, PINK, alpha=0.22, border=PINK)
label(ax, 2.0, 6.3, "Job Description", size=9.5, color=WHITE)
sublabel(ax, 2.0, 6.0, "Plain-language role brief", size=7.5)
sublabel(ax, 2.0, 5.7, "Skills · seniority · domain", size=7)
sublabel(ax, 2.0, 5.4, "Streamlit UI or CLI", size=7)

# Query embedding
box(ax, 4.1, 5.2, 2.6, 1.6, GREEN, alpha=0.22, border=GREEN)
label(ax, 5.4, 6.3, "Query Embedding", size=9.5, color=WHITE)
sublabel(ax, 5.4, 6.0, "text-embedding-3-small", size=7.5)
sublabel(ax, 5.4, 5.7, "Job description → vector", size=7)
sublabel(ax, 5.4, 5.4, "1,536 dimensions", size=7)

# FAISS Retrieval
box(ax, 7.5, 5.2, 2.6, 1.6, AMBER, alpha=0.22, border=AMBER)
label(ax, 8.8, 6.3, "FAISS Retrieval", size=9.5, color=WHITE)
sublabel(ax, 8.8, 6.0, "similarity_search(query, k)", size=7.5)
sublabel(ax, 8.8, 5.7, "Cosine similarity on 67k vecs", size=7)
sublabel(ax, 8.8, 5.4, "Returns top-K chunks", size=7)

# Candidate merger
box(ax, 10.9, 5.2, 2.6, 1.6, TEAL, alpha=0.22, border=TEAL)
label(ax, 12.2, 6.3, "Candidate Merger", size=9.5, color=WHITE)
sublabel(ax, 12.2, 6.0, "Group chunks by resume_id", size=7.5)
sublabel(ax, 12.2, 5.7, "Merge text per candidate", size=7)
sublabel(ax, 12.2, 5.4, "Sorted deterministically", size=7)

# LLM Scorer
box(ax, 14.3, 5.2, 2.9, 1.6, VIOLET, alpha=0.25, border=VIOLET)
label(ax, 15.75, 6.3, "LLM Scorer", size=9.5, color=WHITE)
sublabel(ax, 15.75, 6.0, "gpt-4.1-mini", size=7.5)
sublabel(ax, 15.75, 5.7, "temp=0 · seed=42", size=7)
sublabel(ax, 15.75, 5.4, "Structured JSON output", size=7)

# Arrows — online row 1
for x1, x2, col, lbl in [
    (3.3, 4.1, PINK,   "embed"),
    (6.7, 7.5, GREEN,  "search"),
    (10.1, 10.9, AMBER, "group"),
    (13.5, 14.3, TEAL, "score"),
]:
    arrow(ax, x1, 6.0, x2, 6.0, color=col)
    arrow_label(ax, (x1+x2)/2, 6.17, lbl, color=col)

# ── Row 2: Scoring rubric + Results ──────────────────────────────────────────

# Scoring rubric breakdown (left-center)
box(ax, 0.7, 1.2, 5.5, 3.6, VIOLET, alpha=0.15, border=BORDER)
label(ax, 3.45, 4.55, "Scoring Rubric", size=9.5, color=ACCENT)

rubric = [
    ("17–20", "Excellent match", GREEN,  "Meets almost all requirements"),
    ("13–16", "Good match",      TEAL,   "Meets most requirements"),
    ("8–12",  "Partial match",   AMBER,  "Meets some requirements"),
    ("0–7",   "Not suitable",    PINK,   "Does not meet core requirements"),
]
for i, (score, decision, col, desc) in enumerate(rubric):
    y = 4.1 - i * 0.68
    box(ax, 0.85, y - 0.28, 5.2, 0.55, col, alpha=0.12, border=col, radius=0.15)
    ax.text(1.15, y, score, fontsize=8.5, color=col, fontweight="bold",
            va="center", zorder=5)
    ax.text(2.3, y, decision, fontsize=8, color=WHITE, fontweight="bold",
            va="center", zorder=5)
    ax.text(2.3, y-0.2, desc, fontsize=6.8, color=MUTED,
            va="center", zorder=5)

# Results / output box
box(ax, 7.1, 1.2, 4.0, 3.6, GREEN, alpha=0.15, border=BORDER)
label(ax, 9.1, 4.55, "Ranked Output", size=9.5, color=GREEN)
output_items = [
    ("candidate_id", "Resume identifier", ACCENT),
    ("category",     "Domain / role type", TEAL),
    ("score_sur_20", "Integer score 0–20", AMBER),
    ("decision",     "Verdict label",      GREEN),
    ("justification","Written assessment", PINK),
]
for i, (field, desc, col) in enumerate(output_items):
    y = 4.1 - i * 0.62
    ax.text(7.4, y, f"•", fontsize=10, color=col, va="center", zorder=5)
    ax.text(7.7, y, field, fontsize=8, color=col, fontweight="bold",
            va="center", zorder=5)
    ax.text(7.7, y-0.2, desc, fontsize=6.8, color=MUTED,
            va="center", zorder=5)

# Streamlit UI box
box(ax, 12.0, 1.2, 5.2, 3.6, PINK, alpha=0.15, border=BORDER)
label(ax, 14.6, 4.55, "Streamlit UI", size=9.5, color=PINK)
ui_items = [
    ("Sidebar",          "Shortlist size slider · Role examples · History"),
    ("Hero header",      "Gradient banner · Product description"),
    ("Search form",      "Job description textarea · Submit"),
    ("Metrics row",      "Count · Best score · Avg · Domain"),
    ("Candidate cards",  "Rank · Badge · Score · Progress · Assessment"),
    ("Table view",       "Sortable dataframe with progress column"),
]
for i, (comp, desc) in enumerate(ui_items):
    y = 4.1 - i * 0.55
    ax.text(12.2, y, comp, fontsize=7.5, color=WHITE, fontweight="bold",
            va="center", zorder=5)
    ax.text(12.2, y-0.2, desc, fontsize=6.5, color=MUTED,
            va="center", zorder=5)

# Arrows — LLM to outputs
arrow(ax, 15.75, 5.2, 15.75, 4.8, color=VIOLET)   # down to UI
arrow(ax, 14.3, 5.6, 11.1, 4.8, color=GREEN)       # to ranked output
arrow(ax, 11.1, 4.8, 6.2, 4.8, color=VIOLET)       # to rubric (context)

# FAISS index reuse arrow (from offline to online)
ax.annotate(
    "", xy=(15.75, 7.45), xytext=(15.75, 6.8),
    arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.8,
                    connectionstyle="arc3,rad=0.0"),
    zorder=4,
)
arrow_label(ax, 15.75, 7.12, "load\nindex", color=AMBER)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    ("Dataset / Storage", AMBER),
    ("Embedding layer",   GREEN),
    ("Vector retrieval",  TEAL),
    ("LLM scoring",       VIOLET),
    ("UI / Output",       PINK),
]
lx = 0.75
for i, (lbl, col) in enumerate(legend_items):
    bx = lx + i * 3.35
    box(ax, bx, 0.25, 3.1, 0.55, col, alpha=0.2, border=col, radius=0.18)
    ax.text(bx + 1.55, 0.53, lbl, fontsize=7, color=col,
            fontweight="bold", ha="center", va="center", zorder=5)

plt.tight_layout(pad=0)
plt.savefig("docs/architecture.png", dpi=180, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print("Saved -> docs/architecture.png")
