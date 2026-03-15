"""
CV Matcher — AI-powered resume matching engine.
Run: streamlit run streamlit_app.py
"""
import sys
from html import escape
from pathlib import Path
from statistics import mean

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Matcher",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --accent:     #818CF8;
  --accent-10:  rgba(129,140,248,0.12);
  --accent-20:  rgba(129,140,248,0.22);
  --text:       #F1F0FF;
  --sub:        #C4C2E8;
  --muted:      #8B89B8;
  --border:     rgba(255,255,255,0.10);
  --bg:         rgba(255,255,255,0.05);
  --surface:    rgba(255,255,255,0.07);
  --surface-2:  rgba(255,255,255,0.11);
  --green:      #34D399;
  --green-bg:   rgba(52,211,153,0.12);
  --green-br:   rgba(52,211,153,0.28);
  --teal:       #2DD4BF;
  --teal-bg:    rgba(45,212,191,0.12);
  --teal-br:    rgba(45,212,191,0.28);
  --amber:      #FCD34D;
  --amber-bg:   rgba(252,211,77,0.12);
  --amber-br:   rgba(252,211,77,0.28);
  --red:        #F87171;
  --red-bg:     rgba(248,113,113,0.12);
  --red-br:     rgba(248,113,113,0.28);
  --r:          12px;
  --r-lg:       18px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--text) !important;
}

/* Global text override for dark bg */
p, span, label, li, td, th { color: var(--text) !important; }

#MainMenu, footer, header { visibility: hidden; }

.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 10% 0%,   rgba(124,58,237,0.18) 0%, transparent 55%),
    radial-gradient(ellipse 60% 40% at 90% 10%,  rgba(236,72,153,0.13) 0%, transparent 50%),
    radial-gradient(ellipse 50% 35% at 70% 90%,  rgba(79,70,229,0.12) 0%, transparent 55%),
    radial-gradient(ellipse 40% 30% at 5%  80%,  rgba(16,185,129,0.09) 0%, transparent 50%),
    linear-gradient(150deg, #0F0C29 0%, #1a1040 35%, #24243e 65%, #0d0d1a 100%) !important;
}

.block-container {
  max-width: 1100px !important;
  padding: 2.5rem 2.5rem 5rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(15,12,41,0.72) !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
  backdrop-filter: blur(20px) !important;
}
section[data-testid="stSidebar"] > div { padding: 1.75rem 1.1rem !important; }

section[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  border-radius: 9px !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  background: rgba(255,255,255,0.05) !important;
  color: var(--sub) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  padding: 0.5rem 0.8rem !important;
  text-align: left !important;
  justify-content: flex-start !important;
  transition: all 0.15s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: rgba(129,140,248,0.1) !important;
}

/* Form container */
div[data-testid="stForm"] {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: var(--r-lg) !important;
  padding: 1.75rem !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1) !important;
  backdrop-filter: blur(16px) !important;
}

/* Submit button */
div[data-testid="stForm"] .stButton > button {
  width: 100% !important;
  height: 2.75rem !important;
  border-radius: var(--r) !important;
  border: 0 !important;
  background: var(--accent) !important;
  color: #fff !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
}
div[data-testid="stForm"] .stButton > button:hover { opacity: 0.88 !important; }

/* Textarea */
.stTextArea label {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
}
.stTextArea textarea {
  background: rgba(0,0,0,0.25) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: var(--r) !important;
  font-size: 0.9rem !important;
  line-height: 1.7 !important;
  color: var(--text) !important;
  resize: vertical !important;
}
.stTextArea textarea::placeholder { color: var(--muted) !important; }
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-10) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.25rem !important;
  border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px 8px 0 0 !important;
  padding: 0.55rem 1.1rem !important;
  font-size: 0.83rem !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  background: transparent !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  font-weight: 600 !important;
  border-bottom: 2px solid var(--accent) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }

/* Metrics */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.07) !important;
  border: 1px solid rgba(255,255,255,0.11) !important;
  border-radius: var(--r) !important;
  padding: 1rem 1.1rem !important;
  border-left: 3px solid var(--accent) !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
  backdrop-filter: blur(12px) !important;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.5rem !important;
  font-weight: 700 !important;
  color: var(--accent) !important;
}

/* Progress bar */
.stProgress > div > div {
  background: rgba(79,70,229,0.1) !important;
  border-radius: 999px !important;
  height: 6px !important;
}
.stProgress > div > div > div {
  border-radius: 999px !important;
  background: linear-gradient(90deg, #6366F1, #4F46E5) !important;
  height: 6px !important;
}

/* Bordered containers (candidate cards) */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-radius: var(--r-lg) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  background: rgba(255,255,255,0.06) !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.25) !important;
  backdrop-filter: blur(12px) !important;
}

/* Alerts */
div[data-testid="stAlert"] { border-radius: var(--r) !important; }

/* Reset button */
.stButton > button[kind="secondary"] {
  border-radius: var(--r) !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  color: var(--sub) !important;
  font-size: 0.83rem !important;
  font-weight: 500 !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] { padding: 0.25rem 0 !important; }
.stSlider [data-testid="stThumbValue"] {
  background: var(--accent) !important;
  color: #fff !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
}
.stSlider div[data-testid="stSlider"] > div > div > div {
  background: linear-gradient(90deg, #818CF8, #A78BFA) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 0.9rem 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 999px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def safe(v: object, fallback: str = "—") -> str:
    s = str(v).strip() if v is not None else ""
    return s or fallback


def score_palette(score: int) -> tuple[str, str, str, str]:
    """Return (color, bg, border, label) for a score."""
    if score >= 17:
        return "var(--green)", "var(--green-bg)", "var(--green-br)", "Excellent match"
    if score >= 13:
        return "var(--teal)", "var(--teal-bg)", "var(--teal-br)", "Good match"
    if score >= 8:
        return "var(--amber)", "var(--amber-bg)", "var(--amber-br)", "Partial match"
    return "var(--red)", "var(--red-bg)", "var(--red-br)", "Not suitable"


def badge_html(label: str, color: str, bg: str, border: str) -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;gap:0.35rem;'
        f'padding:0.25rem 0.6rem;border-radius:999px;font-size:0.72rem;font-weight:600;'
        f'color:{color};background:{bg};border:1px solid {border};">'
        f'<span style="width:0.4rem;height:0.4rem;border-radius:50%;background:{color};'
        f'display:inline-block;flex-shrink:0;"></span>'
        f'{escape(label)}</span>'
    )


# ── Pipeline ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_rank_fn():
    from rank_with_llm import rank_candidates_with_llm
    return rank_candidates_with_llm


def run_pipeline(job_description: str, top_k: int) -> list[dict]:
    return _load_rank_fn()(job_description, k=top_k)


# ── Session state ──────────────────────────────────────────────────────────────

for k, v in {"results": None, "last_query": "", "history": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Example searches ───────────────────────────────────────────────────────────

EXAMPLES: dict[str, str] = {
    "Data Scientist": (
        "We are looking for a Data Scientist with 3+ years of experience in Python, SQL, "
        "and machine learning. Strong knowledge of scikit-learn, pandas, and NLP required. "
        "Experience with LLMs and production ML pipelines is a major plus."
    ),
    "HR Recruiter": (
        "Seeking an HR Recruiter / Talent Acquisition Specialist with experience in full-cycle "
        "recruiting, ATS tools, LinkedIn sourcing, onboarding processes, and HR policies."
    ),
    "Senior Python Dev": (
        "Senior Python Developer with solid experience in Django or FastAPI, REST API design, "
        "PostgreSQL, AWS infrastructure, and CI/CD pipelines."
    ),
    "DevOps Engineer": (
        "DevOps Engineer with hands-on experience in Kubernetes, Terraform, CI/CD, AWS or GCP, "
        "and monitoring with Prometheus and Grafana."
    ),
    "Finance Analyst": (
        "Finance Analyst with experience in financial modeling, budgeting, Excel, reporting, "
        "and forecasting. CFA or MBA preferred."
    ),
    "Civil Engineer": (
        "Civil Engineer proficient in AutoCAD, structural design, project management, and site "
        "supervision. Knowledge of local building codes required."
    ),
    "Blockchain Developer": (
        "Blockchain Developer with expertise in Solidity, Web3.js, Ethereum, smart contract "
        "development, DeFi protocols, and security audits."
    ),
    "React / Frontend": (
        "React Developer with TypeScript, Redux, Next.js, REST API integration, Jest testing, "
        "and a strong eye for responsive UI design."
    ),
    "Aviation Operations": (
        "Aviation Operations Specialist with background in ICAO regulations, flight operations, "
        "ground handling, and safety management systems."
    ),
    "Data Analyst": (
        "Data Analyst with strong SQL, Power BI, Excel, and experience building dashboards and "
        "business intelligence reports for executive stakeholders."
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Brand ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;padding:1.25rem;border-radius:16px;
                    background:linear-gradient(135deg,#4F46E5 0%,#7C3AED 60%,#A855F7 100%);
                    position:relative;overflow:hidden;">
          <div style="position:absolute;top:-30px;right:-30px;width:110px;height:110px;
                      border-radius:50%;background:rgba(255,255,255,0.08);pointer-events:none;"></div>
          <div style="position:absolute;bottom:-40px;left:-10px;width:90px;height:90px;
                      border-radius:50%;background:rgba(255,255,255,0.05);pointer-events:none;"></div>
          <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
            <div style="width:28px;height:28px;border-radius:8px;
                        background:rgba(255,255,255,0.2);display:flex;align-items:center;
                        justify-content:center;font-size:0.85rem;">✦</div>
            <span style="font-size:0.7rem;font-weight:800;letter-spacing:0.12em;
                         text-transform:uppercase;color:rgba(255,255,255,0.65);">CV Matcher</span>
          </div>
          <h3 style="margin:0 0 0.3rem;font-size:1.05rem;font-weight:700;color:#fff;line-height:1.2;">
            Candidate search
          </h3>
          <p style="margin:0;font-size:0.78rem;color:rgba(255,255,255,0.65);line-height:1.5;">
            Ranked shortlists from your candidate database.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Shortlist size ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <p style="margin:0 0 0.6rem;font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                  text-transform:uppercase;color:var(--muted);">Shortlist size</p>
        """,
        unsafe_allow_html=True,
    )
    top_k = st.slider(
        "Shortlist size",
        min_value=1,
        max_value=10,
        value=5,
        label_visibility="collapsed",
        help="Number of candidate profiles to retrieve and score.",
    )
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:0.65rem 0.9rem;border-radius:10px;margin-top:0.35rem;
                    background:rgba(129,140,248,0.12);border:1px solid rgba(129,140,248,0.2);">
          <span style="font-size:0.8rem;color:var(--muted);">Top candidates</span>
          <span style="font-size:1.4rem;font-weight:800;color:var(--accent);line-height:1;">
            {top_k}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Quick searches ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <p style="margin:0.2rem 0 0.65rem;font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                  text-transform:uppercase;color:var(--muted);">Quick searches</p>
        """,
        unsafe_allow_html=True,
    )

    _dot_colors = ["#818CF8","#A78BFA","#F472B6","#2DD4BF","#FCD34D"]
    for i, (label, query) in enumerate(EXAMPLES.items()):
        dot = _dot_colors[i % len(_dot_colors)]
        # Dot indicator before the label
        col_dot, col_btn = st.columns([0.08, 1], gap="small")
        with col_dot:
            st.markdown(
                f'<div style="width:7px;height:7px;border-radius:50%;background:{dot};'
                f'margin-top:0.65rem;"></div>',
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                st.session_state.last_query = query
                st.rerun()

    st.divider()

    # ── Recent searches ────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown(
            """
            <p style="margin:0.2rem 0 0.55rem;font-size:0.68rem;font-weight:700;
                      letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);">
              Recent searches
            </p>
            """,
            unsafe_allow_html=True,
        )
        for idx, recent in enumerate(reversed(st.session_state.history[-5:])):
            preview = recent[:38] + ("…" if len(recent) > 38 else "")
            col_n, col_btn = st.columns([0.12, 1], gap="small")
            with col_n:
                st.markdown(
                    f'<p style="margin-top:0.55rem;font-size:0.62rem;font-weight:700;'
                    f'color:var(--muted);text-align:center;">#{idx+1}</p>',
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button(preview, key=f"hist_{hash(recent)}", use_container_width=True):
                    st.session_state.last_query = recent
                    st.rerun()
    else:
        st.markdown(
            """
            <div style="padding:1rem;border-radius:10px;background:rgba(255,255,255,0.04);
                        border:1px dashed rgba(255,255,255,0.1);text-align:center;">
              <p style="margin:0;font-size:0.78rem;color:var(--muted);line-height:1.5;">
                Your recent searches<br>will appear here.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="margin-bottom:2rem;padding:2rem 2.25rem 2rem;border-radius:20px;
                background:linear-gradient(135deg,#4F46E5 0%,#7C3AED 55%,#EC4899 100%);
                position:relative;overflow:hidden;">
      <div style="position:absolute;top:-40px;right:-40px;width:220px;height:220px;
                  border-radius:50%;background:rgba(255,255,255,0.07);pointer-events:none;"></div>
      <div style="position:absolute;bottom:-60px;left:30%;width:160px;height:160px;
                  border-radius:50%;background:rgba(255,255,255,0.06);pointer-events:none;"></div>
      <p style="margin:0 0 0.65rem;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;color:rgba(255,255,255,0.65);">
        Recruiting intelligence
      </p>
      <h1 style="margin:0 0 0.6rem;font-size:2rem;font-weight:700;
                 color:#fff;letter-spacing:-0.025em;line-height:1.1;">
        Find the right candidate
      </h1>
      <p style="margin:0;font-size:0.95rem;color:rgba(255,255,255,0.78);
                line-height:1.65;max-width:520px;">
        Describe the role you're hiring for. We'll match it against our database and return
        a scored shortlist with a written assessment for each profile.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — SEARCH FORM
# ══════════════════════════════════════════════════════════════════════════════

with st.form("match_form"):
    job_description = st.text_area(
        "Job description",
        value=st.session_state.last_query,
        placeholder=(
            "Describe the role: required skills, experience level, domain, and any must-haves. "
            "The more specific, the better the match.\n\n"
            "Example: We are hiring a senior data scientist with strong Python, SQL, and NLP experience. "
            "Familiarity with LLM workflows and production ML systems is a major advantage."
        ),
        height=200,
        label_visibility="collapsed",
    )

    col_btn, col_note = st.columns([1.2, 3], gap="medium")
    with col_btn:
        submitted = st.form_submit_button("Search candidates", use_container_width=True)
    with col_note:
        st.markdown(
            f'<p style="margin:0;font-size:0.82rem;color:var(--muted);padding-top:0.55rem;">'
            f'Will return a shortlist of <b style="color:var(--text);">{top_k} candidate{"s" if top_k != 1 else ""}</b> '
            f'ranked by relevance to your brief.</p>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if submitted:
    if not job_description.strip():
        st.warning("Please enter a job description before searching.")
    else:
        with st.spinner("Searching candidates…"):
            try:
                results = run_pipeline(job_description.strip(), top_k)
                st.session_state.results = results
                st.session_state.last_query = job_description.strip()
                if job_description.strip() not in st.session_state.history:
                    st.session_state.history.append(job_description.strip())
                st.rerun()
            except Exception as exc:
                st.error(f"An error occurred: {exc}")
                st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.results:
    results = st.session_state.results
    scores = [int(r.get("score_sur_20", 0) or 0) for r in results]
    n = len(results)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Result header bar
    col_title, col_reset = st.columns([5, 1], gap="medium")
    with col_title:
        st.markdown(
            f'<p style="margin:0;font-size:0.9rem;color:var(--sub);">'
            f'<b style="color:var(--text);">{n} candidate{"s" if n != 1 else ""}</b> found for this role</p>',
            unsafe_allow_html=True,
        )
    with col_reset:
        if st.button("Clear", key="reset", use_container_width=True):
            st.session_state.results = None
            st.session_state.last_query = ""
            st.rerun()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Candidates", n)
    m2.metric("Best score", f"{max(scores)} / 20")
    m3.metric("Average score", f"{mean(scores):.1f} / 20")
    top_candidate = results[0]
    top_score, top_bg, top_br, top_label = score_palette(int(top_candidate.get("score_sur_20", 0) or 0))
    m4.metric("Top profile", safe(top_candidate.get("category"), "N/A"))

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Tabs
    tab_profiles, tab_table = st.tabs(["Profiles", "Table"])

    with tab_profiles:
        for rank, candidate in enumerate(results, start=1):
            score = int(candidate.get("score_sur_20", 0) or 0)
            color, bg, border, label = score_palette(score)
            cid = escape(safe(candidate.get("candidate_id")))
            cat = escape(safe(candidate.get("category"), "Uncategorized"))
            reason = escape(safe(candidate.get("justification"), "No assessment available."))

            with st.container(border=True):
                left_col, right_col = st.columns([4, 1], gap="medium")

                with left_col:
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
                          <span style="font-size:0.72rem;font-weight:700;color:var(--muted);">#{rank}</span>
                          {badge_html(label, color, bg, border)}
                        </div>
                        <p style="margin:0 0 0.15rem;font-size:1rem;font-weight:600;color:var(--text);">
                          Candidate {cid}
                        </p>
                        <p style="margin:0;font-size:0.83rem;color:var(--muted);">{cat}</p>
                        """,
                        unsafe_allow_html=True,
                    )

                with right_col:
                    st.markdown(
                        f"""
                        <div style="text-align:right;">
                          <p style="margin:0;font-size:2rem;font-weight:700;line-height:1;color:{color};">
                            {score}
                            <span style="font-size:0.85rem;color:var(--muted);font-weight:400;">/20</span>
                          </p>
                          <p style="margin:0.15rem 0 0;font-size:0.7rem;color:var(--muted);">Match score</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.progress(score / 20)

                st.markdown(
                    f"""
                    <div style="margin-top:0.65rem;padding:0.85rem 1rem;border-radius:10px;
                                background:{bg};border:1px solid {border};">
                      <p style="margin:0 0 0.25rem;font-size:0.7rem;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.07em;color:{color};">
                        Assessment
                      </p>
                      <p style="margin:0;font-size:0.875rem;color:var(--sub);line-height:1.7;">
                        {reason}
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_table:
        df = pd.DataFrame(
            [
                {
                    "Rank": i + 1,
                    "Candidate ID": safe(r.get("candidate_id")),
                    "Category": safe(r.get("category"), "Uncategorized"),
                    "Score /20": int(r.get("score_sur_20", 0) or 0),
                    "Decision": safe(r.get("decision")),
                    "Assessment": safe(r.get("justification"), "N/A"),
                }
                for i, r in enumerate(results)
            ]
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score /20": st.column_config.ProgressColumn(
                    "Score /20", min_value=0, max_value=20, format="%d"
                ),
                "Assessment": st.column_config.TextColumn("Assessment", width="large"),
            },
        )

# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    _card_colors = [
        ("linear-gradient(135deg,rgba(99,102,241,0.18),rgba(79,70,229,0.08))", "#818CF8"),
        ("linear-gradient(135deg,rgba(139,92,246,0.18),rgba(124,58,237,0.08))", "#A78BFA"),
        ("linear-gradient(135deg,rgba(236,72,153,0.16),rgba(219,39,119,0.07))", "#F472B6"),
    ]
    for col, step, title, body, (card_bg, step_color) in zip(
        (c1, c2, c3),
        ("1", "2", "3"),
        ("Describe the role", "Get a ranked shortlist", "Review with confidence"),
        (
            "Write the job brief in plain language — skills required, seniority level, and what good looks like for this position.",
            "The system scans the candidate database and returns the profiles most aligned with your requirements.",
            "Each profile comes with a match score out of 20 and a written assessment so you can prioritise quickly.",
        ),
        _card_colors,
    ):
        with col:
            st.markdown(
                f"""
                <div style="padding:1.3rem;border-radius:18px;background:{card_bg};
                            border:1px solid rgba(255,255,255,0.10);">
                  <span style="display:inline-flex;align-items:center;justify-content:center;
                               width:2rem;height:2rem;border-radius:50%;
                               background:{step_color};color:#fff;
                               font-size:0.78rem;font-weight:700;margin-bottom:0.85rem;">
                    {step}
                  </span>
                  <p style="margin:0 0 0.4rem;font-size:0.93rem;font-weight:600;color:var(--text);">{title}</p>
                  <p style="margin:0;font-size:0.84rem;color:var(--muted);line-height:1.65;">{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
