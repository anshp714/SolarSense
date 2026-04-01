"""
Land Suitability Prediction — Streamlit Web App (v2.0)
=======================================================
Research-grade interactive ML application.
Fully decoupled frontend — all ML logic lives in the `backend/` package.
Robust: every data load, model call, and UI render is guarded.

Architecture:
  frontend/app.py  ← you are here (UI only)
  backend/config.py
  backend/data_loader.py
  backend/feature_engineering.py
  backend/model.py
  backend/predictor.py
"""

# ── Standard library ────────────────────────────────────────────────────────
import io
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Add project root to sys.path so `backend` is importable ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Backend imports (guarded so the app degrades gracefully on import errors)
try:
    from backend.config import (
        DATASET_PATH, DISTRICT_DATA_PATH,
        FEATURE_COLS, SOIL_MAPPING, CONDITION_THRESHOLDS,
        MODEL_PATH, confidence_tier,
    )
    from backend.data_loader import (
        load_training_data, load_district_data, data_quality_report
    )
    from backend.feature_engineering import derive_single
    from backend.model import load_or_train_model, load_model_metadata
    from backend.predictor import predict, predict_batch, find_similar_districts
    _BACKEND_OK = True
except Exception as _be:
    _BACKEND_OK = False
    _BACKEND_ERROR = str(_be)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Solar Sense",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ─────────────────────────────────── */
.stApp {
    background: #f8fafc;
    color: #0f172a;
}

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

/* ── Cards (generic) ───────────────────────────── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}
.metric-title {
    font-size: 0.85rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    font-weight: 600;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #0f172a; }

/* ── Result banners ─────────────────────────────── */
.result-suitable {
    background: #f0fdf4;
    border: 1px solid #22c55e;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    color: #166534;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 2px;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
}
.result-not {
    background: #fef2f2;
    border: 1px solid #ef4444;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    color: #991b1b;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 2px;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
}
.result-error {
    background: #fffbeb;
    border: 1px solid #f59e0b;
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    color: #b45309;
    font-size: 1rem;
    font-weight: 600;
}

/* ── Confidence tiers ───────────────────────────── */
.tier-high     { color: #16a34a; font-weight: 700; }
.tier-moderate { color: #d97706; font-weight: 700; }
.tier-low      { color: #dc2626; font-weight: 700; }

/* ── Condition cards ────────────────────────────── */
.pass-card {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    display:flex; align-items:center; gap:12px;
    color: #0f172a;
}
.fail-card {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    display:flex; align-items:center; gap:12px;
    color: #0f172a;
}

/* ── Section header ─────────────────────────────── */
.section-header {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0f172a;
    margin: 32px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid #e2e8f0;
}

/* ── Tabs ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 14px;
    padding: 6px;
    gap: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #64748b;
    font-weight: 600;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    background: #f59e0b !important;
    color: #ffffff !important;
    font-weight: 700;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    background: #f59e0b;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    font-size: 1.05rem;
    font-weight: 700;
    width: 100%;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4);
    background: #d97706;
    color: white;
}

/* ── Misc ────────────────────────────────────────── */
[data-testid="stDataFrame"]  { border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; }
[data-testid="stSlider"] > div { padding: 0 4px; }
[data-testid="stSelectbox"] > div > div {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    color: #0f172a;
}

/* ── Status pills ────────────────────────────────── */
.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.status-ok     { background: #f0fdf4;  color:#16a34a; border:1px solid #16a34a; }
.status-warn   { background: #fffbeb;  color:#d97706; border:1px solid #d97706; }
.status-err    { background: #fef2f2;  color:#dc2626; border:1px solid #dc2626; }

/* ── Quality report badge ────────────────────────── */
.qr-badge {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 0.85rem;
    margin: 4px 0;
    color: #334155;
}

/* ── Landing Page Cards ──────────────────────────── */
.feature-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    height: 100%;
}
.fc-icon {
    font-size: 1.8rem;
    margin-bottom: 16px;
    display: inline-block;
    background: #f0fdf4;
    color: #22c55e;
    padding: 12px;
    border-radius: 12px;
}
.fc-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 12px;
}
.fc-desc {
    color: #475569;
    font-size: 0.95rem;
    line-height: 1.6;
}

.type-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    height: 100%;
    position: relative;
}
.type-card.featured {
    border: 2px solid #f59e0b;
    box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
}
.tc-badge {
    position: absolute;
    top: 16px;
    right: 16px;
    background: #fffbeb;
    color: #d97706;
    font-size: 0.7rem;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 20px;
    text-transform: uppercase;
}
.tc-icon {
    font-size: 2rem;
    color: #0ea5e9;
    margin-bottom: 16px;
}
.tc-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 12px;
}
.tc-desc {
    color: #475569;
    font-size: 0.95rem;
    line-height: 1.5;
    margin-bottom: 20px;
}
.tc-metric {
    border-top: 1px solid #e2e8f0;
    padding-top: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.tc-val {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f172a;
}
.tc-sub {
    font-size: 0.75rem;
    color: #64748b;
}
.tc-growth {
    color: #16a34a;
    font-size: 0.8rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HARD GUARD — backend import failure
# ════════════════════════════════════════════════════════════════════════════
if not _BACKEND_OK:
    st.error(f"❌ **Backend import failed** — the `backend/` package could not be loaded.\n\n```\n{_BACKEND_ERROR}\n```\n\nEnsure you are running from the project root:\n```\nstreamlit run frontend/app.py\n```")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# CACHED DATA & MODEL LOADERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="⏳ Loading training dataset…")
def _load_training(_path_str: str):
    return load_training_data(Path(_path_str))


@st.cache_data(show_spinner="⏳ Loading district data…")
def _load_districts(_path_str: str):
    return load_district_data(Path(_path_str))


@st.cache_resource(show_spinner="🤖 Training Random Forest model…")
def _load_or_train(_df_hash: int, _path_str: str, force: bool = False):
    """Thin wrapper so Streamlit can cache the (model, metrics) pair."""
    df = _load_training(_path_str)
    if df is None:
        return None, None, "error"
    try:
        return load_or_train_model(df, Path(_path_str), force_retrain=force)
    except Exception as exc:
        logger.error("load_or_train_model raised: %s", exc)
        return None, None, "error"


# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ════════════════════════════════════════════════════════════════════════════

df: pd.DataFrame | None = _load_training(str(DATASET_PATH))
df_districts: pd.DataFrame | None = _load_districts(str(DISTRICT_DATA_PATH))

# Compute a lightweight hash for the dataframe identity (needed for cache key)
_df_hash = hash(str(DATASET_PATH)) if df is None else hash(len(df))

model, metrics, model_status = _load_or_train(_df_hash, str(DATASET_PATH))

# ── Derived slider bounds ────────────────────────────────────────────────────
if df is not None:
    elev_min  = float(df["Elevation"].min())
    elev_max  = float(df["Elevation"].max())
    slope_min = float(df["Slope"].min())
    slope_max = float(df["Slope"].max())
    rain_min  = float(df["Rainfall"].min())
    rain_max  = float(df["Rainfall"].max())
else:
    elev_min, elev_max   = 0.0, 3000.0
    slope_min, slope_max = 0.0, 45.0
    rain_min, rain_max   = 200.0, 2000.0

# ── Session state for slider auto-fill ───────────────────────────────────────
for key, default in [("in_elev", 600.0), ("in_slope", 8.0), ("in_rain", 800.0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#e0e0e0", family="Inter"),
    margin=dict(l=20, r=20, t=44, b=20),
)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 12px 0;'>
        <div style='font-size:3.5rem; color:#f59e0b; margin-bottom:8px;'>☀️</div>
        <div style='font-size:1.35rem; font-weight:800; color:#0f172a; letter-spacing: -0.5px;'>SolarSense</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>Random Forest · Fully Offline</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Model status ─────────────────────────────────────────────────────────
    status_html = {
        "loaded":    "<span class='status-pill status-ok'>✅ Model Loaded</span>",
        "retrained": "<span class='status-pill status-ok'>🆕 Model Retrained</span>",
        "error":     "<span class='status-pill status-err'>❌ Model Error</span>",
    }.get(model_status, "<span class='status-pill status-warn'>⚠ Unknown</span>")

    st.markdown(f"**Model Status:** {status_html}", unsafe_allow_html=True)

    if metrics:
        meta = load_model_metadata()
        if meta:
            st.markdown(
                f"<div style='font-size:0.75rem; color:#9e9e9e; margin-top:6px;'>"
                f"Trained: {meta.get('trained_at','?')[:16].replace('T',' ')}<br>"
                f"Samples: {meta.get('n_train',0):,} train / {meta.get('n_test',0):,} test<br>"
                f"Acc: {meta.get('accuracy',0):.3f} &nbsp;|&nbsp; F1: {meta.get('f1',0):.3f} &nbsp;|&nbsp; AUC: {meta.get('roc_auc',0):.3f}"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    if st.button("♻️ Force Retrain Model", key="retrain_btn"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # ── Dataset overview ─────────────────────────────────────────────────────
    st.markdown("**📦 Dataset**")
    if df is not None:
        n_suit = int(df["Suitability"].sum())
        n_ns   = len(df) - n_suit
        st.markdown(
            f"<div style='font-size:0.8rem; color:#ccc;'>"
            f"Rows: <b>{len(df):,}</b><br>"
            f"✅ Suitable: <b>{n_suit:,}</b> ({n_suit/len(df):.1%})<br>"
            f"❌ Not Suitable: <b>{n_ns:,}</b> ({n_ns/len(df):.1%})"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Training data unavailable.")

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem; color:#555; text-align:center;'>v2.0 · March 2026</div>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
# HEADER (Mimicking Top Navigation Menu)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='display:flex; justify-content:space-between; align-items:center; padding: 10px 4px 24px 4px;'>
    <div style='display:flex; align-items:center; gap: 10px;'>
        <div style='font-size: 2rem; color: #f59e0b;'>☀️</div>
        <div style='font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.5px;'>SolarSense</div>
    </div>
    <div style='display:flex; gap: 12px; align-items:center;'>
        <div style='font-size: 0.95rem; font-weight: 600; color: #475569; padding: 8px 16px; cursor: pointer;'>Log In</div>
        <div style='font-size: 0.95rem; font-weight: 700; color: #ffffff; background: #f59e0b; padding: 8px 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(245, 158, 11, 0.25); cursor: pointer;'>Sign Up</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Critical error states ────────────────────────────────────────────────────
if df is None:
    st.error(
        "❌ **Training dataset not found.**  \n"
        f"Expected: `{DATASET_PATH}`  \n"
        "Run `python scripts/generate_dataset.py` to create it."
    )
if model is None:
    st.error("❌ **Model could not be trained.** Check that the dataset is valid.")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# TABS & NEW LANDING ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
tab_home, tab_analyze, tab_dash = st.tabs([
    "🏠 Home",
    "📍 Analyze Location",
    "📊 Dashboard"
])

# Setup aliases so existing code blocks attach correctly:
tab1 = tab_analyze
with tab_dash:
    st.markdown("<div class='section-header'>📊 Administrative Dashboard</div>", unsafe_allow_html=True)
    tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset Explorer",
        "🤖 Model Insights",
        "🔍 Similar Locations",
        "📋 Batch Predict"
    ])

# ════════════════════════════════════════════════════════════════════════════
# HOME TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("""
    <!-- Hero Section -->
    <div style='text-align:center; padding: 60px 20px 80px 20px;'>
        <div style='display:inline-block; background:#f0fdf4; color:#16a34a; padding:6px 16px; border-radius:20px; font-size:0.85rem; font-weight:700; margin-bottom:24px; border:1px solid #22c55e;'>
            🌱 Building a Sustainable Future
        </div>
        <h1 style='font-size:3.5rem; font-weight:900; color:#0f172a; margin-bottom:20px; letter-spacing:-1px;'>
            The Future is <span style='color:#0ea5e9;'>Renewable</span>
        </h1>
        <p style='font-size:1.15rem; color:#475569; max-width:700px; margin:0 auto; line-height:1.6;'>
            Renewable energy is the cornerstone of a sustainable future. Learn how solar, wind, and other clean sources are transforming our world and why the shift matters more than ever.
        </p>
    </div>

    <!-- Why Renewable Energy Matters -->
    <div style='text-align:center; margin-bottom: 40px;'>
        <h2 style='font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:12px;'>Why Renewable Energy Matters</h2>
        <p style='font-size:1.05rem; color:#64748b;'>The transition to clean energy isn't just an environmental choice -- it's an economic imperative.</p>
    </div>
    """, unsafe_allow_html=True)

    hc1, hc2, hc3 = st.columns(3, gap="large")
    with hc1:
        st.markdown("""
        <div class='feature-card'>
            <div class='fc-icon'>🌡️</div>
            <div class='fc-title'>Combat Climate Change</div>
            <div class='fc-desc'>Renewable energy produces little to no greenhouse gas emissions during operation, directly reducing the primary driver of global warming.</div>
        </div>
        <div class='feature-card'>
            <div class='fc-icon'>💧</div>
            <div class='fc-title'>Cleaner Air & Water</div>
            <div class='fc-desc'>Unlike fossil fuels, renewables don't pollute air or contaminate water sources, preventing millions of premature deaths annually.</div>
        </div>
        """, unsafe_allow_html=True)
    with hc2:
        st.markdown("""
        <div class='feature-card'>
            <div class='fc-icon' style='color:#0ea5e9; background:#e0f2fe;'>🛡️</div>
            <div class='fc-title'>Energy Independence</div>
            <div class='fc-desc'>Locally generated renewable energy reduces dependence on imported fossil fuels, enhancing national security and price stability.</div>
        </div>
        <div class='feature-card'>
            <div class='fc-icon' style='color:#8b5cf6; background:#ede9fe;'>♾️</div>
            <div class='fc-title'>Infinite Supply</div>
            <div class='fc-desc'>The sun, wind, and water won't run out. Renewable energy provides a sustainable foundation for humanity's long-term future.</div>
        </div>
        """, unsafe_allow_html=True)
    with hc3:
        st.markdown("""
        <div class='feature-card'>
            <div class='fc-icon' style='color:#ea580c; background:#ffedd5;'>📈</div>
            <div class='fc-title'>Economic Growth</div>
            <div class='fc-desc'>The renewable energy sector creates 3x more jobs per dollar invested than fossil fuels, driving economic development globally.</div>
        </div>
        <div class='feature-card'>
            <div class='fc-icon' style='color:#16a34a; background:#f0fdf4;'>💲</div>
            <div class='fc-title'>Lower Energy Costs</div>
            <div class='fc-desc'>Once installed, renewable energy has near-zero fuel costs. Solar and wind are now cheaper than new coal or gas plants in most regions.</div>
        </div>
        """, unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT SUITABILITY
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Auto-fill expander ───────────────────────────────────────────────────
    if df_districts is not None:
        with st.expander("📍 Auto-Fill from Real Indian District Data", expanded=False):
            st.markdown("Select a state and district to auto-load its historical elevation & rainfall averages.")
            loc1, loc2, loc3 = st.columns([2, 2, 1], gap="small", vertical_alignment="bottom")
            try:
                states = sorted(df_districts["state"].dropna().unique())
                selected_state = loc1.selectbox("State", states, key="st_state")
                districts = sorted(
                    df_districts[df_districts["state"] == selected_state]["district"].dropna().unique()
                )
                selected_district = loc2.selectbox("District", districts, key="st_dist")
                if loc3.button("📥 Load"):
                    rows = df_districts[
                        (df_districts["state"] == selected_state) &
                        (df_districts["district"] == selected_district)
                    ]
                    if rows.empty:
                        st.warning("No data found for that state/district combination.")
                    else:
                        info = rows.iloc[0]
                        st.session_state.in_elev = float(np.clip(info["elevation"], elev_min, elev_max))
                        st.session_state.in_rain = float(np.clip(info["annual_rainfall"], rain_min, rain_max))
                        st.rerun()
            except Exception as exc:
                st.warning(f"Could not render district selector: {exc}")
    else:
        st.info("ℹ️ Indian district data not available. Auto-fill disabled. "
                "Run `scripts/process_india_data.py` to enable this feature.")

    st.markdown("<div class='section-header'>📐 Environmental Parameters</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        elevation = st.slider("⛰️ Elevation (m)",   elev_min,  elev_max,  st.session_state.in_elev, step=10.0, help="Height above sea level")
        slope     = st.slider("📐 Slope (°)",        slope_min, slope_max, st.session_state.in_slope, step=0.5,  help="Terrain steepness in degrees (0° = flat, 45° = steep)")
        rainfall  = st.slider("🌧️ Rainfall (mm/yr)", rain_min,  rain_max,  st.session_state.in_rain,  step=10.0, help="Average annual rainfall")
        soil_type = st.selectbox("🪨 Soil Type", list(SOIL_MAPPING.keys()), help="Select the predominant soil class")

        st.session_state.in_elev  = elevation
        st.session_state.in_slope = slope
        st.session_state.in_rain  = rainfall

    with c2:
        try:
            derived = derive_single(elevation, slope, rainfall)
        except Exception:
            derived = {"temperature": 0, "ndvi": 0, "soil_moisture": 0, "elevation_zone": 0}

        st.markdown("**🔬 Auto-Derived Features** *(live)*")
        d1, d2, d3 = st.columns(3)
        d1.metric("🌡️ Temp",         f"{derived['temperature']:.1f} °C")
        d2.metric("🌿 NDVI",          f"{derived['ndvi']:.3f}")
        d3.metric("💧 Soil Moisture", f"{derived['soil_moisture']:.3f}")

        ez_labels = {0: "Lowland", 1: "Mid-Hill", 2: "Highland", 3: "Alpine"}
        st.caption(f"📏 Elevation Zone: **{ez_labels.get(derived['elevation_zone'], '?')}** (Zone {derived['elevation_zone']})")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict Land Suitability", key="predict_btn")

    if predict_btn:
        with st.spinner("Running inference…"):
            res = predict(model, elevation, slope, rainfall, soil_type)

        st.markdown("<br>", unsafe_allow_html=True)

        if "error" in res:
            st.markdown(f"<div class='result-error'>⚠️ {res['error']}</div>", unsafe_allow_html=True)
        else:
            pred    = res["prediction"]
            conf    = res["confidence"]
            is_suit = pred == 1
            tier    = res["conf_tier"]
            color   = res["conf_color"]

            # ── Result banner ────────────────────────────────────────────────
            cls = "result-suitable" if is_suit else "result-not"
            emoji = "✅" if is_suit else "❌"
            label = "SUITABLE LAND" if is_suit else "NOT SUITABLE LAND"
            st.markdown(
                f"<div class='{cls}'>{emoji} &nbsp; {label} &nbsp;·&nbsp; Confidence: {conf:.1%}</div>",
                unsafe_allow_html=True,
            )

            # ── Low-confidence warning ────────────────────────────────────
            if conf < 0.65:
                st.warning(
                    f"⚠️ **{tier}** ({conf:.1%}) — This is a borderline case. "
                    "Consider refining input parameters or consulting local data."
                )
            elif conf < 0.85:
                st.info(f"ℹ️ **{tier}** ({conf:.1%})")

            st.markdown("<br>", unsafe_allow_html=True)

            col_gauge, col_cards = st.columns([1, 1], gap="large")

            # ── Gauge ────────────────────────────────────────────────────────
            with col_gauge:
                gauge_color = "#4caf50" if is_suit else "#f44336"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf * 100,
                    number={"suffix": "%", "font": {"size": 36, "color": gauge_color}},
                    title={"text": "Confidence Score", "font": {"size": 13, "color": "#9e9e9e"}},
                    gauge={
                        "axis":      {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
                        "bar":       {"color": gauge_color, "thickness": 0.25},
                        "bgcolor":   "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,  65],  "color": "rgba(244,67,54,0.12)"},
                            {"range": [65, 85],  "color": "rgba(255,152,0,0.12)"},
                            {"range": [85, 100], "color": "rgba(76,175,80,0.12)"},
                        ],
                        "threshold": {"line": {"color": gauge_color, "width": 4},
                                      "thickness": 0.8, "value": conf * 100},
                    },
                ))
                fig_gauge.update_layout(**PLOT_LAYOUT, height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Condition cards ───────────────────────────────────────────────
            with col_cards:
                st.markdown("**🔎 Condition Analysis**")
                for cond in res.get("conditions", []):
                    css  = "pass-card" if cond["passed"] else "fail-card"
                    icon = "✅ PASS" if cond["passed"] else "❌ FAIL"
                    st.markdown(
                        f"<div class='{css}'>"
                        f"<span style='font-weight:700;font-size:0.8rem;'>{icon}</span>"
                        f"<span style='font-size:0.88rem;'>{cond['name']}: "
                        f"<b>{cond['val']:.2f}{cond['unit']}</b>"
                        f" &nbsp;(need {cond['op']} {cond['ideal']}{cond['unit']})</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Probability bar chart ─────────────────────────────────────────
            probas = model.predict_proba(
                pd.DataFrame([[
                    elevation, slope, rainfall, SOIL_MAPPING[soil_type],
                    derived["temperature"], derived["ndvi"],
                    derived["soil_moisture"], derived["elevation_zone"],
                ]], columns=FEATURE_COLS)
            )[0]
            fig_bar = go.Figure(go.Bar(
                x=["Not Suitable", "Suitable"],
                y=[round(probas[0], 4), round(probas[1], 4)],
                marker_color=["#f44336", "#4caf50"],
                text=[f"{probas[0]:.1%}", f"{probas[1]:.1%}"],
                textposition="outside",
            ))
            fig_bar.update_layout(
                **PLOT_LAYOUT, height=220,
                title="Class Probability",
                yaxis=dict(range=[0, 1.1], tickformat=".0%"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📊 Explore the Training Dataset</div>", unsafe_allow_html=True)

    if df is None:
        st.error("Training dataset not available.")
    else:
        # ── Data Quality Report ───────────────────────────────────────────────
        with st.expander("🔬 Data Quality Report", expanded=False):
            try:
                qr = data_quality_report(df)
                qc1, qc2, qc3, qc4 = st.columns(4)
                qc1.metric("Total Rows",    f"{qr['n_rows']:,}")
                qc2.metric("Total Columns", qr["n_cols"])
                qc3.metric("Duplicates",    qr["n_dupes"])
                qc4.metric("Missing Cells", qr["total_missing"])

                if qr["missing"]:
                    st.markdown("**Missing values per column:**")
                    for col, cnt in qr["missing"].items():
                        st.markdown(f"<div class='qr-badge'>⚠️ `{col}`: {cnt} missing</div>", unsafe_allow_html=True)
                else:
                    st.success("✅ No missing values detected.")

                if qr.get("outliers"):
                    st.markdown("**IQR outlier counts:**")
                    out_df = pd.DataFrame(
                        qr["outliers"].items(), columns=["Column", "Outlier Count"]
                    )
                    st.dataframe(out_df, use_container_width=True, height=180)
                else:
                    st.success("✅ No outliers detected (IQR method).")

                if "class_balance" in qr:
                    st.markdown(
                        f"**Class ratio** (Suitable / Not Suitable): `{qr['class_ratio']}`"
                    )
            except Exception as exc:
                st.warning(f"Could not generate quality report: {exc}")

        # ── Sample for performance ────────────────────────────────────────────
        df_plot = df.sample(min(6000, len(df)), random_state=42).copy()
        df_plot["Suitability_Label"] = df_plot["Suitability"].map({1: "Suitable", 0: "Not Suitable"})
        colour_map = {"Suitable": "#4caf50", "Not Suitable": "#f44336"}

        # ── Scatter + Pie ─────────────────────────────────────────────────────
        r1c1, r1c2 = st.columns([2, 1])
        try:
            with r1c1:
                num_axes = ["Elevation", "Rainfall", "Slope", "Temperature", "NDVI", "Soil_Moisture"]
                xax = st.selectbox("X Axis", num_axes, index=0, key="xax")
                yax = st.selectbox("Y Axis", num_axes, index=2, key="yax")
                fig_sc = px.scatter(
                    df_plot, x=xax, y=yax,
                    color="Suitability_Label", color_discrete_map=colour_map,
                    opacity=0.55, title=f"{xax} vs {yax}",
                    labels={"Suitability_Label": "Class"},
                )
                fig_sc.update_traces(marker=dict(size=3))
                fig_sc.update_layout(**PLOT_LAYOUT, height=380,
                                     legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig_sc, use_container_width=True)

            with r1c2:
                vc = df["Suitability"].value_counts().reset_index()
                vc.columns = ["class", "count"]
                vc["label"] = vc["class"].map({1: "Suitable", 0: "Not Suitable"})
                fig_pie = px.pie(
                    vc, names="label", values="count",
                    color="label", color_discrete_map=colour_map,
                    title="Dataset Balance", hole=0.55,
                )
                fig_pie.update_traces(
                    textinfo="percent+label",
                    marker=dict(line=dict(color="#12122a", width=2))
                )
                fig_pie.update_layout(**PLOT_LAYOUT, showlegend=False, height=380)
                st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render scatter/pie charts: {exc}")

        # ── Histograms ────────────────────────────────────────────────────────
        st.markdown("**📈 Feature Distributions by Class**")
        num_cols = ["Elevation", "Slope", "Rainfall", "Temperature", "NDVI", "Soil_Moisture"]
        hcols = st.columns(3)
        for i, col in enumerate(num_cols):
            try:
                fig_h = px.histogram(
                    df_plot, x=col, color="Suitability_Label",
                    color_discrete_map=colour_map, barmode="overlay",
                    opacity=0.72, nbins=50, title=col,
                    labels={"Suitability_Label": ""},
                )
                fig_h.update_layout(**PLOT_LAYOUT, height=240,
                                    showlegend=(i == 0),
                                    legend=dict(orientation="h", y=-0.25))
                hcols[i % 3].plotly_chart(fig_h, use_container_width=True)
            except Exception:
                hcols[i % 3].warning(f"Could not render histogram for {col}")

        # ── Correlation heatmap ────────────────────────────────────────────────
        st.markdown("**🔥 Correlation Heatmap**")
        try:
            corr = df[num_cols + ["Suitability"]].corr().round(2)
            fig_heat = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale="RdYlGn", zmid=0,
                text=corr.values.round(2), texttemplate="%{text}",
                textfont={"size": 10}, hoverongaps=False,
            ))
            fig_heat.update_layout(**PLOT_LAYOUT, height=400)
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render heatmap: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS (Research Grade)
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🤖 Model Performance & Research Insights</div>", unsafe_allow_html=True)

    if metrics is None or not metrics:
        st.error("Model metrics are not available.")
    else:
        # ── Row 0: Summary metrics ────────────────────────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        metric_items = [
            (m1, "Accuracy",    metrics.get("accuracy", 0),    ".4f"),
            (m2, "Precision",   metrics.get("precision", 0),   ".4f"),
            (m3, "Recall",      metrics.get("recall", 0),      ".4f"),
            (m4, "F1 Score",    metrics.get("f1", 0),          ".4f"),
            (m5, "ROC-AUC",     metrics.get("roc_auc", 0),     ".4f"),
            (m6, "Brier Score", metrics.get("brier_score", 0), ".4f"),
        ]
        for col, name, val, fmt in metric_items:
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>{name}</div>"
                f"<div class='metric-value' style='font-size:1.5rem;'>{val:{fmt}}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Cross-val summary
        cv_mean = metrics.get("cv_mean", 0)
        cv_std  = metrics.get("cv_std", 0)
        gap     = metrics.get("overfit_gap", 0)
        gap_col = "#4caf50" if gap <= 0.03 else ("#ff9800" if gap <= 0.06 else "#f44336")
        st.markdown(
            f"<div style='margin-top:14px; font-size:0.9rem; color:#bbb;'>"
            f"📊 Cross-Val F1 ({metrics.get('cv_mean',0):.4f} ± {cv_std:.4f} over 5 folds) &nbsp;·&nbsp; "
            f"Train→Test Gap: <span style='color:{gap_col}; font-weight:700;'>{gap:+.4f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Feature Importance + Confusion Matrix ──────────────────────
        ins_col, cm_col = st.columns(2)

        with ins_col:
            try:
                imp_df = pd.DataFrame({
                    "Feature":    metrics["feature_names"],
                    "Importance": metrics["importances"],
                }).sort_values("Importance", ascending=True)
                fig_imp = go.Figure(go.Bar(
                    x=imp_df["Importance"], y=imp_df["Feature"],
                    orientation="h",
                    marker=dict(color=imp_df["Importance"], colorscale="Viridis"),
                    text=[f"{v:.4f}" for v in imp_df["Importance"]],
                    textposition="outside",
                ))
                fig_imp.update_layout(**PLOT_LAYOUT, height=380,
                                      title="Feature Importance (Gini)",
                                      xaxis=dict(title="Importance Score",
                                                 showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                                      yaxis=dict(showgrid=False))
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as exc:
                st.warning(f"Feature importance chart error: {exc}")

        with cm_col:
            try:
                cm = metrics["cm"]
                labels_cm = ["Not Suitable", "Suitable"]
                fig_cm = go.Figure(go.Heatmap(
                    z=cm, x=labels_cm, y=labels_cm,
                    colorscale=[[0, "#12122a"], [0.5, "#3949ab"], [1, "#4caf50"]],
                    showscale=False,
                    text=cm, texttemplate="<b>%{text}</b>",
                    textfont={"size": 22},
                ))
                fig_cm.update_layout(
                    **PLOT_LAYOUT, height=380,
                    title="Confusion Matrix",
                    xaxis=dict(title="Predicted", side="bottom"),
                    yaxis=dict(title="Actual", autorange="reversed"),
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as exc:
                st.warning(f"Confusion matrix error: {exc}")

        # ── Row 2: ROC Curve + Calibration Curve ─────────────────────────────
        roc_col, cal_col = st.columns(2)

        with roc_col:
            try:
                fpr = metrics.get("roc_fpr", [])
                tpr = metrics.get("roc_tpr", [])
                roc_auc = metrics.get("roc_auc", 0.0)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"ROC (AUC = {roc_auc:.4f})",
                    line=dict(color="#667eea", width=2.5),
                    fill="tozeroy", fillcolor="rgba(102,126,234,0.12)",
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="#555", width=1.5),
                    name="Random Classifier",
                    showlegend=True,
                ))
                fig_roc.update_layout(
                    **PLOT_LAYOUT, height=350,
                    title="ROC Curve",
                    xaxis=dict(title="False Positive Rate", range=[0, 1],
                               showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
                    yaxis=dict(title="True Positive Rate", range=[0, 1],
                               showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
                    legend=dict(x=0.55, y=0.08),
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            except Exception as exc:
                st.warning(f"ROC curve error: {exc}")

        with cal_col:
            try:
                frac_pos  = metrics.get("cal_frac_pos", [])
                mean_pred = metrics.get("cal_mean_pred", [])
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=mean_pred, y=frac_pos, mode="lines+markers",
                    name="Calibration Curve",
                    line=dict(color="#4caf50", width=2.5),
                    marker=dict(size=8),
                ))
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="#555", width=1.5),
                    name="Perfectly Calibrated",
                ))
                fig_cal.update_layout(
                    **PLOT_LAYOUT, height=350,
                    title=f"Calibration Curve (Brier = {metrics.get('brier_score',0):.4f})",
                    xaxis=dict(title="Mean Predicted Probability", range=[0, 1],
                               showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
                    yaxis=dict(title="Fraction of Positives", range=[0, 1],
                               showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
                )
                st.plotly_chart(fig_cal, use_container_width=True)
            except Exception as exc:
                st.warning(f"Calibration curve error: {exc}")

        # ── Row 3: Learning Curve ─────────────────────────────────────────────
        st.markdown("**📉 Learning Curve — Overfitting Diagnostic**")
        try:
            lc_sizes  = metrics.get("lc_train_sizes", [])
            lc_train  = metrics.get("lc_train_scores", [])
            lc_val    = metrics.get("lc_val_scores", [])
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=lc_sizes, y=lc_train, mode="lines+markers",
                name="Training F1",
                line=dict(color="#667eea", width=2.5), marker=dict(size=7),
            ))
            fig_lc.add_trace(go.Scatter(
                x=lc_sizes, y=lc_val, mode="lines+markers",
                name="Validation F1",
                line=dict(color="#4caf50", width=2.5), marker=dict(size=7),
            ))
            fig_lc.update_layout(
                **PLOT_LAYOUT, height=320,
                title="Learning Curve (F1 Score)",
                xaxis=dict(title="Training Samples",
                           showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
                yaxis=dict(title="F1 Score", range=[0, 1.05],
                           showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_lc, use_container_width=True)

            # Overfitting commentary
            if len(lc_train) > 0 and len(lc_val) > 0:
                final_gap = lc_train[-1] - lc_val[-1]
                if final_gap > 0.05:
                    st.warning(
                        f"⚠️ **Overfitting signal**: training F1 ({lc_train[-1]:.3f}) "
                        f"exceeds validation F1 ({lc_val[-1]:.3f}) by {final_gap:.3f}. "
                        "Consider further increasing `min_samples_leaf` or reducing `max_depth` in `backend/config.py`."
                    )
                else:
                    st.success(
                        f"✅ **Well-generalised model**: train/val F1 gap = {final_gap:.3f} (< 0.05 threshold)."
                    )
        except Exception as exc:
            st.warning(f"Learning curve error: {exc}")

        # ── Cross-val scores ───────────────────────────────────────────────────
        st.markdown("**🎯 Cross-Validation F1 Scores (5-Fold)**")
        try:
            cv_scores = metrics.get("cv_scores", [])
            fig_cv = go.Figure(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                y=cv_scores,
                marker_color="#667eea",
                text=[f"{s:.4f}" for s in cv_scores],
                textposition="outside",
            ))
            fig_cv.add_hline(
                y=np.mean(cv_scores),
                line_dash="dash", line_color="#ff9800",
                annotation_text=f"Mean={np.mean(cv_scores):.4f}",
                annotation_position="top right",
            )
            fig_cv.update_layout(
                **PLOT_LAYOUT, height=260,
                yaxis=dict(range=[0, 1.1], showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            )
            st.plotly_chart(fig_cv, use_container_width=True)
        except Exception as exc:
            st.warning(f"CV chart error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — FIND SIMILAR LOCATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>🔍 Find Real Indian Districts with Similar Environments</div>", unsafe_allow_html=True)
    st.markdown(
        "Enter target environmental parameters — the system finds the **10 most similar Indian districts** "
        "using MinMax-normalised Euclidean distance on *(Elevation, Rainfall)*."
    )

    if df_districts is None:
        st.error(
            "❌ Indian district data unavailable.  \n"
            f"Expected: `{DISTRICT_DATA_PATH}`  \n"
            "Run `python scripts/process_india_data.py` to generate it."
        )
    else:
        s1, s2 = st.columns(2)
        s_elev = s1.number_input("⛰️ Target Elevation (m)",   min_value=float(elev_min), max_value=float(elev_max), value=600.0, step=10.0)
        s_rain = s2.number_input("🌧️ Target Annual Rainfall (mm)", min_value=float(rain_min), max_value=float(rain_max), value=800.0, step=10.0)

        search_btn = st.button("🔍 Find Similar Districts", key="search_btn")

        if search_btn:
            with st.spinner("Searching…"):
                result_df = find_similar_districts(
                    df_districts, s_elev, s_rain, model=model, n=10
                )

            if result_df is None or result_df.empty:
                st.error("Could not find similar districts. Check that the district CSV is valid.")
            else:
                # Rename for display
                display_cols_map = {
                    "state": "State", "district": "District",
                    "elevation": "Elevation (m)", "annual_rainfall": "Rainfall (mm)",
                    "avg_temp": "Avg Temp (°C)", "latitude": "Lat", "longitude": "Lon",
                    "Match_Distance": "Match Distance",
                    "Predicted_Suitability": "AI Suitability",
                    "Prediction_Confidence": "AI Confidence",
                }
                result_df = result_df.rename(columns=display_cols_map)

                disp_cols = [c for c in display_cols_map.values() if c in result_df.columns]
                st.markdown("<br>**🗺️ Top 10 Matching Districts:**", unsafe_allow_html=True)
                st.dataframe(result_df[disp_cols], use_container_width=True)

                # ── Map ───────────────────────────────────────────────────────
                if "Lat" in result_df.columns and "Lon" in result_df.columns:
                    map_data = result_df.rename(columns={"Lat": "lat", "Lon": "lon"})[
                        ["lat", "lon"]
                    ].dropna()
                    if not map_data.empty:
                        st.markdown("<br>**📍 Geographic Distribution:**", unsafe_allow_html=True)
                        st.map(map_data, zoom=4, color="#ff4b4b", size=4000)
                    else:
                        st.info("Lat/Lon data not available for map display.")

                # ── Download ──────────────────────────────────────────────────
                csv_bytes = result_df[disp_cols].to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Results (CSV)",
                    data=csv_bytes,
                    file_name="similar_districts.csv",
                    mime="text/csv",
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>📋 Batch Land Suitability Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV file with columns: **Elevation, Slope, Rainfall, Soil_Type**  \n"
        "Optional columns (Temperature, NDVI, Soil_Moisture) are derived automatically if missing."
    )

    with st.expander("📄 Expected Format & Example", expanded=False):
        example_df = pd.DataFrame({
            "Elevation": [450, 1200, 2800, 300],
            "Slope":     [5,   12,   35,   3],
            "Rainfall":  [900, 650, 350,  1400],
            "Soil_Type": ["Loamy", "Clay", "Sandy", "Silty"],
        })
        st.dataframe(example_df, use_container_width=True, height=175)
        csv_example = example_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Template", csv_example, "batch_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            df_batch_in = pd.read_csv(uploaded)
            st.markdown(f"**Loaded {len(df_batch_in):,} rows, {len(df_batch_in.columns)} columns.**")
            st.dataframe(df_batch_in.head(5), use_container_width=True, height=200)

            if st.button("🚀 Run Batch Prediction", key="batch_run"):
                with st.spinner(f"Running predictions on {len(df_batch_in):,} rows…"):
                    try:
                        df_results = predict_batch(model, df_batch_in)

                        n_suit = int((df_results["Prediction"] == 1).sum())
                        n_ns   = len(df_results) - n_suit

                        r1, r2, r3 = st.columns(3)
                        r1.metric("Total Rows",    f"{len(df_results):,}")
                        r2.metric("✅ Suitable",    f"{n_suit:,}  ({n_suit/len(df_results):.1%})")
                        r3.metric("❌ Not Suitable", f"{n_ns:,}  ({n_ns/len(df_results):.1%})")

                        # Pie chart of results
                        fig_b = px.pie(
                            names=["Suitable", "Not Suitable"],
                            values=[n_suit, n_ns],
                            color_discrete_sequence=["#4caf50", "#f44336"],
                            hole=0.6,
                            title="Batch Prediction Summary",
                        )
                        fig_b.update_layout(**PLOT_LAYOUT, height=280, showlegend=True)
                        st.plotly_chart(fig_b, use_container_width=True)

                        st.markdown("**Full Results:**")
                        st.dataframe(df_results, use_container_width=True)

                        csv_out = df_results.to_csv(index=False).encode()
                        st.download_button(
                            "⬇️ Download Predictions (CSV)",
                            data=csv_out,
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                        )
                    except ValueError as ve:
                        st.error(f"❌ Input error: {ve}")
                    except Exception as exc:
                        st.error(f"❌ Batch prediction failed: {exc}")
                        logger.error("Batch prediction error: %s", exc, exc_info=True)

        except Exception as exc:
            st.error(f"❌ Could not parse uploaded CSV: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
n_rows_str = f"{len(df):,}" if df is not None else "N/A"
st.markdown(f"""
<div style='text-align:center; color:#444; font-size:0.78rem;
            border-top:1px solid rgba(255,255,255,0.06); padding-top:18px;'>
    🌱 Land Suitability AI v2.0 &nbsp;·&nbsp;
    Random Forest Classifier &nbsp;·&nbsp;
    {n_rows_str} Training Samples &nbsp;·&nbsp;
    Fully Offline &nbsp;·&nbsp;
    Built with Streamlit + scikit-learn
</div>
""", unsafe_allow_html=True)
