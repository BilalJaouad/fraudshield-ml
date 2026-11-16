"""FraudShield / Risk intelligence dashboard — synthetic BAF portfolio demonstration.

This app runs the *saved* CatBoost + preprocessing locally inside Streamlit.
It does not retrain or refit objects. No external prediction API or keys needed.
"""
from __future__ import annotations

import html
import io
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.inference import FraudPredictor
from fraudshield_demo import generate_records, expected_decision_costs, observed_confusion_costs

# ============================================================================
# Theme / model
# ============================================================================
st.set_page_config(
    page_title="FraudShield · Risk Intelligence",
    page_icon="🛡️", layout="wide", initial_sidebar_state="expanded",
)

INK = "#e9f2fa"
MUTED = "#93a6be"
TEAL = "#36e4bc"
BLUE = "#66b9ff"
RED = "#ff7588"
AMBER = "#ffcb75"
PLOT_BG = "rgba(0,0,0,0)"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"] { background:linear-gradient(135deg,#08121f 0%,#0b1726 65%,#0b1b25 100%); color:#e9f2fa; }
[data-testid="stSidebar"] { background:linear-gradient(185deg,#0e2232,#091625); border-right:1px solid #24394a; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stMainBlockContainer"] { padding-top:1.25rem; max-width:1470px; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b8cadb; }
h1,h2,h3,h4 { font-family:'Space Grotesk','DM Sans',sans-serif; letter-spacing:-.035em; }
h1 { font-weight:700; } h3 { font-weight:650; }
[data-testid="stMetric"] { background:#102436; border:1px solid #29445a; border-radius:16px; padding:15px 17px; }
[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; font-weight:700; }
[data-testid="stMetricLabel"] { color:#a9bdd0; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:16px; }
[data-testid="stForm"] { background:#0e2132; border:1px solid #264155; border-radius:18px; padding:16px; }
[data-testid="stExpander"] { border:1px solid #2a4156; border-radius:12px; background:#0d2030; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background:#39dbb5; color:#06291f; border:0; font-weight:800; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover { background:#8bf3d9; color:#06291f; }
.fs-hero { border:1px solid #284a58; border-radius:20px; padding:26px 30px; background:radial-gradient(ellipse at 89% 15%,rgba(57,219,181,.16),transparent 45%),linear-gradient(115deg,#103342,#122039 67%,#112332);margin-bottom:22px; }
.fs-eyebrow { color:#4be6c0; letter-spacing:.17em; font-size:11px; font-weight:800; }
.fs-hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(30px,3.7vw,52px);font-weight:700;letter-spacing:-.055em;margin:8px 0 6px;line-height:1.13;color:#f4fbff; }
.fs-sub {color:#afc6d8;font-size:15px;line-height:1.6;max-width:850px; }
.fs-pill {display:inline-block;border:1px solid #2e8576;background:#123e3b;color:#7cf1d4;border-radius:100px;padding:6px 10px;font-size:12px;font-weight:700;margin-top:12px;}
.fs-brand { font-family:'Space Grotesk',sans-serif; font-weight:700; color:#f0fcff; font-size:25px; letter-spacing:-.055em; }
.fs-brand em {color:#45e4bb;font-style:normal;}
.fs-stat { background:#0e2233; border:1px solid #29465a; padding:17px 18px; border-radius:15px; }
.fs-stat .n { color:#e6f8ff; font:700 29px 'Space Grotesk',sans-serif;letter-spacing:-.035em; }
.fs-stat .k { color:#9bb7c9; font-size:12px; text-transform:uppercase;letter-spacing:.09em; }
.fs-section { font:600 19px 'Space Grotesk',sans-serif;color:#eaf7ff;margin:8px 0; }
.fs-note {color:#a5bed0;font-size:12px;line-height:1.55;}
.fs-tag {display:inline-block;background:#1e3745;border:1px solid #315567;border-radius:9px;padding:6px 9px;color:#bbd4e2;font-size:12px;}
.fs-good { color:#45e4bb; font-weight:800; }.fs-risk { color:#ff9aaa; font-weight:800; }
footer, #MainMenu {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading FraudShield model and preprocessors…")
def load_predictor() -> FraudPredictor:
    return FraudPredictor(Path(__file__).resolve().parent / "artifacts")

try:
    predictor = load_predictor()
except Exception as exc:
    st.error("Cannot load the trusted model artifacts. Check /artifacts and dependency versions (scikit-learn 1.6.1).")
    st.exception(exc)
    st.stop()

schema = predictor.schema
raw = list(schema["raw_features"])
cats = list(schema["categorical_columns"])
numeric = [field for field in raw if field not in cats]
model_threshold = float(predictor.policy["threshold"])
model_version = str(predictor.policy.get("model_version", "v2.0.0"))

HOLDOUT = {"tp": 567, "fp": 1900, "fn": 861, "tn": 93515,
           "ap": 0.2198193581, "auc": 0.896774, "alert_rate": 0.0254742,
           "precision": 0.22983, "recall": 0.39706}

# All values below are historical synthetic-BAF notebook metrics, not live metrics.

def money(amount: float) -> str:
    return f"{amount:,.0f} MAD".replace(",", " ")


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{100*x:.{digits}f}%"


def formatted_label(field: str) -> str:
    special = {"income": "Income (BAF scale)", "name_email_similarity": "Name / email similarity",
               "customer_age": "Age (years)", "credit_risk_score": "Credit risk score",
               "month": "Dataset period (0–7)", "device_fraud_count": "Dataset device fraud count",
               "intended_balcon_amount": "Intended balance amount", "source": "Application channel",
               "device_os": "Device operating system", "prev_address_months_count": "Previous address duration",
               "current_address_months_count": "Current address duration",
               "housing_status": "Housing status", "employment_status": "Employment status"}
    return special.get(field, field.replace("_", " ").title())


def style_chart(fig: go.Figure, height: int = 310) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font={"family":"DM Sans, sans-serif", "color":"#b6cada", "size":12},
        margin=dict(l=18, r=18, t=42, b=18), height=height,
        title_font={"color":"#edf7ff", "size":16},
        legend={"orientation":"h", "y":-0.18, "x":0},
        hoverlabel={"bgcolor":"#132d40"},
    )
    fig.update_xaxes(gridcolor="#244053", zeroline=False, showline=False)
    fig.update_yaxes(gridcolor="#244053", zeroline=False, showline=False)
    return fig


def hero(kicker: str, title: str, description: str, badge: str = "SYNTHETIC-DATA DEMO") -> None:
    st.markdown(
        f'<div class="fs-hero"><div class="fs-eyebrow">{html.escape(kicker)}</div>'
        f'<div class="fs-hero-title">{html.escape(title)}</div>'
        f'<div class="fs-sub">{html.escape(description)}</div>'
        f'<span class="fs-pill">● {html.escape(badge)}</span></div>',
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="fs-stat"><div class="k">{html.escape(title)}</div>'
        f'<div class="n">{html.escape(value)}</div><div class="fs-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def single_risk_chart(score: float, threshold: float) -> go.Figure:
    max_scale = max(0.15, min(1.0, max(score * 1.45, threshold * 3.2)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100,
        number={"suffix":"%", "valueformat":".3f", "font":{"size":38,"color":INK}},
        gauge={"axis":{"range":[0, max_scale * 100],"tickcolor":"#829bb1"},
               "bar":{"color": RED if score >= threshold else TEAL,"thickness":0.3},
               "bgcolor":"#12283a", "borderwidth":0,
               "threshold":{"line":{"color":AMBER,"width":4},"thickness":0.85,"value":threshold*100}},
        title={"text":"Model risk score", "font":{"size":15,"color":MUTED}},
    ))
    return style_chart(fig, 260)


def reconcile_categories(records: list[dict]) -> list[dict]:
    """Respect exactly the categories present in the exported fitted OneHotEncoder."""
    choices = {name: set(map(str, predictor.encoder.categories_[i])) for i, name in enumerate(cats)}
    for record in records:
        for name in cats:
            if record[name] not in choices[name]:
                record[name] = sorted(choices[name])[0]
        if set(record) != set(raw):
            raise ValueError("Synthetic generator does not match the deployed model schema")
    return records


def score_records(records: list[dict]) -> pd.DataFrame:
    outputs = predictor.predict(reconcile_categories(records))
    return pd.DataFrame({
        "risk": [float(o["fraud_probability"]) for o in outputs],
        "decision": [str(o["decision"]) for o in outputs],
        "housing": [str(r["housing_status"]) for r in records],
        "age": [int(r["customer_age"]) for r in records],
        "channel": [str(r["source"]) for r in records],
        "credit_score": [int(r["credit_risk_score"]) for r in records],
    })


def select_demo_candidates(n: int, seed: int, mode: str) -> pd.DataFrame:
    """Select only genuine model scores; optional 'spotlight' deliberately oversamples high scores."""
    if mode == "Risk spotlight (selected)":
        pool = generate_records(max(120, n * 14), seed=seed, profile="watchlist")
        df = score_records(pool)
        # Selection is *deliberately biased*: never treat as a prevalence estimate.
        high_n = max(1, n // 2)
        high = df.nlargest(high_n, "risk")
        low = df.drop(index=high.index).sample(n=n-high_n, random_state=int(seed % 2_147_483_647))
        result = pd.concat([high, low], ignore_index=True)
        return result.sample(frac=1, random_state=int((seed+13) % 2_147_483_647)).reset_index(drop=True)
    elif mode == "Routine traffic":
        profile = "routine"
    else:
        profile = "mixed"
    return score_records(generate_records(n, seed=seed, profile=profile))


def add_history(scored: pd.DataFrame, mode: str) -> None:
    events = st.session_state.get("fs_events", [])
    idx0 = st.session_state.get("fs_event_id", 0)
    time_string = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    for idx, (_, r) in enumerate(scored.iterrows(), start=1):
        events.append({"id": idx0+idx, "time": time_string,
                       "risk": float(r.risk), "housing": str(r.housing),
                       "channel": str(r.channel), "age": int(r.age),
                       "credit_score": int(r.credit_score), "scenario": mode})
    st.session_state.fs_event_id = idx0+len(scored)
    st.session_state.fs_events = events[-400:]


def events_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.fs_events)


if "fs_events" not in st.session_state:
    st.session_state.fs_events = []
if "fs_event_id" not in st.session_state:
    st.session_state.fs_event_id = 0
if "fs_live_running" not in st.session_state:
    st.session_state.fs_live_running = False
if "fs_tick" not in st.session_state:
    st.session_state.fs_tick = 0
if "fs_seed" not in st.session_state:
    st.session_state.fs_seed = 2026

# ============================================================================
# Sidebar / navigation
# ============================================================================
st.sidebar.markdown('<div class="fs-brand">⬡ Fraud<em>Shield</em></div>', unsafe_allow_html=True)
st.sidebar.caption("FRAUD RISK INTELLIGENCE · V2")
st.sidebar.divider()
nav = st.sidebar.radio(
    "Navigation",
    ["◈  Command center", "◉  Live simulator", "▣  Risk assessment",
     "▤  Batch scoring", "◇  Decision economics"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown("<span class='fs-tag'>● SYSTEM ONLINE · LOCAL INFERENCE</span>", unsafe_allow_html=True)
st.sidebar.caption(f"CatBoost {model_version} · 58 engineered features")
st.sidebar.caption(f"Decision threshold · {fmt_pct(model_threshold,2)}")
st.sidebar.markdown("---")
st.sidebar.caption("Synthetic BAF dataset. Educational demonstration, not a real financial decision engine. No applicant records are persisted.")

# ============================================================================
# 1. Command center
# ============================================================================
if "Command center" in nav:
    hero("FRAUDSHIELD / INTELLIGENCE OPERATIONS", "Spot risk. See the trade-offs.",
         "A decision-focused fraud analytics experience built on CatBoost, temporal validation and deployable ML artifacts.",
         "LIVE APPLICATION · HISTORICAL TEST METRICS BELOW")
    test_n = HOLDOUT["tn"] + HOLDOUT["fp"] + HOLDOUT["fn"] + HOLDOUT["tp"]
    base = (HOLDOUT["tp"] + HOLDOUT["fn"]) * 1000
    cost = HOLDOUT["fp"] * 100 + HOLDOUT["fn"] * 1000
    cols = st.columns(4)
    with cols[0]: stat_card("Average precision", "0.220", "Month 7 holdout · historical")
    with cols[1]: stat_card("ROC–AUC", "0.897", "Month 7 holdout · historical")
    with cols[2]: stat_card("Review rate", "2.55%", "2,467 alerts / 96,843 records")
    with cols[3]: stat_card("Cost reduction", f"{100*(base-cost)/base:.1f}%", "Illustrative policy vs no reviews")
    st.caption("Metrics above are frozen notebook results on the **synthetic BAF month-7 split**, not metrics from the current visitors.")
    left, right = st.columns([1.05,0.95], gap="large")
    with left:
        st.markdown("### Decision economics · test set")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Review none", "CatBoost policy"],
                             y=[base, cost], marker_color=["#526e83", TEAL],
                             text=[money(base), money(cost)], textposition="outside", textfont_color=INK))
        fig.update_yaxes(title_text="Illustrative cost (MAD)", range=[0,base*1.2])
        st.plotly_chart(style_chart(fig,320), width="stretch", config={"displayModeBar":False})
        st.caption("Fixed costs: FP = 100 MAD, FN = 1,000 MAD. Hypothetical accounting, not realized bank savings.")
    with right:
        st.markdown("### What the model caught")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=["Fraud applications"],x=[HOLDOUT["tp"]],name="Detected",orientation="h",marker_color=TEAL,
                             text=["567 detected"],textposition="inside"))
        fig.add_trace(go.Bar(y=["Fraud applications"],x=[HOLDOUT["fn"]],name="Missed",orientation="h",marker_color=RED,
                             text=["861 missed"],textposition="inside"))
        fig.update_layout(barmode="stack")
        st.plotly_chart(style_chart(fig,320), width="stretch", config={"displayModeBar":False})
        st.caption("Observed against synthetic labels: fraud recall ≈ 39.7%; most frauds are not identified by this policy.")
    st.markdown("### Session activity")
    history = events_df()
    if history.empty:
        st.info("No synthetic events generated in this browser session. Open **Live simulator** and launch a traffic snapshot.")
    else:
        alert_n = int((history["risk"] >= model_threshold).sum())
        a,b,c = st.columns(3)
        a.metric("Simulated applications", f"{len(history):,}")
        b.metric("Review alerts", f"{alert_n:,}")
        c.metric("Mean predicted score", fmt_pct(float(history["risk"].mean()),3))
        fig = go.Figure(go.Scatter(x=history["id"],y=history["risk"]*100,mode="lines+markers",
                                   line={"color":BLUE,"width":2},marker={"size":4}))
        fig.add_hline(y=model_threshold*100, line_dash="dash",line_color=AMBER,
                      annotation_text="Review threshold", annotation_font_color=AMBER)
        fig.update_yaxes(title_text="Score (%)")
        fig.update_xaxes(title_text="Synthetic application number")
        st.plotly_chart(style_chart(fig,290), width="stretch", config={"displayModeBar":False})
        st.caption("Session-only synthetic scores. They do not measure the prevalence of fraud or production performance.")

# ============================================================================
# 2. Live monitoring simulation
# ============================================================================
elif "Live simulator" in nav:
    hero("FRAUDSHIELD / DEMO TRAFFIC", "Live risk stream.",
         "Generate BAF-like requests, score many applications at once and watch actual CatBoost outputs update in a simulated feed.",
         "BROWSER-SESSION SIMULATION · NOT BANK TRANSACTIONS")
    c_mode, c_batch, c_thresh = st.columns([1.4,0.8,1.2])
    with c_mode:
        traffic_mode = st.selectbox("Synthetic scenario", ["Mixed traffic", "Routine traffic", "Risk spotlight (selected)"],
                                    help="Risk spotlight intentionally selects high-scoring cases from a larger synthetic candidate pool. Not representative of fraud prevalence.")
    with c_batch:
        each_tick = st.selectbox("Requests per update", [8, 16, 32], index=1)
    with c_thresh:
        live_threshold_pct = st.slider("Simulation review threshold (%)", 0.5, 25.0, float(round(model_threshold*100,2)), 0.1)
    live_threshold = live_threshold_pct/100.0
    a,b,c,d = st.columns([1,1,1,1])
    with a:
        if st.button("▶ Start stream", type="primary", width="stretch", disabled=st.session_state.fs_live_running):
            st.session_state.fs_live_running = True
            st.rerun()
    with b:
        if st.button("Ⅱ Pause", width="stretch", disabled=not st.session_state.fs_live_running):
            st.session_state.fs_live_running = False
            st.rerun()
    with c:
        if st.button("⚡ Generate 100 now", width="stretch", disabled=st.session_state.fs_event_id>=400):
            count = min(100, 400-st.session_state.fs_event_id)
            with st.spinner(f"Scoring {count} synthetic candidates…"):
                sample = select_demo_candidates(count, st.session_state.fs_seed + st.session_state.fs_tick*1009, traffic_mode)
                add_history(sample, traffic_mode)
                st.session_state.fs_tick += 1
            st.rerun()
    with d:
        if st.button("↺ Reset session", width="stretch"):
            st.session_state.fs_events = []
            st.session_state.fs_tick = 0
            st.session_state.fs_event_id = 0
            st.session_state.fs_live_running = False
            st.rerun()
    st.caption("Running mode refreshes approximately every 3 seconds while the page is open. Stops automatically after 400 events. No background service or persistence.")
    if traffic_mode == "Risk spotlight (selected)":
        st.warning("**Selection bias:** Risk spotlight scores a larger synthetic candidate pool and deliberately displays its highest-scoring profiles. Alert rates here are *not* representative of a bank's fraud rate.")

    @st.fragment(run_every=3 if st.session_state.fs_live_running else None)
    def render_live_feed():
        if st.session_state.fs_live_running:
            if st.session_state.fs_event_id >= 400:
                st.session_state.fs_live_running = False
                st.rerun()
            take = min(each_tick, 400-st.session_state.fs_event_id)
            sample = select_demo_candidates(take, st.session_state.fs_seed + st.session_state.fs_tick*1009, traffic_mode)
            add_history(sample, traffic_mode)
            st.session_state.fs_tick += 1

        history = events_df()
        if history.empty:
            st.info("Select a scenario, then click **Generate 100 now** or **Start stream**.")
            return
        scores = history["risk"].to_numpy(float)
        review = scores >= live_threshold
        a,b,c,d = st.columns(4)
        a.metric("Requests processed", len(history))
        b.metric("Review alerts", int(review.sum()))
        c.metric("Alert share (selected demo)", fmt_pct(float(review.mean())))
        d.metric("Highest model score", fmt_pct(float(scores.max()), 2))
        left, right = st.columns([1.3,0.7], gap="large")
        with left:
            st.markdown("#### Streaming risk timeline")
            colors = [RED if r else TEAL for r in review]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history["id"],y=scores*100, mode="lines+markers",
                                     line={"color":BLUE,"width":1.7}, marker={"color":colors,"size":6},
                                     name="Fraud risk score"))
            fig.add_hline(y=live_threshold*100, line_dash="dash", line_color=AMBER,
                          annotation_text="Review policy",annotation_font_color=AMBER)
            fig.update_xaxes(title_text="Synthetic request #")
            fig.update_yaxes(title_text="Predicted risk (%)")
            st.plotly_chart(style_chart(fig,315), width="stretch", config={"displayModeBar":False})
        with right:
            st.markdown("#### Decision mix")
            fig = go.Figure(go.Pie(labels=["No alert", "Manual review"],
                                   values=[int((~review).sum()),int(review.sum())], hole=0.72,
                                   marker={"colors":[TEAL,RED]}, textinfo="percent",
                                   sort=False))
            fig.update_layout(showlegend=True)
            st.plotly_chart(style_chart(fig,315), width="stretch", config={"displayModeBar":False})
        st.markdown("#### Recent case queue")
        display = history.tail(30).iloc[::-1].copy()
        display["risk"] = (display["risk"]*100).round(3)
        display["decision"] = np.where(display["risk"] >= live_threshold*100, "REVIEW", "NO ALERT")
        display = display.rename(columns={"id":"Case ID", "time":"Time", "risk":"Risk (%)", "decision":"Action",
                                          "housing":"Housing", "credit_score":"Credit score", "channel":"Channel", "age":"Age", "scenario":"Scenario"})
        st.dataframe(display[["Case ID", "Time", "Risk (%)", "Action", "Age", "Credit score", "Housing", "Channel"]],
                     width="stretch", hide_index=True)
        with st.expander("Cost lens · model-score-weighted proxy (not observed FP/FN)", expanded=False):
            ca, cb = st.columns(2)
            with ca: fp_cost = st.number_input("Cost of an unnecessary review (MAD)",min_value=0, max_value=1000000,value=100,step=25,key="live_fp")
            with cb: fn_cost = st.number_input("Cost of a missed fraud (MAD)", min_value=0,max_value=1000000,value=1000,step=100,key="live_fn")
            costs = expected_decision_costs(scores,live_threshold,fp_cost,fn_cost)
            ca,cb,cc = st.columns(3)
            ca.metric("Estimated unnecessary-review cost",money(costs["expected_fp_cost"]))
            cb.metric("Estimated missed-fraud cost",money(costs["expected_fn_cost"]))
            cc.metric("Combined model-based proxy",money(costs["expected_total_cost"]))
            st.caption("Uses model scores as approximate probabilities; simulated cases have no true fraud labels. These are NOT observed false positives/negatives or measured financial savings.")
        st.download_button("↓ Export session scores (synthetic)", history.to_csv(index=False).encode("utf-8"),
                           file_name="fraudshield_synthetic_stream.csv",mime="text/csv")

    render_live_feed()

# ============================================================================
# 3. Case assessment
# ============================================================================
elif "Risk assessment" in nav:
    hero("FRAUDSHIELD / CASE ANALYSIS", "One case. A clear decision.",
         "Start from a generated synthetic profile, adjust the relevant fields and score it with the deployed model.",
         "31 RAW INPUTS → 58 MODEL FEATURES")

    def load_preset(mode: str) -> None:
        st.session_state.fs_seed += 11
        if mode == "routine":
            records = generate_records(70, st.session_state.fs_seed, "routine")
            scores = score_records(records)["risk"].to_numpy()
            chosen = records[int(np.argmin(scores))]
        else:
            records = generate_records(450, st.session_state.fs_seed, "watchlist")
            scores = score_records(records)["risk"].to_numpy()
            chosen = records[int(np.argmax(scores))]
        for k, v in reconcile_categories([chosen])[0].items():
            st.session_state[f"fs_in_{k}"] = v
        st.session_state.fs_preset_label = "Low-score selection" if mode == "routine" else "High-score selection"
        st.session_state.fs_last_eval = None

    if "fs_in_income" not in st.session_state:
        load_preset("routine")
    st.caption("One-click presets choose the lowest/highest **actual model score** from a synthetic candidate pool; they do not assign a fraudulent label.")
    a,b,c = st.columns([1,1,2])
    with a:
        if st.button("○ Generate routine case",width="stretch"):
            load_preset("routine")
            st.rerun()
    with b:
        if st.button("◇ Find high-score case",type="primary",width="stretch"):
            load_preset("watchlist")
            st.rerun()
    with c:
        st.caption(f'Current preset: **{st.session_state.get("fs_preset_label", "Synthetic case")}**. All values remain editable below.')

    key_fields = ["income", "customer_age", "name_email_similarity", "credit_risk_score",
                  "prev_address_months_count", "current_address_months_count", "housing_status", "device_os", "has_other_cards"]
    other_fields = [field for field in raw if field not in key_fields]
    integer_fields = {"prev_address_months_count", "current_address_months_count", "customer_age",
                      "zip_count_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
                      "credit_risk_score", "email_is_free", "phone_home_valid", "phone_mobile_valid",
                      "bank_months_count", "has_other_cards", "foreign_request", "keep_alive_session",
                      "device_distinct_emails_8w", "device_fraud_count", "month"}
    binary_fields = {"email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards",
                     "foreign_request", "keep_alive_session", "device_fraud_count"}
    number_bounds = {
        "name_email_similarity": (0.0, 1.0),
        "prev_address_months_count": (-1, 400), "current_address_months_count": (-1, 400),
        "days_since_request": (0.0, 80.0), "intended_balcon_amount": (-1.0, 115.0),
        "zip_count_4w": (0, 6500), "velocity_6h": (-150.0, 17000.0),
        "velocity_24h": (1000.0, 10000.0), "velocity_4w": (2500.0, 7500.0),
        "bank_branch_count_8w": (0, 2800), "date_of_birth_distinct_emails_4w": (0, 45),
        "credit_risk_score": (-200, 400), "bank_months_count": (-1, 33),
        "proposed_credit_limit": (190, 2000), "session_length_in_minutes": (-1.0, 90.0),
        "device_distinct_emails_8w": (-1, 2),
    }

    def input_field(field: str) -> None:
        label = formatted_label(field)
        key = "fs_in_"+field
        if field in cats:
            options = [str(x) for x in predictor.encoder.categories_[cats.index(field)]]
            if st.session_state[key] not in options:
                st.session_state[key] = options[0]
            st.selectbox(label, options, key=key, help=f"BAF encoded field: {field}")
        elif field == "income":
            st.selectbox(label, [round(x/10, 1) for x in range(1,10)], key=key,help="BAF normalized income category")
        elif field == "customer_age":
            st.selectbox(label, list(range(10,100,10)), key=key,help="BAF age is grouped into 10-year increments")
        elif field == "month":
            st.selectbox(label,list(range(8)),key=key,help="Relative dataset period, not the current calendar month")
        elif field == "device_fraud_count":
            st.selectbox(label,[0],key=key,disabled=True,help="Constant in the exported synthetic BAF train")
        elif field in binary_fields:
            st.selectbox(label, [0,1],key=key,help=f"BAF feature: {field}")
        else:
            is_int = field in integer_fields
            lower, upper = number_bounds.get(field, (None, None))
            if is_int:
                # Streamlit requires value, bounds and step to use the same numeric type.
                st.session_state[key] = int(st.session_state[key])
                st.number_input(
                    label,
                    min_value=int(lower) if lower is not None else None,
                    max_value=int(upper) if upper is not None else None,
                    value=int(st.session_state[key]),
                    step=1, format="%d", key=key,
                    help=f"BAF feature: {field}",
                )
            else:
                st.session_state[key] = float(st.session_state[key])
                st.number_input(
                    label,
                    min_value=float(lower) if lower is not None else None,
                    max_value=float(upper) if upper is not None else None,
                    value=float(st.session_state[key]),
                    step=0.01, format="%.4f", key=key,
                    help=f"BAF feature: {field}",
                )

    with st.form("case_assessment",border=True):
        st.markdown("#### Essential risk signals")
        cols = st.columns(3, gap="medium")
        for i,name in enumerate(key_fields):
            with cols[i%3]: input_field(name)
        with st.expander("Advanced BAF fields · 22 more signals", expanded=False):
            st.caption("These values are already generated; adjust them only if you want full control of the synthetic request.")
            cols=st.columns(3)
            for i,name in enumerate(other_fields):
                with cols[i%3]: input_field(name)
        submitted = st.form_submit_button("Analyze synthetic application →", type="primary", width="stretch")
    if submitted:
        try:
            record = {key: st.session_state["fs_in_"+key] for key in raw}
            prediction = predictor.predict([record])[0]
            st.session_state.fs_last_eval = prediction
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Could not analyze input: {exc}")

    result = st.session_state.get("fs_last_eval")
    if result:
        score = float(result["fraud_probability"])
        status = "MANUAL REVIEW" if score >= model_threshold else "NO ALERT"
        st.markdown("### Assessment result")
        left,right=st.columns([0.95,1.05], gap="large")
        with left:
            st.plotly_chart(single_risk_chart(score,model_threshold), width="stretch",config={"displayModeBar":False})
        with right:
            st.metric("Suggested action", status)
            st.metric("Decision threshold",fmt_pct(model_threshold,3))
            if score >= model_threshold:
                st.warning("Flagged for human review. This is not proof of fraud.")
            else:
                st.success("No alert generated. A negative prediction does not rule out fraud.")
            st.caption("Scores are model outputs, not verified probabilities for real applicants.")
        with st.expander("Raw prediction JSON", expanded=False): st.json(result)

# ============================================================================
# 4. Batch scoring
# ============================================================================
elif "Batch scoring" in nav:
    hero("FRAUDSHIELD / BULK INFERENCE", "A hundred decisions, one upload.",
         "Upload a small synthetic BAF CSV, score up to 1,000 applications, then export results and inspect policy costs.",
         "IN-MEMORY · CSV UPLOAD · LIMITED TO 1,000 ROWS")
    st.caption("Required columns: all 31 original BAF features; optional `fraud_bool` provides ground-truth labels for a retrospective synthetic-data evaluation.")
    uploaded = st.file_uploader("Synthetic BAF CSV", type=["csv"],help="Do not upload any real banking/customer data.")
    if uploaded is not None:
        try:
            if uploaded.size > 4_000_000:
                raise ValueError("CSV exceeds the 4 MB limit")
            content = uploaded.getvalue()
            upload_fingerprint = (uploaded.name, hashlib.sha256(content).hexdigest())
            if st.session_state.get("fs_batch_fingerprint") != upload_fingerprint:
                st.session_state.fs_batch_fingerprint = upload_fingerprint
                st.session_state.fs_batch_result = None
            df = pd.read_csv(io.BytesIO(content))
            if not 1 <= len(df) <= 1000:
                raise ValueError("Please upload 1–1,000 records")
            missing = sorted(set(raw)-set(df.columns))
            if missing:
                raise ValueError(f"Missing raw fields: {missing}")
            st.success(f"Schema check passed · {len(df)} synthetic rows · {len(raw)} required inputs present")
            if st.button("Analyze file →",type="primary"):
                rows = df[raw].to_dict(orient="records")
                predictions=[]
                for j in range(0,len(rows),125):
                    predictions.extend(predictor.predict(rows[j:j+125]))
                output=df.copy()
                scores=np.asarray([p["fraud_probability"] for p in predictions],dtype=float)
                output["fraud_probability"]=scores
                output["decision"]=np.where(scores>=model_threshold,"manual_review","no_alert")
                st.session_state.fs_batch_result=output
                st.session_state.fs_batch_file_name=uploaded.name
        except (ValueError,TypeError,KeyError,pd.errors.ParserError) as exc:
            st.error(str(exc))
    result = st.session_state.get("fs_batch_result")
    if result is not None:
        st.markdown("### File analysis")
        scores = result["fraud_probability"].to_numpy(dtype=float)
        alerted = scores >= model_threshold
        a,b,c,d=st.columns(4)
        a.metric("Applications",len(result))
        b.metric("Manual reviews",int(alerted.sum()))
        c.metric("Alert share",fmt_pct(float(alerted.mean())))
        d.metric("Average model score",fmt_pct(float(scores.mean()),3))
        f=go.Figure(go.Histogram(x=scores*100,nbinsx=25,marker_color=BLUE))
        f.add_vline(x=model_threshold*100,line_dash="dash",line_color=AMBER,annotation_text="Decision threshold")
        f.update_xaxes(title_text="Predicted risk (%)")
        f.update_yaxes(title_text="Applications")
        st.plotly_chart(style_chart(f,290),width="stretch",config={"displayModeBar":False})
        if "fraud_bool" in result:
            try:
                labels = pd.to_numeric(result["fraud_bool"],errors="raise").to_numpy()
                stat = observed_confusion_costs(labels,scores,model_threshold,100,1000)
                st.markdown("#### Ground-truth evaluation · labeled synthetic CSV")
                a,b,c,d=st.columns(4)
                a.metric("True positives",stat["tp"])
                b.metric("False positives",stat["fp"])
                c.metric("False negatives",stat["fn"])
                d.metric("True negatives",stat["tn"])
                st.info(f'At FP = 100 MAD and FN = 1,000 MAD: observed cost on this labeled file = **{money(stat["observed_total_cost"])}**. No-review baseline = **{money(stat["no_alert_baseline"])}**.')
            except (ValueError,TypeError):
                st.warning("fraud_bool exists but is not a fully valid binary label column; no confusion matrix calculated.")
        else:
            st.caption("No fraud_bool labels supplied: false positive/false negative counts cannot be measured from scores alone.")
        display = result[["fraud_probability","decision"]].copy()
        display.insert(0,"case",np.arange(1,len(display)+1))
        display["fraud_probability"]=(display["fraud_probability"]*100).round(3)
        display.columns=["Case", "Predicted score (%)", "Decision"]
        st.dataframe(display.head(200), width="stretch",hide_index=True)
        st.download_button("↓ Download scored CSV",result.to_csv(index=False).encode("utf-8"),
                           file_name="fraudshield_scored.csv",mime="text/csv")
        st.caption("Uploaded content is processed within the current app process and only kept temporarily in this user's session state; do not use personal banking data.")

# ============================================================================
# 5. Economics / explainability
# ============================================================================
else:
    hero("FRAUDSHIELD / DECISION SCIENCE", "The cost of being wrong.",
         "Compare false-alarm friction with undetected fraud cost using the frozen synthetic holdout confusion matrix.",
         "POLICY ANALYSIS · HISTORICAL LAB EVALUATION")
    a,b=st.columns(2)
    with a:
        cfp=st.number_input("Cost of a false positive · unnecessary review (MAD)",min_value=0,max_value=1000000,value=100,step=25)
    with b:
        cfn=st.number_input("Cost of a false negative · missed fraud (MAD)",min_value=0,max_value=10000000,value=1000,step=100)
    policy_cost=HOLDOUT["fp"]*cfp+HOLDOUT["fn"]*cfn
    baseline_cost=(HOLDOUT["tp"]+HOLDOUT["fn"])*cfn
    gain=baseline_cost-policy_cost
    r1,r2,r3=st.columns(3)
    with r1: stat_card("Policy cost",money(policy_cost),"1,900 FP + 861 FN (frozen holdout)")
    with r2: stat_card("No-alert policy",money(baseline_cost),"1,428 missed synthetic frauds")
    with r3: stat_card("Illustrative difference",money(gain),"Positive means policy is cheaper")
    st.caption("Costs above are hypothetical and applied to a **fixed** test-set confusion matrix. Changing costs here does not re-optimize the threshold or retrain CatBoost.")
    left,right=st.columns([1.0,1.0],gap="large")
    with left:
        st.markdown("### Confusion matrix · BAF test month 7")
        z=[[HOLDOUT["tn"],HOLDOUT["fp"]],[HOLDOUT["fn"],HOLDOUT["tp"]]]
        fig=go.Figure(go.Heatmap(z=z,x=["No alert","Review"],y=["Actual legitimate","Actual fraud"],
                                 colorscale=[[0,"#142a3a"],[1,"#37ddb8"]],showscale=False,
                                 text=[[f"{n:,}" for n in row] for row in z],texttemplate="%{text}",
                                 textfont={"size":17,"color":INK}, hovertemplate="%{x} / %{y}: %{z:,}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_chart(fig,330),width="stretch",config={"displayModeBar":False})
    with right:
        st.markdown("### Relative cost components")
        fig=go.Figure()
        fig.add_trace(go.Bar(x=["Unnecessary review", "Missed fraud"],
                             y=[HOLDOUT["fp"]*cfp,HOLDOUT["fn"]*cfn],
                             marker_color=[BLUE,RED], text=[money(HOLDOUT["fp"]*cfp),money(HOLDOUT["fn"]*cfn)],
                             textposition="outside",textfont_color=INK))
        fig.update_yaxes(title_text="Hypothetical cost (MAD)")
        st.plotly_chart(style_chart(fig,330),width="stretch",config={"displayModeBar":False})
    st.markdown("### What powers the predictions")
    left,right=st.columns([1.1,0.9],gap="large")
    with left:
        # Stored notebook importance, not causal effects; no recalculation required at deployment.
        importance=[("Housing status · BA",9.09),("Device OS · Windows",9.04),("Has other cards",7.87),
                    ("Previous address missing",7.18),("Current address months",7.09),
                    ("Home phone valid",6.16),("Keep-alive session",5.05),("Name/email similarity",5.03)]
        arr=pd.DataFrame(importance,columns=["Feature","Importance"]).sort_values("Importance")
        fig=go.Figure(go.Bar(y=arr["Feature"],x=arr["Importance"],orientation="h",marker_color=TEAL))
        fig.update_xaxes(title_text="CatBoost feature importance (not causal)")
        st.plotly_chart(style_chart(fig,345),width="stretch",config={"displayModeBar":False})
    with right:
        st.markdown("**Pipeline integrity**")
        st.markdown("- 31 raw fields → 58 engineered features\n- Four fitted imputers/scalers and one OneHotEncoder\n- CatBoost V2: 330 trees\n- Frozen decision threshold: ~6.03%\n- Time-split test: dataset month 7")
        st.warning("The `month` feature is dataset-relative. This prototype is not validated for future calendar periods, live customers or banking decisions.")
        with st.expander("Inspect schema and deployment policy",expanded=False):
            st.json({"raw_features":schema["raw_features"],"model_features":schema["model_features"],"policy":predictor.policy})
    st.caption("Data provenance: BAF synthetic dataset; all historical metrics and importances come from the user's evaluation notebook. No causal interpretation or commercial financial claim.")

st.divider()
st.caption("FRAUDSHIELD  ·  RESEARCH / PORTFOLIO DEMO  ·  Synthetic applications only  ·  No real-time bank feed  ·  No automated adverse decisions")
