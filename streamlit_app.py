"""FraudShield — local CatBoost inference on synthetic BAF records.
Deploy as streamlit_app.py from the existing fraudshield-ml GitHub repo.
No external API, credentials, or user data storage is needed.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from api.inference import FraudPredictor

st.set_page_config(page_title="FraudShield | Fraud Risk Intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b1020; color: #f1f5f9; }
[data-testid="stSidebar"] { background: #10182b; }
h1,h2,h3 { letter-spacing: -.035em; }
[data-testid="stMetric"] { background: #172238; padding: 14px; border: 1px solid #24324b; border-radius: 12px; }
.fs-hero {background:linear-gradient(110deg,#152b38,#142142 65%,#1a1b3e); border:1px solid #2b4259; border-radius:18px; padding:26px; margin:12px 0 20px;}
.fs-eyebrow { color:#79e1c0; font-weight:700; letter-spacing:.12em; font-size:12px; }
.fs-muted { color:#a5b4cc; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading FraudShield inference artifacts…")
def get_predictor() -> FraudPredictor:
    return FraudPredictor(Path(__file__).resolve().parent / "artifacts")

try:
    predictor = get_predictor()
except Exception as exc:
    st.error("The model could not be loaded. Ensure the trusted Kaggle artifacts are in the artifacts/ directory and versions match training.")
    st.exception(exc)
    st.stop()

schema = predictor.schema
raw = schema["raw_features"]
cat_cols = schema["categorical_columns"]
numeric_cols = [name for name in raw if name not in cat_cols]
threshold = float(predictor.policy["threshold"])

st.sidebar.markdown("## 🛡️ FraudShield")
st.sidebar.caption("FRAUD RISK INTELLIGENCE · DEMO")
page = st.sidebar.radio("Workspace", ["Risk assessment", "Batch analysis", "Model overview"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Model: " + str(predictor.policy.get("model_version", "v2")))
st.sidebar.caption("Synthetic BAF data only · Not for real financial decisions")

st.markdown("""<div class="fs-hero"><div class="fs-eyebrow">FRAUDSHIELD / RESEARCH DEMONSTRATION</div>
<h1 style="margin:8px 0">Risk intelligence, clearly explained.</h1>
<div class="fs-muted">CatBoost inference with the exact saved preprocessing pipeline. Scoring runs within this Streamlit app, without an external API.</div></div>""", unsafe_allow_html=True)

if page == "Risk assessment":
    st.subheader("Single application assessment")
    st.caption("Enter a complete synthetic BAF record. All numerical defaults are demonstration placeholders, not realistic banking profiles.")
    with st.form("assessment"):
        st.markdown("#### Applicant & financial data")
        record = {}
        groups = [numeric_cols[:9], numeric_cols[9:18], numeric_cols[18:]]
        for idx, group in enumerate(groups):
            with st.expander(["Identity and finances", "Activity signals", "Additional signals"][idx], expanded=(idx==0)):
                left, right = st.columns(2)
                for i, key in enumerate(group):
                    host = left if i % 2 == 0 else right
                    with host:
                        if key in ("month", "device_fraud_count"):
                            st.caption(f"{key}: dataset-specific feature")
                        record[key] = st.number_input(key, value=0.0, step=1.0, format="%.3f", key="field_"+key)
        with st.expander("Categorical information", expanded=True):
            left, right = st.columns(2)
            for i, key in enumerate(cat_cols):
                choices = [str(v) for v in predictor.encoder.categories_[cat_cols.index(key)]]
                with (left if i % 2 == 0 else right):
                    record[key] = st.selectbox(key, choices, key="field_"+key)
        submitted = st.form_submit_button("Analyze synthetic application", type="primary", use_container_width=True)
    if submitted:
        try:
            prediction = predictor.predict([record])[0]
        except (ValueError, TypeError, KeyError) as err:
            st.error(f"Input validation failed: {err}")
        else:
            score = prediction["fraud_probability"]
            c1,c2,c3 = st.columns(3)
            c1.metric("Predicted risk", f"{score*100:.3f}%")
            c2.metric("Alert threshold", f"{threshold*100:.3f}%")
            c3.metric("Decision", "REVIEW" if prediction["decision"]=="manual_review" else "NO ALERT")
            st.progress(min(1.0, score), text="Predicted probability (not a calibrated guarantee)")
            if prediction["decision"]=="manual_review":
                st.warning("The model score exceeds the configured review threshold. This is not proof of fraud.")
            else:
                st.success("No manual-review alert triggered for this synthetic example. Fraud is not ruled out.")
            st.json(prediction)

elif page == "Batch analysis":
    st.subheader("Batch risk assessment")
    st.caption("Upload a CSV containing all 31 original raw BAF input columns. Data is processed in memory and is not persisted by this app.")
    file = st.file_uploader("Synthetic records CSV", type=["csv"])
    if file is not None:
        if file.size > 5_000_000:
            st.error("The upload must be 5 MB or smaller.")
        else:
            try:
                df = pd.read_csv(io.BytesIO(file.getvalue()))
                if len(df) > 1000:
                    raise ValueError("For this demo, submit at most 1,000 records per upload")
                if len(df) == 0:
                    raise ValueError("The CSV is empty")
                missing = sorted(set(raw)-set(df.columns))
                if missing:
                    raise ValueError(f"Missing columns: {missing}")
                # Drop unneeded fields such as target fraud_bool; only the required raw columns enter the model.
                if st.button("Run batch prediction", type="primary"):
                    results = []
                    for start in range(0, len(df), 100):
                        records = df[raw].iloc[start:start+100].to_dict("records")
                        results.extend(predictor.predict(records))
                    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Applications", len(result_df))
                    c2.metric("Manual-review alerts", int((result_df["decision"]=="manual_review").sum()))
                    c3.metric("Average score", f'{result_df["fraud_probability"].mean()*100:.3f}%')
                    st.dataframe(result_df[["fraud_probability", "decision"]].head(100), use_container_width=True)
                    st.download_button("Download scored CSV", result_df.to_csv(index=False).encode("utf-8"),
                                       "fraudshield_scored.csv", "text/csv")
            except (ValueError, TypeError, KeyError, pd.errors.ParserError) as err:
                st.error(str(err))

else:
    st.subheader("Model overview")
    a,b,c = st.columns(3)
    a.metric("Input variables", len(raw))
    b.metric("Engineered features", len(schema["model_features"]))
    c.metric("Policy threshold", f"{threshold*100:.3f}%")
    st.markdown("**Inference**: CatBoostClassifier, stored as a native .cbm model; five saved preprocessing transformers. Models are not refitted during requests.")
    st.info("Dataset-specific variables such as 'month' limit generalization. This is a synthetic-data portfolio demonstrator, not a production credit or fraud system.")
    with st.expander("Raw feature schema"):
        st.write(raw)
    with st.expander("Decision policy"):
        st.json(predictor.policy)
