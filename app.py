"""
Tumour classifier bench: score five saved models on an uploaded test CSV.

The models are loaded from model/ and only used for prediction; nothing is retrained here.
Run locally with:  streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from utils import load_metadata, load_models, score_model, validate_upload

ROOT = Path(__file__).resolve().parent

# Palette taken from H&E histology staining: hematoxylin (nuclei) and eosin (cytoplasm).
HEMATOXYLIN = "#3B2E5A"
EOSIN = "#C9577A"
SLATE = "#6F86A6"
SLIDE = "#FAF8FB"
INK = "#221C2E"

st.set_page_config(page_title="Tumour classifier bench", page_icon="🔬", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Spectral:wght@500;700&family=Work+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Work Sans', sans-serif; }}
    h1, h2, h3 {{ font-family: 'Spectral', serif !important; color: {HEMATOXYLIN}; letter-spacing: -0.01em; }}
    h1 {{ font-size: 2.4rem !important; line-height: 1.15 !important; }}
    .lede {{ max-width: 68ch; color: {INK}; opacity: .8; font-size: 1.02rem; margin-top: -.4rem; }}
    div[data-testid="stMetric"] {{
        background: white; border-left: 4px solid {HEMATOXYLIN};
        padding: .7rem .9rem; border-radius: 2px 8px 8px 2px;
    }}
    div[data-testid="stMetricLabel"] p {{ color: {SLATE}; font-weight: 600; }}
    div[data-testid="stMetricValue"] {{ font-family: 'Spectral', serif; color: {HEMATOXYLIN}; }}
    .stTabs [aria-selected="true"] {{ color: {EOSIN} !important; }}
    .note {{ font-size: .88rem; color: {SLATE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifacts():
    meta = load_metadata()
    return meta, load_models(meta)


meta, models = get_artifacts()

# ---------------------------------------------------------------- header
st.title("Tumour classifier bench")
st.markdown(
    "<p class='lede'>Upload held-out samples from the Breast Cancer Wisconsin (Diagnostic) "
    "dataset and see how five pre-trained classifiers separate malignant from benign "
    "fine-needle aspirates. Models are only used for prediction here, never retrained.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Test data")
    uploaded = st.file_uploader("Upload a test CSV", type=["csv"])
    use_bundled = False
    if uploaded is None:
        use_bundled = st.toggle("No file? Use the bundled test_data.csv")
    st.download_button(
        "Download test_data.csv",
        data=(ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        help="The 20% split held out during training. Upload it back to reproduce the README numbers.",
    )
    st.header("Model")
    model_name = st.selectbox("Classifier to inspect", list(models.keys()))
    st.markdown(
        f"<p class='note'>Positive class: malignant (M). The file needs the "
        f"{len(meta['features'])} feature columns and a <code>{meta['target']}</code> column.</p>",
        unsafe_allow_html=True,
    )

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source = uploaded.name
elif use_bundled:
    raw = pd.read_csv(ROOT / "test_data.csv")
    source = "test_data.csv (bundled)"
else:
    st.info("Upload a test CSV in the sidebar, or switch on the bundled test data to start.")
    st.stop()

try:
    X, y = validate_upload(raw, meta)
except ValueError as err:
    st.error(f"This file can't be scored. {err}")
    st.stop()

st.caption(
    f"Scoring {len(X)} rows from {source}: {int(y.sum())} malignant, {int((1 - y).sum())} benign."
)


@st.cache_data(show_spinner=False)
def score_all(X_df: pd.DataFrame, y_s: pd.Series) -> dict:
    return {name: score_model(m, X_df, y_s) for name, m in models.items()}


results = score_all(X, y)
chosen = results[model_name]

tab_model, tab_compare, tab_rows = st.tabs(
    [f"{model_name}", "Compare all five", "Row-level predictions"]
)

# ---------------------------------------------------------------- selected model
with tab_model:
    cols = st.columns(6)
    for col, (label, value) in zip(cols, chosen["metrics"].items()):
        col.metric(label, "n/a" if np.isnan(value) else f"{value:.4f}")

    params = meta["best_params"].get(model_name, {})
    if params:
        pretty = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in params.items())
        st.markdown(f"<p class='note'>Tuned hyper-parameters: {pretty}</p>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Confusion matrix")
        cm = chosen["cm"]
        cmap = LinearSegmentedColormap.from_list("stain", [SLIDE, "#B9A9D6", HEMATOXYLIN])
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.imshow(cm, cmap=cmap)
        names = ["Benign", "Malignant"]
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names)
        ax.set_xlabel("Predicted", color=INK)
        ax.set_ylabel("Actual", color=INK)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18,
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"<p class='note'>{fn} malignant case(s) were called benign (missed), "
            f"and {fp} benign case(s) were flagged as malignant.</p>",
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Classification report")
        report = chosen["report"].copy()
        report["support"] = report["support"].astype(int)
        st.dataframe(report.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                          "f1-score": "{:.4f}"}), width="stretch")

        if chosen["roc"] is not None:
            fpr, tpr, _ = chosen["roc"]
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.plot([0, 1], [0, 1], ls=":", color=SLATE, lw=1)
            ax2.plot(fpr, tpr, color=EOSIN, lw=2.2)
            ax2.fill_between(fpr, tpr, alpha=.12, color=EOSIN)
            ax2.set_xlabel("False positive rate")
            ax2.set_ylabel("True positive rate")
            ax2.set_title(f"ROC curve, AUC {chosen['metrics']['AUC']:.4f}", fontsize=10, color=INK)
            for s in ("top", "right"):
                ax2.spines[s].set_visible(False)
            fig2.tight_layout()
            st.pyplot(fig2)

# ---------------------------------------------------------------- comparison
with tab_compare:
    table = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T
    table.index.name = "ML Model Name"
    st.subheader("All models on this file")
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="#EBD3DC"),
        width="stretch",
    )
    st.markdown("<p class='note'>Shaded cells mark the best score in each column.</p>",
                unsafe_allow_html=True)

    metric = st.radio("Metric to chart", list(table.columns), horizontal=True, index=5)
    ordered = table[metric].sort_values()
    fig3, ax3 = plt.subplots(figsize=(7, 2.8))
    colors = [EOSIN if n == model_name else "#B9A9D6" for n in ordered.index]
    ax3.barh(ordered.index, ordered.values, color=colors)
    lo = max(0, ordered.min() - 0.05)
    ax3.set_xlim(lo, 1.0)
    for i, v in enumerate(ordered.values):
        ax3.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    ax3.set_xlabel(metric)
    fig3.tight_layout()
    st.pyplot(fig3)

    st.download_button("Download this table as CSV", table.round(4).to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

# ---------------------------------------------------------------- rows
with tab_rows:
    out = pd.DataFrame({
        "actual": np.where(y == 1, "M", "B"),
        "predicted": np.where(chosen["y_pred"] == 1, "M", "B"),
        "score_malignant": np.round(chosen["y_score"], 4),
    }, index=X.index)
    out["correct"] = out["actual"] == out["predicted"]
    only_wrong = st.checkbox("Show misclassified rows only")
    view = out[~out["correct"]] if only_wrong else out
    st.dataframe(pd.concat([view, X.loc[view.index]], axis=1), width="stretch", height=380)
    st.markdown(
        "<p class='note'>score_malignant is a probability for most models and the SVM "
        "decision-function value for the Support Vector Machine.</p>",
        unsafe_allow_html=True,
    )
