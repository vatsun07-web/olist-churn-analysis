import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Churn Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme / CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background-color: #0F1923; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #131E2B;
    border-right: 1px solid #1E3048;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #8DA4BE;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 4px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #131E2B 0%, #1A2840 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label {
    color: #5B8DB8 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E8A838 !important;
    font-size: 28px !important;
    font-weight: 700;
}

/* Section headers */
.section-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 700;
    color: #2E86AB;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1E3048;
}

/* Page title */
.page-title {
    font-size: 32px;
    font-weight: 700;
    color: #F0F6FF;
    line-height: 1.1;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 14px;
    color: #5B8DB8;
    margin-bottom: 28px;
    font-weight: 400;
}

/* Risk tier badges */
.badge-high   { background:#3D1515; color:#FF6B6B; border:1px solid #FF6B6B33; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-medium { background:#3D2B0A; color:#E8A838; border:1px solid #E8A83833; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-low    { background:#0A2B1E; color:#4CAF8A; border:1px solid #4CAF8A33; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }

/* Insight card */
.insight-card {
    background: linear-gradient(135deg, #131E2B, #1A2840);
    border: 1px solid #1E3A5F;
    border-left: 3px solid #2E86AB;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 13px;
    color: #A8C4E0;
    line-height: 1.6;
}
.insight-card strong { color: #E8A838; }

/* Plotly chart background override */
.js-plotly-plot { border-radius: 12px; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #131E2B;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1E3048;
}
.stTabs [data-baseweb="tab"] {
    color: #5B8DB8;
    font-size: 13px;
    font-weight: 500;
    border-radius: 6px;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #1E3A5F !important;
    color: #F0F6FF !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ────────────────────────────────────────────────────────────
PLOT_BG    = "#0F1923"
PAPER_BG   = "#131E2B"
GRID_COLOR = "#1E3048"
TEXT_COLOR = "#8DA4BE"
TEAL       = "#2E86AB"
GOLD       = "#E8A838"
RED        = "#FF6B6B"
GREEN      = "#4CAF8A"
NAVY       = "#1B2A4A"

def base_layout(title="", height=380):
    return dict(
        title=dict(text=title, font=dict(color="#F0F6FF", size=14, family="DM Sans"), x=0.02),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COLOR, family="DM Sans", size=12),
        height=height,
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    )

# ─── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    risk     = pd.read_parquet("outputs/06_risk_segments.parquet")
    shap     = pd.read_parquet("outputs/06_shap_values.parquet")
    preds    = pd.read_parquet("outputs/05_predictions.parquet")
    models   = pd.read_parquet("outputs/05_model_comparison.parquet")
    eda      = pd.read_parquet("outputs/03_eda_outputs.parquet")
    master   = pd.read_parquet("outputs/02_master_table.parquet")
    return risk, shap, preds, models, eda, master

risk, shap, preds, models, eda, master = load_data()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 28px 0;'>
        <div style='font-size:22px; font-weight:700; color:#F0F6FF; line-height:1.2;'>Olist Churn<br>Intelligence</div>
        <div style='font-size:11px; color:#2E86AB; margin-top:6px; letter-spacing:2px; text-transform:uppercase;'>Portfolio Project 4</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p>Navigation</p>", unsafe_allow_html=True)
    page = st.radio("", [
        "📊  Overview",
        "🔍  Customer Explorer",
        "🤖  Model Performance",
        "💡  SHAP Drivers",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<p>Risk Tier Filter</p>", unsafe_allow_html=True)
    tier_filter = st.multiselect(
        "", options=["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#2E86AB; line-height:1.8;'>
        <div>Dataset: Olist Brazilian E-Commerce</div>
        <div>Eligible customers: 74,899</div>
        <div>Test set: 13,203</div>
        <div>Model: Tuned Logistic Regression</div>
        <div style='margin-top:8px;'>
            <a href='https://github.com/vatsun07-web/olist-churn-analysis'
               style='color:#2E86AB; text-decoration:none;'>GitHub →</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Filtered data ───────────────────────────────────────────────────────────
risk_f = risk[risk["risk_tier"].isin(tier_filter)] if tier_filter else risk

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "📊  Overview":

    st.markdown('<div class="page-title">Business Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Customer retention landscape across 74,899 eligible Olist customers</div>', unsafe_allow_html=True)

    # ── KPI row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Eligible Customers", "74,899")
    k2.metric("Base Churn Rate", "97.82%")
    k3.metric("Retention Rate", "2.18%")
    k4.metric("Repeat Buyers", "2,801")
    k5.metric("Model ROC-AUC", "0.626")

    st.markdown('<div class="section-header">Churn Rate by Dimension</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "State", "Category", "Payment", "Delay", "Item Count", "Time"
    ])

    # ── State ──
    with tab1:
        state_df = eda[eda["breakout"] == "customer_state"].copy()
        state_df = state_df.sort_values("churn_rate")
        fig = go.Figure(go.Bar(
            x=state_df["churn_rate"],
            y=state_df["group_value"],
            orientation="h",
            marker=dict(
                color=state_df["churn_rate"],
                colorscale=[[0, GREEN], [0.5, TEAL], [1, RED]],
                showscale=True,
                colorbar=dict(title="Churn %", tickfont=dict(color=TEXT_COLOR))
            ),
            text=[f"{v:.1f}%" for v in state_df["churn_rate"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            customdata=state_df["n"],
            hovertemplate="<b>%{y}</b><br>Churn rate: %{x:.2f}%<br>n = %{customdata:,}<extra></extra>"
        ))
        layout = base_layout("Churn Rate by Customer State (%)", height=580)
        layout["xaxis"]["range"] = [95, 101]
        layout["xaxis"]["title"] = "Churn Rate (%)"
        layout["yaxis"]["title"] = ""
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-card"><strong>Lowest churn:</strong> MT (96.77%), RO (96.94%), AC (96.97%) &nbsp;·&nbsp; <strong>Highest churn:</strong> RR (100%), AM (99.10%), SE (98.82%)<br>Largest state SP shows 97.71% on 30,596 customers — the reliable benchmark.</div>', unsafe_allow_html=True)

    # ── Category ──
    with tab2:
        cat_df = eda[eda["breakout"] == "product_category_name_english"].copy()
        cat_df = cat_df.sort_values("churn_rate").tail(15)
        fig = go.Figure(go.Bar(
            x=cat_df["churn_rate"],
            y=cat_df["group_value"],
            orientation="h",
            marker=dict(color=TEAL, opacity=0.85),
            text=[f"{v:.2f}%" for v in cat_df["churn_rate"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            customdata=cat_df["n"],
            hovertemplate="<b>%{y}</b><br>Churn: %{x:.2f}%<br>n = %{customdata:,}<extra></extra>"
        ))
        layout = base_layout("Churn Rate by Product Category — Top 15 (%)", height=460)
        layout["xaxis"]["range"] = [95, 100]
        layout["xaxis"]["title"] = "Churn Rate (%)"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-card"><strong>2.73pp spread</strong> — the largest single categorical signal found. <strong>bed_bath_table</strong> retains best (96.21%); <strong>cool_stuff</strong> churns hardest (98.94%). Lifestyle and home goods create repeat purchase intent.</div>', unsafe_allow_html=True)

    # ── Payment ──
    with tab3:
        pay_df = eda[eda["breakout"] == "primary_payment_type"].copy()
        pay_df = pay_df.sort_values("churn_rate")
        colors = [GREEN if r < 97 else TEAL if r < 98 else RED for r in pay_df["churn_rate"]]
        fig = go.Figure(go.Bar(
            x=pay_df["group_value"],
            y=pay_df["churn_rate"],
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.2f}%" for v in pay_df["churn_rate"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=11),
            customdata=pay_df["n"],
            hovertemplate="<b>%{x}</b><br>Churn: %{y:.2f}%<br>n = %{customdata:,}<extra></extra>"
        ))
        layout = base_layout("Churn Rate by Payment Method (%)", height=360)
        layout["yaxis"]["range"] = [94, 99.5]
        layout["yaxis"]["title"] = "Churn Rate (%)"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-card"><strong>Voucher users churn least</strong> at 95.65% — counter-intuitive but consistent. Voucher usage signals engaged, deal-seeking shoppers who return. Debit card users churn hardest at 97.94%.</div>', unsafe_allow_html=True)

    # ── Delay ──
    with tab4:
        delay_df = eda[eda["breakout"] == "delay_bucket"].copy()
        order_map = {"Early": 0, "On-time (0-2d)": 1, "Late (3-7d)": 2, "Very late (>7d)": 3}
        delay_df["order"] = delay_df["group_value"].map(order_map)
        delay_df = delay_df.sort_values("order")
        colors_d = [GREEN, TEAL, GOLD, RED]
        fig = go.Figure(go.Bar(
            x=delay_df["group_value"],
            y=delay_df["churn_rate"],
            marker=dict(color=colors_d, opacity=0.85),
            text=[f"{v:.2f}%" for v in delay_df["churn_rate"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=11),
            customdata=delay_df["n"],
            hovertemplate="<b>%{x}</b><br>Churn: %{y:.2f}%<br>n = %{customdata:,}<extra></extra>"
        ))
        layout = base_layout("Churn Rate by Delivery Delay Bucket (%)", height=360)
        layout["yaxis"]["range"] = [96.5, 99]
        layout["yaxis"]["title"] = "Churn Rate (%)"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-card"><strong>91.2% of orders arrived early.</strong> Non-monotonic pattern — on-time (0–2d) customers churn least (97.48%); very late (>7d) churn most (98.24%). Delivery delay alone is not a strong predictor, but late delivery is a compounding risk factor.</div>', unsafe_allow_html=True)

    # ── Item count ──
    with tab5:
        item_df = eda[eda["breakout"] == "item_count"].copy()
        item_df["group_value"] = item_df["group_value"].astype(str)
        fig = go.Figure(go.Bar(
            x=item_df["group_value"],
            y=item_df["churn_rate"],
            marker=dict(
                color=item_df["churn_rate"],
                colorscale=[[0, GREEN], [1, RED]],
                reversescale=True,
                showscale=False
            ),
            text=[f"{v:.2f}%" for v in item_df["churn_rate"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=11),
            customdata=item_df["n"],
            hovertemplate="<b>%{x} items</b><br>Churn: %{y:.2f}%<br>n = %{customdata:,}<extra></extra>"
        ))
        layout = base_layout("Churn Rate by Order Size (Item Count)", height=360)
        layout["yaxis"]["range"] = [94, 99]
        layout["xaxis"]["title"] = "Item Count"
        layout["yaxis"]["title"] = "Churn Rate (%)"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-card"><strong>Item count is the strongest univariate signal</strong> (r_pb = −0.0214, p &lt; 0.001). Single-item orders: 97.92% churn. 3+ item orders: ~96%. More items in a basket = stronger purchase intent = more likely to return.</div>', unsafe_allow_html=True)

    # ── Time ──
    with tab6:
        time_df = eda[eda["breakout"] == "purchase_quarter"].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_df["group_value"],
            y=time_df["churn_rate"],
            mode="lines+markers",
            line=dict(color=TEAL, width=2.5),
            marker=dict(color=GOLD, size=8, line=dict(color=TEAL, width=2)),
            fill="tozeroy",
            fillcolor="rgba(46,134,171,0.08)",
            customdata=time_df["n"],
            hovertemplate="<b>%{x}</b><br>Churn: %{y:.2f}%<br>n = %{customdata:,}<extra></extra>"
        ))
        layout = base_layout("Churn Rate by Purchase Quarter", height=360)
        layout["yaxis"]["title"] = "Churn Rate (%)"
        layout["xaxis"]["title"] = "Quarter"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-card">2016Q4 shows highest churn (98.81%) on only 253 customers — sparse cohort, noisy estimate. Stable period 2017Q2–2018Q1 shows churn between 97.53%–97.88%, confirming the structural ceiling.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Customer Explorer":

    st.markdown('<div class="page-title">Customer Risk Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Explore individual churn probability scores and risk profiles across 13,203 test customers</div>', unsafe_allow_html=True)

    # ── KPIs ──
    total = len(risk_f)
    high  = len(risk_f[risk_f["risk_tier"] == "High Risk"])
    med   = len(risk_f[risk_f["risk_tier"] == "Medium Risk"])
    low   = len(risk_f[risk_f["risk_tier"] == "Low Risk"])
    actual_churn = risk_f["churn"].mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers Shown", f"{total:,}")
    k2.metric("High Risk", f"{high:,}")
    k3.metric("Medium Risk", f"{med:,}")
    k4.metric("Low Risk", f"{low:,}")
    k5.metric("Actual Churn Rate", f"{actual_churn:.1f}%")

    st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        tier_counts = risk_f["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        color_map = {"High Risk": RED, "Medium Risk": GOLD, "Low Risk": GREEN}
        fig = go.Figure(go.Bar(
            x=tier_counts["tier"],
            y=tier_counts["count"],
            marker=dict(color=[color_map.get(t, TEAL) for t in tier_counts["tier"]], opacity=0.85),
            text=tier_counts["count"],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR),
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>"
        ))
        fig.update_layout(**base_layout("Customer Count by Risk Tier", height=320))
        fig.update_layout(yaxis_title="Customers", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tier_churn = risk_f.groupby("risk_tier")["churn"].mean().reset_index()
        tier_churn["churn_pct"] = tier_churn["churn"] * 100
        fig = go.Figure(go.Bar(
            x=tier_churn["risk_tier"],
            y=tier_churn["churn_pct"],
            marker=dict(color=[color_map.get(t, TEAL) for t in tier_churn["risk_tier"]], opacity=0.85),
            text=[f"{v:.1f}%" for v in tier_churn["churn_pct"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR),
            hovertemplate="<b>%{x}</b><br>Actual churn: %{y:.2f}%<extra></extra>"
        ))
        fig.update_layout(**base_layout("Actual Churn Rate by Risk Tier", height=320))
        fig.update_layout(yaxis_title="Churn Rate (%)", yaxis_range=[95, 101], xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # ── Probability histogram ──
    st.markdown('<div class="section-header">Churn Probability Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for tier, color in [("High Risk", RED), ("Medium Risk", GOLD), ("Low Risk", GREEN)]:
        subset = risk_f[risk_f["risk_tier"] == tier]
        fig.add_trace(go.Histogram(
            x=subset["churn_proba"],
            name=tier,
            marker_color=color,
            opacity=0.7,
            nbinsx=40,
            hovertemplate=f"<b>{tier}</b><br>Probability: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>"
        ))
    layout = base_layout("Churn Probability Score Distribution by Risk Tier", height=340)
    layout["barmode"] = "overlay"
    layout["xaxis"]["title"] = "Churn Probability"
    layout["yaxis"]["title"] = "Customer Count"
    layout["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Customer lookup ──
    st.markdown('<div class="section-header">Individual Customer Lookup</div>', unsafe_allow_html=True)

    merged = risk_f.merge(
        master[["customer_unique_id", "customer_state", "product_category_name_english",
                "avg_item_price", "review_score", "primary_payment_type", "max_installments"]],
        on="customer_unique_id", how="left"
    )

    col_search, col_tier = st.columns([2, 1])
    with col_tier:
        tier_sel = st.selectbox("Filter by tier", ["All"] + ["High Risk", "Medium Risk", "Low Risk"])
    with col_search:
        search_pool = merged if tier_sel == "All" else merged[merged["risk_tier"] == tier_sel]
        cid = st.selectbox(
            "Select Customer ID",
            options=search_pool["customer_unique_id"].values[:200],
            format_func=lambda x: x[:20] + "..."
        )

    if cid:
        row = merged[merged["customer_unique_id"] == cid].iloc[0]
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Churn Probability", f"{row['churn_proba']:.3f}")
        r2.metric("Risk Tier", row["risk_tier"])
        r3.metric("Actual Outcome", "Churned ✗" if row["churn"] == 1 else "Retained ✓")
        r4.metric("State", str(row.get("customer_state", "—")))
        r5.metric("Review Score", f"{row.get('review_score', 0):.1f} ★" if pd.notna(row.get("review_score")) else "—")

        d1, d2, d3 = st.columns(3)
        d1.metric("Category", str(row.get("product_category_name_english", "—")))
        d2.metric("Avg Item Price", f"R$ {row.get('avg_item_price', 0):.2f}")
        d3.metric("Payment Method", str(row.get("primary_payment_type", "—")))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":

    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Comparison of Logistic Regression, Random Forest, and LightGBM — time-based train/test split</div>', unsafe_allow_html=True)

    # ── Model comparison table ──
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    display_models = models.copy()
    display_models["tuned"] = display_models["tuned"].map({True: "✓ Tuned", False: "—"})
    display_models.columns = ["Model", "ROC-AUC", "PR-AUC", "F1 (Churn)", "F1 (Retained)", "Recall (Retained)", "Precision (Retained)", "Tuned"]
    display_models = display_models.round(4)

    def highlight_winner(row):
        if "Logistic" in str(row["Model"]) and row["Tuned"] == "✓ Tuned":
            return [f"background-color: #1E3A5F; color: #E8A838"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_models.style.apply(highlight_winner, axis=1),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="insight-card"><strong>Logistic Regression wins</strong> with ROC-AUC 0.626 — higher than Random Forest (0.584) and LightGBM (0.599) despite being the simplest model. This tells us the signal in this dataset is largely linear. The AUC ceiling (≈0.65 theoretical max given r_max = 0.07) is a data structure issue, not a model failure.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">ROC-AUC vs PR-AUC Comparison</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        model_names = models["model"].tolist()
        roc_vals = models["roc_auc"].tolist()
        colors_m = [GOLD if (m == "Logistic Regression" and t) else TEAL
                    for m, t in zip(models["model"], models["tuned"])]
        fig = go.Figure(go.Bar(
            x=model_names,
            y=roc_vals,
            marker=dict(color=colors_m, opacity=0.85),
            text=[f"{v:.4f}" for v in roc_vals],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR),
            hovertemplate="<b>%{x}</b><br>ROC-AUC: %{y:.4f}<extra></extra>"
        ))
        layout = base_layout("ROC-AUC by Model", height=340)
        layout["yaxis"]["range"] = [0.55, 0.65]
        layout["yaxis"]["title"] = "ROC-AUC"
        layout["shapes"] = [dict(type="line", x0=-0.5, x1=3.5, y0=0.75, y1=0.75,
                                  line=dict(color=RED, width=1.5, dash="dash"))]
        layout["annotations"] = [dict(x=3.4, y=0.752, text="Target 0.75",
                                       font=dict(color=RED, size=11), showarrow=False)]
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pr_vals = models["pr_auc"].tolist()
        fig = go.Figure(go.Bar(
            x=model_names,
            y=pr_vals,
            marker=dict(color=colors_m, opacity=0.85),
            text=[f"{v:.4f}" for v in pr_vals],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR),
            hovertemplate="<b>%{x}</b><br>PR-AUC: %{y:.4f}<extra></extra>"
        ))
        layout = base_layout("PR-AUC by Model (Retained class)", height=340)
        layout["yaxis"]["title"] = "PR-AUC"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    # ── Recall vs Precision ──
    st.markdown('<div class="section-header">Recall vs Precision — Retained Class</div>', unsafe_allow_html=True)

    fig = go.Figure()
    for _, row in models.iterrows():
        is_winner = row["model"] == "Logistic Regression" and row["tuned"]
        fig.add_trace(go.Scatter(
            x=[row["precision_retained"]],
            y=[row["recall_retained"]],
            mode="markers+text",
            name=row["model"] + (" ★" if is_winner else ""),
            marker=dict(
                size=18 if is_winner else 14,
                color=GOLD if is_winner else TEAL,
                symbol="star" if is_winner else "circle",
                line=dict(color="#0F1923", width=2)
            ),
            text=[row["model"].split()[0]],
            textposition="top center",
            textfont=dict(color=TEXT_COLOR, size=11)
        ))
    layout = base_layout("Precision vs Recall — Retained Class (churn=0)", height=380)
    layout["xaxis"]["title"] = "Precision (Retained)"
    layout["yaxis"]["title"] = "Recall (Retained)"
    layout["showlegend"] = True
    layout["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-card"><strong>Why the AUC is 0.626, not 0.75+:</strong> The maximum feature-target correlation is r = 0.07 (item_count), giving a theoretical AUC ceiling of ≈0.65. With only 243 retained customers in the test set (1.84% of 13,203), the minority class is extremely sparse. This is a data structure ceiling, not a model failure — a richer feature set (browsing history, app engagement, return history) would be what moves the needle, not a different algorithm.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP DRIVERS
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡  SHAP Drivers":

    st.markdown('<div class="page-title">SHAP Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">What drives churn probability — LinearExplainer applied to 13,203 test customers</div>', unsafe_allow_html=True)

    shap_cols = [c for c in shap.columns if c != "customer_unique_id"]
    mean_shap = shap[shap_cols].abs().mean().sort_values(ascending=False)

    # ── Global importance ──
    st.markdown('<div class="section-header">Global Feature Importance — Mean |SHAP|</div>', unsafe_allow_html=True)

    top_n = st.slider("Show top N features", 5, len(mean_shap), 15, 1)
    top_features = mean_shap.head(top_n)

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure(go.Bar(
            x=top_features.values[::-1],
            y=top_features.index[::-1],
            orientation="h",
            marker=dict(
                color=top_features.values[::-1],
                colorscale=[[0, TEAL], [0.5, GOLD], [1, RED]],
                showscale=True,
                colorbar=dict(title="Mean |SHAP|", tickfont=dict(color=TEXT_COLOR))
            ),
            text=[f"{v:.4f}" for v in top_features.values[::-1]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>"
        ))
        layout = base_layout(f"Top {top_n} Features by Mean Absolute SHAP Value", height=max(380, top_n * 28))
        layout["xaxis"]["title"] = "Mean |SHAP Value|"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 5 Drivers**")
        for i, (feat, val) in enumerate(top_features.head(5).items()):
            descriptions = {
                "category_churn_rate":    "What was bought matters most",
                "max_installments":       "More instalments = less churn",
                "seller_state_churn_rate":"Seller geography signal",
                "log_avg_item_price":     "Higher price = less repeat intent",
                "state_churn_rate":       "Regional retention patterns",
                "recency_days":           "How recently they ordered",
                "delivery_delay_days":    "Delivery experience",
                "review_score":           "Satisfaction signal",
                "item_count":             "Order size protective effect",
                "used_voucher":           "Voucher users engage more",
            }
            desc = descriptions.get(feat, "Feature contribution")
            st.markdown(f"""
            <div class="insight-card" style="margin:4px 0; padding:10px 14px;">
                <strong style="color:#2E86AB;">#{i+1} {feat}</strong><br>
                <span style="font-size:11px;">{desc}</span><br>
                <span style="color:#E8A838; font-family:'DM Mono',monospace; font-size:12px;">{val:.5f}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── SHAP distribution by feature ──
    st.markdown('<div class="section-header">SHAP Value Distribution — Select Feature</div>', unsafe_allow_html=True)

    selected_feat = st.selectbox("Choose feature to explore", options=list(mean_shap.index))

    col1, col2 = st.columns(2)

    with col1:
        shap_vals = shap[selected_feat]
        fig = go.Figure(go.Histogram(
            x=shap_vals,
            marker=dict(color=TEAL, opacity=0.8),
            nbinsx=50,
            hovertemplate="SHAP value: %{x:.4f}<br>Count: %{y}<extra></extra>"
        ))
        layout = base_layout(f"SHAP Distribution: {selected_feat}", height=320)
        layout["xaxis"]["title"] = "SHAP Value (positive = more churn)"
        layout["yaxis"]["title"] = "Count"
        fig.add_vline(x=0, line=dict(color=GOLD, width=1.5, dash="dash"))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if selected_feat in shap.columns and selected_feat != "customer_unique_id":
            shap_risk = shap[["customer_unique_id", selected_feat]].merge(
                risk[["customer_unique_id", "risk_tier"]], on="customer_unique_id"
            )
            fig = go.Figure()
            for tier, color in [("High Risk", RED), ("Medium Risk", GOLD), ("Low Risk", GREEN)]:
                subset = shap_risk[shap_risk["risk_tier"] == tier][selected_feat]
                fig.add_trace(go.Box(
                    y=subset,
                    name=tier,
                    marker_color=color,
                    line=dict(color=color),
                    fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)"
                ))
            layout = base_layout(f"SHAP by Risk Tier: {selected_feat}", height=320)
            layout["yaxis"]["title"] = "SHAP Value"
            layout["showlegend"] = True
            layout["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR))
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    # ── Business interpretation ──
    st.markdown('<div class="section-header">Business Interpretation</div>', unsafe_allow_html=True)

    insights = [
        ("🛍 Product Category", "category_churn_rate",
         "The single strongest driver. What a customer bought predicts whether they'll return. Lifestyle and home goods (bed_bath_table, furniture_decor) create repeat intent; discretionary one-off categories (electronics, cool_stuff) do not."),
        ("💳 Payment Instalments", "max_installments",
         "More instalments = lower churn. A customer financing a purchase over 6–10 months is demonstrating commitment. <strong>Actionable:</strong> offer instalment options proactively to high-risk segments."),
        ("🌎 Seller Geography", "seller_state_churn_rate",
         "The seller's state carries structural churn signal. Sellers in certain states serve customer bases with lower repeat intent — likely a product mix and logistics effect."),
        ("💰 Item Price", "log_avg_item_price",
         "Higher-priced single purchases signal discretionary buying — the customer came for one specific expensive item and has no structural reason to return. Lower-priced everyday goods drive repeat traffic."),
        ("📍 Customer State", "state_churn_rate",
         "Regional retention patterns are real and stable. MT, RO, AC retain better; RR, AM, SE churn harder. Geography is a structural predictor, not an actionable lever — but it informs targeting budget allocation.")
    ]

    for icon_title, feat, text in insights:
        mean_val = mean_shap.get(feat, 0)
        rank = list(mean_shap.index).index(feat) + 1 if feat in mean_shap.index else "—"
        st.markdown(f"""
        <div class="insight-card">
            <strong>{icon_title}</strong> &nbsp;·&nbsp;
            <span style="color:#2E86AB; font-family:'DM Mono',monospace; font-size:11px;">rank #{rank} &nbsp;|&nbsp; mean |SHAP| = {mean_val:.5f}</span><br>
            {text}
        </div>
        """, unsafe_allow_html=True)
