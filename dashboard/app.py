
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import shap
from pathlib import Path

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="HR Attrition Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colors ────────────────────────────────────
STAYED = "#2ECC71"
LEFT   = "#E74C3C"
BLUE   = "#2980B9"
ORANGE = "#E67E22"

# ── Paths ─────────────────────────────────────
BASE = Path(__file__).parent.parent

# ── Load Data & Models ────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(BASE / "data/ibm_hr_attrition.csv")

@st.cache_resource
def load_model():
    artifact = joblib.load(BASE / "models/best_model.pkl")
    scaler   = joblib.load(BASE / "models/scaler.pkl")
    with open(BASE / "models/feature_names.json") as f:
        features = json.load(f)
    return artifact, scaler, features

df                         = load_data()
artifact, scaler, features = load_model()
model                      = artifact["model"]
threshold                  = artifact["threshold"]

# ── Sidebar ───────────────────────────────────
st.sidebar.title("👥 HR Attrition")
st.sidebar.markdown("**IBM HR Analytics Dataset**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview",
     "EDA Insights",
     "Model Performance",
     "SHAP Explainability",
     "Recommendations"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dataset:** 1,470 employees  \n"
    f"**Model:** XGBoost  \n"
    f"**AUC:** {artifact['auc']:.3f}  \n"
    f"**Threshold:** {threshold:.2f}")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.title("👥 IBM HR Attrition Analysis Dashboard")
    st.markdown(
        "End-to-end ML project analysing employee attrition "
        "using the IBM HR Analytics dataset.")
    st.markdown("---")

    total      = len(df)
    left_count = (df["Attrition"] == "Yes").sum()
    rate       = left_count / total * 100
    avg_sal    = df["MonthlyIncome"].mean()
    rep_cost   = avg_sal * 12 * 0.75
    total_cost = left_count * rep_cost

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees",  f"{total:,}")
    col2.metric("Attrition Count",  f"{left_count:,}")
    col3.metric("Attrition Rate",   f"{rate:.1f}%",
                delta="↑ vs 13% industry avg",
                delta_color="inverse")
    col4.metric("Est. Annual Cost", f"${total_cost/1e6:.1f}M")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Attrition Split")
        counts = df["Attrition"].value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.55,
            marker_colors=[STAYED, LEFT]))
        fig.update_layout(height=300, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("By Department")
        dept = (df.groupby("Department")["Attrition"]
                  .apply(lambda x: (x=="Yes").mean()*100)
                  .reset_index()
                  .rename(columns={"Attrition":"Rate"}))
        fig = px.bar(dept, x="Rate", y="Department",
                     orientation="h", color="Rate",
                     color_continuous_scale=["#2ECC71","#E74C3C"])
        fig.update_layout(height=300, margin=dict(t=20,b=20),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("Overtime Impact")
        ot = (df.groupby("OverTime")["Attrition"]
                .apply(lambda x: (x=="Yes").mean()*100)
                .reset_index()
                .rename(columns={"Attrition":"Rate"}))
        fig = px.bar(ot, x="OverTime", y="Rate",
                     color="OverTime",
                     color_discrete_map={"No": STAYED, "Yes": LEFT})
        fig.update_layout(height=300, margin=dict(t=20,b=20),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔑 Key Findings")
    f1, f2, f3, f4 = st.columns(4)
    f1.info("**Overtime** workers leave at **30.5%** vs 10.4%")
    f2.warning("**Sales dept** has highest attrition at **20.6%**")
    f3.error("**18-25 age group** attrition at **34.8%**")
    f4.success("**Stock options** reduce attrition to **9.9%**")

# ============================================================
# PAGE 2 — EDA INSIGHTS
# ============================================================
elif page == "EDA Insights":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Deep dive into attrition drivers.")
    st.markdown("---")

    dept_filter = st.selectbox(
        "Filter by Department",
        ["All"] + list(df["Department"].unique()))

    dff = df.copy() if dept_filter == "All" else           df[df["Department"] == dept_filter].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Income by Attrition")
        fig = px.box(dff, x="Attrition", y="MonthlyIncome",
                     color="Attrition",
                     color_discrete_map={"No": STAYED, "Yes": LEFT})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Attrition by Age Group")
        bins   = [18, 25, 35, 45, 55, 65]
        labels = ["18-25","26-35","36-45","46-55","55+"]
        dff["AgeGroup"] = pd.cut(dff["Age"],
                                  bins=bins, labels=labels)
        age_attr = (dff.groupby("AgeGroup", observed=True)["Attrition"]
                      .apply(lambda x: (x=="Yes").mean()*100)
                      .reset_index()
                      .rename(columns={"Attrition":"Rate"}))
        fig = px.bar(age_attr, x="AgeGroup", y="Rate",
                     color="Rate",
                     color_continuous_scale=["#2ECC71","#E74C3C"])
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Job Satisfaction vs Attrition")
        js = (dff.groupby("JobSatisfaction")["Attrition"]
                .apply(lambda x: (x=="Yes").mean()*100)
                .reset_index()
                .rename(columns={"Attrition":"Rate"}))
        fig = px.bar(js, x="JobSatisfaction", y="Rate",
                     color="Rate",
                     color_continuous_scale=["#2ECC71","#E74C3C"],
                     labels={"JobSatisfaction":
                             "Job Satisfaction (1=Low, 4=High)"})
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Attrition by Job Role")
        role = (dff.groupby("JobRole")["Attrition"]
                   .apply(lambda x: (x=="Yes").mean()*100)
                   .reset_index()
                   .rename(columns={"Attrition":"Rate"})
                   .sort_values("Rate", ascending=True))
        fig = px.bar(role, x="Rate", y="JobRole",
                     orientation="h", color="Rate",
                     color_continuous_scale=["#2ECC71","#E74C3C"])
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("XGBoost selected as best model.")
    st.markdown("---")

    scores = pd.DataFrame({
        "Model":    ["Logistic Regression",
                     "Random Forest", "XGBoost ★"],
        "ROC-AUC":  [0.7869, 0.7971, 0.8007],
        "F1 Score": [0.4590, 0.3797, 0.5106],
        "Recall":   [0.60,   0.32,   0.51],
        "Precision":[0.37,   0.47,   0.51],
    })
    st.subheader("Model Comparison Scorecard")
    st.dataframe(scores, use_container_width=True,
                 hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC-AUC Comparison")
        fig = px.bar(scores, x="Model", y="ROC-AUC",
                     color="Model",
                     color_discrete_sequence=[BLUE, BLUE, LEFT])
        fig.update_layout(height=350, showlegend=False,
                          yaxis_range=[0.7, 0.85])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Threshold Tuning Impact")
        thresh = pd.DataFrame({
            "Threshold":        ["Default (0.50)", "Tuned (0.28)"],
            "F1 Score":         [0.4571, 0.5106],
            "Recall":           [0.34,   0.51],
            "Employees Caught": [16,     24],
        })
        st.dataframe(thresh, use_container_width=True,
                     hide_index=True)
        st.info(
            "Lowering threshold from 0.50 → 0.28 caught "
            "**8 more at-risk employees** with no retraining.")

    st.markdown("---")
    st.subheader("Confusion Matrix — XGBoost (Tuned)")
    col1, col2 = st.columns([1, 2])
    with col1:
        cm = pd.DataFrame({
            "":             ["Actually Stayed", "Actually Left"],
            "Pred. Stayed": [240, 23],
            "Pred. Left":   [7,   24],
        })
        st.dataframe(cm, use_container_width=True,
                     hide_index=True)
        st.success("**24/47** at-risk employees identified")
        st.error("**23/47** employees missed")
    with col2:
        st.markdown("""
        **Reading the matrix:**
        - ✅ **240** → Correctly predicted Stayed
        - ✅ **24** → Correctly predicted Left
        - ❌ **7** → False alarm
        - ❌ **23** → Missed departures

        > Catching 24 at-risk employees at $58,526
        > replacement cost = **$1.4M** in avoided costs.
        """)

# ============================================================
# PAGE 4 — SHAP EXPLAINABILITY
# ============================================================
elif page == "SHAP Explainability":
    st.title("🔎 SHAP Explainability")
    st.markdown(
        "SHAP values explain **why** the model flags each employee.")
    st.markdown("---")

    X_test = np.load(BASE / "models/X_test_scaled.npy")

    @st.cache_resource
    def compute_shap(_model, _X):
        explainer   = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(_X)
        return shap_values

    with st.spinner("Computing SHAP values..."):
        shap_values = compute_shap(model, X_test)

    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[::-1][:15]
    top_feats = [features[i] for i in top_idx]
    top_vals  = mean_shap[top_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Attrition Drivers")
        fig = px.bar(
            x=top_vals[::-1], y=top_feats[::-1],
            orientation="h", color=top_vals[::-1],
            color_continuous_scale=["#2980B9","#E74C3C"],
            labels={"x": "Mean |SHAP Value|", "y": "Feature"})
        fig.update_layout(height=500,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Driver Insights")
        st.markdown("""
        🔴 **OverTime (0.855)**
        Strongest predictor. Overtime employees are
        **2.9x** more likely to leave.

        🔴 **StockOptionLevel (0.499)**
        43% of staff have zero options — hidden retention lever.

        🔴 **JobSatisfaction (0.387)**
        Score 1 → 22.8% attrition vs 11.3% at Score 4.

        🔴 **YearsWithCurrManager (0.370)**
        Long tenure under same manager signals stagnation.
        """)
        st.info(
            "💡 SHAP revealed StockOptionLevel as #2 driver — "
            "missed by EDA alone.")

    st.markdown("---")
    st.subheader("Individual Employee Explanation")

    xgb_proba = model.predict_proba(X_test)[:, 1]
    emp_idx   = st.slider("Select Employee", 0,
                           len(X_test)-1, 200)
    prob      = xgb_proba[emp_idx]
    emp_shap  = shap_values[emp_idx]

    col1, col2 = st.columns(2)
    with col1:
        risk = "🔴 HIGH RISK" if prob >= threshold                else "🟢 LOW RISK"
        st.metric("Attrition Probability", f"{prob*100:.1f}%")
        st.markdown(f"**Risk Level:** {risk}")

    with col2:
        st.markdown("**Pushing toward LEAVING 🔴**")
        for i in np.argsort(emp_shap)[::-1][:5]:
            st.markdown(
                f"`{features[i][:30]}` → +{emp_shap[i]:.3f}")
        st.markdown("**Pushing toward STAYING 🟢**")
        for i in np.argsort(emp_shap)[:5]:
            st.markdown(
                f"`{features[i][:30]}` → {emp_shap[i]:.3f}")

# ============================================================
# PAGE 5 — RECOMMENDATIONS
# ============================================================
elif page == "Recommendations":
    st.title("💼 Business Recommendations")
    st.markdown(
        "Data-driven interventions to recover **$9.62M** annually.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.error("**Current Annual Cost**\n\n# $13.87M")
    col2.success("**Potential Savings**\n\n# $9.62M")
    col3.info("**Cost Reduction**\n\n# 69.4%")

    st.markdown("---")
    st.subheader("The Four Key Interventions")

    recs = [
        ("⏰ Overtime Control", LEFT,
         "28% of staff work overtime at **2.9x** risk.",
         "Cap overtime. Redistribute workload. Hire where overloaded.",
         "$2.23M"),
        ("💰 Salary Review", ORANGE,
         "Bottom 25% earn <$2,911/month, leave at **29.3%**.",
         "Benchmark salaries. Adjust bottom quartile.",
         "$1.90M"),
        ("🎯 Career Development", BLUE,
         "729 employees ≤35 leave at **21.9%** vs 10.4%.",
         "Mentoring program. Internal mobility portal.",
         "$2.34M"),
        ("📈 Stock Options", STAYED,
         "43% have zero options, leave at **24.4%**.",
         "Expand eligibility. Highest ROI intervention.",
         "$3.15M"),
    ]

    for title, color, finding, action, saving in recs:
        with st.expander(f"{title} — Est. Saving: {saving}"):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Finding:** {finding}")
            c1.markdown(f"**Action:** {action}")
            c2.metric("Estimated Annual Saving", saving)

    st.markdown("---")
    st.subheader("30-60-90 Day Action Plan")
    st.image(str(BASE / "report/action_plan.png"),
             use_container_width=True)

    st.markdown("---")
    st.subheader("Risk Group Comparison")
    st.image(str(BASE / "report/recommendations.png"),
             use_container_width=True)

# ── Footer ────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px'>"
    "IBM HR Attrition Analysis · XGBoost + SHAP · Streamlit"
    "</div>",
    unsafe_allow_html=True)
