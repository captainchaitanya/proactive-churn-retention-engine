# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import plotly.graph_objects as go

# --- Load Artifacts ---
model = joblib.load('churn_model.pkl')
explainer = joblib.load('shap_explainer.pkl')
X_test = joblib.load('X_test.pkl')

# --- Page Configuration ---
st.set_page_config(page_title="SME Retention Engine", layout="wide")
st.title("🚀 Proactive SME Cardholder Retention Engine")

# --- Helper Function for Tiers and Recommendations ---
def get_tier_and_recommendation(probability):
    """Assigns a risk tier and a recommended action based on churn probability."""
    if probability > 0.7:
        return "Tier 1: High Risk", "Urgent: Assign to senior account manager for personalized outreach."
    elif probability > 0.5:
        return "Tier 2: Medium Risk", "Proactive Outreach: Assign to account manager for a check-in call and offer."
    else:
        return "Tier 3: Low Risk", "Monitor: Enroll in automated marketing campaign to maintain engagement."

# --- Create Prediction DataFrame ---
@st.cache_data
def get_prediction_data():
    """Generates predictions and supplemental data for the dashboard."""
    predictions_proba = model.predict_proba(X_test)[:, 1]
    results_df = X_test.copy()
    results_df['churn_probability'] = predictions_proba
    tiers_and_recs = results_df['churn_probability'].apply(get_tier_and_recommendation)
    results_df['tier'], results_df['recommendation'] = zip(*tiers_and_recs)
    return results_df.sort_values(by='churn_probability', ascending=False)

results_df_sorted = get_prediction_data()


# --- Main Dashboard View ---
st.header("Prioritized At-Risk SME Accounts")
st.write("This table lists SME accounts prioritized by their churn risk. Select an index from the dropdown below to see a detailed analysis.")
st.dataframe(results_df_sorted[['churn_probability', 'tier', 'recommendation', 'recency', 'frequency', 'monetary']])

st.markdown("---")

# --- Deep Dive Analysis View ---
st.header("Deep Dive Seller Analysis")
selected_seller_index = st.selectbox("Select a Seller Index to Analyze:", results_df_sorted.index)

if selected_seller_index is not None:
    seller_data = X_test.loc[[selected_seller_index]]
    
    # **FIX 1**: Removed the `[1]` indexer. The explainer is returning a single array.
    shap_values_for_seller = explainer.shap_values(seller_data)
    
    seller_prob = results_df_sorted.loc[selected_seller_index, 'churn_probability']
    seller_tier = results_df_sorted.loc[selected_seller_index, 'tier']
    seller_rec = results_df_sorted.loc[selected_seller_index, 'recommendation']

    st.subheader(f"Seller Profile (Index: {selected_seller_index})")
    st.markdown(f"**Churn Probability:** `{seller_prob:.2%}`")
    st.markdown(f"**Prioritization Tier:** `{seller_tier}`")
    st.markdown(f"**Recommended Action:** `{seller_rec}`")

    # --- SHAP Waterfall Plot ---
    st.subheader("Key Drivers of Churn Risk (SHAP Analysis)")
    st.write("This chart shows how each feature pushes the model's prediction from a base value to the final output. Red bars increase churn risk; blue bars decrease it.")
    
    # **FIX 2**: Changed `explainer.expected_value[1]` to `explainer.expected_value`
    # as it's now a single value, not a list.
    fig = go.Figure(go.Waterfall(
        orientation = "h",
        measure = ["relative"] * len(X_test.columns) + ["total"],
        y = list(X_test.columns) + ["Final Prediction"],
        x = list(shap_values_for_seller.flatten()) + [shap_values_for_seller.sum() + explainer.expected_value],
        base = explainer.expected_value,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="Churn Risk Drivers", showlegend=False, yaxis_title="Features")
    st.plotly_chart(fig, use_container_width=True)