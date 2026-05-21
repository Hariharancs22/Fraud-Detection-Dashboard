import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# --- Configuration ---
st.set_page_config(page_title="FraudOps Dashboard", page_icon="🛡️", layout="wide")

# --- Data Loading (Cached for performance) ---
@st.cache_data
def load_data():
    return pd.read_csv('dashboard_data.csv')

@st.cache_resource
def load_models():
    model = joblib.load('model.pkl')
    explainer = joblib.load('explainer.pkl')
    return model, explainer

df = load_data()
model, explainer = load_models()

# Ensure we have our derived columns from Task 5 for visualization
if 'Fraud_Probability' not in df.columns:
    df['Fraud_Probability'] = model.predict_proba(df.drop(['True_Label', 'Risk_Tier', 'Fraud_Probability'], axis=1, errors='ignore'))[:, 1]

def assign_tier(prob):
    if prob >= 0.75: return 'Critical Risk'
    elif prob >= 0.40: return 'Suspicious'
    else: return 'Clear'

if 'Risk_Tier' not in df.columns:
    df['Risk_Tier'] = df['Fraud_Probability'].apply(assign_tier)

# --- Sidebar Navigation & Filters ---
st.sidebar.title("🛡️ FraudOps Navigation")
page = st.sidebar.radio("Go to", ["Overview & Insights", "Transaction Explorer", "SHAP Explainer"])

st.sidebar.markdown("---")
st.sidebar.header("Global Filters")
risk_filter = st.sidebar.multiselect("Filter by Risk Tier", ['Clear', 'Suspicious', 'Critical Risk'], default=['Clear', 'Suspicious', 'Critical Risk'])

# Apply filters
filtered_df = df[df['Risk_Tier'].isin(risk_filter)]

# --- PAGE 1: Overview & Insights ---
if page == "Overview & Insights":
    st.title("Fraud Operations Overview")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_txns = len(filtered_df)
    total_fraud = len(filtered_df[filtered_df['Fraud_Probability'] >= 0.30]) # Using your optimal 0.30 threshold!
    fraud_rate = (total_fraud / total_txns) * 100 if total_txns > 0 else 0
    avg_fraud_amt = filtered_df[filtered_df['Fraud_Probability'] >= 0.30]['TransactionAmt'].mean()

    col1.metric("Total Transactions", f"{total_txns:,}")
    col2.metric("Detected Frauds (Threshold 0.30)", f"{total_fraud:,}")
    col3.metric("Detection Rate", f"{fraud_rate:.2f}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amt:.2f}" if not np.isnan(avg_fraud_amt) else "$0.00")

    st.markdown("---")
    
    # Visualizations (Task 7 Requirements)
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("Risk Tier Distribution")
        tier_counts = filtered_df['Risk_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Risk_Tier', 'Count']
        fig_donut = px.pie(tier_counts, values='Count', names='Risk_Tier', hole=0.5, 
                           color='Risk_Tier', color_discrete_map={'Clear':'#2ECC71', 'Suspicious':'#F1C40F', 'Critical Risk':'#E74C3C'})
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with row1_col2:
        st.subheader("Transaction Amount Distribution (Log Scale)")
        fig_hist = px.histogram(filtered_df, x='TransactionAmt', color='Risk_Tier', log_y=True, nbins=50,
                                color_discrete_map={'Clear':'#2ECC71', 'Suspicious':'#F1C40F', 'Critical Risk':'#E74C3C'})
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Fraud Patterns: Transaction Amount vs Hour of Day")
    # Removed the size='TransactionAmt' parameter so negative scaled values don't crash the plot
    fig_scatter = px.scatter(filtered_df, x='HourOfDay', y='TransactionAmt', color='Fraud_Probability', 
                             hover_data=['TransactionAmt'], 
                             color_continuous_scale='Reds', title="Interactive Risk Scatter Plot (Bonus Task)")
    st.plotly_chart(fig_scatter, use_container_width=True)


# --- PAGE 2: Transaction Explorer ---
elif page == "Transaction Explorer":
    st.title("Live Transaction Explorer")
    st.write("Search and filter recent transactions. Live risk scores are calculated by the XGBoost model.")
    
    # Display dataframe with specific columns for readability
    display_cols = ['TransactionAmt', 'HourOfDay', 'Fraud_Probability', 'Risk_Tier']
    # Add a few other raw columns just for flavor if they exist
    for col in ['ProductCD', 'DeviceType']:
        if col in filtered_df.columns: display_cols.append(col)
        
    st.dataframe(filtered_df[display_cols].sort_values('Fraud_Probability', ascending=False), use_container_width=True)


# --- PAGE 3: SHAP Explainer ---
elif page == "SHAP Explainer":
    st.title("Explainable AI (SHAP)")
    st.write("Understand exactly why the model flagged a specific transaction.")
    
    # Let user select a row index from the sample data
    selected_idx = st.selectbox("Select a Transaction Index to Explain:", filtered_df.index)
    
    if st.button("Generate Explanation"):
        with st.spinner("Calculating SHAP values..."):
            # Get the exact row of features used by the model
            feature_cols = [c for c in filtered_df.columns if c not in ['True_Label', 'Fraud_Probability', 'Risk_Tier']]
            transaction_features = filtered_df.loc[[selected_idx], feature_cols]
            
            # Calculate SHAP
            shap_vals = explainer(transaction_features)
            prob = filtered_df.loc[selected_idx, 'Fraud_Probability']
            tier = filtered_df.loc[selected_idx, 'Risk_Tier']
            
            st.markdown(f"### Risk Score: **{prob:.4f}** ({tier})")
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_vals[0], show=False)
            st.pyplot(fig)
            
            # Plain English Explanation
            st.markdown("### Plain English Analysis")
            if tier == 'Critical Risk':
                st.error("🚨 **High Fraud Probability:** The red bars in the chart above push the risk score heavily toward fraud. This is typically driven by abnormal transaction times, unusual transaction amounts, or high-risk device signatures.")
            elif tier == 'Suspicious':
                st.warning("⚠️ **Review Required:** This transaction has conflicting signals. While some elements align with normal behavior (blue bars), certain anomalies (red bars) elevated the risk enough to require manual analyst review.")
            else:
                st.success("✅ **Legitimate Transaction:** The blue bars dominate the decision, indicating that this transaction perfectly aligns with safe, historical customer patterns.")