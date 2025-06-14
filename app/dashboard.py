#Main code
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model, predict_fraud

st.set_page_config(layout="wide")
st.title("🚨 Spot the Scam - Job Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload a job postings CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # st.subheader("Uploaded Data (Top 10 Rows)")
    st.subheader("Uploaded Data")
    # st.dataframe(df.head(10))
    # st.dataframe(df)

    model = load_model()
    preds, probs = predict_fraud(model, df)

    df['fraud_probability'] = probs
    df['fraud_prediction'] = preds

    # st.subheader("Fraud Prediction Results")
    # # st.dataframe(df[['title', 'fraud_probability', 'fraud_prediction']].head(10))
    # st.dataframe(df[['title', 'fraud_probability', 'fraud_prediction']])

    st.subheader("Fraud Prediction Results")
    st.dataframe(df[['title', 'fraud_probability', 'fraud_prediction']])

    # ✅ Add Download Button
    csv = df.to_csv(index=False)
    st.download_button(
    label="📥 Download Full Results as CSV",
    data=csv,
    file_name='scam_predictions.csv',
    mime='text/csv'
    )

    # Visualizations
    st.subheader("🔍 Visual Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Fraud Probability Histogram")
        plt.figure(figsize=(6, 4))
        sns.histplot(df['fraud_probability'], bins=20, kde=True, color='orange')
        st.pyplot(plt.gcf())

    with col2:
        st.write("### Fraud vs Real Pie Chart")
        labels = ['Real', 'Fraud']
        values = df['fraud_prediction'].value_counts().sort_index()

        # ✅ Fix: Proper pie chart rendering
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures the pie is circular
        st.pyplot(fig)

    # 🚀 NEW GRAPHS BELOW
    st.subheader("📊 Additional Data Insights")

    col3, _, _ = st.columns([1, 0.1, 0.1])  # Optional spacing with narrow columns

    with col3:
        st.write("### Top Job Titles in Fraudulent Listings")
        top_titles = df[df['fraud_prediction'] == 1]['title'].value_counts().head(10)
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        top_titles.plot(kind='barh', color='red', ax=ax_bar)
        ax_bar.set_xlabel("Count")
        ax_bar.set_ylabel("Job Title")
        ax_bar.set_title("Top 10 Fraudulent Job Titles")
        st.pyplot(fig_bar)

    # NEW GRAPH - Scatter Plot for Title Length vs Fraud Probability
    col4, _, _ = st.columns([1, 0.1, 0.1])  # Optional spacing for layout alignment

    with col4:
        st.write("### Title Length vs Fraud Probability (Scatter Plot)")
        df['title_length'] = df['title'].apply(lambda x: len(str(x)))
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
        ax_scatter.scatter(df['title_length'], df['fraud_probability'], alpha=0.5, color='purple')
        ax_scatter.set_xlabel("Title Length")
        ax_scatter.set_ylabel("Fraud Probability")
        ax_scatter.set_title("Title Length vs Fraud Probability")
        st.pyplot(fig_scatter)

    # NEW GRAPH - Heatmap of Numerical Feature Correlations
    col5, _, _ = st.columns([1, 0.1, 0.1])  # Optional spacing

    with col5:
        st.write("### Heatmap of Numerical Feature Correlations")
        numeric_cols = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_cols.empty:
            fig_heat, ax_heat = plt.subplots(figsize=(6, 4))
            sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax_heat)
            st.pyplot(fig_heat)
        else:
            st.info("No numerical columns available for heatmap.")

    # NEW GRAPH - Time Series Analysis (if 'date' column exists)
    # col6, _, _ = st.columns([1, 0.1, 0.1])  # Optional spacing

    # with col6:
    #     if 'date' in df.columns:
    #         st.write("### Time Series: Fraudulent Posts Over Time")
    #         df['location'] = pd.to_datetime(df['location'], errors='coerce')
    #         ts_df = df.dropna(subset=['location'])
    #         ts_counts = ts_df.groupby(ts_df['location'].dt.to_period('M'))['fraud_prediction'].sum()
    #         ts_counts.index = ts_counts.index.to_timestamp()

    #         fig_ts, ax_ts = plt.subplots()
    #         ts_counts.plot(ax=ax_ts, marker='o', linestyle='-', color='teal')
    #         ax_ts.set_ylabel("Number of Fraudulent Posts")
    #         ax_ts.set_xlabel("location")
    #         ax_ts.set_title("Monthly Fraudulent Posts")
    #         st.pyplot(fig_ts)
    #     else:
    #         st.info("No 'date' column found for time series analysis.")
