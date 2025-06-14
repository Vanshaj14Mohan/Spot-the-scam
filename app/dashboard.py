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
    st.subheader("Uploaded Data (Top 10 Rows)")
    st.dataframe(df.head(10))

    model = load_model()
    preds, probs = predict_fraud(model, df)

    df['fraud_probability'] = probs
    df['fraud_prediction'] = preds

    st.subheader("Fraud Prediction Results")
    st.dataframe(df[['title', 'fraud_probability', 'fraud_prediction']].head(10))

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

