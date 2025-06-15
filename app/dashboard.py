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
    st.subheader("Uploaded Data")
    model = load_model()
    preds, probs = predict_fraud(model, df)

    df['fraud_probability'] = probs
    df['fraud_prediction'] = preds

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

    # 🚀 Some more graphs
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

    #col4- Scatter Plot for Title Length vs Fraud Probability
    col4, _, _ = st.columns([1, 0.1, 0.1])  # Optional spacing for layout alignment

    with col4:
        st.write("### Title Length vs Fraud Probability (Scatter Plot)")
        df['title_length'] = df['title'].apply(lambda x: len(str(x)))
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 4))
        ax_scatter.scatter(df['title_length'], df['fraud_probability'], alpha=0.5, color='purple')
        ax_scatter.set_xlabel("Title Length")
        ax_scatter.set_ylabel("Fraud Probability")
        ax_scatter.set_title("Title Length vs Fraud Probability")
        st.pyplot(fig_scatter)

    #col5 - Heatmap of Numerical Feature Correlations
    col5, _, _ = st.columns([1, 0.1, 0.1])  # Optional spacing

    with col5:
        st.write("### Heatmap of Numerical Feature Correlations")
        numeric_cols = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_cols.empty:
            fig_heat, ax_heat = plt.subplots(figsize=(10, 4))
            sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax_heat)
            st.pyplot(fig_heat)
        else:
            st.info("No numerical columns available for heatmap.")

    #col6 - Location-based Fraudulent Job Count
    col6, _, _ = st.columns([1, 0.1, 0.1])  # Optional spacing

    with col6:
        st.write("### Top Locations with Fraudulent Listings")
        top_locations = df[df['fraud_prediction'] == 1]['location'].value_counts().head(10)

        fig_loc, ax_loc = plt.subplots(figsize=(10, 4))
        top_locations.plot(kind='barh', color='crimson', ax=ax_loc)
        ax_loc.set_xlabel("Count")
        ax_loc.set_ylabel("Location")
        ax_loc.set_title("Top 10 Fraud-Prone Job Locations")
        st.pyplot(fig_loc)

    #col7- col7: Boxplot of Fraud Probability by Employment Type
    col7, _, _ = st.columns([1, 0.1, 0.1])

    with col7:
        st.write("### Fraud Probability by Employment Type")
        if 'employment_type' in df.columns:
            fig_box, ax_box = plt.subplots(figsize=(9, 3))
            sns.boxplot(
                x='employment_type',
                y='fraud_probability',
                hue='employment_type', 
                data=df,
                ax=ax_box,
                palette='Set2',
                legend=False # ✅ Avoids extra legend
            )
            ax_box.set_xlabel("Employment Type")
            ax_box.set_ylabel("Fraud Probability")
            ax_box.set_title("Fraud Probability Distribution Across Employment Types")
            plt.xticks(rotation=45)
            st.pyplot(fig_box)
        else:
            st.info("Column 'employment_type' not found in dataset.")


    #col8: Description Word Count Distribution
    col8, _, _ = st.columns([1, 0.1, 0.1])

    with col8:
        st.write("### Description Word Count Distribution")
        if 'description' in df.columns:
            df['desc_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
            fig_wc, ax_wc = plt.subplots(figsize=(9, 3))
            sns.histplot(df['desc_word_count'], bins=30, kde=True, color='slateblue', ax=ax_wc)
            ax_wc.set_xlabel("Word Count")
            ax_wc.set_ylabel("Number of Listings")
            ax_wc.set_title("Distribution of Word Count in Job Descriptions")
            st.pyplot(fig_wc)
        else:
            st.info("Column 'description' not found in dataset.")

