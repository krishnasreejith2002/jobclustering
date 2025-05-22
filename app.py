import streamlit as st
import pandas as pd
import joblib
import re

# Load saved models once
@st.cache_resource
def load_models():
    kmeans = joblib.load('job_cluster_model.pkl')
    vectorizer = joblib.load('skill_vectorizer.pkl')
    return kmeans, vectorizer

kmeans, vectorizer = load_models()

# Text cleaning function
def clean_skills(skills_text):
    if not isinstance(skills_text, str):
        return ""
    return re.sub(r'[^a-zA-Z, ]', '', skills_text.lower())

st.title("Job Clustering on Karkidi.com Listings")

uploaded_file = st.file_uploader("Upload your scraped jobs CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data:", df.head())

    if 'Skills' not in df.columns:
        st.error("CSV file must contain 'Skills' column.")
    else:
        df['Cleaned_Skills'] = df['Skills'].apply(clean_skills)
        X = vectorizer.transform(df['Cleaned_Skills'])
        df['Cluster'] = kmeans.predict(X)

        st.write("Clustered Jobs:")
        st.dataframe(df[['Title', 'Company', 'Skills', 'Cluster']])

        # Let user filter by cluster
        selected_cluster = st.selectbox("Filter by Cluster", sorted(df['Cluster'].unique()))
        filtered_df = df[df['Cluster'] == selected_cluster]
        st.write(f"Jobs in Cluster {selected_cluster}:", filtered_df[['Title', 'Company', 'Skills']])
