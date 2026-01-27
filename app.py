import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load Model
model = pickle.load(open("models/best_model.pkl", "rb"))

# Load Dataset for Feature Names
data = pd.read_csv("dataset/digital_footprint_dataset.csv")
features = data.drop("risk_label", axis=1).columns

st.set_page_config(page_title="Digital Footprint Risk Analyzer", layout="wide")

st.title("🔐 Digital Footprint Risk Analyzer")
st.write("AI Powered Privacy Risk Prediction & Explainable Analysis System")

# -------- Sidebar Input -------- #

st.sidebar.header("User Digital Behavior Input")

screen_time = st.sidebar.slider("Daily Screen Time (Hours)",1,15,5)
public_profile = st.sidebar.selectbox("Public Profile",["No","Yes"])
location = st.sidebar.selectbox("Location Sharing",["Off","On"])
twofa = st.sidebar.selectbox("Two Factor Authentication",["Enabled","Disabled"])
password = st.sidebar.slider("Password Strength",1,10,6)
platforms = st.sidebar.slider("Social Platforms Used",1,8,3)
posts = st.sidebar.slider("Public Posts",0,200,30)
permissions = st.sidebar.slider("App Permissions",1,20,6)
breach = st.sidebar.selectbox("Data Breach Exposure",["No","Yes"])

# Convert Inputs
public_profile = 1 if public_profile == "Yes" else 0
location = 1 if location == "On" else 0
twofa = 1 if twofa == "Enabled" else 0
breach = 1 if breach == "Yes" else 0

input_data = np.array([[screen_time, public_profile, location, twofa,
                        password, platforms, posts, permissions, breach]])

# -------- Prediction -------- #

if st.sidebar.button("Analyze Risk"):

    probabilities = model.predict_proba(input_data)

    risk_score = round(probabilities[0][2] * 100, 2)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Privacy Risk Score")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Risk Level (%)"},
            gauge={'axis': {'range': [0,100]}}
        ))

        st.plotly_chart(fig)

        if risk_score >= 70:
            st.error("🚨 HIGH RISK — Immediate Action Required")
        elif risk_score >= 40:
            st.warning("⚠ MEDIUM RISK — Improve Privacy Settings")
        else:
            st.success("✅ LOW RISK — Good Digital Behavior")

    # -------- Explainable AI -------- #

    with col2:
        st.subheader("🧠 AI Explainability")

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            importance = abs(model.coef_[0])

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig2 = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Impact on Privacy Risk"
        )

        st.plotly_chart(fig2)

    # -------- Recommendations -------- #

    st.subheader("🔍 Privacy Improvement Suggestions")

    if location == 1:
        st.write("• Disable public location sharing")

    if public_profile == 1:
        st.write("• Switch social profiles to private")

    if password <= 5:
        st.write("• Use stronger passwords")

    if permissions > 10:
        st.write("• Reduce unnecessary app permissions")

    if breach == 1:
        st.write("• Change all passwords immediately")

    if twofa == 0:
        st.write("• Enable Two Factor Authentication")
