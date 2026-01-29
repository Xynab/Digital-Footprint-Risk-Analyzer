import os
from datetime import datetime
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ------------------ File Paths ------------------

USERS_FILE = "users/users.csv"
HISTORY_FILE = "history/results.csv"

# ------------------ Create Files If Not Exist ------------------

os.makedirs("users", exist_ok=True)
os.makedirs("history", exist_ok=True)

if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USERS_FILE, index=False)

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["username", "risk_score", "risk_level", "date_time"]).to_csv(HISTORY_FILE, index=False)

# ------------------ Helper Functions ------------------

def register_user(username, password):
    df = pd.read_csv(USERS_FILE)
    df.loc[len(df)] = [username, password]
    df.to_csv(USERS_FILE, index=False)

def validate_user(username, password):
    df = pd.read_csv(USERS_FILE)
    user = df[(df["username"] == username) & (df["password"] == password)]
    return not user.empty

def save_history(username, score, level):
    df = pd.read_csv(HISTORY_FILE)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [username, score, level, time]
    df.to_csv(HISTORY_FILE, index=False)

# ------------------ Load Model & Dataset ------------------

model = pickle.load(open("models/best_model.pkl", "rb"))
data = pd.read_csv("dataset/digital_footprint_dataset.csv")
features = data.drop("risk_label", axis=1).columns

# ------------------ Page Config ------------------

st.set_page_config(page_title="Digital Footprint Risk Analyzer", layout="wide")

st.title("🔐 Digital Footprint Risk Analyzer")
st.write("AI Powered Privacy Risk Prediction System")

# ------------------ LOGIN SYSTEM ------------------

st.sidebar.header("🔐 Login System")

login_username = st.sidebar.text_input("Login Username")
login_password = st.sidebar.text_input("Password", type="password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.sidebar.button("Login"):
    if validate_user(login_username, login_password):
        st.session_state.logged_in = True
        st.session_state.user = login_username
        st.sidebar.success("Login Successful")
    else:
        st.sidebar.error("Invalid Credentials")

if st.sidebar.button("Register"):
    register_user(login_username, login_password)
    st.sidebar.success("Registration Successful")

if not st.session_state.logged_in:
    st.stop()

# ------------------ Sidebar Inputs ------------------

st.sidebar.header("User Digital Behavior Input")

demo_username = st.sidebar.text_input("Username (Demo Mode)")

screen_time = st.sidebar.slider("Daily Screen Time (Hours)", 1, 15, 5)
public_profile = st.sidebar.selectbox("Public Profile", ["No", "Yes"])
location = st.sidebar.selectbox("Location Sharing", ["Off", "On"])
twofa = st.sidebar.selectbox("Two Factor Authentication", ["Enabled", "Disabled"])
password = st.sidebar.slider("Password Strength", 1, 10, 6)
platforms = st.sidebar.slider("Social Platforms Used", 1, 8, 3)
posts = st.sidebar.slider("Public Posts", 0, 200, 30)
permissions = st.sidebar.slider("App Permissions", 1, 20, 6)
breach = st.sidebar.selectbox("Data Breach Exposure", ["No", "Yes"])

# Convert Inputs
public_profile = 1 if public_profile == "Yes" else 0
location = 1 if location == "On" else 0
twofa = 1 if twofa == "Enabled" else 0
breach = 1 if breach == "Yes" else 0

# ------------------ Prediction Button ------------------

if st.sidebar.button("Analyze Risk"):

    input_data = np.array([[screen_time, public_profile, location, twofa,
                            password, platforms, posts, permissions, breach]])

    probabilities = model.predict_proba(input_data)
    ml_risk_score = probabilities[0][2] * 100

    # ------------------ Username Risk Logic ------------------

    username_score = 0

    if len(demo_username) < 6:
        username_score += 20

    if any(char.isdigit() for char in demo_username):
        username_score += 15

    common_names = ["admin", "user", "test", "syed123"]

    if demo_username.lower() in common_names:
        username_score += 30

    # ------------------ Final Risk ------------------

    final_risk = round(min(100, ml_risk_score + username_score), 2)

    if final_risk >= 70:
        risk_level = "HIGH"
    elif final_risk >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Save History
    save_history(st.session_state.user, final_risk, risk_level)

    col1, col2 = st.columns(2)

    # ------------------ Risk Meter ------------------

    with col1:

        st.subheader("📊 Privacy Risk Score")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_risk,
            title={'text': "Risk Level (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        st.plotly_chart(fig)

        if risk_level == "HIGH":
            st.error("🚨 HIGH RISK")
        elif risk_level == "MEDIUM":
            st.warning("⚠ MEDIUM RISK")
        else:
            st.success("✅ LOW RISK")

    # ------------------ Explainable AI ------------------

    with col2:

        st.subheader("🧠 Risk Factor Importance")

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            importance = abs(model.coef_[0])

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=True)

        fig2 = px.bar(imp_df, x="Importance", y="Feature", orientation="h")

        st.plotly_chart(fig2)

    # ------------------ Safety Tips ------------------

    st.subheader("✅ Privacy Safety Tips")

    if public_profile == 1:
        st.write("• Set social profiles to private")

    if location == 1:
        st.write("• Turn off public location sharing")

    if password <= 5:
        st.write("• Use stronger passwords")

    if permissions > 10:
        st.write("• Reduce unnecessary app permissions")

    if twofa == 0:
        st.write("• Enable Two Factor Authentication")

    if breach == 1:
        st.write("• Change passwords immediately")

# ------------------ User History Section ------------------

st.subheader("📁 Your Risk Analysis History")

history_df = pd.read_csv(HISTORY_FILE)
user_history = history_df[history_df["username"] == st.session_state.user]

st.dataframe(user_history)
