import streamlit as st
import numpy as np
import pickle
import gzip
import os
from openai import OpenAI

# === Load model and scaler with safe pathing ===
base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "compressed_xgb_grid.pkl.gz")
scaler_path = os.path.join(base_path, "scaler.pkl")

with gzip.open(model_path, "rb") as file:
    model = pickle.load(file)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# === OpenRouter API Key Setup ===
api_key = st.secrets["openrouter_api_key"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# === Streamlit App Layout ===
st.set_page_config(page_title="Heart Risk Predictor", layout="wide")

# === Sidebar Chatbot ===
st.sidebar.title("🧠 AI Assistant")
st.sidebar.write("Ask me anything about your heart disease risk prediction.")
chat_input = st.sidebar.text_input("💬 Your Question")

def get_llm_response(prompt):
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://yourappname.streamlit.app",
                "X-Title": "HeartRiskPredictorApp",
            },
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

if chat_input:
    st.sidebar.write("🗣️ AI is thinking...")
    response = get_llm_response(chat_input)
    st.sidebar.write(response)

# === Title and Intro ===
st.markdown("<h1 style='color:blue'>Heart Disease Risk Prediction 💓</h1>", unsafe_allow_html=True)
st.write("Please go through the steps below to get your prediction.")

# === Session State Setup ===
if "step" not in st.session_state:
    st.session_state.step = 1

def reset_step():
    st.session_state.step = 1

# === Step 1 - Demographic Info ===
if st.session_state.step == 1:
    st.subheader("🧍 Demographic Info")
    age = st.slider("Age", 1, 100, 30)
    sex = st.radio("Sex", ["Male", "Female"])
    education = st.selectbox("Education Level", [1, 2, 3, 4])

    if st.button("Next ➡️"):
        st.session_state.age = age
        st.session_state.sex = sex
        st.session_state.education = education
        st.session_state.step = 2

# === Step 2 - Lifestyle ===
elif st.session_state.step == 2:
    st.subheader("🚬 Lifestyle & Habits")
    cigs_per_day = st.number_input("Cigarettes Per Day", min_value=0, max_value=50, value=0)

    if st.button("Next ➡️"):
        st.session_state.cigs_per_day = cigs_per_day
        st.session_state.step = 3

# === Step 3 - Medical History ===
elif st.session_state.step == 3:
    st.subheader("📋 Medical History")
    bp_meds = st.radio("Blood Pressure Meds?", ["Yes", "No"])
    prevalent_stroke = st.radio("Prevalent Stroke?", ["Yes", "No"])
    prevalent_hyp = st.radio("Prevalent Hypertension?", ["Yes", "No"])
    diabetes = st.radio("Diabetes?", ["Yes", "No"])

    if st.button("Next ➡️"):
        st.session_state.bp_meds = bp_meds
        st.session_state.prevalent_stroke = prevalent_stroke
        st.session_state.prevalent_hyp = prevalent_hyp
        st.session_state.diabetes = diabetes
        st.session_state.step = 4

# === Step 4 - Current Medical + Prediction ===
elif st.session_state.step == 4:
    st.subheader("🏥 Medical Data")
    tot_chol = st.number_input("Total Cholesterol", 0.0, 600.0, 200.0)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    heart_rate = st.slider("Heart Rate", 40, 150, 75)
    glucose = st.number_input("Glucose", 40.0, 400.0, 100.0)
    pulse_pressure = st.number_input("Pulse Pressure", 0.0, 100.0, 40.0)

    if st.button("🎯 Predict"):
        # Encode inputs
        sex_int = 1 if st.session_state.sex == "Male" else 0
        bp_meds_int = 1 if st.session_state.bp_meds == "Yes" else 0
        prevalent_stroke_int = 1 if st.session_state.prevalent_stroke == "Yes" else 0
        prevalent_hyp_int = 1 if st.session_state.prevalent_hyp == "Yes" else 0
        diabetes_int = 1 if st.session_state.diabetes == "Yes" else 0

        input_data = np.array([
            st.session_state.age,
            st.session_state.education,
            sex_int,
            st.session_state.cigs_per_day,
            bp_meds_int,
            prevalent_stroke_int,
            prevalent_hyp_int,
            diabetes_int,
            tot_chol,
            bmi,
            heart_rate,
            glucose,
            pulse_pressure
        ])

        scaled_input = scaler.transform(input_data.reshape(1, -1))
        prediction = model.predict(scaled_input)

        st.success("✅ Prediction Complete!")
        st.subheader("🩺 Result:")
        if prediction[0] == 1:
            st.error("⚠️ You may be at risk of heart disease.")
        else:
            st.success("🎉 You are not likely at risk of heart disease.")

    st.markdown("---")
    if st.button("🔄 Restart"):
        reset_step()
