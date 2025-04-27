import streamlit as st
import numpy as np
import pickle
import gzip
import os
import io
from openai import OpenAI
from fpdf import FPDF

# === Load Model and Scaler ===
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "compressed_xgb_grid.pkl.gz")
scaler_path = os.path.join(base_path, "scaler.pkl")

with gzip.open(model_path, "rb") as file:
    model = pickle.load(file)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# === OpenRouter API Key ===
api_key = st.secrets["openrouter_api_key"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# === App Configuration ===
st.set_page_config(page_title="Heart Health Risk Analysis", page_icon="💖", layout="wide")

# === Header Section with Power BI Link ===
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="color: #FF4B4B; font-size: 50px;">Heart Health Risk Analyzer 💖</h1>
        <a href="https://app.powerbi.com/links/lmizRoRfol?ctid=6ab2fdb5-cc9e-42b3-8fb9-08ee58b9c4a1&pbi_source=linkShare" target="_blank">
            <button style="background-color: #00BFFF; color: white; padding: 10px 20px; border: none; border-radius: 8px; font-size: 18px;">📊 View Chart</button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# === Sidebar Assistant ===
with st.sidebar:
    st.title("💬 AI Assistant")
    st.caption("Need help or have questions? Just ask!")
    chat_query = st.text_input("Type your question here:")

    if chat_query:
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://yourappname.streamlit.app",
                        "X-Title": "HeartRiskPredictorApp",
                    },
                    model="deepseek/deepseek-chat-v3-0324:free",
                    messages=[{"role": "user", "content": chat_query}]
                )
                st.success(response.choices[0].message.content)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# === Collect User Inputs ===
st.subheader("👤 Personal & Lifestyle Details")

with st.expander("📄 Basic Information"):
    age = st.slider("Age", 1, 100, 30)
    sex = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["1 (High School)", "2 (Diploma)", "3 (Graduate)", "4 (Post Graduate/Doctorate)"])

with st.expander("🚬 Lifestyle Habits"):
    cigs_per_day = st.slider("Cigarettes Smoked Per Day", 0, 50, 0)

with st.expander("🩺 Medical History"):
    bp_meds = st.radio("Are you taking blood pressure medication?", ["Yes", "No"])
    prevalent_stroke = st.radio("Have you ever had a stroke?", ["Yes", "No"])
    prevalent_hyp = st.radio("Do you have hypertension?", ["Yes", "No"])
    diabetes = st.radio("Do you have diabetes?", ["Yes", "No"])

with st.expander("🏥 Current Medical Stats"):
    tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, max_value=600.0, value=200.0)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
    heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 75)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=40.0, max_value=400.0, value=100.0)
    pulse_pressure = st.number_input("Pulse Pressure (mm Hg)", min_value=0.0, max_value=100.0, value=40.0)

# === Prediction Section ===
st.markdown("---")
centered_col = st.columns([1, 3, 1])[1]

risk_prediction = None

with centered_col:
    if st.button("🎯 Predict My Risk", use_container_width=True):
        sex_int = 1 if sex == "Male" else 0
        bp_meds_int = 1 if bp_meds == "Yes" else 0
        prevalent_stroke_int = 1 if prevalent_stroke == "Yes" else 0
        prevalent_hyp_int = 1 if prevalent_hyp == "Yes" else 0
        diabetes_int = 1 if diabetes == "Yes" else 0

        input_data = np.array([
            age,
            int(education[0]),
            sex_int,
            cigs_per_day,
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
        risk_prediction = prediction[0]

        st.subheader("🔎 Your Risk Assessment:")
        if risk_prediction == 1:
            st.error("⚠️ High Risk of Heart Disease. Please consult a medical professional!")
        else:
            st.success("✅ Low Risk! Keep maintaining a healthy lifestyle!")

# === Personalized Recommendations ===
if risk_prediction is not None:
    with st.expander("💡 Personalized Recommendations"):
        if risk_prediction == 1:
            st.markdown("""
            - **Visit your doctor** for a complete cardiac evaluation.
            - **Quit smoking** immediately if you smoke.
            - **Adopt a heart-healthy diet**: more vegetables, fruits, lean proteins.
            - **Exercise regularly**: minimum 30 minutes a day.
            - **Manage your stress** through meditation or yoga.
            """)
        else:
            st.markdown("""
            - **Keep up your healthy habits!** 🎉
            - Continue eating nutritious food.
            - Stay physically active.
            - Regular health checkups are still important!
            - Avoid smoking or excessive alcohol consumption.
            """)

# === FAQ and Educational Section ===
with st.expander("📚 Heart Health FAQs"):
    st.markdown("""
    **Q: What are the major risk factors for heart disease?**  
    A: Smoking, high blood pressure, high cholesterol, diabetes, and obesity.

    **Q: Can heart disease be reversed?**  
    A: With early detection and major lifestyle changes, it's possible to significantly reduce risks.

    **Q: How often should I get screened?**  
    A: Adults should check cholesterol every 4-6 years, blood pressure yearly, and glucose levels every 3 years.

    **Q: Does family history matter?**  
    A: Yes. Genetics can influence heart disease risk, so share family history with your doctor.
    """)

# === Export and Share Results (Updated) ===
if risk_prediction is not None:
    with st.expander("📤 Export and Share Your Results"):
        if st.button("📄 Download My Report (PDF)"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Heart Risk Prediction Report", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
            pdf.cell(200, 10, txt=f"Sex: {sex}", ln=True)
            pdf.cell(200, 10, txt=f"Education: {education}", ln=True)
            pdf.cell(200, 10, txt=f"Cigarettes Per Day: {cigs_per_day}", ln=True)
            pdf.cell(200, 10, txt=f"Taking BP Medication: {bp_meds}", ln=True)
            pdf.cell(200, 10, txt=f"History of Stroke: {prevalent_stroke}", ln=True)
            pdf.cell(200, 10, txt=f"Hypertension: {prevalent_hyp}", ln=True)
            pdf.cell(200, 10, txt=f"Diabetes: {diabetes}", ln=True)
            pdf.cell(200, 10, txt=f"Total Cholesterol: {tot_chol} mg/dL", ln=True)
            pdf.cell(200, 10, txt=f"BMI: {bmi}", ln=True)
            pdf.cell(200, 10, txt=f"Heart Rate: {heart_rate} bpm", ln=True)
            pdf.cell(200, 10, txt=f"Glucose Level: {glucose} mg/dL", ln=True)
            pdf.cell(200, 10, txt=f"Pulse Pressure: {pulse_pressure} mm Hg", ln=True)
            pdf.ln(10)
            pdf.set_font("Arial", "B", size=12)
            pdf.cell(200, 10, txt=f"Heart Risk Status: {'High Risk' if risk_prediction==1 else 'Low Risk'}", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt="Thank you for using Heart Health Risk Analyzer! Stay healthy! 💖")

            buffer = io.BytesIO()
            pdf.output(buffer)
            buffer.seek(0)

            st.download_button(
                label="📥 Click to Download Report",
                data=buffer,
                file_name="heart_risk_report.pdf",
                mime="application/pdf"
            )

# === Footer ===
st.markdown("---")
st.caption("Built with ❤️ | Powered by Machine Learning and OpenAI")
import streamlit as st
import numpy as np
import pickle
import gzip
import os
import io
from openai import OpenAI
from fpdf import FPDF

# === Load Model and Scaler ===
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "compressed_xgb_grid.pkl.gz")
scaler_path = os.path.join(base_path, "scaler.pkl")

with gzip.open(model_path, "rb") as file:
    model = pickle.load(file)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# === OpenRouter API Key ===
api_key = st.secrets["openrouter_api_key"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# === App Configuration ===
st.set_page_config(page_title="Heart Health Risk Analysis", page_icon="💖", layout="wide")

# === Header Section with Power BI Link ===
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="color: #FF4B4B; font-size: 50px;">Heart Health Risk Analyzer 💖</h1>
        <a href="https://app.powerbi.com/links/lmizRoRfol?ctid=6ab2fdb5-cc9e-42b3-8fb9-08ee58b9c4a1&pbi_source=linkShare" target="_blank">
            <button style="background-color: #00BFFF; color: white; padding: 10px 20px; border: none; border-radius: 8px; font-size: 18px;">📊 View Chart</button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# === Sidebar Assistant ===
with st.sidebar:
    st.title("💬 AI Assistant")
    st.caption("Need help or have questions? Just ask!")
    chat_query = st.text_input("Type your question here:")

    if chat_query:
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://yourappname.streamlit.app",
                        "X-Title": "HeartRiskPredictorApp",
                    },
                    model="deepseek/deepseek-chat-v3-0324:free",
                    messages=[{"role": "user", "content": chat_query}]
                )
                st.success(response.choices[0].message.content)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# === Collect User Inputs ===
st.subheader("👤 Personal & Lifestyle Details")

with st.expander("📄 Basic Information"):
    age = st.slider("Age", 1, 100, 30)
    sex = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["1 (High School)", "2 (Diploma)", "3 (Graduate)", "4 (Post Graduate/Doctorate)"])

with st.expander("🚬 Lifestyle Habits"):
    cigs_per_day = st.slider("Cigarettes Smoked Per Day", 0, 50, 0)

with st.expander("🩺 Medical History"):
    bp_meds = st.radio("Are you taking blood pressure medication?", ["Yes", "No"])
    prevalent_stroke = st.radio("Have you ever had a stroke?", ["Yes", "No"])
    prevalent_hyp = st.radio("Do you have hypertension?", ["Yes", "No"])
    diabetes = st.radio("Do you have diabetes?", ["Yes", "No"])

with st.expander("🏥 Current Medical Stats"):
    tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, max_value=600.0, value=200.0)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
    heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 75)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=40.0, max_value=400.0, value=100.0)
    pulse_pressure = st.number_input("Pulse Pressure (mm Hg)", min_value=0.0, max_value=100.0, value=40.0)

# === Prediction Section ===
st.markdown("---")
centered_col = st.columns([1, 3, 1])[1]

risk_prediction = None

with centered_col:
    if st.button("🎯 Predict My Risk", use_container_width=True):
        sex_int = 1 if sex == "Male" else 0
        bp_meds_int = 1 if bp_meds == "Yes" else 0
        prevalent_stroke_int = 1 if prevalent_stroke == "Yes" else 0
        prevalent_hyp_int = 1 if prevalent_hyp == "Yes" else 0
        diabetes_int = 1 if diabetes == "Yes" else 0

        input_data = np.array([
            age,
            int(education[0]),
            sex_int,
            cigs_per_day,
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
        risk_prediction = prediction[0]

        st.subheader("🔎 Your Risk Assessment:")
        if risk_prediction == 1:
            st.error("⚠️ High Risk of Heart Disease. Please consult a medical professional!")
        else:
            st.success("✅ Low Risk! Keep maintaining a healthy lifestyle!")

# === Personalized Recommendations ===
if risk_prediction is not None:
    with st.expander("💡 Personalized Recommendations"):
        if risk_prediction == 1:
            st.markdown("""
            - **Visit your doctor** for a complete cardiac evaluation.
            - **Quit smoking** immediately if you smoke.
            - **Adopt a heart-healthy diet**: more vegetables, fruits, lean proteins.
            - **Exercise regularly**: minimum 30 minutes a day.
            - **Manage your stress** through meditation or yoga.
            """)
        else:
            st.markdown("""
            - **Keep up your healthy habits!** 🎉
            - Continue eating nutritious food.
            - Stay physically active.
            - Regular health checkups are still important!
            - Avoid smoking or excessive alcohol consumption.
            """)

# === FAQ and Educational Section ===
with st.expander("📚 Heart Health FAQs"):
    st.markdown("""
    **Q: What are the major risk factors for heart disease?**  
    A: Smoking, high blood pressure, high cholesterol, diabetes, and obesity.

    **Q: Can heart disease be reversed?**  
    A: With early detection and major lifestyle changes, it's possible to significantly reduce risks.

    **Q: How often should I get screened?**  
    A: Adults should check cholesterol every 4-6 years, blood pressure yearly, and glucose levels every 3 years.

    **Q: Does family history matter?**  
    A: Yes. Genetics can influence heart disease risk, so share family history with your doctor.
    """)

# === Export and Share Results (Updated) ===
if risk_prediction is not None:
    with st.expander("📤 Export and Share Your Results"):
        if st.button("📄 Download My Report (PDF)"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Heart Risk Prediction Report", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
            pdf.cell(200, 10, txt=f"Sex: {sex}", ln=True)
            pdf.cell(200, 10, txt=f"Education: {education}", ln=True)
            pdf.cell(200, 10, txt=f"Cigarettes Per Day: {cigs_per_day}", ln=True)
            pdf.cell(200, 10, txt=f"Taking BP Medication: {bp_meds}", ln=True)
            pdf.cell(200, 10, txt=f"History of Stroke: {prevalent_stroke}", ln=True)
            pdf.cell(200, 10, txt=f"Hypertension: {prevalent_hyp}", ln=True)
            pdf.cell(200, 10, txt=f"Diabetes: {diabetes}", ln=True)
            pdf.cell(200, 10, txt=f"Total Cholesterol: {tot_chol} mg/dL", ln=True)
            pdf.cell(200, 10, txt=f"BMI: {bmi}", ln=True)
            pdf.cell(200, 10, txt=f"Heart Rate: {heart_rate} bpm", ln=True)
            pdf.cell(200, 10, txt=f"Glucose Level: {glucose} mg/dL", ln=True)
            pdf.cell(200, 10, txt=f"Pulse Pressure: {pulse_pressure} mm Hg", ln=True)
            pdf.ln(10)
            pdf.set_font("Arial", "B", size=12)
            pdf.cell(200, 10, txt=f"Heart Risk Status: {'High Risk' if risk_prediction==1 else 'Low Risk'}", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt="Thank you for using Heart Health Risk Analyzer! Stay healthy! 💖")

            buffer = io.BytesIO()
            pdf.output(buffer)
            buffer.seek(0)

            st.download_button(
                label="📥 Click to Download Report",
                data=buffer,
                file_name="heart_risk_report.pdf",
                mime="application/pdf"
            )

# === Footer ===
st.markdown("---")
st.caption("Built with ❤️ | Powered by Machine Learning and OpenAI")
