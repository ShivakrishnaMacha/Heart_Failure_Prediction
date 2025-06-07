# 🫀 Heart Failure Prediction (Framingham Heart Study)

This project predicts the 10-year risk of coronary heart disease (CHD) using health data from an ongoing cardiovascular study in Framingham, Massachusetts. By applying various classification models, we aim to identify high-risk individuals early and support preventive healthcare.

🔗 **Live App on Streamlit:**  
[👉 Heart Disease Risk Predictor](https://heartfailureprediction-gorv3hkqohajttbgpv932z.streamlit.app/)
![Image](https://github.com/user-attachments/assets/9dd05b6f-bf33-4d5f-92f2-114066bf12ba)


---

## 📌 Objective

Build and evaluate classification machine learning models that predict whether a patient will develop CHD within the next 10 years based on medical, demographic, and behavioral features.

---

## 📊 Dataset Overview

- **Source:** Framingham Heart Study
- **Records:** ~4,000 patients
- **Features:** 15 variables
  - **Demographic:**  
    - `sex`: Male or Female  
    - `age`: Age in years  
    - `education`: Education level (1 to 4)  
  - **Behavioral:**  
    - `is_smoking`: Smoker (YES/NO)  
    - `cigsPerDay`: Cigarettes per day (numeric)  
  - **Medical History:**  
    - `BPMeds`: On BP medication (0 or 1)  
    - `prevalentStroke`: Previous stroke (0 or 1)  
    - `prevalentHyp`: Hypertensive (0 or 1)  
    - `diabetes`: Diabetic (0 or 1)  
  - **Current Medical:**  
    - `totChol`: Total cholesterol  
    - `sysBP`: Systolic blood pressure  
    - `diaBP`: Diastolic blood pressure  
    - `BMI`: Body Mass Index  
    - `heartRate`: Heart rate  
    - `glucose`: Glucose level  

---

## 📈 Exploratory Data Analysis (EDA) Insights

### Categorical Features:
- The target variable `TenYearCHD` is imbalanced; fewer positive cases.
- Education level 1 shows a higher proportion of CHD risk.
- Male patients have a greater risk despite fewer numbers compared to females.
- Smokers and non-smokers show similar CHD risk proportions.
- Very few at-risk patients are on blood pressure medication.
- Most at-risk individuals are not diabetic.

### Continuous Features:
- Many features show **positive skewness** and required transformation.
- Risk-prone age range: **51–63 years**
- Smokers at risk usually smoke **<5 cigarettes/day**
- Systolic BP: **120–150 mmHg**, Diastolic BP: **75–85 mmHg**
- BMI: **24–26**, Heart Rate: **70–80 bpm**, Glucose: **50–100 mg/dL**

---

## 🛠️ Feature Engineering

- Dropped `is_smoking` due to high correlation with `cigsPerDay`.
- Introduced `pulse_pressure = sysBP - diaBP`.
- Applied `log1p` transformation to correct skewness in continuous variables.
- Handled outliers by retaining data within ±3 standard deviations.
- Addressed class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.
- Data split into **train/test sets**, scaled with **MinMaxScaler**.

---

## 🤖 Machine Learning Models Used

Five classification models were trained and evaluated:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
5. XGBoost

- **Hyperparameter tuning**: Performed using `GridSearchCV`
- **Cross-validation**: 5-fold cross-validation for reliable performance estimation
- **Metric of focus**: **Recall** — critical in medical diagnosis to minimize false negatives

---

## 🏆 Model Performance

- **XGBoost (after tuning)** delivered the **best recall score (0.98)** on test data.
- This means 98 out of 100 high-risk individuals were correctly identified.

---

## 📊 Confusion Matrix & Feature Importance

- Final model shows a strong ability to correctly identify positive cases.
- Top contributing features:
  - `age`
  - `diabetes`
  - `heartRate`
- Least contributing features:
  - `BMI`
  - `totChol`

---

## ✅ Conclusion

- A comprehensive ML pipeline was developed for predicting CHD risk:
  - Data preprocessing: Imputation, scaling, transformation
  - Feature engineering: Pulse pressure, outlier handling, correlation management
  - Class imbalance handling via SMOTE
  - Model evaluation focused on **Recall** to prioritize patient safety
- The final tuned XGBoost model proved to be the most effective
- High recall performance makes it a **useful tool for early CHD risk screening**

---

## 🗂️ Repository Contents

- `notebooks/`: EDA, modeling, and tuning notebooks
- `app/`: Streamlit web app files
- `cardiovascular_risk_data.csv`: Original dataset
- `FINAL PPT CAPSTONE.pptx`: Project presentation
- `requirements.txt`: Python dependencies

---

## 🛠️ Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Imbalanced-learn (SMOTE)
- Streamlit

---

## 🌐 Live Web App

🎯 Check out the interactive predictor here:  
[https://heartfailureprediction-gorv3hkqohajttbgpv932z.streamlit.app/](https://heartfailureprediction-gorv3hkqohajttbgpv932z.streamlit.app/)

---

## 📬 Contact

For questions or collaborations, reach out via GitHub or email.

---

## 📚 Acknowledgments

- Framingham Heart Study for the dataset
- Scikit-learn and XGBoost communities
- Streamlit for enabling rapid model deployment
