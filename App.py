# app.py
import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Step 1: Load trained pipeline
# ------------------------------
# We use the pipeline you saved earlier (preprocessing + RandomForest model)
pipeline = joblib.load("heart_disease_pipeline.pkl")

# ------------------------------
# Step 2: UI Header
# ------------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient health data and get a real-time prediction")

# ------------------------------
# Step 3: User Input Fields
# ------------------------------
age = st.number_input("Age", 20, 100, 55)

sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])

cp = st.selectbox("Chest Pain Type (1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic)", [1, 2, 3, 4])

trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)

chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (0=no, 1=yes)", [0, 1])

restecg = st.selectbox("Resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)", [0, 1, 2])

thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 220, 150)

exang = st.selectbox("Exercise Induced Angina (0=no, 1=yes)", [0, 1])

oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)

slope = st.selectbox("Slope of ST Segment (1=upsloping, 2=flat, 3=downsloping)", [1, 2, 3])

ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

thal = st.selectbox("Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)", [3, 6, 7])

# Collect all inputs in a dictionary
sample = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}
sample_df = pd.DataFrame([sample])

# ------------------------------
# Step 4: Prediction
# ------------------------------
if st.button("🔍 Predict"):
    prediction = pipeline.predict(sample_df)[0]
    probability = pipeline.predict_proba(sample_df)[0][1]  # prob of disease

    if prediction == 1:
        st.error(f"⚠️ Patient is **likely to have Heart Disease** with probability {probability*100:.1f}%")
    else:
        st.success(f"✅ Patient is **healthy** with probability {(1-probability)*100:.1f}%")

# ------------------------------
# Step 5: Data Visualization
# ------------------------------
st.subheader("📊 Explore Heart Disease Trends")
uploaded_file = st.file_uploader("Upload dataset (CSV) to explore trends", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:", data.head())

    # Show simple statistics
    st.write("Disease distribution:")
    st.bar_chart(data["target"].value_counts())

    # Correlation heatmap (optional, simple)
    st.write("Feature correlation with Heart Disease:")
    st.bar_chart(data.corr()["target"].sort_values(ascending=False))
