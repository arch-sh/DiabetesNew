import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load model and dataset
model = pickle.load(open('model.pkl', 'rb'))
dataset = pd.read_csv('diabetes.csv')

# Extract the relevant features used for prediction
dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values

# Fit scaler to the dataset
sc = MinMaxScaler(feature_range=(0, 1))
sc.fit(dataset_X)

# Streamlit UI
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction System")
st.markdown("Enter the values below to predict diabetes risk:")

# Input fields (in order: [1, 2, 5, 7])
glucose = st.number_input("Glucose Level", min_value=0.0, format="%.2f")
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, format="%.2f")
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
age = st.number_input("Age", min_value=0.0, format="%.2f")

if st.button("Predict"):
    input_features = np.array([[glucose, blood_pressure, bmi, age]])
    scaled_input = sc.transform(input_features)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("⚠️ You have Diabetes, please consult a Doctor.")
    else:
        st.success("✅ You don't have Diabetes.")