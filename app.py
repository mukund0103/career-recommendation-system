import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# Load model and encoders
# ---------------------------
model = joblib.load("career_model.joblib")
mlb_skills = joblib.load("mlb_skills.joblib")
mlb_interests = joblib.load("mlb_interests.joblib")
label_encoder = joblib.load("label_encoder.joblib")
feature_names = joblib.load("feature_names.joblib")  # List of all features from training

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Career Recommendation System", layout="centered")
st.title("🎯 Career Recommendation System")

st.write("Enter your details below to get a career recommendation.")

# Age input
age = st.number_input("Enter your Age:", min_value=15, max_value=60, value=22)

# Education input
education_cols = [col for col in feature_names if col.startswith("Education_")]
education_options = [col.replace("Education_", "") for col in education_cols]
education = st.selectbox("Select your Education:", education_options)

# Skills input
skills_list = list(mlb_skills.classes_)
skills = st.multiselect("Select your Skills:", skills_list)

# Interests input
interests_list = list(mlb_interests.classes_)
interests = st.multiselect("Select your Interests:", interests_list)

# Predict button
if st.button("Predict Career"):
    if not education or not skills or not interests:
        st.error("Please fill in all fields!")
    else:
        # ---------------------------
        # Build input vector
        # ---------------------------
        input_df = pd.DataFrame(columns=feature_names)
        input_df.loc[0] = 0  # initialize all columns to 0

        # Age
        input_df['Age'] = age

        # Education
        edu_col = f"Education_{education}"
        if edu_col in input_df.columns:
            input_df[edu_col] = 1

        # Skills
        skills_encoded = mlb_skills.transform([skills])
        skills_cols = ["skill_" + col for col in mlb_skills.classes_]
        for i, col in enumerate(skills_cols):
            if col in input_df.columns:
                input_df[col] = skills_encoded[0][i]

        # Interests
        interests_encoded = mlb_interests.transform([interests])
        interests_cols = ["interest_" + col for col in mlb_interests.classes_]
        for i, col in enumerate(interests_cols):
            if col in input_df.columns:
                input_df[col] = interests_encoded[0][i]

        # ---------------------------
        # Prediction
        # ---------------------------
        prediction_encoded = model.predict(input_df.values)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"✅ Recommended Career: **{prediction}**")
