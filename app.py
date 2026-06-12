import streamlit as st
import pandas as pd
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# ==================================================
# LOAD ARTIFACTS
# ==================================================

model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
columns = joblib.load("artifacts/columns.pkl")

# ==================================================
# TITLE
# ==================================================

st.title("🚢 Titanic Survival Prediction System")

st.markdown(
    """
Predict whether a passenger would survive the Titanic disaster
based on passenger details.
"""
)

# ==================================================
# USER INPUTS
# ==================================================

st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    age = st.slider(
        "Age",
        min_value=1,
        max_value=80,
        value=25
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=50.0
    )

with col2:

    sibsp = st.number_input(
        "Siblings / Spouse",
        min_value=0,
        max_value=10,
        value=0
    )

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    embarked = st.selectbox(
        "Embarked",
        ["C", "Q", "S"]
    )

# ==================================================
# FEATURE ENGINEERING
# ==================================================

family_size = sibsp + parch + 1

is_alone = int(family_size == 1)

# AgeGroup

if age <= 12:
    age_group = "Child"

elif age <= 18:
    age_group = "Teen"

elif age <= 35:
    age_group = "YoungAdult"

elif age <= 60:
    age_group = "Adult"

else:
    age_group = "Senior"

# FareGroup

if fare <= 8:
    fare_group = "Low"

elif fare <= 15:
    fare_group = "Medium"

elif fare <= 32:
    fare_group = "High"

else:
    fare_group = "Premium"

# ==================================================
# CREATE INPUT DATAFRAME
# ==================================================

input_df = pd.DataFrame({
    "pclass": [pclass],
    "sex": [sex],
    "age": [age],
    "sibsp": [sibsp],
    "parch": [parch],
    "fare": [fare],
    "embarked": [embarked],
    "FamilySize": [family_size],
    "IsAlone": [is_alone],
    "AgeGroup": [age_group],
    "FareGroup": [fare_group]
})

# ==================================================
# SAME PREPROCESSING AS TRAINING
# ==================================================

input_df = pd.get_dummies(
    input_df,
    drop_first=True
)

# Add missing columns

for col in columns:

    if col not in input_df.columns:
        input_df[col] = 0

# Ensure same order

input_df = input_df[columns]

# ==================================================
# SCALE
# ==================================================

input_scaled = scaler.transform(
    input_df
)

# ==================================================
# PREDICTION
# ==================================================

if st.button("Predict Survival"):

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    st.divider()

    if prediction == 1:

        st.success(
            f"✅ Passenger Likely Survived"
        )

        st.metric(
            "Survival Probability",
            f"{probability:.2%}"
        )

    else:

        st.error(
            f"❌ Passenger Likely Did Not Survive"
        )

        st.metric(
            "Death Probability",
            f"{(1 - probability):.2%}"
        )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Titanic Survival Prediction System | Machine Learning Project"
)