import streamlit as st
import pandas as pd
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Titanic Survival Prediction System",
    page_icon="🚢",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 3.2rem;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD ARTIFACTS
# ==================================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Project Information")

st.sidebar.success(
    "End-to-End Machine Learning Project"
)

st.sidebar.markdown("""
### Model Details

**Model:** Support Vector Machine (SVM)

**Test Accuracy:** 83.24%

### Features Used

- Passenger Class
- Gender
- Age
- Fare
- Family Information
- Embarked Port

### Tech Stack

- Python
- Pandas
- Scikit-Learn
- Streamlit
""")

# ==================================================
# HERO SECTION
# ==================================================

st.markdown("""
# 🚢 Titanic Survival Prediction System

Predict whether a passenger would have survived the Titanic disaster using Machine Learning.

---
""")

# ==================================================
# INPUT SECTION
# ==================================================

st.subheader("Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    age = st.slider(
        "Age",
        min_value=1,
        max_value=80,
        value=25
    )

with col2:

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=50.0
    )

with col3:

    embarked = st.selectbox(
        "Embarked",
        ["C", "Q", "S"]
    )

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

# ==================================================
# FEATURE ENGINEERING
# ==================================================

family_size = sibsp + parch + 1

is_alone = int(family_size == 1)

# Age Group

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

# Fare Group

if fare <= 8:
    fare_group = "Low"

elif fare <= 15:
    fare_group = "Medium"

elif fare <= 32:
    fare_group = "High"

else:
    fare_group = "Premium"

# ==================================================
# PASSENGER SUMMARY
# ==================================================

st.markdown("### Passenger Summary")

summary_df = pd.DataFrame({
    "Feature": [
        "Passenger Class",
        "Gender",
        "Age",
        "Fare",
        "Family Size",
        "Embarked"
    ],
    "Value": [
        pclass,
        sex,
        age,
        fare,
        family_size,
        embarked
    ]
})

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)

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

for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

# ==================================================
# SCALE
# ==================================================

input_scaled = scaler.transform(input_df)

# ==================================================
# PREDICTION
# ==================================================

if st.button("Predict Survival"):

    with st.spinner("Analyzing passenger information..."):

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

    st.divider()

    if prediction == 1:

        st.success(
            "✅ Passenger Likely Survived"
        )

        st.metric(
            "Survival Probability",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

    else:

        st.error(
            "❌ Passenger Likely Did Not Survive"
        )

        st.metric(
            "Death Probability",
            f"{(1 - probability):.2%}"
        )

        st.progress(float(1 - probability))

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown("""
### About This Project

This application predicts whether a passenger would survive the Titanic disaster using Machine Learning.

**Workflow**

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Cross Validation
- Hyperparameter Tuning
- Streamlit Deployment

**Final Model:** Support Vector Machine (SVM)

**Dataset:** Titanic Dataset
""")