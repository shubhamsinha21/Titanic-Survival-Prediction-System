# Titanic Survival Prediction System

An end-to-end Machine Learning project that predicts whether a passenger would survive the Titanic disaster based on passenger information.

---

## Project Overview

This project uses the Titanic dataset to build a machine learning model capable of predicting passenger survival.

The project covers the complete machine learning workflow including:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training
- Model Evaluation
- Cross Validation
- Hyperparameter Tuning
- Model Deployment using Streamlit

---

## Problem Statement

Predict whether a passenger would survive the Titanic disaster using passenger demographic and travel information.

Target Variable:

- Survived
    - 1 = Survived
    - 0 = Did Not Survive

---

## Dataset Features

- Passenger Class (Pclass)
- Gender (Sex)
- Age
- Siblings / Spouse Aboard (SibSp)
- Parents / Children Aboard (Parch)
- Fare
- Embarked Port

Engineered Features:

- FamilySize
- IsAlone
- AgeGroup
- FareGroup

---

## Exploratory Data Analysis

Key Insights:

- Female passengers had significantly higher survival rates.
- First-class passengers were more likely to survive.
- Younger passengers had better survival chances.
- Fare showed a positive relationship with survival.
- Family size influenced survival probability.

---

## Models Trained

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

---

## Final Model

Support Vector Machine (SVM)

Test Accuracy:

83.24%

The baseline SVM model outperformed the tuned model on the test dataset and was therefore selected as the final deployment model.

---

## Technologies Used

Python

Libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Streamlit
- Joblib

---

## Project Workflow

Data Collection

↓

Data Cleaning

↓

Exploratory Data Analysis

↓

Feature Engineering

↓

Data Preprocessing

↓

Model Training

↓

Model Evaluation

↓

Model Saving

↓

Streamlit Application

↓

Deployment

---

## Installation

Clone Repository

```bash
git clone https://github.com/yourusername/titanic-survival-prediction-system.git
```

Move into Project Directory

```bash
cd titanic-survival-prediction-system
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit Application

```bash
streamlit run app.py
```

---

## Screenshots

Add application screenshots here after deployment.

---

## Future Improvements

- Model Explainability
- Advanced Feature Engineering
- Docker Support
- Cloud Deployment
- Automated ML Pipeline

---

## Author

Shubham Sinha | AI Engineer