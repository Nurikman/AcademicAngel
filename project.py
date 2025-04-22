# app.py
# Streamlit application for student GPA prediction, SHAP explanation, and AI-driven study advice

"""
Installation:
    pip install streamlit xgboost shap openai pandas matplotlib

Set environment variable:
    export OPENAI_API_KEY="your_openai_api_key"
    (Windows PowerShell: setx OPENAI_API_KEY "your_openai_api_key")

Run:
    streamlit run app.py
"""



import os
import pandas as pd
import xgboost as xgb
import shap
import openai
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI


from typing import Tuple, Dict

@st.cache_data
def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load student performance data and prepare features and target.
    """
    df = pd.read_csv(path, sep=',')
    df.columns = df.columns.str.strip()
    X = df.drop(columns=['StudentID', 'GradeClass', 'GPA'])
    y = df['GPA']
    return X, y

@st.cache_data
def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """
    Instantiate and train XGBRegressor with predefined hyperparameters.
    """
    model = xgb.XGBRegressor(
        subsample=1.0,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

@st.cache_resource
def make_explainer(_model: xgb.XGBRegressor, X: pd.DataFrame) -> shap.Explainer:
    """
    Create a SHAP explainer using the full training set as background.
    """
    explainer = shap.Explainer(_model, X)
    return explainer

def predict_and_explain(
    inputs: Dict[str, float],
    model: xgb.XGBRegressor,
    explainer: shap.Explainer
) -> Tuple[float, pd.DataFrame, shap.Explanation]:
    """
    Predict GPA and compute SHAP explanations for a single student input.

    Returns:
        pred_gpa: float
        contrib_df: DataFrame of feature, value, shap_value (filtered & sorted)
        shap_exp: Explanation object for waterfall plot (filtered)
    """
    feat_df = pd.DataFrame([inputs])
    # reorder columns to match the training order
    feat_df = feat_df[model.feature_names_in_]    
    pred_gpa = float(model.predict(feat_df)[0])

    # Compute SHAP values
    shap_vals = explainer(feat_df)

    # Build contributions DataFrame
    contrib = pd.DataFrame({
        'feature': feat_df.columns,
        'value': feat_df.iloc[0].values,
        'shap_value': shap_vals.values[0]
    })
    # Filter out static demographics
    contrib = contrib[~contrib['feature'].isin(['Age', 'Gender', 'Ethnicity'])]
    # Sort by absolute impact
    contrib = contrib.reindex(contrib.shap_value.abs().sort_values(ascending=False).index)

    # Prepare Explanation for waterfall (filter mask)
    mask = [f not in ('Age', 'Gender', 'Ethnicity') for f in feat_df.columns]
    expl_sub = shap.Explanation(
        values       = shap_vals.values[0][mask],
        base_values  = shap_vals.base_values[0],
        data         = feat_df.values[0][mask],
        feature_names= feat_df.columns[mask]
    )

    return pred_gpa, contrib, expl_sub

def get_advice(pred_gpa: float, contrib_df: pd.DataFrame) -> str:
    """
    Generate personalized study advice using OpenAI ChatCompletion API.
    """
    os.environ["OPENAI_API_KEY"] = "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"
    openai_client = OpenAI()
    openai_client.api_key = os.getenv("OPENAI_API_KEY")


    table_md = contrib_df.to_markdown(index=False)

    system_msg = (
        "You are a friendly and supportive academic advisor speaking directly to a student. "
        "First, share the student’s predicted GPA. Then highlight what the student is already doing well, "
        "and finally suggest concrete improvements to raise the GPA further."
    )
    user_msg = (
        f"Hi! Based on your profile, your predicted GPA is **{pred_gpa:.2f}**.\n\n"
        "Here are the key factors influencing your GPA (positive values help, negative values hurt):\n\n"
        f"{table_md}\n\n"
        "Please:\n"
        "1. Affirm where you’re doing well.\n"
        "2. Recommend 3–5 specific actions to improve weaker areas.\n"
        "3. Explain why each action matters in terms of these features.\n\n"
        "Write in a supportive, student‑friendly tone."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return ""

# === Streamlit App ===

def main():
    st.title("Student GPA Predictor & Academic Advisor")
    st.markdown(
        "Fill out the form below to predict your GPA, see what factors help or hurt your score, and get personalized advice!"
    )

    # Load and train
    X, y = load_data('Student_performance_data.txt')
    model = train_model(X, y)
    explainer = make_explainer(model, X)

    # Sidebar form
    with st.form("survey_form"):
        st.subheader("Student Survey")
        age = st.slider("Age:", min_value=15, max_value=18, value=16)
        gender = st.radio("Gender:", options={"Male": 0, "Female": 1})
        ethnicity = st.selectbox(
            "Ethnicity:", options={
                "Caucasian": 0,
                "African American": 1,
                "Asian": 2,
                "Other": 3
            }
        )
        parental_education = st.selectbox(
            "Parental Education:",
            options={"None": 0, "High School": 1, "Some College": 2, "Bachelor's": 3, "Higher": 4}
        )
        study_time = st.slider("Study Time (hours/week):", 0, 20, 5)
        absences = st.slider("Absences during year:", 0, 30, 3)
        tutoring = st.checkbox("Tutoring (Yes)?")
        extracurricular = st.checkbox("Extracurricular Activities?")
        sports = st.checkbox("Sports Participation?")
        music = st.checkbox("Music Activities?")
        volunteering = st.checkbox("Volunteering?")
        parental_support = st.selectbox(
            "Parental Support Level:",
            options={"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
        )

        submitted = st.form_submit_button("Predict & Advise")

        if submitted:
            # map dropdown/radio selections to numeric codes
            gender_map = {"Male": 0, "Female": 1}
            ethnicity_map = {
                "Caucasian": 0,
                "African American": 1,
                "Asian": 2,
                "Other": 3
            }
            educ_map = {
                "None": 0,
                "High School": 1,
                "Some College": 2,
                "Bachelor's": 3,
                "Higher": 4
            }
            support_map = {
                "None": 0,
                "Low": 1,
                "Moderate": 2,
                "High": 3,
                "Very High": 4
            }

            gender = gender_map[gender]
            ethnicity = ethnicity_map[ethnicity]
            parental_education = educ_map[parental_education]
            parental_support = support_map[parental_support]

            # Now build the numeric-only inputs dict
            inputs = {
                'Age': age,
                'Gender': gender,
                'Ethnicity': ethnicity,
                'ParentalEducation': parental_education,
                'StudyTimeWeekly': study_time,
                'Absences': absences,
                'Tutoring': int(tutoring),
                'Extracurricular': int(extracurricular),
                'Sports': int(sports),
                'Music': int(music),
                'Volunteering': int(volunteering),
                'ParentalSupport': parental_support
            }
            pred_gpa, contrib_df, shap_exp = predict_and_explain(inputs, model, explainer)
            st.metric("Predicted GPA", f"{pred_gpa:.2f}")


            # Predict and explain

            st.subheader("Top Influential Factors")
            st.table(contrib_df[['feature', 'value', 'shap_value']].head(10))

            st.subheader("SHAP Waterfall Explanation")
            plt.figure(figsize=(8, 4))
            shap.plots.waterfall(shap_exp, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

            st.subheader("Personalized Advice")
            advice = get_advice(pred_gpa, contrib_df)
            st.markdown(advice)

if __name__ == "__main__":
    main()
