import streamlit as st
import pickle
import json
import numpy as np
import random
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Load model, encoders, questions, and data for IG calculation
@st.cache_resource(show_spinner=False)
def load_resources():
    with open("model.pkl", "rb") as f:
        model, encoders, target_encoder = pickle.load(f)
    with open("questions.json", "r") as f:
        questions = json.load(f)
    # Load training data for IG calculation
    data = pd.read_csv("training_data.csv")  # Provide your CSV here
    return model, encoders, target_encoder, questions, data

model, encoders, target_encoder, questions, data = load_resources()

st.set_page_config(page_title="Career Path Predictor with Feature Selection", layout="centered")

st.title("🎯 Career Path Predictor with Feature Selection")

st.markdown("""
This app predicts your ideal career field using a Decision Tree model.
You can select features based on their **Information Gain** to customize the questionnaire.
""")

# Separate features and target for IG
target_col = "Predicted_Career_Field"  # Your dataset target column
X = data.drop(columns=[target_col])
y = data[target_col]

# Encode target for IG calculation
le_target = target_encoder

# Compute Information Gain (Mutual Information) for each feature
st.sidebar.header("Feature Selection")

# Encode categorical features temporarily for IG
X_encoded = X.copy()
for col in X_encoded.columns:
    if col in encoders:
        X_encoded[col] = encoders[col].transform(X_encoded[col])
    else:
        # If numeric, keep as is
        try:
            X_encoded[col] = X_encoded[col].astype(float)
        except:
            X_encoded[col] = 0  # fallback

mi_scores = mutual_info_classif(X_encoded, le_target.transform(y), discrete_features='auto')
mi_df = pd.DataFrame({"feature": X_encoded.columns, "info_gain": mi_scores})
mi_df = mi_df.sort_values(by="info_gain", ascending=False).reset_index(drop=True)

# Slider for IG threshold
threshold = st.sidebar.slider(
    "Information Gain Threshold",
    min_value=0.0,
    max_value=mi_df["info_gain"].max(),
    value=0.01,
    step=0.005,
    help="Select minimum Information Gain threshold to include features."
)

# Filter features by threshold
selected_features = mi_df[mi_df["info_gain"] >= threshold]["feature"].tolist()

st.sidebar.markdown(f"**Selected Features ({len(selected_features)}):**")
for f in selected_features:
    st.sidebar.write(f"- {f.replace('_', ' ').title()}")

if len(selected_features) == 0:
    st.warning("No features selected. Please lower the threshold.")
    st.stop()

# Show questions only for selected features
st.subheader("Please answer the following questions:")

responses = {}
for feature in selected_features:
    if feature in questions:
        q = random.choice(questions[feature])
        response = st.radio(
            label=f"**{feature.replace('_', ' ').title()}**: {q['question']}",
            options=q["options"],
            key=feature
        )
        responses[feature] = response
    else:
        # For features without questions, ask simple input (optional)
        responses[feature] = st.text_input(f"Enter value for {feature.replace('_', ' ').title()}:", key=feature)

if st.button("Predict Career Field"):
    # Validate all selected features answered
    if len(responses) < len(selected_features):
        st.warning("Please answer all questions.")
    else:
        try:
            input_vector = []
            for f in selected_features:
                val = responses[f]
                if f in encoders:
                    val = encoders[f].transform([val])[0]
                else:
                    try:
                        val = float(val)
                    except:
                        val = 0
                input_vector.append(val)

            prediction = model.predict([input_vector])[0]
            career = target_encoder.inverse_transform([prediction])[0]

            st.success(f"### Your predicted career field is: **{career}**")
            st.balloons()

        except Exception as e:
            st.error(f"Prediction error: {e}")

# Footer
st.markdown("---")
st.markdown("© 2025 Career Path Predictor with Feature Selection")
