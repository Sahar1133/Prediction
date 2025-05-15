import json
import pickle
import random
import streamlit as st

# Load model and encoders
with open("model.pkl", "rb") as f:
    model, encoders, target_encoder = pickle.load(f)

# Load questions
with open("questions.json") as f:
    questions = json.load(f)

st.title("Career Path Predictor")

responses = {}

# Ask one random question per feature
for feature, q_list in questions.items():
    q = random.choice(q_list)
    response = st.radio(q["question"], q["options"], key=feature)
    responses[feature] = response

if st.button("Predict Career Field"):
    input_data = []
    for feature in questions:
        val = responses[feature]
        if feature in encoders:
            val = encoders[feature].transform([val])[0]
        input_data.append(val)
    
    pred = model.predict([input_data])[0]
    career = target_encoder.inverse_transform([pred])[0]
    st.success(f"Your predicted career field is: **{career}**")