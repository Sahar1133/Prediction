import pandas as pd
import numpy as np
import pickle
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_excel("new updated data.xlsx")
df = df.dropna()

# Define target and features
target_col = "Predicted_Career_Field"
features = [col for col in df.columns if col != target_col]

X = df[features].copy()
y = df[target_col].copy()

# Encode categorical features using LabelEncoder and save encoders
encoders = {}
for col in features:
    if X[col].dtype == "object" or isinstance(X[col].iloc[0], str):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

# Encode target column
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model, encoders, and target encoder
with open("model.pkl", "wb") as f:
    pickle.dump((model, encoders, target_encoder), f)

# Generate 5 questions per feature randomly from unique options
questions = {}
for feature in features:
    unique_vals = df[feature].dropna().astype(str).unique()
    for i in range(5):
        question_text = f"Question {i+1}: Choose your preference for {feature.replace('_', ' ')}"
        options = np.random.choice(unique_vals, size=min(4, len(unique_vals)), replace=False).tolist()
        questions.setdefault(feature, []).append({
            "question": question_text,
            "options": options
        })

# Save questions to JSON
with open("questions.json", "w") as f:
    json.dump(questions, f, indent=4)

print("Model and questions files have been saved.")