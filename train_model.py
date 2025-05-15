import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel("new updated data.xlsx")

# Drop missing values
df = df.dropna()

# Features and target
target_column = "Predicted_Career_Field"
features = [col for col in df.columns if col != target_column]

X = df[features]
y = df[target_column]

# Encode categorical features
encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump((model, encoders, target_encoder), f)

# Generate questions
questions = {}
for feature in features:
    options = df[feature].astype(str).unique()
    sampled_options = np.random.choice(options, min(4, len(options)), replace=False)
    q_list = []
    for i in range(5):
        q_list.append({
            "question": f"What best describes your preference for {feature.replace('_', ' ')}? (Q{i+1})",
            "options": sampled_options.tolist()
        })
    questions[feature] = q_list

# Save questions
with open("questions.json", "w") as f:
    json.dump(questions, f, indent=2)

print("Model and questions generated successfully.")
