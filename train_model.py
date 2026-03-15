import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("dataset/creditcard.csv")

# Features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Scale Amount
scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(X[["Amount"]])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_smote, y_train_smote)

# Evaluate
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/fraud_model.pkl")

print("Model saved successfully")