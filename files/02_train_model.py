"""
Train the loan-approval model.

Protected attributes (protected_group, gender) are explicitly EXCLUDED from
the feature set, consistent with fair-lending practice (ECOA/Reg B) - they
are only retained downstream, in the test split, for the independent
fairness audit performed by the validation team in script 03.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("/home/claude/ai_model_validation_project/data/loan_applications.csv")

FEATURES = [
    "credit_score", "annual_income", "debt_to_income", "employment_length_years",
    "age", "num_open_accounts", "delinquencies_2yr", "loan_amount_requested", "zip_tier",
]
TARGET = "loan_approved"

X = df[FEATURES]
y = df[TARGET]
protected = df[["protected_group", "gender"]]

X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X, y, protected, test_size=0.25, random_state=7, stratify=y
)

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clf", GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, random_state=7
    )),
])
pipe.fit(X_train, y_train)

joblib.dump(pipe, "/home/claude/ai_model_validation_project/outputs/model.joblib")
X_test.assign(loan_approved=y_test).join(prot_test).to_csv(
    "/home/claude/ai_model_validation_project/outputs/test_set.csv", index=False
)
X_train.assign(loan_approved=y_train).join(prot_train).to_csv(
    "/home/claude/ai_model_validation_project/outputs/train_set.csv", index=False
)

with open("/home/claude/ai_model_validation_project/outputs/features.json", "w") as f:
    json.dump(FEATURES, f)

print("train size:", X_train.shape, "test size:", X_test.shape)
print("train approval rate:", y_train.mean().round(4))
print("test approval rate:", y_test.mean().round(4))
