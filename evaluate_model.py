
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(BASE_DIR, "Career_Path_Dataset_with_Career.csv")
)

features = [
    "Technical_Skills",
    "Soft_Skills",
    "Fields_of_Interest",
    "Preferred_Work_Style",
    "Work_Type_Interest"
]

X = df[features].fillna("")
y = df["Suggested_Career_Path"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=1/3,
    random_state=42,
    stratify=y
)

# Load the compressed Random Forest model
model = joblib.load(
    os.path.join(BASE_DIR, "best_career_model_compressed.pkl")
)

predictions = model.predict(X_test)

# Majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

print("Random Forest accuracy:",
      accuracy_score(y_test, predictions))

print("Random Forest weighted F1:",
      f1_score(y_test, predictions, average="weighted"))

print("Baseline accuracy:",
      accuracy_score(y_test, baseline_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))
