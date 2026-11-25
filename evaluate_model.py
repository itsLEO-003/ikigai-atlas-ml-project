
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Career_Path_Dataset_with_Career.csv")

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

# Load our trained Random Forest
model = joblib.load("best_career_model.pkl")
predictions = model.predict(X_test)

# Majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

print("Random Forest accuracy:",
      accuracy_score(y_test, predictions))

print("Baseline accuracy:",
      accuracy_score(y_test, baseline_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))
