
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load dataset
df = pd.read_csv(
    os.path.join(BASE_DIR, "Career_Path_Dataset_with_Career.csv")
)

# 2. Select questionnaire-related inputs
features = [
    "Technical_Skills",
    "Soft_Skills",
    "Fields_of_Interest",
    "Preferred_Work_Style",
    "Work_Type_Interest"
]

X = df[features].fillna("")
y = df["Suggested_Career_Path"]

# 3. Split: 2/3 training, 1/3 testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=1/3,
    random_state=42,
    stratify=y
)

# 4. Convert text answers into numerical features
preprocessor = ColumnTransformer([
    ("technical", TfidfVectorizer(max_features=1000, min_df=2), "Technical_Skills"),
    ("soft", TfidfVectorizer(max_features=1000, min_df=2), "Soft_Skills"),
    ("interests", TfidfVectorizer(max_features=100, min_df=2), "Fields_of_Interest"),
    ("work_style", OneHotEncoder(handle_unknown="ignore"),
     ["Preferred_Work_Style"]),
    ("work_type", OneHotEncoder(handle_unknown="ignore"),
     ["Work_Type_Interest"])
])

# 5. Compare three ML algorithms
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=3,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=20,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1
    ),

    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
best_score = -1
best_model = None
best_name = ""

for name, algorithm in models.items():
    print(f"\nTraining {name}...")

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", algorithm)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Weighted F1": f1
    })

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

# 6. Display comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 7. Save results and best model
results_df.to_csv(
    os.path.join(BASE_DIR, "model_comparison.csv"),
    index=False
)

joblib.dump(
    best_model,
    os.path.join(BASE_DIR, "best_career_model.pkl")
)

print(f"\nBest model: {best_name}")
print("Saved: model_comparison.csv")
print("Saved: best_career_model.pkl")
