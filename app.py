
# from flask import Flask, request, jsonify
# import joblib

# app = Flask(__name__)

# Load the trained machine learning pipeline
# model = joblib.load("best_career_model.pkl")


# @app.route("/")
# def home():
    # return "Ikigai Atlas ML Backend is running!"


# if __name__ == "__main__":
    # app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "best_career_model.pkl"))


@app.route("/")
def home():
    return "Ikigai Atlas ML Backend is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Prepare input in the same format used during training
        input_data = pd.DataFrame([{
            "Technical_Skills": data.get("Technical_Skills", ""),
            "Soft_Skills": data.get("Soft_Skills", ""),
            "Fields_of_Interest": data.get("Fields_of_Interest", ""),
            "Preferred_Work_Style": data.get("Preferred_Work_Style", ""),
            "Work_Type_Interest": data.get("Work_Type_Interest", "")
        }])

        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Get the three highest-scoring careers
        classes = model.classes_
        top_indices = probabilities.argsort()[-3:][::-1]

        results = []

        for index in top_indices:
            results.append({
                "career": str(classes[index]),
                "model_score": round(float(probabilities[index]) * 100, 2)
            })

        return jsonify({"careers": results})

    except Exception as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

