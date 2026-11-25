# Ikigai Atlas – ML-Based Career Exploration System

## Project Overview

Ikigai Atlas is a machine-learning-based career exploration application. It analyzes users' selected technical skills, soft skills, interests, preferred work styles, and work types to generate three predicted career paths.

The project combines a web-based questionnaire, a trained machine learning model, and a Flask backend.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Flask
- HTML, CSS, JavaScript
- TF-IDF Vectorization
- One-Hot Encoding
- Render (Backend Deployment)

## Machine Learning Methodology

Five input features are used:

- Technical Skills
- Soft Skills
- Fields of Interest
- Preferred Work Style
- Work Type Interest

The target variable is `Suggested_Career_Path`.

Three classification algorithms were implemented and compared:

1. Decision Tree
2. Random Forest
3. K-Nearest Neighbors (KNN)

The dataset was divided using a stratified split:

- Training: approximately 2/3
- Testing: approximately 1/3

TF-IDF vectorization was used for text features, while One-Hot Encoding was used for categorical features.

Random Forest was selected based on its highest weighted F1-score.

## Model Evaluation

| Metric | Result |
|---|---:|
| Random Forest Accuracy | 31.80% |
| Random Forest Weighted F1 | 31.72% |
| Majority-Class Baseline Accuracy | 5.10% |

The model was evaluated on 5,000 test samples across 22 career classes.

These results indicate modest predictive performance. The model scores represent predicted class probabilities, not validated measures of personal career suitability.

## Application Architecture

1. User completes the career questionnaire.
2. Frontend converts selected answers into model input features.
3. Flask backend receives the input.
4. The trained Random Forest model generates predictions.
5. The application displays the top three predicted career paths.

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python app.py
```

Open `Smart_career_prediction.html` in a browser.

## Dataset

Dataset: [Career Path Prediction – Mendeley Data](https://data.mendeley.com/datasets/cns7h4b9g7/1)

## Deployment

Backend hosted on Render:

https://ikigai-atlas-backend.onrender.com/

## Project Limitations

- Predictive performance is modest.
- Results depend on the dataset and selected input features.
- Model scores should not be interpreted as professional career guidance.
- The system is intended as an exploratory career-support tool, not a definitive career assessment.

## Project Purpose

This project demonstrates the implementation, comparison, evaluation, and deployment of machine learning classification algorithms in a practical web application.
