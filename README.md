# Ikigai Atlas — ML-Based Career Exploration System

**An interactive career exploration application powered by Machine Learning.**

[Live Application](https://itsleo-003.github.io/ikigai-atlas-ml-project/) | [GitHub Repository](https://github.com/itsLEO-003/ikigai-atlas-ml-project) | [Backend API](https://ikigai-atlas-backend.onrender.com/)

---

## 1. Problem Statement

Choosing a career path can be difficult for students because they may not know how their skills, interests, and work preferences relate to different professional fields.

Ikigai Atlas aims to address this problem through a machine-learning-based career exploration system.

The application collects information through an interactive questionnaire and uses a trained classification model to generate three predicted career paths.

The system is designed as an exploratory tool to help users consider possible career directions, not as a definitive career assessment.

## 2. Project Objectives

- Develop an interactive career exploration application.
- Analyze technical skills, soft skills, interests, and work preferences.
- Implement and compare multiple machine learning classification algorithms.
- Evaluate model performance using appropriate classification metrics.
- Integrate the selected model with a Flask backend.
- Deploy the application for public access.

## 3. Dataset

**Dataset:** Career Path Prediction

**Source:** [Mendeley Data — Career Path Prediction](https://data.mendeley.com/datasets/cns7h4b9g7/1)

Dataset characteristics:
- 15,000 records
- 19 columns
- 22 career classes

**Target variable:** `Suggested_Career_Path`

The following five features were selected for model training:

1. `Technical_Skills`
2. `Soft_Skills`
3. `Fields_of_Interest`
4. `Preferred_Work_Style`
5. `Work_Type_Interest`

Missing values in the selected features were handled by replacing them with empty strings.

The dataset is not included in the repository. It can be obtained from the original source above.

## 4. Machine Learning Methodology

The dataset was divided using a stratified train-test split:

- Training set: approximately 2/3 (10,000 records)
- Testing set: approximately 1/3 (5,000 records)
- Random state: 42

### Data preprocessing

- TF-IDF vectorization was applied to text-based features.
- One-Hot Encoding was applied to categorical work-preference features.
- A Scikit-learn ColumnTransformer was used to combine the preprocessing operations.

### Algorithms implemented

Three classification algorithms were trained and compared:

1. Decision Tree Classifier
2. Random Forest Classifier
3. K-Nearest Neighbors (KNN)

The model with the highest weighted F1-score was selected.

**Selected model: Random Forest Classifier**

## 5. Experimental Results

The models were evaluated on the same test dataset.

| Algorithm | Accuracy | Weighted F1 |
|---|---:|---:|
| Decision Tree | 31.76% | 31.38% |
| Random Forest | 31.80% | 31.72% |
| KNN | 31.94% | 31.23% |

### Baseline comparison

A majority-class Dummy Classifier was used as a baseline.

| Model | Accuracy |
|---|---:|
| Random Forest | 31.80% |
| Majority-class baseline | 5.10% |

Random Forest was selected because it achieved the highest weighted F1-score, although KNN achieved slightly higher accuracy.

The results indicate that the model performs better than the majority-class baseline, but its overall predictive performance remains modest.

## 6. Application Architecture

The application consists of three main components:

**Frontend**
- HTML
- CSS
- JavaScript
- Interactive questionnaire and results interface

**Backend**
- Python
- Flask
- Flask-CORS
- Pandas
- Joblib

**Machine Learning**
- Scikit-learn
- TF-IDF Vectorization
- One-Hot Encoding
- Random Forest Classification

### Prediction workflow

1. The user completes the questionnaire.
2. The frontend converts selected responses into five model input features.
3. The frontend sends the data to the Flask prediction API.
4. The backend processes the input using the trained model.
5. The model generates class probabilities.
6. The application displays the three highest-scoring predicted career paths.

The displayed model scores represent predicted class probabilities. They are not measures of personal career suitability.

## 7. Model Optimization

The trained model was compressed using Joblib compression.

| Model artifact | Size |
|---|---:|
| Original model | 12.02 MB |
| Compressed model | 1.38 MB |

The compressed model is used by the deployed backend to reduce storage and memory requirements.

## 8. Deployment

The application is deployed using:

- **Frontend:** GitHub Pages
- **Backend:** Render

Live application:

https://itsleo-003.github.io/ikigai-atlas-ml-project/

Backend:

https://ikigai-atlas-backend.onrender.com/

## 9. Repository Structure

```text
ikigai-atlas-ml-project/
│
├── index.html
├── Smart_career_prediction.html
├── app.py
├── train_model.py
├── evaluate_model.py
├── compress_model.py
├── best_career_model_compressed.pkl
├── model_comparison.csv
├── requirements.txt
├── render.yaml
└── .gitignore
```

## 10. Running the Project Locally

### Requirements

- Python 3.11
- pip
- The original dataset CSV
- Project source files

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the models

```bash
python train_model.py
```

### Compress the selected model

```bash
python compress_model.py
```

### Evaluate the model

```bash
python evaluate_model.py
```

### Start the Flask backend

```bash
python app.py
```

The backend runs at:

`http://127.0.0.1:5000`

Open `Smart_career_prediction.html` in your browser to use the frontend. For local development, configure its API endpoint to use the local Flask backend.

## 11. Limitations and Future Scope

### Current limitations

- The model has modest predictive performance.
- Predictions depend on the dataset's representation of career paths.
- Only five selected features are used for model prediction.
- Predicted probabilities should not be interpreted as validated career suitability scores.

### Possible future improvements

- Explore additional and more diverse career datasets.
- Improve feature engineering and model performance.
- Investigate additional classification algorithms.
- Incorporate user feedback into future model development.
- Improve the quality and interpretability of career recommendations.

## 12. Acknowledgements

The project uses the Career Path Prediction dataset published on Mendeley Data.

Machine learning implementation and evaluation were carried out using Python and Scikit-learn.

---

**Project:** Ikigai Atlas — ML-Based Career Exploration System  
**Purpose:** Academic Machine Learning Semester Project
