import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_excel(
    r"C:\Users\LENOVO\Downloads\final_basic_emotion_dataset.xlsx"
)

print("Dataset Loaded Successfully!\n")

print(df.head())

# ==============================
# FEATURES AND LABELS
# ==============================

X = df["Comment"]
y = df["Emotion"]

# ==============================
# TEXT VECTORIZATION
# ==============================

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

print("\nText Vectorization Completed!")

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Completed!")

# ==============================
# HYPERPARAMETER TUNING
# ==============================

parameters = {

    'C': [0.1, 1, 10],

    'solver': ['liblinear', 'lbfgs'],

    'max_iter': [1000, 2000]
}

grid_search = GridSearchCV(

    LogisticRegression(),

    parameters,

    cv=5,

    scoring='accuracy',

    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best Model

model = grid_search.best_estimator_

print("\nBest Parameters:")

print(grid_search.best_params_)

# ==============================
# CROSS VALIDATION
# ==============================

cv_scores = cross_val_score(

    model,

    X_vectorized,

    y,

    cv=5
)

print("\nCross Validation Scores:")

print(cv_scores)

print("\nAverage CV Accuracy:")

print(cv_scores.mean())

# ==============================
# MODEL TRAINING
# ==============================

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==============================
# MODEL PREDICTION
# ==============================

y_pred = model.predict(X_test)

# ==============================
# MODEL ACCURACY
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")

print(accuracy)

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ==============================
# SAMPLE PREDICTION
# ==============================

sample = ["This video is absolutely amazing"]

sample_vector = vectorizer.transform(sample)

prediction = model.predict(sample_vector)

print("\nSample Prediction:")

print("Comment:", sample[0])

print("Predicted Emotion:", prediction[0])

# ==============================
# CURRENT DIRECTORY
# ==============================

print("\nCurrent Working Directory:")

print(os.getcwd())

# ==============================
# SAVE MODEL AND VECTORIZER
# ==============================

joblib.dump(
    model,
    r"C:\Users\LENOVO\Downloads\model.pkl"
)

joblib.dump(
    vectorizer,
    r"C:\Users\LENOVO\Downloads\vectorizer.pkl"
)

print("\nFiles Saved Successfully!")

print("\nSaved Files:")

print("model.pkl")
print("vectorizer.pkl")