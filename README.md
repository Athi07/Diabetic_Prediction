# Diabetic_Prediction Using Machine Learning

This project is focused on predicting whether a patient is likely to develop diabetes or not, based on various medical predictor variables. Machine learning classification techniques are applied to build the model and assess its accuracy.

## Dataset

The dataset used for this project is the **Pima Indians Diabetes Database**, which is available from the UCI Machine Learning Repository. It consists of 768 records and 8 attributes related to diabetes outcomes.

### Features:
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skinfold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic factor)
- `Age`: Age in years

### Target:
- `Outcome`: 0 or 1 (indicates if the patient has diabetes)

## Project Workflow

1. **Data Preprocessing**:
   - Handle missing values (if any)
   - Feature scaling
   - Train-test split (typically 70/30 or 80/20 split)

2. **Exploratory Data Analysis (EDA)**:
   - Visualize distributions of features
   - Correlation matrix for feature relationships
   - Basic statistics (mean, median, etc.)

3. **Model Building**:
   - Applied machine learning classification algorithms:
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
   - Train the models and evaluate performance using:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - ROC AUC score

4. **Model Evaluation**:
   - Compare the performance of different models.
   - Choose the best-performing model based on accuracy and other evaluation metrics.
   - Perform cross-validation and hyperparameter tuning to improve performance.

5. **Prediction**:
   - Test the final model on new unseen data.
   - Predict whether a patient is likely to develop diabetes.

## Libraries Used

The following Python libraries are used in this project:
- `pandas`: For data manipulation and analysis
- `numpy`: For numerical operations
- `matplotlib` & `seaborn`: For data visualization
- `scikit-learn`: For machine learning algorithms and evaluation metrics
- `jupyter`: To run the notebook environment

## Sample Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Load the dataset
data = pd.read_csv("Your Dataset Path")

# Data overview
print(data.head())
print(data.info())

# Check for missing values (if any)
print(data.isnull().sum())

# Data Preprocessing
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']               # Target

# Splitting data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (important for models like SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

# Support Vector Classifier
svc = SVC(probability=True)
svc.fit(X_train_scaled, y_train)
y_pred_svc = svc.predict(X_test_scaled)

# Model Evaluation
def evaluate_model(y_test, y_pred, model_name):
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Evaluate Logistic Regression
evaluate_model(y_test, y_pred_logreg, "Logistic Regression")

# Evaluate Random Forest
evaluate_model(y_test, y_pred_rfc, "Random Forest")

# Evaluate SVC
evaluate_model(y_test, y_pred_svc, "Support Vector Classifier")

# ROC AUC Score for Random Forest
rfc_roc_auc = roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1])
print(f"Random Forest ROC AUC Score: {rfc_roc_auc:.4f}")

# Plotting the feature importance (for Random Forest)
importances = rfc.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], color="b", align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=45)
plt.show()

# Make a prediction on a new patient data (example)
new_patient = np.array([[2, 130, 80, 0, 0, 32.1, 0.472, 45]])  # Sample patient data
new_patient_scaled = scaler.transform(new_patient)

# Predict using the best model (Random Forest in this case)
prediction = rfc.predict(new_patient_scaled)
print(f"Diabetes Prediction for the new patient: {'Diabetic' if prediction[0] == 1 else 'Non-diabetic'}")

