import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load data
DATA_PATH = r"C:\Users\oggus\Downloads\vscodes\ml_project\credit card fraud detection\creditcard.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale Time & Amount
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Base model
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# Hyperparameter search space
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2"]
}

# Randomized Search
search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,
    scoring="recall",   # IMPORTANT: optimize recall
    cv=3,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_smote, y_train_smote)

print("Best parameters found:")
print(search.best_params_)

# Evaluate best model on test data
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred))
