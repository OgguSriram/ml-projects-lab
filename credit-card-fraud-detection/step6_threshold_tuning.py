import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, classification_report
from imblearn.over_sampling import SMOTE

# Load data
DATA_PATH = r"C:\Users\oggus\Downloads\vscodes\ml_project\credit card fraud detection\creditcard.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop('Class', axis=1)
y = df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_smote, y_train_smote)

# Probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Find threshold with precision >= 0.80 and maximum recall
min_precision = 0.80

valid_idxs = np.where(precision[:-1] >= min_precision)[0]

if len(valid_idxs) == 0:
    print("No threshold satisfies the minimum precision requirement.")
else:
    best_idx = valid_idxs[np.argmax(recall[valid_idxs])]
    best_threshold = thresholds[best_idx]

    print("Selected threshold:", best_threshold)
    print("Precision at threshold:", precision[best_idx])
    print("Recall at threshold:", recall[best_idx])

    # Apply new threshold
    y_pred_tuned = (y_probs >= best_threshold).astype(int)

    print("\nClassification Report (Tuned Threshold):")
    print(classification_report(y_test, y_pred_tuned))

# Apply new threshold
y_pred_tuned = (y_probs >= best_threshold).astype(int)

print("\nClassification Report (Tuned Threshold):")
print(classification_report(y_test, y_pred_tuned))
