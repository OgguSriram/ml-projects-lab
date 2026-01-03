import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
os.makedirs("results", exist_ok=True)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
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

# Scale Time & Amount
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Final tuned model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_smote, y_train_smote)

# Probabilities & tuned threshold
y_probs = model.predict_proba(X_test)[:, 1]
FINAL_THRESHOLD = 0.44
y_pred = (y_probs >= FINAL_THRESHOLD).astype(int)

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Final Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()

# -------------------------
# Precision-Recall Curve
# -------------------------
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(5, 4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("results/precision_recall_curve.png")
plt.show()

# -------------------------
# Save Final Metrics
# -------------------------
report = classification_report(y_test, y_pred)

with open("results/final_metrics.txt", "w") as f:
    f.write("Final Threshold: 0.44\n\n")
    f.write(report)

print("Final evaluation completed.")
print(report)
