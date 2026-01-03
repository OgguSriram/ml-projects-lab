import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
DATA_PATH = r"C:\Users\oggus\Downloads\vscodes\ml_project\credit card fraud detection\creditcard.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df['Class'].value_counts())

print("\nClass distribution (percentage):")
print(df['Class'].value_counts(normalize=True) * 100)

# Plot class imbalance
plt.figure()
df['Class'].value_counts().plot(kind='bar')
plt.title("Class Distribution (0 = Normal, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# -------------------------------
# PART 2: Amount & Time Analysis
# -------------------------------

fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

print("\nTransaction Amount statistics:")
print("Fraud Amount Summary:")
print(fraud['Amount'].describe())

print("\nNormal Amount Summary:")
print(normal['Amount'].describe())

# Amount distribution
plt.figure()
plt.hist(normal['Amount'], bins=50, alpha=0.7, label='Normal')
plt.hist(fraud['Amount'], bins=50, alpha=0.7, label='Fraud')
plt.legend()
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

# Time distribution
plt.figure()
plt.hist(normal['Time'], bins=50, alpha=0.7, label='Normal')
plt.hist(fraud['Time'], bins=50, alpha=0.7, label='Fraud')
plt.legend()
plt.title("Transaction Time Distribution")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency")
plt.show()
