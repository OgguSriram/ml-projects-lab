import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
DATA_PATH = r"C:\Users\oggus\Downloads\vscodes\ml_project\credit card fraud detection\creditcard.csv"
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Stratified split (VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())

# Scale Time and Amount only
scaler = StandardScaler()

X_train[['Time', 'Amount']] = scaler.fit_transform(
    X_train[['Time', 'Amount']]
)

X_test[['Time', 'Amount']] = scaler.transform(
    X_test[['Time', 'Amount']]
)

print("\nPreprocessing completed successfully.")
