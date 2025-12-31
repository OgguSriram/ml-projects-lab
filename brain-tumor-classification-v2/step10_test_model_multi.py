import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ======================
# CONFIG
# ======================
MODEL_PATH = "brain_tumor_resnet50_final.keras"
TEST_DIR = "dataset/brisc2025/classification_task/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================
# LOAD MODEL
# ======================
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# ======================
# TEST DATA GENERATOR
# ======================
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ======================
# PREDICTION
# ======================
pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# ======================
# CLASSIFICATION REPORT
# ======================
print("\nFINAL CLASSIFICATION REPORT:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – ResNet50 Final")
plt.tight_layout()
plt.show()
