# Pneumonia Detection from Chest X-Ray Images

## Project Overview
This project implements an end-to-end deep learning pipeline to detect pneumonia from chest X-ray images.  
The complete workflow covers data preprocessing, model building, training, evaluation, optimization using transfer learning, and real-world prediction.

The project was developed step by step, starting from a basic CNN model and later improved using a pre-trained MobileNetV2 architecture.

---

## Dataset
- Source: Kaggle – Chest X-Ray Pneumonia Dataset
- Classes:
  - NORMAL
  - PNEUMONIA
- Images are grayscale chest X-rays resized to 224×224.

*(Dataset is not included in this repository due to size and licensing constraints.)*

---

## Project Workflow

1. **Data Visualization**
   - Loaded and visualized sample X-ray images.
   - Verified class distribution and image integrity.

2. **Data Preprocessing**
   - Image rescaling (pixel normalization).
   - Data augmentation to improve generalization.
   - Training and validation data generators created.

3. **CNN Model from Scratch**
   - Built a custom convolutional neural network.
   - Model learned basic visual patterns from X-rays.

4. **Model Training**
   - Trained using binary cross-entropy loss.
   - Observed learning progression across epochs.

5. **Model Evaluation**
   - Evaluated on unseen test data.
   - Generated confusion matrix and classification report.
   - Achieved ~80% accuracy with high pneumonia recall.

6. **Transfer Learning (MobileNetV2)**
   - Used pre-trained MobileNetV2 as feature extractor.
   - Trained only the final layers.
   - Achieved improved performance with faster convergence.

7. **Model Saving and Loading**
   - Saved trained model in modern `.keras` format.
   - Reloaded model to verify persistence.

8. **Real-World Prediction**
   - Tested model on individual chest X-ray images.
   - Model successfully predicted pneumonia with high confidence.

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Special emphasis was placed on **high recall for pneumonia**, which is clinically important to avoid missing positive cases.

---

## Technologies Used
- Python
- TensorFlow & Keras
- NumPy
- scikit-learn
- Matplotlib

---

## Key Learnings
- End-to-end deep learning workflow
- Importance of data preprocessing
- CNN fundamentals
- Transfer learning benefits
- Model evaluation using real metrics
- Model deployment readiness

---

## Disclaimer
This project is for educational and research purposes only and should not be used for medical diagnosis.
