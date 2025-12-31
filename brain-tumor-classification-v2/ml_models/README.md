# Model Information – Brain Tumor Classification (v2)

## Overview
This directory documents the trained deep learning model used in the
Brain Tumor Classification v2 project.

The actual model file is **not included** in this repository due to
GitHub file size limitations.

---

## Model Architecture
- **Backbone:** ResNet50 (ImageNet pretrained)
- **Input size:** 224 × 224 × 3
- **Output classes:** 4
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor

---

## Training Strategy
- Fine-tuned the **last 80 layers** of ResNet50
- Data augmentation applied during training
- **EarlyStopping** to prevent overfitting
- **ReduceLROnPlateau** for adaptive learning rate control
- Optimizer: Adam
- Initial learning rate: `5e-6`

---

## Best Model Details
- **Best epoch:** 20
- **Validation accuracy:** ~79%
- **Test accuracy:** ~79%
- **Model filename (local):**
