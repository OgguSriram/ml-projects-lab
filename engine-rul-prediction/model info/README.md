# Trained Model Information

The trained machine learning model is not included in this repository due to
GitHub file size limitations.

## Model Details
- Algorithm: Random Forest Regressor
- Feature Engineering: Rolling mean & standard deviation
- Best Performance:
  - MAE ≈ 2.10 cycles
  - RMSE ≈ 5.18 cycles
  - R² ≈ 0.994

## How to Recreate the Model
Run the following scripts in order:
1. step3_eda.py
2. step4_create_rul.py
3. step5_train_test_split.py
4. step8b_feature_engineering.py
5. step9_save_load_model.py

This will regenerate the trained model locally.
