# Multimodal Housing Price Prediction

## Objective

Predict house prices using:

1. House Images
2. Structured Housing Data

The project combines CNN-extracted image features with tabular features for multimodal learning.

---

## Dataset

Custom Housing Dataset

Columns:

- image_name
- bedrooms
- bathrooms
- area
- price

---

## Technologies

- TensorFlow/Keras
- MobileNetV2
- Scikit-Learn
- OpenCV
- Pandas
- NumPy

---

## Workflow

Housing Image
      ↓
CNN Feature Extraction
      ↓
Image Embeddings

Housing Data
      ↓
Scaling

Feature Fusion
(Image + Tabular)
      ↓
Random Forest Regressor
      ↓
House Price Prediction

---

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## Run

```bash
pip install -r requirements.txt

python multimodal_housing.py
```

---

## Output

```text
multimodal_model.pkl
scaler.pkl
```

Reusable multimodal housing price prediction model.
