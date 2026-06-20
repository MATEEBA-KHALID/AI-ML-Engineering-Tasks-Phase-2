import os
import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================
# Load Dataset
# =====================================

data = pd.read_csv("housing_data.csv")

IMAGE_FOLDER = "house_images"

# =====================================
# Pretrained CNN
# =====================================

cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# =====================================
# Extract Image Features
# =====================================

image_features = []

print("Extracting image features...")

for image_name in data["image_name"]:

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    image = cv2.imread(image_path)

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = preprocess_input(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    features = cnn.predict(
        image,
        verbose=0
    )

    image_features.append(
        features.flatten()
    )

image_features = np.array(
    image_features
)

print(
    "Image Feature Shape:",
    image_features.shape
)

# =====================================
# Tabular Features
# =====================================

tabular_features = data[
    ["bedrooms",
     "bathrooms",
     "area"]
].values

target = data["price"].values

# =====================================
# Scale Tabular Data
# =====================================

scaler = StandardScaler()

tabular_features = scaler.fit_transform(
    tabular_features
)

# =====================================
# Feature Fusion
# =====================================

X = np.concatenate(
    [
        tabular_features,
        image_features
    ],
    axis=1
)

y = target

print(
    "Combined Shape:",
    X.shape
)

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# Train Regressor
# =====================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# =====================================
# Predictions
# =====================================

predictions = model.predict(
    X_test
)

# =====================================
# Evaluation
# =====================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = root_mean_squared_error(
    y_test,
    predictions
)

print("\nResults")
print("=" * 40)

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

# =====================================
# Save Model
# =====================================

joblib.dump(
    model,
    "multimodal_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nModel Saved Successfully!")