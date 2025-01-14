# -*- coding: utf-8 -*-
"""Dry_Beans.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iPulbN7ab4foTXbYCMDy8fY7N5QKjmS6

Load Data
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "Dry_Bean_Dataset.xlsx"
dry_bean_data = pd.read_excel(file_path)

# Prepare data: split features and labels
X = dry_bean_data.drop(columns=["Class"])  # Features
y = dry_bean_data["Class"]                 # Labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)
nn_model = MLPClassifier(random_state=42, max_iter=500)

# Train models
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

# Predict on test set
rf_preds = rf_model.predict(X_test)
svm_preds = svm_model.predict(X_test)
nn_preds = nn_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_preds)
svm_accuracy = accuracy_score(y_test, svm_preds)
nn_accuracy = accuracy_score(y_test, nn_preds)

# Print accuracies
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"Neural Network Accuracy: {nn_accuracy:.2f}")

# Find the best model
best_model = None
best_accuracy = 0
model_name = ""

if rf_accuracy > best_accuracy:
    best_model = rf_model
    best_accuracy = rf_accuracy
    model_name = "RandomForest"

if svm_accuracy > best_accuracy:
    best_model = svm_model
    best_accuracy = svm_accuracy
    model_name = "SVM"

if nn_accuracy > best_accuracy:
    best_model = nn_model
    best_accuracy = nn_accuracy
    model_name = "NeuralNetwork"

print(f"Best Model: {model_name} with accuracy: {best_accuracy:.2f}")

# Save the best model and label encoder
joblib.dump(best_model, f"{model_name}_best_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Feature importance from Random Forest
feature_importance = rf_model.feature_importances_
features = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance from Random Forest")
plt.show()

# Correlation matrix
corr_matrix = X.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Matrix")
plt.show()

# Add the class labels back to the dataset for grouped analysis
dry_bean_data["Class_encoded"] = y_encoded

# Plot the distribution of key size features for each class
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Area distribution
sns.boxplot(x="Class_encoded", y="Area", data=dry_bean_data, ax=axes[0, 0])
axes[0, 0].set_title("Area Distribution by Bean Type")

# Perimeter distribution
sns.boxplot(x="Class_encoded", y="Perimeter", data=dry_bean_data, ax=axes[0, 1])
axes[0, 1].set_title("Perimeter Distribution by Bean Type")

# MajorAxisLength distribution
sns.boxplot(x="Class_encoded", y="MajorAxisLength", data=dry_bean_data, ax=axes[1, 0])
axes[1, 0].set_title("Major Axis Length Distribution by Bean Type")

# AspectRation distribution
sns.boxplot(x="Class_encoded", y="AspectRation", data=dry_bean_data, ax=axes[1, 1])
axes[1, 1].set_title("Aspect Ratio Distribution by Bean Type")

plt.tight_layout()
plt.show()

import cv2
import numpy as np

# Function to process image and extract features
def extract_features_from_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Thresholding to binary image
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour corresponds to the bean
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Fit an ellipse to extract major and minor axis lengths
    ellipse = cv2.fitEllipse(largest_contour)
    major_axis_length = max(ellipse[1])
    minor_axis_length = min(ellipse[1])

    # Compute aspect ratio
    aspect_ratio = major_axis_length / minor_axis_length

    # Compute compactness (area / perimeter^2)
    compactness = area / (perimeter ** 2)

    return np.array([area, perimeter, major_axis_length, minor_axis_length, aspect_ratio, compactness])

""" Hàm Trích Xuất Đặc Trưng"""

def extract_full_features_from_image(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Chuyển ảnh sang nhị phân
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Tìm đường viền
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lấy đường viền lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)

    # Tính toán các đặc trưng hiện có
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    ellipse = cv2.fitEllipse(largest_contour)
    major_axis_length = max(ellipse[1])
    minor_axis_length = min(ellipse[1])

    # Các đặc trưng hiện có
    aspect_ratio = major_axis_length / minor_axis_length
    compactness = area / (perimeter ** 2)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)

    # Tính các đặc trưng khác
    convex_hull = cv2.convexHull(largest_contour)
    convex_area = cv2.contourArea(convex_hull)
    equivalent_diameter = np.sqrt(4 * area / np.pi)

    # Thêm các đặc trưng khác (ví dụ hệ số hình dạng, độ nhọn)

    # Tạo mảng đặc trưng
    features = np.array([area, perimeter, major_axis_length, minor_axis_length, aspect_ratio, compactness,
                         circularity, eccentricity, convex_area, equivalent_diameter])

    # Đảm bảo có đủ 16 đặc trưng
    while len(features) < 16:
        features = np.append(features, 0)  # Thêm các đặc trưng bằng 0 hoặc tính toán thêm

    return features

"""Dự Đoán Lớp Đậu"""

# Tạo từ điển ánh xạ loại đậu với mô tả của chúng
bean_descriptions = {
    'DERMASON': "DERMASON is a small green bean, commonly used in salads and soups. It has a round shape and a slightly smooth surface.",
    'PINK': "PINK beans are small, round, and have a pinkish color. They are often used in stews and chili.",
    'CANNELLINI': "CANNELLINI beans are large white beans with a creamy texture. They are popular in Italian dishes.",
    'SEKECI': "SEKECI beans are medium-sized, oval, and often have a dark brown color. They are used in various traditional dishes.",
    'SORA': "SORA beans are small, dark beans with a shiny surface. They are often used in desserts and savory dishes.",
    'BENGAL': "BENGAL beans are medium-sized, with a light brown color and a slightly flattened shape. They are used in curries.",
    'BLACK': "BLACK beans are small and shiny with a black color. They are commonly used in Mexican cuisine."
}

# Hàm dự đoán loại đậu từ ảnh
def predict_bean_from_image(image_path, model):
    # Trích xuất các đặc trưng đầy đủ từ hình ảnh
    features = extract_full_features_from_image(image_path)

    # Reshape các đặc trưng để khớp với đầu vào của mô hình
    features = features.reshape(1, -1)

    # Dự đoán lớp
    predicted_class = model.predict(features)

    # Giải mã lớp dự đoán (biến từ nhãn mã hóa trở lại tên lớp gốc)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    # Lấy mô tả cho loại đậu dự đoán
    description = bean_descriptions.get(predicted_label[0], "Description not available.")

    return predicted_label[0], description

# Sử dụng ví dụ
image_path = '/content/bean_beers1-768x614.jpg'
predicted_bean_type, predicted_bean_description = predict_bean_from_image(image_path, rf_model)
print(f"The predicted bean type is: {predicted_bean_type}")
print(f"Description: {predicted_bean_description}")