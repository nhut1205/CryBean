import os
import google.generativeai as genai
import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Thiết lập khóa API cho Google AI
os.environ["GEMINI_API_KEY"] = "AIzaSyAxgyZvlkYk8hdfDAGnlFpDIbqumDfrSDA"

# Global variables for models and label encoder
rf_model = None
svm_model = None
nn_model = None
label_encoder = None

# Configure Google AI API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Hàm khởi tạo mô hình và bộ mã hóa nhãn
def load_models():
    global rf_model, label_encoder  # Declare as global to modify the variables
    rf_model = joblib.load("RandomForest_best_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

# Tạo từ điển ánh xạ loại đậu với mô tả của chúng
bean_descriptions = {
    'DERMASON': "DERMASON là loại đậu có kích thước nhỏ, thường được sử dụng trong các món salad và súp.",
    'SEKECI': "Đậu SEKECI có kích thước trung bình, hình bầu dục và thường có màu nâu đậm.",
    'SORA': "Đậu SORA là loại đậu nhỏ, màu đen sẫm với bề mặt bóng loáng.",
    'BENGAL': "Đậu BENGAL có kích thước trung bình, với màu nâu nhạt.",
    'HOROZ' : "Đậu HOROZ có đặc điểm là hạt to, hình bầu dục và có thể có màu nâu hoặc nâu nhạt",
    'SEKER' : "Đậu Đen có kích thước nhỏ, bề mặt bóng loáng và có màu đen."
}   

# Hàm trích xuất các đặc trưng từ ảnh
def extract_full_features_from_image(image):
    # Các bước xử lý hình ảnh như cũ
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < 500:
        return None

    perimeter = cv2.arcLength(largest_contour, True)
    ellipse = cv2.fitEllipse(largest_contour)
    major_axis_length = max(ellipse[1])
    minor_axis_length = min(ellipse[1])

    aspect_ratio = major_axis_length / minor_axis_length
    compactness = area / (perimeter ** 2)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)

    convex_hull = cv2.convexHull(largest_contour)
    convex_area = cv2.contourArea(convex_hull)
    equivalent_diameter = np.sqrt(4 * area / np.pi)

    # Thêm các đặc trưng bổ sung để đảm bảo có 16 đặc trưng
    bounding_rect = cv2.boundingRect(largest_contour)
    rect_area = bounding_rect[2] * bounding_rect[3]
    extent = area / rect_area  # Đặc trưng extent
    solidity = area / convex_area  # Đặc trưng solidity
    roundness = (4 * area) / (np.pi * (major_axis_length ** 2))  # Đặc trưng roundness
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Tạo mảng đặc trưng với 16 đặc trưng
    features = np.array([area, perimeter, major_axis_length, minor_axis_length, 
                         aspect_ratio, compactness, circularity, eccentricity, 
                         convex_area, equivalent_diameter, rect_area, extent,
                         solidity, roundness, hu_moments[0], hu_moments[1]])

    return features


# Hàm dự đoán loại đậu từ ảnh
def predict_bean_from_image(image, model):
    features = extract_full_features_from_image(image)
    
    # Nếu không có đặc trưng (không tìm thấy contour phù hợp), trả về "Không phải đậu"
    if features is None:
        return "Không phải đậu", "The image does not contain a recognizable bean."

    features = features.reshape(1, -1)

    predicted_class = model.predict(features)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    description = bean_descriptions.get(predicted_label[0], "Description not available.")
    
    return predicted_label[0], description

# Hàm chuẩn bị dữ liệu huấn luyện
def prepare_training_data():
    global label_encoder, rf_model, svm_model, nn_model  # Declare models as global variables before assignment

    # Load the dataset from the provided file path
    file_path = "Dry_Bean_Dataset.xlsx"  # Ensure that the file path is correct
    dry_bean_data = pd.read_excel(file_path)

    # Show the first few rows of the dataset to understand its structure
    st.write(dry_bean_data.head())

    # Prepare data: split features and labels
    X = dry_bean_data.drop(columns=["Class"])  # Features
    y = dry_bean_data["Class"]                   # Labels

    label_encoder = LabelEncoder()                # Assign value to global variable
    y_encoded = label_encoder.fit_transform(y)    # Encode labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)  # Assign value to global variable
    svm_model = SVC(random_state=42)                      # Assign value to global variable
    nn_model = MLPClassifier(random_state=42, max_iter=500)  # Assign value to global variable

    # Train the models
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    nn_model.fit(X_train, y_train)

    # Make predictions on the test set
    rf_preds = rf_model.predict(X_test)
    svm_preds = svm_model.predict(X_test)
    nn_preds = nn_model.predict(X_test)

    # Calculate accuracy
    rf_accuracy = accuracy_score(y_test, rf_preds)
    svm_accuracy = accuracy_score(y_test, svm_preds)
    nn_accuracy = accuracy_score(y_test, nn_preds)

    # Display accuracy
    st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")
    st.write(f"SVM Accuracy: {svm_accuracy:.2f}")
    st.write(f"Neural Network Accuracy: {nn_accuracy:.2f}")

    # Feature importance from Random Forest
    feature_importance = rf_model.feature_importances_
    features = X.columns

    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance from Random Forest")
    st.pyplot(plt)  # Display the plot in Streamlit

    # Correlation matrix
    corr_matrix = X.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Feature Correlation Matrix")
    st.pyplot(plt)  # Display the heatmap in Streamlit

# Giao diện Streamlit
st.title("Bean Classification App")
st.write("Select an option from the menu on the left.")

# Load models initially
load_models()

# Menu lựa chọn
option = st.sidebar.selectbox("Choose an option:", ["Prepare Training Data", "Chatbot", "Classify Bean from Image"])

if option == "Classify Bean from Image":
    st.write("Upload an image of a bean to classify it.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Chuyển đổi từ BGR sang RGB trước khi hiển thị
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predicted_bean_type, predicted_bean_description = predict_bean_from_image(image, rf_model)

        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)
        st.write(f"The predicted bean type is: {predicted_bean_type}")
        st.write(f"Description: {predicted_bean_description}")

elif option == "Prepare Training Data":
    st.write("Preparing the training data...")
    prepare_training_data()  # Gọi hàm chuẩn bị dữ liệu

elif option == "Chatbot":
    st.write("This section is for the chatbot. You can ask questions about beans or classification.")
    user_input = st.text_area("Type your question for the chatbot:", key="user_input")  # Add a unique key
    
    if st.button("Send", key="send_button"):  # Add a unique key
        if user_input:
            # Tạo một đối tượng chat session và gửi input của người dùng đến GPT
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(user_input)
            st.write("Chatbot response:")
            st.write(response.text)
        else:
            st.write("Please enter a question.")

