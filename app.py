import streamlit as st
import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# -------------------------
# Load dataset function
# -------------------------
DATASET_DIR = "data"
CATEGORIES = ["cats", "dogs"]
IMG_SIZE = 64

def load_data():
    data = []
    labels = []
    for category in CATEGORIES:
        folder = os.path.join(DATASET_DIR, category)
        if not os.path.exists(folder):
            continue
        label = CATEGORIES.index(category)
        for file in os.listdir(folder):
            try:
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img.flatten())
                labels.append(label)
            except:
                continue
    return np.array(data), np.array(labels)

# -------------------------
# Train model (or load if exists)
# -------------------------
MODEL_FILE = "svm_model.pkl"

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained! Accuracy: {acc:.2f}")
    joblib.dump(model, MODEL_FILE)

# -------------------------
# Streamlit Web App
# -------------------------
st.title("🐱🐶 Cat vs Dog Classifier (SVM)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    st.image(img, caption="Uploaded Image", use_container_width=True)

    
    # Predict
    features = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    st.subheader("Prediction:")
    st.write(f"Class: **{CATEGORIES[prediction]}**")
    st.write(f"Confidence: {prob[prediction]*100:.2f}%")
