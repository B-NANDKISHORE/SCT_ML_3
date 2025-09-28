# 🚀 Image Classification with SVM: A Streamlit Application
Tagline: A Python project that leverages the power of Support Vector Machines (SVM) to classify images and provides a user-friendly interface using Streamlit.

# Description
This project aims to build an image classification application using Support Vector Machines (SVM) and Streamlit. The application allows users to upload an image, and the model predicts the class label based on a pre-trained SVM model. The project is designed to demonstrate the effectiveness of SVM in image classification tasks and provide a user-friendly interface for users to interact with the model.

# Features
Image Classification: The application can classify images into predefined classes using a pre-trained SVM model.

User-Friendly Interface: The application uses Streamlit to provide a user-friendly interface for users to interact with the model.

Image Preprocessing: The application can read and preprocess images using OpenCV and PIL.

Model Training: The application can train an SVM model using a dataset of labeled images.

Model Evaluation: The application can evaluate the performance of the SVM model using metrics such as accuracy.

Deployment: The application can be deployed as a web application using Streamlit.

Command-Line Interface: The application provides a command-line interface for users to train and evaluate the model.

Support for Multiple Image Formats: The application supports multiple image formats, including JPEG, PNG, and BMP.

# Tech Stack
Technology  

Python	    3.8

Streamlit	  1.11

OpenCV	    4.5.1

PIL	        8.3.2

scikit-learn	1.0.2

joblib	    1.1.0

# Project Structure

app.py
requirements.txt
README.md

app.py: The main application file that uses Streamlit to provide a user-friendly interface.

requirements.txt: The file that lists the dependencies required to run the application.

# How to Run

Install the dependencies: Run pip install -r requirements.txt to install the dependencies.

Train the model: Run python train_svm.py to train the SVM model.

Run the application: Run streamlit run app.py to run the application.

Deploy the application: Use Streamlit to deploy the application as a web application.

# Testing Instructions
Test the application: Use the command-line interface to test the application by uploading an image and verifying the predicted class label.
