# Streamlit app
import streamlit as st
import torch
from PIL import Image
import json
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split, SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
# import other necessary modules

def get_model(device):
    """
    Function to get the model.
    """
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, 2)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = CNN().to(device)
    return model

# Load your trained model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)  # your get_model function
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to process an uploaded image
def process_image(uploaded_file):
    # Open the image file
    img = Image.open(uploaded_file)

    # Convert grayscale or images with alpha channel to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # You should load your saved transforms from 'transform_parameters.json'
    with open('transform_parameters.json') as f:
        transform_parameters = json.load(f)
    mean = transform_parameters['mean']
    std = transform_parameters['std']
    transform = transforms.Compose([
        transforms.Resize(size=(56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

# Function to get predictions
def predict(model, image):
    # Check the device the model is on
    device = next(model.parameters()).device
    # Move the image tensor to the device the model is on
    image = image.to(device)
    outputs = model(image)
    # Calculate probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1)
    highest_prob, predicted = torch.max(probs, 1)
    return highest_prob, predicted

def plot_image(image_path, caption):
    """
    Function to display an image with a caption
    """
    image = Image.open(image_path)
    st.image(image, caption=caption, use_column_width=True)

def main():
    st.title("Pneumonia Detection from Chest X-Ray Images")

    # Create a mapping for classes
    class_names = {0: 'Normal', 1: 'Pneumonia'}

    # Load the model
    model = load_model('saved_models/best_model.pth')  # update the path

    # Upload image and preprocess it
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Chest X-Ray.', use_column_width=True)
        if st.button('Predict'):
            image = process_image(uploaded_file)
            highest_prob, predicted = predict(model, image)
            predicted_class = class_names[predicted.item()]
            # Use markdown to display results in a larger font
            st.markdown(f"<h2 style='text-align: center; color: red;'>Prediction: {predicted_class}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: white;'>Probability: {highest_prob.item()*100:.2f}%</h3>", unsafe_allow_html=True)

    # Adding another page in the app for model details
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Details"])

    if page == "Model Details":
        st.header("Model Details")
        
        # Model Introduction
        st.subheader("Introduction:")
        st.write("""
        Image Classification is an approach that is being widely utilised for different purposes in the realm of medical imaging and diagnostics. Pneumonia, a potentially life-threatening respiratory infection, poses a significant global health challenge, and rapid and accurate diagnosis is paramount for effective treatment.
        """)
        
        # Dataset details
        st.subheader("Dataset:")
        st.write("""
        - The dataset is divided into three folders - 'train', 'test', and 'val', and it contains subfolders for different image categories, specifically 'Pneumonia' and 'Normal'. 
        - These chest X-ray images were obtained from pediatric patients aged one to five years old.
        """)
        st.markdown("[Link to the dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)")
        
        # Display the images: Confusion Matrix, Validation-Train Accuracy and Validation-Train Loss
        plot_image("/Users/ganeshghimire/Documents/github/capstone/confusion_matrix.png", "Confusion Matrix")


        # Classification Report
        st.subheader("Classification Report:")
        st.write("""
        ```
        Classification Report:
                   precision    recall  f1-score   support
              Normal       0.73      0.87      0.79       234
           Pneumonia       0.91      0.80      0.85       390
            accuracy                           0.83       624
           macro avg       0.82      0.84      0.82       624
        weighted avg       0.84      0.83      0.83       624
        ```
        """)

        # Model Architecture
        st.subheader("Model Architecture:")
        st.write("""
        The Convolutional Neural Network (CNN) architecture consists of four convolutional layers followed by pooling layers. After the convolutional layers, there are three fully connected layers with a dropout layer in between to prevent overfitting. The model outputs two classes: 'Normal' and 'Pneumonia'. This architecture was trained using a batch size of 64, learning rate of 0.001, and for a total of 12 epochs. The training and validation accuracy achieved is displayed in the provided plots.
        """)
        plot_image("validation_train accuracy.png", "Validation-Train Accuracy")
        plot_image("validation_train loss.png", "Validation-Train Loss")

# The main entry of the application
if __name__ == "__main__":
    main()