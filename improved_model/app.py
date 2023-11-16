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
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import json
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# import other necessary modules

def get_model(device):
    """
    Function to get the model.
    """
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=7, padding=3)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=7, padding=3)
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

            self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
            self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
            self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

            self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv7 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv9 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(128*7*7, 128)
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(128, 2)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool1(x)

            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool2(x)

            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = self.pool3(x)

            x = F.relu(self.conv7(x))
            x = F.relu(self.conv8(x))
            x = self.pool4(x)

            x = F.relu(self.conv9(x))
            x = F.relu(self.conv10(x))
            x = self.pool5(x)

            x = self.adaptive_pool(x)
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = CNNModel().to(device)
    return model

# Load your trained model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)  # your get_model function
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Global variables for mean and std
mean = None
std = None

# Function to process an uploaded image
def process_image(uploaded_file, device):
    global mean, std
    # Open the image file
    img = Image.open(uploaded_file)

    # Convert grayscale or images with alpha channel to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Load your saved transforms from 'transform_parameters.json'
    with open('transform_parameters.json') as f:
        transform_parameters = json.load(f)
    mean = transform_parameters['mean']
    std = transform_parameters['std']
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),  # Modify the size to 224x224 or another size that is compatible with your model architecture
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image_tensor = transform(img).unsqueeze(0).to(device)
    return image_tensor

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

# Function to get predictions
def predict(model, image):
    # No need to check the device again, as it's already checked in the process_image function
    outputs = model(image)
    # Calculate probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1)
    highest_prob, predicted = torch.max(probs, 1)
    return highest_prob, predicted

def apply_grad_cam(image_tensor, model, target_layer, mean, std, uploaded_file):
    model_output = model(image_tensor)
    predicted_class = model_output.argmax(dim=1).item()

    target_layers = [target_layer]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=image_tensor.is_cuda)
    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]

    # Ensure mean and std are numpy arrays
    mean = np.array(mean)
    std = np.array(std)

    # Convert the image tensor from CUDA to CPU and then to a numpy array if it's not already on the CPU
    image_numpy = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_numpy = (image_numpy * std + mean) * 255
    image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)

    grayscale_cam = cv2.resize(grayscale_cam, (image_numpy.shape[1], image_numpy.shape[0]))
    cam_image = show_cam_on_image(image_numpy/255.0, grayscale_cam, use_rgb=True)
    # Convert the PIL image to numpy array to get its size
    pil_image = Image.open(uploaded_file)
    original_size = pil_image.size  # Original size is in (width, height)
    
    # Resize Grad-CAM image to match the original image size
    cam_image_resized = cv2.resize(cam_image, original_size)
    return cam_image

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

        # In the main function, within the 'Predict' button conditional
        if st.button('Predict'):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image = process_image(uploaded_file, device)
            highest_prob, predicted = predict(model, image)
            predicted_class = class_names[predicted.item()]
            st.markdown(f"<h2 style='text-align: center; color: red;'>Prediction: {predicted_class}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: white;'>Probability: {highest_prob.item()*100:.2f}%</h3>", unsafe_allow_html=True)

            # Call apply_grad_cam with mean, std, and uploaded_file
            cam_image = apply_grad_cam(image, model, model.conv10, mean, std, uploaded_file)
            st.image(cam_image, caption="Grad-CAM Visualization", use_column_width=True)



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
        plot_image("confusion_matrix.png", "Confusion Matrix")


        # Classification Report
        st.subheader("Classification Report:")
        st.write("""
        ```
        Classification Report:
                           precision recall    f1-score   support
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
        The CNNModel architecture is a convolutional neural network designed for image classification tasks. It starts with an input convolutional layer with a kernel size of 7x7 and padding of 3, which increases the depth from 3 channels to 8 while preserving the spatial dimensions due to padding. This is followed by a second convolutional layer with the same specifications, which helps in learning more complex features without reducing the size of the feature maps.

Subsequent layers include a series of convolution and max pooling layers that progressively increase the depth while reducing the spatial dimensions. Specifically, the network employs max pooling with a 3x3 kernel and a stride of 3 after the first two convolutional layers, followed by two sets of convolutional layers with 5x5 kernels and 3x3 pooling, and finally two sets with 3x3 kernels and 2x2 pooling. These pooling layers serve to reduce the dimensionality, condensing the feature representations and making the network more computationally efficient.

The network then transitions from convolutional layers to fully connected layers through an adaptive average pooling layer that outputs a fixed-size 7x7 feature map regardless of the input size. This allows for flexibility in the input image dimensions and ensures the following fully connected layers have a consistent input size.

Flattening the output of the adaptive pooling layer, the network proceeds to a dense layer with 128 units, including a dropout layer with a rate of 0.2 to prevent overfitting by randomly omitting some of the features during training. The final fully connected layer outputs the logits for two classes, which can be converted into probabilities via a softmax function.
        """)
        plot_image("validation_train accuracy.png", "Validation-Train Accuracy")
        plot_image("validation_train loss.png", "Validation-Train Loss")

# The main entry of the application
if __name__ == "__main__":
    main()
