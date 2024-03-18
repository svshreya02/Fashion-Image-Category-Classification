from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import io
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
from sklearn.cluster import KMeans

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Classification and Color Extraction API!"}
# Define transforms exactly as in your notebook
transform = transforms.Compose([
    transforms.Resize((60, 80)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load the label encoder using the labels.txt file
def load_label_encoder(labels_file):
    labels_df = pd.read_csv(labels_file, header=None)
    label_encoder = LabelEncoder()
    labels_df[1] = label_encoder.fit_transform(labels_df[1])
    return label_encoder

# Load label encoder
label_encoder = load_label_encoder('/efs/users/Shreya_Sivakumar/TASK_1/data/labels.txt')

# Initialize and load the model
def initialize_model(num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load('/efs/users/Shreya_Sivakumar/TASK_1/models/torch_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Adjust the number of classes as per your labels
model = initialize_model(num_classes=len(label_encoder.classes_))


@app.post("/predict-category/")
async def predict_image_category(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_label = label_encoder.inverse_transform([predicted.item()])
    return JSONResponse(content={"predicted_category": predicted_label[0]})

@app.post("/extract-colors/")
async def extract_image_colors(file: UploadFile = File(...), num_colors: int = 5):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    colors = extract_colors(image, num_colors=num_colors)
    return JSONResponse(content={"colors": colors.tolist()})

# Updated extract_colors to work directly with PIL images instead of file paths
def extract_colors(image, num_colors=5):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    return colors.astype(int)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
