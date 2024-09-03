import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import csv


model_path = 'models/mobilenet.pth'
mobilenet = torch.load(model_path)
mobilenet.eval()


# Define the normalization mean and std values for MobileNet
# For grayscale images, use the mean and std values for each channel
# Even though we have only one channel, we can reuse the same values used for RGB channels
# mobilenet_mean = [0.485]
# mobilenet_std = [0.229]

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    # transforms.Normalize(mean=mobilenet_mean, std=mobilenet_std)  # Normalize with mean and std
])

# Load the grayscale image
img_path = 'data/fear1_mesh.jpg'


def extract_features(img_path, model):
    image = Image.open(img_path).convert('L')  # Convert to grayscale
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0)  # Add a batch dimension

    # Extract features
    with torch.no_grad():
        features = model(transformed_image)
    return features.squeeze().numpy()  # Convert to numpy array and remove batch dimension


features =  extract_features(img_path, mobilenet)

csv_file_path = 'data/features.csv'



# Open the CSV file for writing
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    header = ['image_name', 'emotion'] + [f'feature_{i}' for i in range(features.shape[0])]  
    writer.writerow(header)
    

    emotion = 'neutral'
    features = extract_features(img_path, mobilenet)
    row = [img_path, emotion] + features.tolist()
    writer.writerow(row)

    # Loop through images and emotions, extract features, and write to CSV
    # for img_path, emotion in images_and_emotions:
    #     features = extract_features(img_path, modified_mobilenet)
    #     row = [img_path, emotion] + features.tolist()
    #     writer.writerow(row)
