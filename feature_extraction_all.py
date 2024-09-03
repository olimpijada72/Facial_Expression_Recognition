import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import csv
from tqdm import tqdm
import os


emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Loading CNN model which we will use to extract features
model_path = 'models/mobilenet.pth'
mobilenet = torch.load(model_path)
mobilenet.eval()


# Define the normalization mean and std values for MobileNet
# For grayscale images, use the mean and std values for each channel
# Even though we have only one channel, we can reuse the same values used for RGB channels
mobilenet_mean = [0.485]
mobilenet_std = [0.229]

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=mobilenet_mean, std=mobilenet_std)  # Normalize with mean and std
])




def extract_features(img_path, model):
    image = Image.open(img_path).convert('L')  # Convert to grayscale
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0)  

    # Extract features
    with torch.no_grad():
        features = model(transformed_image)
    return features.squeeze().numpy()  # Convert to numpy array and remove batch dimension



csv_file_path = 'data/features_2.csv'



# Open the CSV file for writing
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    number_of_features = 1000
    header = ['image_name', 'emotion'] + [f'feature_{i}' for i in range(number_of_features)]  
    writer.writerow(header)
    

    for emotion in emotions:
        folder_path = os.path.join('data/mesh_faces', emotion)
        for image_id in tqdm(os.listdir(folder_path)):
            image_name = os.path.join(folder_path, image_id)
            features = extract_features(image_name, mobilenet)
            row = [image_id, emotion] + features.tolist()
            writer.writerow(row)





