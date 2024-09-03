import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

# Load the pre-trained MobileNet model
mobilenet = models.mobilenet_v2(pretrained=True)

# Modify the first convolutional layer to accept grayscale images
# The original conv1 layer has weight shape [out_channels, in_channels, kernel_size, kernel_size]
# We need to change the in_channels from 3 to 1
original_conv1 = mobilenet.features[0][0]
new_conv1 = torch.nn.Conv2d(
    in_channels=1, 
    out_channels=original_conv1.out_channels, 
    kernel_size=original_conv1.kernel_size, 
    stride=original_conv1.stride, 
    padding=original_conv1.padding, 
    bias=original_conv1.bias is not None
)

# Copy the weights from the original conv1 layer to the new conv1 layer
new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
if original_conv1.bias is not None:
    new_conv1.bias.data = original_conv1.bias.data

# Replace the first convolutional layer in the model
mobilenet.features[0][0] = new_conv1


model_path = 'models/mobilenet.pth'
torch.save(mobilenet, model_path)

