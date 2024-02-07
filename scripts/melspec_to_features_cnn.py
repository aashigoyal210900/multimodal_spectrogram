import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = models.resnet18(pretrained=True)
        # Modify the first layer to accept 1 channel input (for grayscale spectrograms)
        self.features.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final layer to output desired feature size
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.softmax(x)
        return x

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),           # Convert to tensor
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def extract_features_from_folder(input_folder):
    # Initialize the model
    model = CNN(num_classes=3)  # 3 classes for HAPPY, SAD, ANGRY
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # List all files in the input folder
    files = os.listdir(input_folder)
    
    # Iterate over files in the folder
    for filename in files:
        if filename.endswith(".png"):  # Assuming mel spectrograms are stored as PNG files
            input_path = os.path.join(input_folder, filename)
            img_tensor = preprocess_image(input_path)
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                output_features = model(img_tensor)
            print(f"Features extracted for {filename}: {output_features}")

            
if __name__ == "__main__":
    # Check if input arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python melspec_to_features_cnn.py input_folder")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        sys.exit(1)
    
def main():
    extract_features_from_folder(input_folder)

main()
