#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


get_ipython().system('python3 -m pip install einops')
get_ipython().system('python3 -m pip install facenet-pytorch')
get_ipython().system('python3 -m pip install face_alignment')
get_ipython().system('python3 -m pip install self_attention_cv')


# In[3]:


import cv2
import gc
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from einops.layers.torch import Rearrange
from einops import rearrange
from facenet_pytorch import MTCNN
from self_attention_cv import TransformerEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import face_alignment
import dlib
import requests


# In[4]:


def extract_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    mid_frame_index = frame_count // 2  # Index of the frame in the middle of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
    ret, frame = cap.read()
    if ret:
        cap.release()
        return frame
    else:
        cap.release()
        return None


# In[5]:


def detect_face(frame):
    mtcnn = MTCNN()
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        # Assuming only one face in the frame
        box = boxes[0]
        x1, y1, x2, y2 = box
        # Crop the frame to the detected face
        cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
        return cropped_frame
    else:
        return None


# In[6]:


# Function to download the pretrained face alignment model if it doesn't exist
def download_face_alignment_model(url, save_path):
    if not os.path.exists(save_path):
        print("Downloading pretrained face alignment model...")
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

# Specify the URL of the pretrained face alignment model
face_alignment_model_url = "https://github.com/1adrianb/face-alignment-models/releases/download/2.0.1/2DFAN4-11f355bf06.pth.tar"

# Download the pretrained face alignment model if it doesn't exist
face_alignment_model_path = os.path.abspath("2DFAN4-11f355bf06.pth.tar")
download_face_alignment_model(face_alignment_model_url, face_alignment_model_path)

# Initialize face alignment model
fa = face_alignment.FaceAlignment(2, flip_input=False,device='cpu')  # 2 corresponds to 2D landmarks

def align_face(frame):
    # Perform face alignment
    aligned_faces = fa.get_landmarks(frame)
    if aligned_faces is not None:
        aligned_face = aligned_faces[0]  # Assuming only one face in the frame
        return aligned_face
    else:
        return None


# In[7]:


# Define the preprocessing functions for video frames and spectrograms
def preprocess_image(frame):
    frame_pil = Image.fromarray(frame.astype('uint8'))
    frame_pil = frame_pil.convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalize using ImageNet mean and std
    ])
    frame_tensor = transform(frame_pil)
    return frame_tensor


# In[8]:


def preprocess_spectrogram(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ViT input size
        transforms.ToTensor(),           # Convert to tensor
    ])
    img_tensor = transform(img)
    return img_tensor


# In[9]:


def load_spectrogram_dataset(input_folder):
    X = []
    y = []
    # List all files in the input folder
    files = os.listdir(input_folder)
    # Iterate over files in the folder
    for filename in files:
        if filename.endswith(".png"):  # Assuming mel spectrograms are stored as PNG files
            input_path = os.path.join(input_folder, filename)
            img_tensor = preprocess_spectrogram(input_path)
            X.append(img_tensor)
            # Extract label from filename (assuming filename is in format "abc_IEO_label_xyz.png")
            label = filename.split("_")[2]
            if label == "HAP":
                y.append(0)
            elif label == "SAD":
                y.append(1)
            elif label == "ANG":
                y.append(2)
            elif label == "DIS":
                y.append(3)
            elif label == "FEA":
                y.append(4)
            elif label == "NEU":
                y.append(5)
    return X, y


# In[10]:


def load_dataset(input_folder):
    X = []
    y = []
    video_files = [file for file in os.listdir(input_folder) if file.endswith(".flv")]
    for video_file in tqdm(video_files):
        video_path = os.path.join(input_folder, video_file)
        frame = extract_frame(video_path)
        if frame is not None:
            cropped_face = detect_face(frame)
            if cropped_face is not None:
                preprocessed_face = preprocess_image(cropped_face)
                X.append(preprocessed_face)
                label = video_file.split("_")[2].split(".")[0]  # Adjusted to handle different file extensions
                if label == "HAP":
                    y.append(0)
                elif label == "SAD":
                    y.append(1)
                elif label == "ANG":
                    y.append(2)
                elif label == "DIS":
                    y.append(3)
                elif label == "FEA":
                    y.append(4)
                elif label == "NEU":
                    y.append(5)
            else:
                print(f"No face detected in {video_file}. Skipping.")
        else:
            print(f"Failed to extract frame from {video_file}. Skipping.")
    return X, y


# In[11]:


# Define the ConcatDataset class to concatenate video frame and spectrogram tensors
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, X1, X2, y, modality='multimodal', fullscale=False):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.modality = modality
        self.fullscale = fullscale
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if not self.fullscale:
          img1 = self.X1[idx]
          img2 = self.X2[idx]
          label = self.y[idx]
        else:
          img1 = torch.from_numpy(self.X1[idx]).float()  # Convert numpy array to torch tensor
          img2 = torch.from_numpy(self.X2[idx]).float()  # Convert numpy array to torch tensor
          label = torch.tensor(self.y[idx])  # Convert numpy array to torch tensor

        concatenated_img = torch.cat((img1, img2), dim=0)  # Concatenate along 0 dimension
        if self.modality == 'visual':
          return img1, label
        if self.modality == 'audio':
          return img2, label
        return concatenated_img, label # concatenate modalities


# In[12]:


def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = correct_preds / total_preds
    return epoch_loss, accuracy


# In[13]:


def test_model(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct_preds / total_preds
    return epoch_loss, accuracy


# In[14]:


class ViT(nn.Module):
    # ViT architecture adapted from here - https://theaisummer.com/vision-transformer/
    def __init__(self, *,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=6, # full-scale CREMA-D
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0.4, transformer=None, classification=True):
        """
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible'
        self.p = patch_dim
        self.classification = classification
        tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)
        if self.classification:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, dim))
            self.mlp_head = nn.Linear(dim, num_classes)
        else:
            self.pos_emb1D = nn.Parameter(torch.randn(tokens, dim))

        if transformer is None:
            self.transformer = TransformerEncoder(dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

    def expand_cls_to_batch(self, batch):
        """
        Args:
            batch: batch size
        Returns: cls token expanded to the batch size
        """
        return self.cls_token.expand([batch, -1, -1])

    def forward(self, img, mask=None):
        batch_size = img.shape[0]
        img_patches = rearrange(
            img, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)
        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        if self.classification:
            img_patches = torch.cat(
                (self.expand_cls_to_batch(batch_size), img_patches), dim=1)

        patch_embeddings = self.emb_dropout(img_patches + self.pos_emb1D)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)

        if self.classification:
            # we index only the cls token for classification.
            return self.mlp_head(y[:, 0, :])
        else:
            return y


# In[15]:


_fullscale = True # Run fullscale experiment?

# Define input_folder and input_folder_spec
if _fullscale:
  input_folder = '/Users/aashigoyal/Desktop/vscode/aashimain/csci535/final_temp/videos_fullscale'
  input_folder_spec = '/Users/aashigoyal/Desktop/vscode/aashimain/csci535/final_temp/melspec_fullscale'
else:
  input_folder = '/content/drive/MyDrive/csci535/videos'
  input_folder_spec = '/content/drive/MyDrive/csci535/melspec'

# Check if input folder exists
if not os.path.exists(input_folder):
    print("Input folder does not exist.")
    sys.exit(1)
# Check if input folder exists
if not os.path.exists(input_folder_spec):
    print("Input folder does not exist.")
    sys.exit(1)

# Load dataset and split into train and test sets

if not _fullscale:
  X, y = load_dataset(input_folder)
  X_spec, y_spec = load_spectrogram_dataset(input_folder_spec)

else:
  # Load numpy arrays with memory-mapping
  X = np.load('/Users/aashigoyal/Desktop/vscode/aashimain/csci535/final_temp/X.npy', mmap_mode='r')
  y = np.load('/Users/aashigoyal/Desktop/vscode/aashimain/csci535/final_temp/y.npy', mmap_mode='r')
  X_spec = np.load('/Users/aashigoyal/Desktop/vscode/aashimain/csci535/final_temp/X_spec.npy', mmap_mode='r')
  y_spec = np.load('/Users/aashigoyal/Desktop/vscode/aashimain/csci535/final_temp/y_spec.npy', mmap_mode='r')

# Split the data into train and test sets
print(f"Total number of samples: {len(X)}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Number of train samples (video): {len(X_train)}", f"Number of test samples: {len(X_test)}")
X_train_spec, X_test_spec, y_train_spec, y_test_spec = train_test_split(X_spec, y_spec, test_size=0.3, random_state=42)
print(f"Number of train samples (audio): {len(X_train_spec)}", f"Number of test samples: {len(X_test_spec)}")


# In[16]:


# Save X, y, X_spec, y_spec
# np.save('X.npy', np.array(X))
# np.save('y.npy', np.array(y))
# np.save('X_spec.npy', np.array(X_spec))
# np.save('y_spec.npy', np.array(y_spec))
# !cp 'X.npy' '/content/drive/MyDrive/csci535/models/'
# !cp 'y.npy' '/content/drive/MyDrive/csci535/models/'
# !cp 'X_spec.npy' '/content/drive/MyDrive/csci535/models/'
# !cp 'y_spec.npy' '/content/drive/MyDrive/csci535/models/'


# In[17]:


def train_ViT(_modality):
  # Adjust input channels as per modality
  if _modality == 'multimodal':
    _input_channels = 2
  else:
    _input_channels = 1

  # Initialize the ViT model
  model = ViT(img_dim=224,  # Image dimension
              in_channels=_input_channels,  # Number of input channels
              patch_dim=16,  # Patch dimension
              num_classes=6,  # 6 classes for HAPPY, SAD, ANGRY, DISGUST, FEAR, NEUTRAL
              dim=768,  # Dimensionality of the token embeddings
              blocks=6,  # Number of transformer blocks
              heads=4,  # Number of attention heads
              dim_linear_block=1024,  # Dimensionality of the linear block
              dropout=0.4,  # Dropout rate
              classification=True)  # Whether or not to include a classification token

  # Define device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # Define loss function and optimizer
  _lr = 0.0001
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=_lr)

  # Concatenate datasets if multimodal
  train_dataset = ConcatDataset(X_train, X_train_spec, y_train, modality = _modality, fullscale = _fullscale)
  test_dataset = ConcatDataset(X_test, X_test_spec, y_test, modality = _modality, fullscale = _fullscale)

  # Create data loaders
  _bs = 8

  # Create data loaders
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_bs, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_bs)

  print(f"\n\nBatch size: {_bs}", f"lr: {_lr}")

  # Training loop
  num_epochs = 50
  print(f"Training ViT for \"{_modality}\" pipeline ...\n------------------------------------------------\n")
  for epoch in range(num_epochs):
      print("Epoch " + str(epoch))
      train_loss, train_accuracy = train_model(model, criterion, optimizer, train_loader, device)
      test_loss, test_accuracy = test_model(model, criterion, test_loader, device)
      print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

  # Save the model
  if _modality == 'multimodal':
    torch.save(model.state_dict(), 'ViT_audio_video_fullscale_'+str(num_epochs)+'_'+str(_bs)+'_'+str(_lr))
  elif _modality == 'audio':
    torch.save(model.state_dict(), 'ViT_audio_fullscale_'+str(num_epochs)+'_'+str(_bs)+'_'+str(_lr))
  elif _modality == 'visual':
    torch.save(model.state_dict(), 'ViT_video_fullscale_'+str(num_epochs)+'_'+str(_bs)+'_'+str(_lr))
  else:
    print("Improper modality provided!")

  return train_loss, train_accuracy, test_loss, test_accuracy


# In[18]:


# Define modalities
# _modality = ['visual', 'audio', 'multimodal']


# In[19]:


# Train ViT
# scores = {}
# for _m in _modality:
#   train_loss, train_accuracy, test_loss, test_accuracy = train_ViT(_m)
#   scores[_m] = [train_loss, train_accuracy, test_loss, test_accuracy]


# In[20]:


# # Print results
# print("\nResults\n---------------\n")
# for _m, val in scores.items():
#   print(f"Modality: {_m}, Train Loss: {val[0]:.4f}, Train Accuracy: {val[1]:.4f}, Test Loss: {val[2]:.4f}, Test Accuracy: {val[3]:.4f}")


# In[21]:


# # Copy trained models to GDrive
# !cp 'ViT_audio_video_fullscale_50_16_0.0001' '/content/drive/MyDrive/csci535/models'
# !cp 'ViT_audio_fullscale_50_16_0.0001' '/content/drive/MyDrive/csci535/models'
# !cp 'ViT_video_fullscale_50_16_0.0001' '/content/drive/MyDrive/csci535/models'


# In[22]:


def cross_test_ViT(_modality_model, _modality_features):
  _input_channels = 1
  model = ViT(img_dim=224,  # Image dimension
              in_channels=_input_channels,  # Number of input channels
              patch_dim=16,  # Patch dimension
              num_classes=6,  # 6 classes for HAPPY, SAD, ANGRY, DISGUST, FEAR, NEUTRAL
              dim=768,  # Dimensionality of the token embeddings
              blocks=6,  # Number of transformer blocks
              heads=4,  # Number of attention heads
              dim_linear_block=1024,  # Dimensionality of the linear block
              dropout=0.4,  # Dropout rate
              classification=True)  # Whether or not to include a classification token

  state_dict = torch.load('/Users/aashigoyal/Desktop/vscode/aashimain/csci535/final_temp/models/ViT_' + _modality_model + '_fullscale_50_8_0.0001',map_location=torch.device('cpu'))
  model.load_state_dict(state_dict)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  # Define loss function and optimizer
  _lr = 0.0001
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=_lr)

  test_dataset = ConcatDataset(X_test, X_test_spec, y_test, modality = _modality_features, fullscale = _fullscale)

  # Create data loaders
  _bs = 16

  # Create data loaders
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_bs)
  print(f"\nTesting {_modality_model}-trained ViT pipeline on {_modality_features} features ...\n---------------------------------------------------------\n")
  test_loss, test_accuracy = test_model(model, criterion, test_loader, device)
  print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# In[23]:


# Peform cross-test
cross_test_ViT('audio', 'visual')
cross_test_ViT('video', 'audio')


# In[24]:


# Perform straight-test for sanity check
cross_test_ViT('audio', 'audio')
cross_test_ViT('video', 'visual')


# In[25]:


# Clear memory
gc.collect()


# In[ ]:




