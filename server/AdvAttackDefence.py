# %%
# 1. Download Dataset
#Data Source: https://advnet.seas.upenn.edu/#:~:text=AdvNet%20is%20a%20dataset%20of,without%20any%20stickers%20on%20them

from io import BytesIO

import argparse
import subprocess
import time
import gdown
import os
import requests
import sys
import foolbox as fb


sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Train or Load Model with Attack and Defense Options")
parser.add_argument("--train", action="store_true", help="Train the model from scratch")
parser.add_argument("--attack", choices=["fgsm", "pgd", "deepfool", "all"], default="all", help="Select attack type")
parser.add_argument("--defense", choices=["bitdepth", "binary", "none", "jpeg", "all"], default="all", help="Select defense type")
args = parser.parse_args()

# Initialize Attack and Defense Lists
attacks = [] 
defense_opts = []

if args.attack in ['all']:
    attacks = ['fgsm', 'pgd', 'deepfool']
else:
    attacks = [args.attack] 

if args.defense in ['all']:
    defense_opts = ['vanilla', 'bitdepth', 'binary', 'jpeg']
elif args.defense == 'none':
    defense_opts = ['vanilla']
else:
    defense_opts = [args.defense] 

# url = f"https://drive.google.com/uc?export=download&id=13NdhIvPgzOQoRg9A-xUUXSsfxVwPrEUV"
# output_path = "../content/LisaCnn.zip"


# %%
# 2. Extract the Downloaded Dataset"""

import zipfile
import os

#CHANGE THIS
#zip_path = "C:/Users/himsi/source/repos/PythonTester/LisaCnn.zip"  # Update with the correct path on your local machine
zip_path = "../content/LisaCnn.zip"
#CHANGE THIS
extract_dir = "../content/LisaCnn"  # Correct extraction directory for local machine

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction completed.")

# %%
# 3. Check Folder Structure

for root, dirs, files in os.walk(extract_dir):
    print(root, ":", len(files), "files")


# %%
# 4. Organize the Dataset"""

import shutil

# Paths
original_root = "../content/LisaCnn/LisaCnn"
clean_root = "../content/LisaCnn_Clean"
adv_root = "../content/LisaCnn_Adv"  # New folder for adversarial images

# Categories and their corresponding "clean" and "adversarial" folders
categories = {
    "Speed35Signs": ("Speed35Signs/Clean", "Speed35Signs/Adv"),
    "StopSigns": ("StopSigns/CleanStop", "StopSigns/LisaAdvStop/AdvStop"),
    "YieldSigns": ("YieldSigns/CleanYield", "YieldSigns/AdvYield")
}

# Create dataset structures
os.makedirs(clean_root, exist_ok=True)
os.makedirs(adv_root, exist_ok=True)

for category, (clean_folder, adv_folder) in categories.items():
    clean_source = os.path.join(original_root, clean_folder)
    adv_source = os.path.join(original_root, adv_folder)

    clean_target = os.path.join(clean_root, category)
    adv_target = os.path.join(adv_root, category)

    os.makedirs(clean_target, exist_ok=True)
    os.makedirs(adv_target, exist_ok=True)

    # Copy clean images
    for file in os.listdir(clean_source):
        file_path = os.path.join(clean_source, file)
        if os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(clean_target, file))

    # Copy adversarial images
    for file in os.listdir(adv_source):
        file_path = os.path.join(adv_source, file)
        if os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(adv_target, file))

print("Clean and adversarial datasets structured successfully.")

# %%
# 5. Analyze Image Sizes and Color Channels"""

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define dataset path
dataset_root = "../content/LisaCnn/LisaCnn"

# Clean image directories
clean_folders = {
    "Speed35Signs": "Speed35Signs/Clean",
    "StopSigns": "StopSigns/CleanStop",
    "YieldSigns": "YieldSigns/CleanYield"
}

# Store statistics
image_stats = []

# Analyze each category
for category, folder in clean_folders.items():
    folder_path = os.path.join(dataset_root, folder)
    if os.path.exists(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode  # 'RGB', 'L' (grayscale), etc.
                    image_stats.append((category, width, height, mode))
            except Exception as e:
                print(f"Could not open {image_path}: {e}")

# Convert to DataFrame
df_stats = pd.DataFrame(image_stats, columns=["Category", "Width", "Height", "Mode"])
print(df_stats)

# Plot Image Size Distributions
plt.figure(figsize=(10, 5))

# Plot width vs. height
plt.scatter(df_stats["Width"], df_stats["Height"], alpha=0.5, label="Image Sizes")
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Distribution")
plt.legend()
plt.show()

import random

# Number of images per category to display
num_show = 3

# Plot some sample images
plt.figure(figsize=(15, 6))

for i, (category, folder) in enumerate(clean_folders.items()):
    folder_path = os.path.join(dataset_root, folder)
    if os.path.exists(folder_path):
        image_files = os.listdir(folder_path)
        random.shuffle(image_files)
        for j in range(num_show):
            img_path = os.path.join(folder_path, image_files[j])
            try:
                img = Image.open(img_path)
                plt.subplot(len(clean_folders), num_show, i * num_show + j + 1)
                plt.imshow(img)
                plt.title(f"{category}")
                plt.axis('off')
            except Exception as e:
                print(f"Could not open {img_path}: {e}")

plt.suptitle("Sample Images from Each Category", fontsize=16)
plt.tight_layout()
plt.show()

# %%
# 6. Load Dataset Using PyTorch's ImageFolder"""


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define image transformation: only ToTensor (no normalization)
transform = transforms.ToTensor()

# Load dataset using ImageFolder with simple transform
dataset = datasets.ImageFolder(root=clean_root, transform=transform)

# Split dataset into train (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Dataloaders ready (no normalization).")

# %%
# 7. Import Dependencies"""

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# 8. Define & Modify ResNet-18"""

class CustomResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(weights=None)  # Load ResNet-18 without pretrained weights

        # Modify first layer to accept 32x32 images (instead of 224x224)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove max pooling (since 32x32 is small)
        self.model.maxpool = nn.Identity()

        # Modify the final fully connected layer for 3 classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Instantiate model
model = CustomResNet(num_classes=3).to(device)

# %%
# 9. Define Loss & Optimizer"""
model_path = "models/resnet18_traffic_signs.pth"

if args.train:
    print("Training model...")
    criterion = nn.CrossEntropyLoss()  # Multi-class classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adam optimizer with weight decay

    # %%
    # 10. Train the Model"""

    num_epochs = 5  # Number of training epochs

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

    print("Training complete.")

    # %%
    #11. Evaluate the Model
    types = ['clean'] + attacks
    acc = {t: {d: 0 for d in defense_opts} for t in types}  # Accuracy counters
    count = 0  # Counter for number of processed images/batches

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # %%
    # 12. Save the Model"""

    torch.save(model.state_dict(), "models/resnet18_traffic_signs.pth")
    print("Model saved.")

else:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model from disk.")
    else:
        raise FileNotFoundError("Model not found. Use --train to train it first.")

# %%
import torch
import torch.nn.functional as F
import foolbox as fb

# Define the FGSM attack function (Fast Gradient Sign Method)
def fgsm_attack(model, image, label, epsilon):
    model.eval() 
    image = image.clone().detach().requires_grad_(True)  
    
    output = model(image)
    
    loss = F.cross_entropy(output, label)
    model.zero_grad()  

    loss.backward()
    
    data_grad = image.grad.data
    adv_image = image + epsilon * data_grad.sign()
    
    adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image.detach() 

# PGD Attack Function (Projected Gradient Descent)
def pgd_attack(model, image, label, epsilon, alpha, iterations):
    model.eval() 
    
    image_adv = image.clone().detach().requires_grad_(True)  
    original_image = image.clone().detach() 
    for _ in range(iterations):
        
        output = model(image_adv)
        
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        
        if image_adv.grad is None:
            break
        
        with torch.no_grad():
            image_adv = image_adv + alpha * image_adv.grad.sign()
            image_adv = torch.min(torch.max(image_adv, original_image - epsilon), original_image + epsilon)
            image_adv = torch.clamp(image_adv, 0, 1) 

    return image_adv.detach()  

# DeepFool Attack Function 
def deepfool_attack(model, image, label, epsilons=1e-4):  # Default epsilon value for perturbation magnitude
    model.eval() 
    
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=image.device)
    
    attack = fb.attacks.deepfool.L2DeepFoolAttack()
    
    adv_image, _, _ = attack(fmodel, image, label, epsilons=epsilons) 
    return adv_image



# %%
# PGD Attack Parameters
epsilon = 12 / 255  # Increased perturbation magnitude
alpha = 6 / 255     # Increased step size for PGD attack
iterations = 50    # Number of iterations for PGD attack

# Example: Get one image and label from your validation set
images, labels = next(iter(val_loader))
image = images[1].unsqueeze(0).clone().detach().to(device)
label = labels[1].unsqueeze(0).clone().detach().to(device)

# Compute FGSM adversarial images
adv_image_fgsm = fgsm_attack(model, image, label, epsilon)  # Ensure FGSM is computed here

# Compute PGD adversarial images
adv_image_pgd = pgd_attack(model, image, label, epsilon, alpha, iterations)  # PGD computation

# Get predictions for original image, FGSM, and PGD adversarial image
model.eval()  # Set model to evaluation mode for inference
with torch.no_grad():
    pred_clean = model(image).argmax(dim=1).item()
    pred_fgsm = model(adv_image_fgsm).argmax(dim=1).item()  # Use adv_image_fgsm
    pred_pgd = model(adv_image_pgd).argmax(dim=1).item()  # Use adv_image_pgd

# Visualize the original, FGSM, and PGD adversarial images
image_vis = image.squeeze().detach().cpu()
adv_vis_fgsm = adv_image_fgsm.squeeze().detach().cpu()
adv_vis_pgd = adv_image_pgd.squeeze().detach().cpu()

plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(image_vis.permute(1, 2, 0).numpy())
plt.title(f"Original Label: {label.item()} | Pred: {pred_clean}")
plt.axis('off')

# FGSM Adversarial Image
plt.subplot(1, 3, 2)
plt.imshow(adv_vis_fgsm.permute(1, 2, 0).numpy())
plt.title(f"FGSM Label: {label.item()} | Pred: {pred_fgsm}")
plt.axis('off')

# PGD Adversarial Image
plt.subplot(1, 3, 3)
plt.imshow(adv_vis_pgd.permute(1, 2, 0).numpy())
plt.title(f"PGD Label: {label.item()} | Pred: {pred_pgd}")
plt.axis('off')

plt.tight_layout()
plt.show()

# Visualize the difference between original and adversarial images
diff_fgsm = (adv_image_fgsm - image).squeeze().detach().cpu()
diff_pgd = (adv_image_pgd - image).squeeze().detach().cpu()

# Plot the differences
plt.figure(figsize=(12, 8))

# Difference between Original and FGSM
plt.subplot(1, 2, 1)
plt.imshow(diff_fgsm.permute(1, 2, 0).numpy())
plt.title("Difference: FGSM")
plt.axis('off')

# Difference between Original and PGD
plt.subplot(1, 2, 2)
plt.imshow(diff_pgd.permute(1, 2, 0).numpy())
plt.title("Difference: PGD")
plt.axis('off')

plt.tight_layout()
plt.show()

# %%
def bit_depth_reduction(img, bits=3):
  levels = 2 ** bits
  return torch.round(img * (levels - 1)) / (levels - 1)

def binary_filter(img, threshold=0.5):
  return torch.relu(torch.sign(img - threshold))

def jpeg_compression(img, quality=75):
    # Check if the input image is in the correct shape
    if img.ndimension() == 4:  # If the image has 4 dimensions (batch_size, channels, height, width)
        img = img[0]  # Select the first image from the batch

    # Convert the selected image to a PIL Image
    img_pil = transforms.ToPILImage()(img.cpu())
    
    # Apply JPEG compression
    buf = BytesIO()
    img_pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    comp = Image.open(buf)
    
    # Return the compressed image as a tensor
    return transforms.ToTensor()(comp).unsqueeze(0).to(device)


correct_clean, correct_fgsm, correct_pgd, correct_deepfool, correct_bit_fgsm, correct_binfilter_fgsm, correct_bit_pgd, correct_binfilter_pgd, correct_bit_deepfool, correct_binfilter_deepfool, total = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
bit_levels = [1, 2, 3, 4, 5, 6, 7]
jpeg_qualities = [50, 60, 70, 80, 90]

binary_results_fgsm = {t: 0 for t in thresholds}
bit_results_fgsm = {b: 0 for b in bit_levels}

binary_results_pgd = {t: 0 for t in thresholds}
bit_results_pgd = {b: 0 for b in bit_levels}

binary_results_deepfool = {t: 0 for t in thresholds}
bit_results_deepfool = {b: 0 for b in bit_levels}

jpeg_results_fgsm = {q: 0 for q in jpeg_qualities}
jpeg_results_pgd = {q: 0 for q in jpeg_qualities}
jpeg_results_deepfool = {q: 0 for q in jpeg_qualities}

# Iterate over the validation set
for images, labels in val_loader:
    images, labels = images.to(device), labels.to(device)

    # Clean prediction
    with torch.no_grad():
        outputs_clean = model(images)
        _, preds_clean = torch.max(outputs_clean, 1)
    
    correct_clean += (preds_clean == labels).sum().item()

    # Generate FGSM and PGD adversarial examples
    if args.attack in ["fgsm", "all"]:
        adv_images_fgsm = fgsm_attack(model, images, labels, epsilon)
        with torch.no_grad():
            outputs_fgsm = model(adv_images_fgsm)
            _, preds_fgsm = torch.max(outputs_fgsm, 1)
        correct_fgsm += (preds_fgsm == labels).sum().item()

        # Apply defenses (bit-depth reduction, binary filter, JPEG)
        if args.defense in ["bitdepth", "all"]:
            for b in bit_levels:
                reduced_fgsm_adv_images = bit_depth_reduction(adv_images_fgsm, b)
                with torch.no_grad():
                    outputs_bit = model(reduced_fgsm_adv_images)
                    _, preds_bit = torch.max(outputs_bit, 1)
                    bit_results_fgsm[b] += (preds_bit == labels).sum().item()

        if args.defense in ["binary", "all"]:
            for t in thresholds:
                filtered_fgsm_adv_images = binary_filter(adv_images_fgsm, t)
                with torch.no_grad():
                    outputs_binfilter = model(filtered_fgsm_adv_images)
                    _, preds_binfilter = torch.max(outputs_binfilter, 1)
                    binary_results_fgsm[t] += (preds_binfilter == labels).sum().item()

        # Apply JPEG compression defense
        if args.defense in ["jpeg", "all"]:
            for q in jpeg_qualities:
                jpeg_fgsm_adv_images = jpeg_compression(adv_images_fgsm, quality=q)
                with torch.no_grad():
                    outputs_jpeg = model(jpeg_fgsm_adv_images)
                    _, preds_jpeg = torch.max(outputs_jpeg, 1)
                    jpeg_results_fgsm[q] += (preds_jpeg == labels).sum().item()

    # Repeat the same for PGD and DeepFool
    if args.attack in ["pgd", "all"]:
        adv_images_pgd = pgd_attack(model, images, labels, epsilon, alpha, iterations)
        with torch.no_grad():
            outputs_pgd = model(adv_images_pgd)
            _, preds_pgd = torch.max(outputs_pgd, 1)
        correct_pgd += (preds_pgd == labels).sum().item()

        # Apply defenses (bit-depth reduction, binary filter, JPEG)
        if args.defense in ["bitdepth", "all"]:
            for b in bit_levels:
                reduced_pgd_adv_images = bit_depth_reduction(adv_images_pgd, b)
                with torch.no_grad():
                    outputs_bit = model(reduced_pgd_adv_images)
                    _, preds_bit = torch.max(outputs_bit, 1)
                    bit_results_pgd[b] += (preds_bit == labels).sum().item()

        if args.defense in ["binary", "all"]:
            for t in thresholds:
                filtered_pgd_adv_images = binary_filter(adv_images_pgd, t)
                with torch.no_grad():
                    outputs_binfilter = model(filtered_pgd_adv_images)
                    _, preds_binfilter = torch.max(outputs_binfilter, 1)
                    binary_results_pgd[t] += (preds_binfilter == labels).sum().item()

        # Apply JPEG compression defense
        if args.defense in ["jpeg", "all"]:
            for q in jpeg_qualities:
                jpeg_pgd_adv_images = jpeg_compression(adv_images_pgd, quality=q)
                with torch.no_grad():
                    outputs_jpeg = model(jpeg_pgd_adv_images)
                    _, preds_jpeg = torch.max(outputs_jpeg, 1)
                    jpeg_results_pgd[q] += (preds_jpeg == labels).sum().item()

    if args.attack in ["deepfool", "all"]:
        adv_images_deepfool = deepfool_attack(model, images, labels, epsilons=1e-1)
        with torch.no_grad():
            outputs_deepfool = model(adv_images_deepfool)
            _, preds_deepfool = torch.max(outputs_deepfool, 1)

        correct_deepfool += (preds_deepfool == labels).sum().item()

        # Apply defenses (bit-depth reduction, binary filter, JPEG)
        if args.defense in ["bitdepth", "all"]:
            for b in bit_levels:
                reduced_deepfool_adv_images = bit_depth_reduction(adv_images_deepfool, b)
                with torch.no_grad():
                    outputs_bit = model(reduced_deepfool_adv_images)
                    _, preds_bit = torch.max(outputs_bit, 1)
                    bit_results_deepfool[b] += (preds_bit == labels).sum().item()

        if args.defense in ["binary", "all"]:
            for t in thresholds:
                filtered_deepfool_adv_images = binary_filter(adv_images_deepfool, t)
                with torch.no_grad():
                    outputs_binfilter = model(filtered_deepfool_adv_images)
                    _, preds_binfilter = torch.max(outputs_binfilter, 1)
                    binary_results_deepfool[t] += (preds_binfilter == labels).sum().item()

        # Apply JPEG compression defense
        if args.defense in ["jpeg", "all"]:
            for q in jpeg_qualities:
                jpeg_deepfool_adv_images = jpeg_compression(adv_images_deepfool, quality=q)
                with torch.no_grad():
                    outputs_jpeg = model(jpeg_deepfool_adv_images)
                    _, preds_jpeg = torch.max(outputs_jpeg, 1)
                    jpeg_results_deepfool[q] += (preds_jpeg == labels).sum().item()

    total += labels.size(0)  # Accumulate total number of images processed

# Calculate accuracies for JPEG compression defense
val_acc_clean = 100 * correct_clean / total
val_acc_fgsm = 100 * correct_fgsm / total
val_acc_pgd = 100 * correct_pgd / total
val_acc_deepfool = 100 * correct_deepfool / total

print(f"Clean Accuracy on full val set: {val_acc_clean:.2f}%")

if args.attack in ["fgsm", "all"]:
    print(f"FGSM Accuracy on full val set: {val_acc_fgsm:.2f}%")
    
    # Results for bit-depth reduction defense
    if args.defense in ["bitdepth", "all"]:
        for b in bit_levels:
            bit_fgsm_acc = 100 * bit_results_fgsm[b] / total
            print(f"FGSM + Bit-Depth Reduction (bits={b}) : {bit_fgsm_acc:.2f}%")
    
    # Results for binary filter defense
    if args.defense in ["binary", "all"]:
        for t in thresholds:
            binary_fgsm_acc = 100 * binary_results_fgsm[t] / total
            print(f"FGSM + Binary Filter (threshold={t}) : {binary_fgsm_acc:.2f}%")
    
    # Results for JPEG compression defense
    if args.defense in ["jpeg", "all"]:
        for q in jpeg_qualities:
            jpeg_fgsm_acc = 100 * jpeg_results_fgsm[q] / total
            print(f"FGSM + JPEG Compression (quality={q}) : {jpeg_fgsm_acc:.2f}%")
            
if args.attack in ["pgd", "all"]:
    print(f"PGD Accuracy on full val set: {val_acc_pgd:.2f}%")
    
    # Results for bit-depth reduction defense
    if args.defense in ["bitdepth", "all"]:
        for b in bit_levels:
            bit_pgd_acc = 100 * bit_results_pgd[b] / total
            print(f"PGD + Bit-Depth Reduction (bits={b}) : {bit_pgd_acc:.2f}%")
    
    # Results for binary filter defense
    if args.defense in ["binary", "all"]:
        for t in thresholds:
            binary_pgd_acc = 100 * binary_results_pgd[t] / total
            print(f"PGD + Binary Filter (threshold={t}) : {binary_pgd_acc:.2f}%")
    
    # Results for JPEG compression defense
    if args.defense in ["jpeg", "all"]:
        for q in jpeg_qualities:
            jpeg_pgd_acc = 100 * jpeg_results_pgd[q] / total
            print(f"PGD + JPEG Compression (quality={q}) : {jpeg_pgd_acc:.2f}%")
            
if args.attack in ["deepfool", "all"]:
    print(f"DeepFool Accuracy on full val set: {val_acc_deepfool:.2f}%")
    
    # Results for bit-depth reduction defense
    if args.defense in ["bitdepth", "all"]:
        for b in bit_levels:
            bit_deepfool_acc = 100 * bit_results_deepfool[b] / total
            print(f"DeepFool + Bit-Depth Reduction (bits={b}) : {bit_deepfool_acc:.2f}%")
    
    # Results for binary filter defense
    if args.defense in ["binary", "all"]:
        for t in thresholds:
            binary_deepfool_acc = 100 * binary_results_deepfool[t] / total
            print(f"DeepFool + Binary Filter (threshold={t}) : {binary_deepfool_acc:.2f}%")
    
    # Results for JPEG compression defense
    if args.defense in ["jpeg", "all"]:
        for q in jpeg_qualities:
            jpeg_deepfool_acc = 100 * jpeg_results_deepfool[q] / total
            print(f"DeepFool + JPEG Compression (quality={q}) : {jpeg_deepfool_acc:.2f}%")

# %%
import matplotlib.pyplot as plt
def visualize_attack(image, adv_image, attack_name="DeepFool"):
    # Convert to numpy and plot images
    image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    adv_image = adv_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(adv_image)
    plt.title(f"{attack_name} Adversarial Image")
    plt.axis('off')
    
    plt.show()

# Visualize a sample clean image and its DeepFool adversarial example
image_sample = images[0].unsqueeze(0).to(device)
label_sample = labels[0].unsqueeze(0).to(device)

# Perform DeepFool attack on the sample
adv_image_deepfool = deepfool_attack(model, image_sample, label_sample, epsilons=1e-2)  # Increase epsilon here

# Visualize
visualize_attack(image_sample, adv_image_deepfool, attack_name="DeepFool")

def show_defence_example(original, fgsm, pgd, deepfool, 
                          bit_reduced_list_fgsm, binary_filtered_list_fgsm, 
                          bit_reduced_list_pgd, binary_filtered_list_pgd, 
                          bit_reduced_list_deepfool, binary_filtered_list_deepfool, 
                          jpeg_fgsm, jpeg_pgd, jpeg_deepfool, 
                          bit_levels, thresholds, jpeg_qualities, index=0):
    total_cols = 1
    if args.defense in ["bitdepth", "all"]:
        total_cols += len(bit_levels)
    if args.defense in ["binary", "all"]:
        total_cols += len(thresholds)
    if args.defense in ["jpeg", "all"]:
        total_cols += len(jpeg_qualities)

    rows = 1
    if args.attack in ["fgsm", "all"]:
        rows += 1
    if args.attack in ["pgd", "all"]:
        rows += 1
    if args.attack in ["deepfool", "all"]:  # Include DeepFool in the visualization
        rows += 1

    plt.figure(figsize=(3 * total_cols, 3 * rows))

    # Row 1: Original
    plt.subplot(rows, total_cols, total_cols // 2 + 1)
    plt.imshow(original[index].permute(1, 2, 0).cpu().numpy())
    plt.title("Original")
    plt.axis("off")

    plot_pos = total_cols + 1

    # FGSM attack visualization
    if args.attack in ["fgsm", "all"]:
        plt.subplot(rows, total_cols, plot_pos)
        plt.imshow(fgsm[index].permute(1, 2, 0).cpu().numpy())
        plt.title("FGSM Adv")
        plt.axis("off")
        plot_pos += 1

        if args.defense in ["bitdepth", "all"]:
            for i, img in enumerate(bit_reduced_list_fgsm):
                plt.subplot(rows, total_cols, plot_pos)
                plt.imshow(img[index].permute(1, 2, 0).cpu().numpy())
                plt.title(f"FGSM Bit {bit_levels[i]}")
                plt.axis("off")
                plot_pos += 1

        if args.defense in ["binary", "all"]:
            for i, img in enumerate(binary_filtered_list_fgsm):
                plt.subplot(rows, total_cols, plot_pos)
                plt.imshow(img[index].permute(1, 2, 0).cpu().numpy())
                plt.title(f"FGSM Thresh {thresholds[i]}")
                plt.axis("off")
                plot_pos += 1

        # JPEG Compression Defense
        if args.defense in ["jpeg", "all"]:
            for i, img in enumerate(jpeg_fgsm):
                plt.subplot(rows, total_cols, plot_pos)
                plt.imshow(img[index].permute(1, 2, 0).cpu().numpy())
                plt.title(f"FGSM JPEG {jpeg_qualities[i]}")
                plt.axis("off")
                plot_pos += 1

    # PGD attack visualization
    if args.attack in ["pgd", "all"]:
        plt.subplot(rows, total_cols, plot_pos)
        plt.imshow(pgd[index].permute(1, 2, 0).cpu().numpy())
        plt.title("PGD Adv")
        plt.axis("off")
        plot_pos += 1

        if args.defense in ["bitdepth", "all"]:
            for i, img in enumerate(bit_reduced_list_pgd):
                plt.subplot(rows, total_cols, plot_pos)
                plt.imshow(img[index].permute(1, 2, 0).cpu().numpy())
                plt.title(f"PGD Bit {bit_levels[i]}")
                plt.axis("off")
                plot_pos += 1

        if args.defense in ["binary", "all"]:
            for i, img in enumerate(binary_filtered_list_pgd):
                plt.subplot(rows, total_cols, plot_pos)
                plt.imshow(img[index].permute(1, 2, 0).cpu().numpy())
                plt.title(f"PGD Thresh {thresholds[i]}")
                plt.axis("off")
                plot_pos += 1

        # JPEG Compression Defense
        if args.defense in ["jpeg", "all"]:
            for i, img in enumerate(jpeg_pgd):
                plt.subplot(rows, total_cols, plot_pos)
                plt.imshow(img[index].permute(1, 2, 0).cpu().numpy())
                plt.title(f"PGD JPEG {jpeg_qualities[i]}")
                plt.axis("off")
                plot_pos += 1

    # DeepFool attack visualization
    if args.attack in ["deepfool", "all"]:
        plt.subplot(rows, total_cols, plot_pos)
        plt.imshow(deepfool[index].permute(1, 2, 0).cpu().numpy())
        plt.title("DeepFool Adv")
        plt.axis("off")
        plot_pos += 1

        if args.defense in ["bitdepth", "all"]:
            for i, img in enumerate(bit_reduced_list_deepfool):
                plt.subplot(rows, total_cols, plot_pos)
                plt.imshow(img[index].permute(1, 2, 0).cpu().numpy())
                plt.title(f"DeepFool Bit {bit_levels[i]}")
                plt.axis("off")
                plot_pos += 1

        if args.defense in ["binary", "all"]:
            for i, img in enumerate(binary_filtered_list_deepfool):
                plt.subplot(rows, total_cols, plot_pos)
                plt.imshow(img[index].permute(1, 2, 0).cpu().numpy())
                plt.title(f"DeepFool Thresh {thresholds[i]}")
                plt.axis("off")
                plot_pos += 1

        # JPEG Compression Defense
        if args.defense in ["jpeg", "all"]:
            for i, img in enumerate(jpeg_deepfool):
                plt.subplot(rows, total_cols, plot_pos)
                plt.imshow(img[index].permute(1, 2, 0).cpu().numpy())
                plt.title(f"DeepFool JPEG {jpeg_qualities[i]}")
                plt.axis("off")
                plot_pos += 1

    plt.tight_layout()
    plt.show()


# %%
bit_reduced_images_fgsm = [bit_depth_reduction(adv_images_fgsm, b) for b in bit_levels] if args.attack in ["fgsm", "all"] and args.defense in ["bitdepth", "all"] else []
binary_filtered_images_fgsm = [binary_filter(adv_images_fgsm, t) for t in thresholds] if args.attack in ["fgsm", "all"] and args.defense in ["binary", "all"] else []

bit_reduced_images_pgd = [bit_depth_reduction(adv_images_pgd, b) for b in bit_levels] if args.attack in ["pgd", "all"] and args.defense in ["bitdepth", "all"] else []
binary_filtered_images_pgd = [binary_filter(adv_images_pgd, t) for t in thresholds] if args.attack in ["pgd", "all"] and args.defense in ["binary", "all"] else []

bit_reduced_images_deepfool = [bit_depth_reduction(adv_images_deepfool, b) for b in bit_levels] if args.attack in ["deepfool", "all"] and args.defense in ["bitdepth", "all"] else []
binary_filtered_images_deepfool = [binary_filter(adv_images_deepfool, t) for t in thresholds] if args.attack in ["deepfool", "all"] and args.defense in ["binary", "all"] else []

jpeg_fgsm_images = [jpeg_compression(adv_images_fgsm, quality=q) for q in jpeg_qualities] if args.attack in ["fgsm", "all"] and args.defense in ["jpeg", "all"] else []
jpeg_pgd_images = [jpeg_compression(adv_images_pgd, quality=q) for q in jpeg_qualities] if args.attack in ["pgd", "all"] and args.defense in ["jpeg", "all"] else []
jpeg_deepfool_images = [jpeg_compression(adv_images_deepfool, quality=q) for q in jpeg_qualities] if args.attack in ["deepfool", "all"] and args.defense in ["jpeg", "all"] else []

show_defence_example(
    images,
    adv_images_fgsm if args.attack in ["fgsm", "all"] else None,
    adv_images_pgd if args.attack in ["pgd", "all"] else None,
    adv_images_deepfool if args.attack in ["deepfool", "all"] else None,
    bit_reduced_images_fgsm,
    binary_filtered_images_fgsm,
    bit_reduced_images_pgd,
    binary_filtered_images_pgd,
    bit_reduced_images_deepfool,
    binary_filtered_images_deepfool,
    jpeg_fgsm_images,
    jpeg_pgd_images,
    jpeg_deepfool_images,
    bit_levels,
    thresholds,
    jpeg_qualities
)

# %%
print(f"Clean Accuracy on full val set: {val_acc_clean:.2f}%")

if args.attack in ["fgsm", "all"]:
    print(f"FGSM Accuracy on full val set: {val_acc_fgsm:.2f}%")
    
    # Best Bit-Depth Reduction
    if args.defense in ["bitdepth", "all"]:
        best_bit, best_bit_acc = None, 0
        for b in bit_levels:
            acc = 100 * bit_results_fgsm[b] / total
            if acc > best_bit_acc:
                best_bit_acc = acc
                best_bit = b
        print(f"FGSM + Best Bit-Depth Reduction (bits={best_bit}) : {best_bit_acc:.2f}%")

    # Best Binary Filter
    if args.defense in ["binary", "all"]:
        best_thresh, best_thresh_acc = None, 0
        for t in thresholds:
            acc = 100 * binary_results_fgsm[t] / total
            if acc > best_thresh_acc:
                best_thresh_acc = acc
                best_thresh = t
        print(f"FGSM + Best Binary Filter (threshold={best_thresh}) : {best_thresh_acc:.2f}%")

    # Best JPEG Compression
    if args.defense in ["jpeg", "all"]:
        best_jpeg, best_jpeg_acc = None, 0
        for q in jpeg_qualities:
            acc = 100 * jpeg_results_fgsm[q] / total
            if acc > best_jpeg_acc:
                best_jpeg_acc = acc
                best_jpeg = q
        print(f"FGSM + Best JPEG Compression (quality={best_jpeg}) : {best_jpeg_acc:.2f}%")
        
if args.attack in ["pgd", "all"]:
    print(f"PGD Accuracy on full val set: {val_acc_pgd:.2f}%")
    
    # Best Bit-Depth Reduction
    if args.defense in ["bitdepth", "all"]:
        best_bit, best_bit_acc = None, 0
        for b in bit_levels:
            acc = 100 * bit_results_pgd[b] / total
            if acc > best_bit_acc:
                best_bit_acc = acc
                best_bit = b
        print(f"PGD + Best Bit-Depth Reduction (bits={best_bit}) : {best_bit_acc:.2f}%")

    # Best Binary Filter
    if args.defense in ["binary", "all"]:
        best_thresh, best_thresh_acc = None, 0
        for t in thresholds:
            acc = 100 * binary_results_pgd[t] / total
            if acc > best_thresh_acc:
                best_thresh_acc = acc
                best_thresh = t
        print(f"PGD + Best Binary Filter (threshold={best_thresh}) : {best_thresh_acc:.2f}%")

    # Best JPEG Compression
    if args.defense in ["jpeg", "all"]:
        best_jpeg, best_jpeg_acc = None, 0
        for q in jpeg_qualities:
            acc = 100 * jpeg_results_pgd[q] / total
            if acc > best_jpeg_acc:
                best_jpeg_acc = acc
                best_jpeg = q
        print(f"PGD + Best JPEG Compression (quality={best_jpeg}) : {best_jpeg_acc:.2f}%")
        
if args.attack in ["deepfool", "all"]:
    print(f"DeepFool Accuracy on full val set: {val_acc_deepfool:.2f}%")
    
    # Best Bit-Depth Reduction
    if args.defense in ["bitdepth", "all"]:
        best_bit, best_bit_acc = None, 0
        for b in bit_levels:
            acc = 100 * bit_results_deepfool[b] / total
            if acc > best_bit_acc:
                best_bit_acc = acc
                best_bit = b
        print(f"DeepFool + Best Bit-Depth Reduction (bits={best_bit}) : {best_bit_acc:.2f}%")

    # Best Binary Filter
    if args.defense in ["binary", "all"]:
        best_thresh, best_thresh_acc = None, 0
        for t in thresholds:
            acc = 100 * binary_results_deepfool[t] / total
            if acc > best_thresh_acc:
                best_thresh_acc = acc
                best_thresh = t
        print(f"DeepFool + Best Binary Filter (threshold={best_thresh}) : {best_thresh_acc:.2f}%")

    # Best JPEG Compression
    if args.defense in ["jpeg", "all"]:
        best_jpeg, best_jpeg_acc = None, 0
        for q in jpeg_qualities:
            acc = 100 * jpeg_results_deepfool[q] / total
            if acc > best_jpeg_acc:
                best_jpeg_acc = acc
                best_jpeg = q
        print(f"DeepFool + Best JPEG Compression (quality={best_jpeg}) : {best_jpeg_acc:.2f}%")


# %%
