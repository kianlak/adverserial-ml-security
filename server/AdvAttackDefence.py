#!/usr/bin/env python3
"""
Script: adv_attack_defense.py
Description: Train or load a ResNet-18 on LisaCNN with adversarial attacks (FGSM, PGD, DeepFool)
and defenses (bit-depth reduction, binary filter, JPEG compression). Prints comparative results.
Includes visualization of attack/defense effects for selected samples.
"""
import argparse, subprocess, sys, os, zipfile, shutil, time
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Foolbox for DeepFool
try:
    import foolbox as fb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "foolbox"])
    import foolbox as fb


def parse_args():
    parser = argparse.ArgumentParser(description="Train or Load Model with Attack and Defense Options")
    parser.add_argument("--train", action="store_true", help="Train the model from scratch")
    parser.add_argument("--attack", choices=["fgsm", "pgd", "deepfool", "all"], default="all",
                        help="Select attack type")
    parser.add_argument("--defense", choices=["none", "bitdepth", "binary", "jpeg", "all"], default="all",
                        help="Select defense type")
    parser.add_argument("--data-zip", type=str, default="../LisaCnn.zip",
                        help="Path to dataset zip")
    parser.add_argument("--data-dir", type=str, default="../LisaCnn",
                        help="Directory to extract dataset")
    parser.add_argument("--model-path", type=str, default="models/resnet18_traffic_signs.pth",
                        help="Path to save/load model")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()


args = parse_args()
# Set device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Extract dataset
if not os.path.exists(args.data_dir):
    print("Extracting dataset...")
    with zipfile.ZipFile(args.data_zip, 'r') as zip_ref:
        zip_ref.extractall(args.data_dir)
    print("Extraction completed.")
else:
    print("Dataset already extracted.")

# 2. Organize dataset
tidy_clean = os.path.join(args.data_dir, "LisaCnn_Clean")
tidy_adv = os.path.join(args.data_dir, "LisaCnn_Adv")
original_root = os.path.join(args.data_dir, "LisaCnn")  # Adjust if needed
categories = {
    "Speed35Signs": ("Speed35Signs/Clean", "Speed35Signs/Adv"),
    "StopSigns": ("StopSigns/CleanStop", "StopSigns/LisaAdvStop/AdvStop"),
    "YieldSigns": ("YieldSigns/CleanYield", "YieldSigns/AdvYield")
}
os.makedirs(tidy_clean, exist_ok=True)
os.makedirs(tidy_adv, exist_ok=True)
for cat, (clean_f, adv_f) in categories.items():
    src_c = os.path.join(original_root, clean_f)
    src_a = os.path.join(original_root, adv_f)
    tgt_c = os.path.join(tidy_clean, cat)
    tgt_a = os.path.join(tidy_adv, cat)
    os.makedirs(tgt_c, exist_ok=True)
    os.makedirs(tgt_a, exist_ok=True)
    for f in os.listdir(src_c): shutil.copy(os.path.join(src_c, f), tgt_c)
    for f in os.listdir(src_a): shutil.copy(os.path.join(src_a, f), tgt_a)
print("Dataset structured.")

# 3. Load Dataset
transform = transforms.ToTensor()
dataset = datasets.ImageFolder(root=tidy_clean, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
print("DataLoaders ready.")

# 4. Define model
class CustomResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x): return self.model(x)

model = CustomResNet().to(device)

# 5. Train or load
if args.train:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(5):
        model.train()
        total_l, corr, tot = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()
            total_l += loss.item()
            preds = outs.argmax(1)
            corr += (preds==lbls).sum().item()
            tot += lbls.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_l/len(train_loader):.4f}, Acc: {100*corr/tot:.2f}%")
    torch.save(model.state_dict(), args.model_path)
    print("Model saved.")
else:
    assert os.path.exists(args.model_path), "Model not found, use --train."
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded.")

# 6. Attack functions
def fgsm_attack(model, image, label, eps):
    model.eval()
    img = image.clone().detach().requires_grad_(True).to(device)
    lbl = label.to(device)
    out = model(img)
    loss = F.cross_entropy(out, lbl)
    model.zero_grad(); loss.backward()
    grad = img.grad.data
    return torch.clamp(img + eps * grad.sign(), 0, 1).detach()

def pgd_attack(model, image, label, eps, alpha, iters):
    model.eval()
    img_adv = image.clone().detach().requires_grad_(True).to(device)
    orig = image.clone().detach().to(device)
    for _ in range(iters):
        out = model(img_adv)
        loss = F.cross_entropy(out, label.to(device))
        model.zero_grad(); loss.backward()
        with torch.no_grad():
            img_adv = img_adv + alpha * img_adv.grad.sign()
            img_adv = torch.max(torch.min(img_adv, orig+eps), orig-eps)
            img_adv = torch.clamp(img_adv, 0,1)
        img_adv.requires_grad_(True)
    return img_adv.detach()

def deepfool_attack(model, image, label):
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=image.device)
    attack = fb.attacks.deepfool.L2DeepFoolAttack()
    _, clipped, _ = attack(fmodel, image, label, epsilons=None)  # <--- fix
    return clipped


# 7. Defenses
def bit_depth_reduction(img, bits=3):
    levels = 2**bits
    return torch.round(img*(levels-1))/(levels-1)

def binary_filter(img, threshold=0.5):
    return torch.relu(torch.sign(img - threshold))

def jpeg_compression(img, quality=75):
    img_pil = transforms.ToPILImage()(img.squeeze().cpu())
    buf = BytesIO()
    img_pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    comp = Image.open(buf)
    return transforms.ToTensor()(comp).unsqueeze(0).to(device)

# 8. Evaluation Setup
eps, alpha, iters = 12/255, 6/255, 50
attacks = []
defense_opts = []
if args.attack in ['all']:
    attacks = ['fgsm','pgd','deepfool']
else:
    attacks = [args.attack]
if args.defense in ['all']:
    defense_opts = ['vanilla','bitdepth','binary','jpeg']
elif args.defense == 'none':
    defense_opts = ['vanilla']
else:
    defense_opts = [args.defense]

# Initialize counters
types = ['clean'] + attacks
acc = {t: {d:0 for d in defense_opts} for t in types}
count=0

for imgs, lbls in val_loader:
    for i in range(imgs.size(0)):
        img = imgs[i].unsqueeze(0).to(device)
        lbl = lbls[i].unsqueeze(0).to(device)
        # Clean
        out = model(img)
        pred = out.argmax(1)
        for d in defense_opts:
            if d=='vanilla':
                acc['clean'][d] += (pred==lbl).item()
            elif d=='bitdepth':
                acc['clean'][d] += (model(bit_depth_reduction(img)).argmax(1)==lbl).item()
            elif d=='binary':
                acc['clean'][d] += (model(binary_filter(img)).argmax(1)==lbl).item()
            elif d=='jpeg':
                acc['clean'][d] += (model(jpeg_compression(img)).argmax(1)==lbl).item()
        # Adversarial
        for atk in attacks:
            if atk=='fgsm': adv = fgsm_attack(model,img,lbl,eps)
            if atk=='pgd': adv = pgd_attack(model,img,lbl,eps,alpha,iters)
            if atk=='deepfool': adv = deepfool_attack(model,img,lbl)
            for d in defense_opts:
                if d=='vanilla':
                    acc[atk][d] += (model(adv).argmax(1)==lbl).item()
                elif d=='bitdepth':
                    acc[atk][d] += (model(bit_depth_reduction(adv)).argmax(1)==lbl).item()
                elif d=='binary':
                    acc[atk][d] += (model(binary_filter(adv)).argmax(1)==lbl).item()
                elif d=='jpeg':
                    acc[atk][d] += (model(jpeg_compression(adv)).argmax(1)==lbl).item()
        count+=1

# Compute and print results
print(f"{'Attack':<12}{''.join([f'{d.capitalize():>10}' for d in defense_opts])}")
for t in types:
    row = f"{t:<12}"
    for d in defense_opts:
        row += f"{(acc[t][d]/count*100):>10.2f}"
    print(row)
    
  
# 9. Visualize results  
def show_defence_example(original, fgsm, pgd, bit_reduced_list_fgsm, binary_filtered_list_fgsm, bit_reduced_list_pgd, binary_filtered_list_pgd, bit_levels, thresholds, index=0):
    total_cols = 1
    if args.defense in ["bitdepth", "all"]:
        total_cols += len(bit_levels)
    if args.defense in ["binary", "all"]:
        total_cols += len(thresholds)

    rows = 1
    if args.attack in ["fgsm", "all"]:
        rows += 1
    if args.attack in ["pgd", "all"]:
        rows += 1

    plt.figure(figsize=(3 * total_cols, 3 * rows))

    # Row 1: Original
    plt.subplot(rows, total_cols, total_cols // 2 + 1)
    plt.imshow(original[index].permute(1, 2, 0).cpu().numpy())
    plt.title("Original")
    plt.axis("off")

    plot_pos = total_cols + 1

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

    plt.tight_layout()
    plt.show()

# Auto-execute visualization after evaluation
if args.attack in ["fgsm", "all"] or args.attack in ["pgd", "all"]:
    try:
        bit_reduced_images_fgsm = [bit_depth_reduction(adv_images_fgsm, b) for b in bit_levels] if 'fgsm' in attacks else []
        binary_filtered_images_fgsm = [binary_filter(adv_images_fgsm, t) for t in thresholds] if 'fgsm' in attacks else []
        bit_reduced_images_pgd = [bit_depth_reduction(adv_images_pgd, b) for b in bit_levels] if 'pgd' in attacks else []
        binary_filtered_images_pgd = [binary_filter(adv_images_pgd, t) for t in thresholds] if 'pgd' in attacks else []
        show_defence_example(
            images,
            adv_images_fgsm if 'fgsm' in attacks else None,
            adv_images_pgd if 'pgd' in attacks else None,
            bit_reduced_images_fgsm,
            binary_filtered_images_fgsm,
            bit_reduced_images_pgd,
            binary_filtered_images_pgd,
            bit_levels,
            thresholds
        )
    except Exception as e:
        print("Visualization skipped due to error:", e)
