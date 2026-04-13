import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNet

# --- CONFIGURATION ---
DATA_DIR = "dataset"
OUTPUT_DIR = "Final_Uniform_Results"  # New folder for consistent images
MODEL_PATH = "my_unet_model.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_uniform_visuals():
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Generating Uniform Heatmaps on: {device}")
    
    model = UNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("❌ Model not found!")
        return
    model.eval()

    # 2. Process Every City
    cities = [f for f in os.listdir(os.path.join(DATA_DIR, "time1")) if f.endswith('.png')]
    
    for city_file in tqdm(cities):
        city_name = city_file.split('.')[0]
        
        # Load Images
        t1_path = os.path.join(DATA_DIR, "time1", city_file)
        t2_path = os.path.join(DATA_DIR, "time2", city_file)
        
        img1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
        
        i1 = cv2.resize(img1, (256, 256)) / 255.0
        i2 = cv2.resize(img2, (256, 256)) / 255.0
        
        t1 = torch.tensor(i1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        t2 = torch.tensor(i2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # 3. Predict & Normalize
        with torch.no_grad():
            out = model(t1, t2)
            prob_map = torch.sigmoid(out).cpu().numpy().squeeze()
            
        # FORCE HEATMAP: Normalize 0 to 1 regardless of confidence
        # This ensures even faint signals in Paris look the same style as strong signals in Mumbai
        norm_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)
        
        # 4. Save Image
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.title(f"{city_name} - 2015", fontsize=10)
        plt.imshow(img1, cmap="gray")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"{city_name} - 2018", fontsize=10)
        plt.imshow(img2, cmap="gray")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Urbanization Probability", fontsize=10)
        # Use 'inferno' (Black->Orange->Yellow) for everything
        plt.imshow(norm_map, cmap="inferno") 
        plt.axis('off')

        plt.savefig(os.path.join(OUTPUT_DIR, f"{city_name}_Result.png"), bbox_inches='tight', dpi=150)
        plt.close()

    print(f"✅ All images saved to folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    make_uniform_visuals()