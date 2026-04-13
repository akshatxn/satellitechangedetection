import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNet

# --- CONFIGURATION ---
DATA_DIR = "dataset"            # Folder containing time1, time2
OUTPUT_DIR = "Final_Presentation_Results" # Where to save the images
MODEL_PATH = "my_unet_model.pth"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_all():
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Processing on device: {device}")
    
    model = UNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("❌ Model not found! Train it first.")
        return
    model.eval()

    # 2. Get List of Cities
    cities = [f for f in os.listdir(os.path.join(DATA_DIR, "time1")) if f.endswith('.png')]
    print(f"🚀 Found {len(cities)} cities. Generating results...")

    # 3. Loop Through Every City
    for city_file in tqdm(cities):
        city_name = city_file.split('.')[0]
        
        # Load Images
        t1_path = os.path.join(DATA_DIR, "time1", city_file)
        t2_path = os.path.join(DATA_DIR, "time2", city_file)
        
        img1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize & Normalize
        i1 = cv2.resize(img1, (256, 256)) / 255.0
        i2 = cv2.resize(img2, (256, 256)) / 255.0
        
        t1 = torch.tensor(i1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        t2 = torch.tensor(i2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            out = model(t1, t2)
            prob_map = torch.sigmoid(out).cpu().numpy().squeeze()

        # --- SMART VISUALIZATION LOGIC ---
        max_conf = prob_map.max()
        
        # Scenario A: AI is confident (>10%) -> Use Standard Black/White
        if max_conf > 0.1:
            visual_mode = "Binary Mask (Standard)"
            # Create a clean binary mask
            mask = (prob_map > 0.1).astype(np.float32)
            display_map = mask # Black and White
            cmap = "gray"
            
        # Scenario B: AI is unsure (<10%) -> Use Heatmap (The Orange Fix)
        else:
            visual_mode = "Enhanced Heatmap (Low Confidence)"
            # Normalize to stretch values to 0-1 range
            norm_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)
            display_map = norm_map
            cmap = "inferno" # This makes it Orange/Purple

        # --- SAVE THE IMAGE ---
        plt.figure(figsize=(12, 4))
        
        # Time 1
        plt.subplot(1, 3, 1)
        plt.title(f"{city_name} - 2015", fontsize=10)
        plt.imshow(img1, cmap="gray")
        plt.axis('off')

        # Time 2
        plt.subplot(1, 3, 2)
        plt.title(f"{city_name} - 2018", fontsize=10)
        plt.imshow(img2, cmap="gray")
        plt.axis('off')

        # Result
        plt.subplot(1, 3, 3)
        plt.title(f"Detected Changes\n({visual_mode})", fontsize=10)
        plt.imshow(display_map, cmap=cmap)
        plt.axis('off')

        # Save to folder
        save_path = os.path.join(OUTPUT_DIR, f"Result_{city_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close() # Close memory to prevent crashing

    print(f"\n✅ DONE! All images are saved in the folder: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    process_all()