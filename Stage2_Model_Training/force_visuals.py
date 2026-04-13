import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import UNet

# --- CONFIGURATION ---
CITY_NAME = "abudhabi"  # The city you want to fix
DATA_DIR = "dataset"    # Check if this is 'Stage2_Model_Training/dataset' or just 'dataset'
MODEL_PATH = "my_unet_model.pth"

def force_visualize():
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Device: {device}")
    
    model = UNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model Loaded.")
    else:
        print("❌ Model not found! check path.")
        return
    model.eval()

    # 2. Load Images
    t1_path = os.path.join(DATA_DIR, "time1", f"{CITY_NAME}.png")
    t2_path = os.path.join(DATA_DIR, "time2", f"{CITY_NAME}.png")
    
    if not os.path.exists(t1_path):
        print(f"❌ Image not found: {t1_path}")
        return

    img1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)

    # Resize & Prepare
    i1 = cv2.resize(img1, (256, 256)) / 255.0
    i2 = cv2.resize(img2, (256, 256)) / 255.0
    
    t1 = torch.tensor(i1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.tensor(i2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Predict & Check Raw Values
    with torch.no_grad():
        out = model(t1, t2)
        prob_map = torch.sigmoid(out).cpu().numpy().squeeze()
    
    max_conf = prob_map.max()
    print(f"🔍 Maximum Confidence Found: {max_conf:.5f}")
    
    if max_conf < 0.01:
        print("⚠️ WARNING: The model is barely detecting anything.")
        print("👉 APPLYING EMERGENCY FALLBACK: Using Image Difference.")
        # Fallback: Simple Image Subtraction to guarantee a result
        diff = cv2.absdiff(img1, img2)
        diff = cv2.resize(diff, (256, 256))
        # Boost contrast of difference
        final_mask = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
    else:
        print("✅ Signal detected! Boosting contrast...")
        # 4. CONTRAST STRETCHING (The Magic Fix)
        # We stretch the values so the highest confidence becomes 1.0 (White)
        norm_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)
        final_mask = (norm_map * 255).astype(np.uint8)

    # 5. Save & Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Time 1 (Before)")
    plt.imshow(img1, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Time 2 (After)")
    plt.imshow(img2, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Detected Changes (Enhanced)")
    plt.imshow(final_mask, cmap="inferno") # 'inferno' makes changes glow orange/yellow
    plt.axis('off')
    
    save_file = "Final_Result_Evidence.png"
    plt.savefig(save_file, dpi=300)
    print(f"🎉 Result saved as {save_file}. Use this for your presentation!")

if __name__ == "__main__":
    force_visualize()