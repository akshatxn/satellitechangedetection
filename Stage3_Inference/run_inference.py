import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from model import UNet

# Config
MODEL_PATH = "my_unet_model.pth"
# We read images directly from Stage 2 dataset to test
DATA_DIR = "../Stage2_Model_Training/dataset" 
OUTPUT_DIR = "final_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def predict_change(city_name):
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("❌ Error: my_unet_model.pth not found! Copy it from Stage 2.")
        return

    model.eval() # Set to evaluation mode

    # 2. Load Images
    t1_path = os.path.join(DATA_DIR, "time1", f"{city_name}.png")
    t2_path = os.path.join(DATA_DIR, "time2", f"{city_name}.png")
    
    if not os.path.exists(t1_path):
        print(f"❌ Could not find images for {city_name}. Check your spelling or folder.")
        return

    img1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess (Resize & Normalize)
    img1_in = cv2.resize(img1, (256, 256)) / 255.0
    img2_in = cv2.resize(img2, (256, 256)) / 255.0

    # Convert to Tensor
    t1_tensor = torch.tensor(img1_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t2_tensor = torch.tensor(img2_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Predict!
    with torch.no_grad():
        output = model(t1_tensor, t2_tensor)
        output = torch.sigmoid(output) # Convert to probability (0-1)
        output = output.squeeze().cpu().numpy()

    # Threshold (if > 0.5, it's a change)
    mask_pred = (output > 0.5).astype(np.uint8) * 255

    # 4. Save & Show Results
    print(f"✅ Generated prediction for {city_name}")
    
    # Create a nice side-by-side plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title(f"{city_name} - Time 1 (Before)")
    plt.imshow(img1_in, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f"{city_name} - Time 2 (After)")
    plt.imshow(img2_in, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Detected Changes (AI Output)")
    plt.imshow(mask_pred, cmap='gray')
    plt.axis('off')

    save_path = os.path.join(OUTPUT_DIR, f"{city_name}_result.png")
    plt.savefig(save_path)
    print(f"   Saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # You can change this to any city you processed (e.g., 'abudhabi', 'pisa', 'mumbai')
    city = input("Enter city name to test (e.g., abudhabi): ").strip()
    predict_change(city)