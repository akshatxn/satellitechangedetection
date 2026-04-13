import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import UNet

# --- CONFIGURATION ---
DATA_DIR = "dataset"
MODEL_PATH = "my_unet_model.pth"
OUTPUT_FILENAME = "Urbanization_Leaderboard.png"

# --- 1. LOAD MODEL ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Model loaded successfully on {device}")
else:
    print("❌ Model not found! Please check the file path.")
    exit()
model.eval()

# --- 2. CALCULATE SCORES ---
results = []
cities = [f.split('.')[0] for f in os.listdir(os.path.join(DATA_DIR, "time1")) if f.endswith('.png')]

print(f"🚀 Scanning {len(cities)} cities for urbanization...")

for city in tqdm(cities):
    # Load Images
    t1_path = os.path.join(DATA_DIR, "time1", f"{city}.png")
    t2_path = os.path.join(DATA_DIR, "time2", f"{city}.png")
    
    # Read & Preprocess
    img1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 256x256 (Model Requirement)
    i1 = cv2.resize(img1, (256, 256)) / 255.0
    i2 = cv2.resize(img2, (256, 256)) / 255.0
    
    # Convert to Tensor
    t1 = torch.tensor(i1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.tensor(i2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Inference (Get Probability Map)
    with torch.no_grad():
        out = model(t1, t2)
        prob_map = torch.sigmoid(out).cpu().numpy().squeeze()
        
    # --- SCORING LOGIC ---
    # We count a pixel as "New Construction" if model confidence > 20%
    # Threshold 0.2 captures even early stages of construction
    construction_pixels = np.sum(prob_map > 0.2)
    total_pixels = 256 * 256
    
    # Urbanization Score = (New Pixels / Total Pixels) * 100
    score = (construction_pixels / total_pixels) * 100
    results.append({"City": city, "Score": score})

# --- 3. GENERATE GRAPH ---
df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create Horizontal Bar Chart
ax = sns.barplot(x="Score", y="City", data=df, palette="magma")

# Add Title & Labels
plt.title("Urbanization Growth Rate by City (2015 - 2018)", fontsize=14, fontweight='bold')
plt.xlabel("New Construction (% of Land Area)", fontsize=12)
plt.ylabel("Region", fontsize=12)

# ADD SCORES TO BARS (The crucial part)
for i, v in enumerate(df["Score"]):
    ax.text(v + 0.1, i, f"{v:.2f}%", color='black', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_FILENAME, dpi=300)
print(f"🎉 Graph saved as '{OUTPUT_FILENAME}'. Open it to see the rankings!")