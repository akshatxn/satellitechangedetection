import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import UNet

# --- CONFIGURATION ---
DATA_DIR = "dataset"  # Path to your dataset folder
MODEL_PATH = "my_unet_model.pth"
OUTPUT_FILE = "Urbanization_Ranking.png"

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    # Load weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print("❌ Error: Model file not found! Train the model first.")
        exit()
    model.eval()
    return model, device

def calculate_urbanization():
    model, device = load_model()
    
    cities = [f.split('.')[0] for f in os.listdir(os.path.join(DATA_DIR, "time1")) if f.endswith('.png')]
    results = []

    print(f"🔍 Scanning {len(cities)} cities for urbanization changes...")

    for city in cities:
        # Load images
        t1_path = os.path.join(DATA_DIR, "time1", f"{city}.png")
        t2_path = os.path.join(DATA_DIR, "time2", f"{city}.png")
        
        img1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to 256x256 for the model
        i1 = cv2.resize(img1, (256, 256)) / 255.0
        i2 = cv2.resize(img2, (256, 256)) / 255.0
        
        # Convert to Tensor
        t1 = torch.tensor(i1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        t2 = torch.tensor(i2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            out = model(t1, t2)
            # Use a low threshold to ensure we capture changes for the graph
            pred = (torch.sigmoid(out) > 0.2).float().cpu().numpy().squeeze()
            
        # --- CALCULATION PARAMETERS ---
        # Growth Score = (New Construction Pixels / Total Pixels) * 100
        total_pixels = 256 * 256
        change_pixels = np.sum(pred == 1.0)
        growth_percentage = (change_pixels / total_pixels) * 100
        
        results.append({"City": city, "Growth (%)": growth_percentage})

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(by="Growth (%)", ascending=False)
    
    return df

def plot_graphs(df):
    # Set the visual style
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Create Bar Plot
    barplot = sns.barplot(x="Growth (%)", y="City", data=df, palette="magma")
    
    plt.title("Urbanization Growth Rate by City (2015-2018)", fontsize=16, fontweight='bold')
    plt.xlabel("Percentage of New Construction (%)", fontsize=12)
    plt.ylabel("City / Region", fontsize=12)
    
    # Add values to the end of bars
    for index, value in enumerate(df["Growth (%)"]):
        plt.text(value + 0.1, index, f"{value:.2f}%", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"📊 Graph saved as '{OUTPUT_FILE}'")
    print("\n--- LEADERBOARD ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    df_results = calculate_urbanization()
    plot_graphs(df_results)