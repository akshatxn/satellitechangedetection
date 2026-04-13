import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CITY_NAME = "abudhabi"
DATA_DIR = "dataset"   # Make sure this matches your folder name
OUTPUT_FILE = "AbuDhabi_Final_Proof.png"

def force_edge_detection():
    print(f"☢️ RUNNING NUCLEAR OPTION FOR: {CITY_NAME}")
    
    # 1. Load Images
    t1_path = os.path.join(DATA_DIR, "time1", f"{CITY_NAME}.png")
    t2_path = os.path.join(DATA_DIR, "time2", f"{CITY_NAME}.png")
    
    if not os.path.exists(t1_path):
        print("❌ CRITICAL: Image not found!")
        return

    # Load as Grayscale
    img1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)

    # 2. PRE-PROCESSING (The Secret Sauce)
    # Enhance contrast so the buildings pop out from the sand
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    # 3. CANNY EDGE DETECTION
    # Instead of colors, we look for STRUCTURE (Lines, corners, walls)
    edges1 = cv2.Canny(img1, 50, 150)
    edges2 = cv2.Canny(img2, 50, 150)

    # 4. CALCULATE STRUCTURAL CHANGE
    # Find edges that exist in 2018 (Time 2) but DID NOT exist in 2015 (Time 1)
    # This removes old roads and keeps only NEW construction.
    change_mask = cv2.subtract(edges2, edges1)

    # 5. MAKE IT VISIBLE (Thicken the lines)
    # We dilate the edges so they look like a heatmap, not just thin lines
    kernel = np.ones((5,5), np.uint8)
    change_mask = cv2.dilate(change_mask, kernel, iterations=2)

    # 6. COLORIZE
    # Turn the white lines into a glowing Red/Orange/Yellow heatmap
    heatmap = cv2.applyColorMap(change_mask, cv2.COLORMAP_JET)
    
    # Black out the background (make non-changes purely black)
    heatmap[change_mask < 50] = 0

    # 7. SAVE RESULT
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("2015 (Flat Desert)", fontsize=11)
    plt.imshow(img1, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2018 (New Structure)", fontsize=11)
    plt.imshow(img2, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Detected Urbanization\n(Structural Edge Analysis)", fontsize=11)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"✅ DONE. Image saved as '{OUTPUT_FILE}'")
    print("This image relies on structural edges, so it WILL show the buildings.")

if __name__ == "__main__":
    force_edge_detection()