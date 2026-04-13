import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
from scipy.ndimage import center_of_mass

# ==========================================
# 1. YOUR EXACT FOLDER PATHS
# ==========================================
DIR_IMAGES = r"C:\Users\wwwaa\OneDrive\Desktop\SatelliteChangeDetection\Onera Satellite Change Detection dataset - Images"
DIR_TRAIN_LABELS = r"C:\Users\wwwaa\OneDrive\Desktop\SatelliteChangeDetection\Onera Satellite Change Detection dataset - Train Labels"
DIR_TEST_LABELS = r"C:\Users\wwwaa\OneDrive\Desktop\SatelliteChangeDetection\Onera Satellite Change Detection dataset - Test Labels"
DIR_PREDICTIONS = r"C:\Users\wwwaa\OneDrive\Desktop\SatelliteChangeDetection\My_AI_Predictions" 

# The 3 cities for the IEEE paper
cities = ["dubai", "montpellier", "abudhabi"]

columns = ["T1 (Baseline)", "T2 (Follow-up)", "Ground Truth", "EF-FCN (Bad Baseline)", "RGB U-Net (Good Baseline)", "Proposed (Ours)"]
rows = ["Arid Biome\n(Dubai)", "Temperate Biome\n(Montpellier)", "Dense Urban\n(Abu Dhabi)"]
CROP_SIZE = 256 

# ==========================================
# 2. OSCD SPECIFIC HELPER FUNCTIONS
# ==========================================
def find_label_path(city):
    train_path = os.path.join(DIR_TRAIN_LABELS, city, "cm", "cm.png")
    test_path = os.path.join(DIR_TEST_LABELS, city, "cm", "cm.png")
    if os.path.exists(train_path): return train_path
    if os.path.exists(test_path): return test_path
    return None

def find_best_crop_center(mask):
    if mask is None or np.sum(mask) == 0:
        return 250, 250
    cy, cx = center_of_mass(mask > 0)
    return int(cy), int(cx)

def load_oscd_rgb(city_path, time_folder, cy, cx):
    base_folder = os.path.join(city_path, time_folder)
    if not os.path.exists(base_folder):
        base_folder = os.path.join(city_path, f"{time_folder}_rect")
        
    b4_path = os.path.join(base_folder, "B04.tif")
    b3_path = os.path.join(base_folder, "B03.tif")
    b2_path = os.path.join(base_folder, "B02.tif")
    
    if not (os.path.exists(b4_path) and os.path.exists(b3_path) and os.path.exists(b2_path)):
        return np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        
    r = tiff.imread(b4_path)
    g = tiff.imread(b3_path)
    b = tiff.imread(b2_path)
    
    half = CROP_SIZE // 2
    h, w = r.shape
    startY = max(0, min(cy - half, h - CROP_SIZE))
    startX = max(0, min(cx - half, w - CROP_SIZE))
    
    r_crop = r[startY:startY+CROP_SIZE, startX:startX+CROP_SIZE]
    g_crop = g[startY:startY+CROP_SIZE, startX:startX+CROP_SIZE]
    b_crop = b[startY:startY+CROP_SIZE, startX:startX+CROP_SIZE]
    
    rgb = np.dstack((r_crop, g_crop, b_crop))
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip(rgb, p2, p98)
    rgb = (rgb - p2) / (p98 - p2 + 1e-8) * 255.0
    return rgb.astype(np.uint8)

def load_mask_crop(filepath, cy, cx):
    if not filepath or not os.path.exists(filepath):
        return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    half = CROP_SIZE // 2
    h, w = img.shape
    startY = max(0, min(cy - half, h - CROP_SIZE))
    startX = max(0, min(cx - half, w - CROP_SIZE))
    
    crop = img[startY:startY+CROP_SIZE, startX:startX+CROP_SIZE]
    return (crop > 0).astype(np.uint8) * 255

# ==========================================
# 3. MAIN LOOP: ASSEMBLE THE GRID
# ==========================================
fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(20, 10))

for i, city in enumerate(cities):
    print(f"Processing city: {city}...")
    
    city_img_path = os.path.join(DIR_IMAGES, city)
    gt_path = find_label_path(city)
    
    ef_path   = os.path.join(DIR_PREDICTIONS, f"{city}_effcn_pred.png")
    unet_path = os.path.join(DIR_PREDICTIONS, f"{city}_unet_pred.png")
    ours_path = os.path.join(DIR_PREDICTIONS, f"{city}_ours_pred.png")
    
    if gt_path:
        full_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        cy, cx = find_best_crop_center(full_gt)
    else:
        cy, cx = 500, 500
        print(f"  -> WARNING: Ground truth missing for {city}.")

    t1_crop   = load_oscd_rgb(city_img_path, "imgs_1", cy, cx) 
    t2_crop   = load_oscd_rgb(city_img_path, "imgs_2", cy, cx) 
    gt_crop   = load_mask_crop(gt_path, cy, cx)
    ef_crop   = load_mask_crop(ef_path, cy, cx)
    unet_crop = load_mask_crop(unet_path, cy, cx)
    ours_crop = load_mask_crop(ours_path, cy, cx)
    
    row_images = [t1_crop, t2_crop, gt_crop, ef_crop, unet_crop, ours_crop]
    
    for j, img in enumerate(row_images):
        ax = axes[i, j]
        
        if np.max(img) == 0 and j > 2: 
            ax.text(0.5, 0.5, "Missing\nPrediction", ha='center', va='center', color='red', fontweight='bold')
            ax.set_facecolor('#eeeeee')
        else:
            if img.ndim == 2:
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(img)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        if i == 0:
            ax.set_title(columns[j], fontsize=14, fontweight='bold', pad=15)
        if j == 0:
            ax.set_ylabel(rows[i], fontsize=14, fontweight='bold', labelpad=15)

plt.tight_layout()

output_filename = "ieee_oscd_comparison_grid.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS! Image saved to your folder as: {output_filename}")