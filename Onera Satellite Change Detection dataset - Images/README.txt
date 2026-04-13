import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
from scipy.ndimage import center_of_mass

# ==========================================
# 1. CONFIGURATION: SET YOUR FOLDER PATHS
# ==========================================
# Change these to the actual paths on your computer or Kaggle!
DIR_NORMAL = r"C:\path\to\your\normal_folder"        # Contains city folders -> time1 / time2
DIR_GT     = r"C:\path\to\your\test_folder"          # Contains ground truth .cm or .tif masks
DIR_PRED   = r"C:\path\to\your\predictions_folder"   # Where your AI saved its black & white outputs

# Define the 3 cities we are using for the IEEE paper
cities = ["dubai", "montpellier", "abudhabi"]

# Titles for the grid
columns = ["T1 (Baseline)", "T2 (Follow-up)", "Ground Truth", "EF-FCN (Bad Baseline)", "RGB U-Net (Good Baseline)", "Proposed (Ours)"]
rows = ["Arid Biome\n(Dubai)", "Temperate Biome\n(Montpellier)", "Dense Urban\n(Abu Dhabi)"]

# Size of the patch to crop for the paper (256x256 is standard)
CROP_SIZE = 256 

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def normalize_image(img):
    """Satellite .tif images are often 16-bit and look black. This makes them look bright and normal."""
    if img is None: return np.zeros((CROP_SIZE, CROP_SIZE, 3))
    # Drop pure outliers and scale to 0-255
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip(img, p2, p98)
    img = (img - p2) / (p98 - p2 + 1e-8) * 255.0
    return img.astype(np.uint8)

def find_best_crop_center(mask):
    """Automatically finds the center of the largest cluster of changed buildings!"""
    if mask is None or np.sum(mask) == 0:
        return mask.shape[0]//2, mask.shape[1]//2 # Fallback to true center
    
    # Find the center of mass of the white pixels (changes)
    cy, cx = center_of_mass(mask > 0)
    return int(cy), int(cx)

def load_and_crop(filepath, cy, cx, is_mask=False):
    """Loads a .tif/.cm/.png file and crops a perfectly aligned 256x256 patch."""
    if not os.path.exists(filepath):
        print(f"MISSING: {filepath}")
        return np.zeros((CROP_SIZE, CROP_SIZE)) if is_mask else np.zeros((CROP_SIZE, CROP_SIZE, 3))
    
    # Read image (handles .tif, .cm, .png)
    if filepath.endswith('.tif'):
        img = tiff.imread(filepath)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    # Calculate crop boundaries
    half = CROP_SIZE // 2
    h, w = img.shape[:2]
    
    # Ensure we don't crop outside the image boundaries
    startY = max(0, min(cy - half, h - CROP_SIZE))
    startX = max(0, min(cx - half, w - CROP_SIZE))
    
    crop = img[startY:startY+CROP_SIZE, startX:startX+CROP_SIZE]
    
    if not is_mask:
        # Assuming Sentinel-2 RGB bands (adjust if your TIF has different band orders)
        if len(crop.shape) == 3 and crop.shape[2] >= 3:
            crop = crop[:, :, :3] # Take first 3 bands (RGB)
        crop = normalize_image(crop)
    else:
        # Binarize masks to pure black and white
        crop = (crop > 0).astype(np.uint8) * 255
        
    return crop

# ==========================================
# 3. MAIN LOOP: ASSEMBLE THE GRID
# ==========================================
fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(20, 10))

for i, city in enumerate(cities):
    print(f"Processing city: {city}...")
    
    # Define exact paths for this city (ADJUST FILENAMES IF YOURS DIFFER)
    # E.g., OSCD often uses 'B04.tif', 'B03.tif', 'B02.tif' or 'TCI.tif'. Assuming 'TCI.tif' here.
    t1_path   = os.path.join(DIR_NORMAL, city, "time1", "TCI.tif") # Change TCI.tif to your actual RGB file name
    t2_path   = os.path.join(DIR_NORMAL, city, "time2", "TCI.tif")
    gt_path   = os.path.join(DIR_GT, city, "cm", "cm.png")         # Change to cm.tif if necessary
    
    # Your model predictions (You must generate these and put them in DIR_PRED)
    ef_path   = os.path.join(DIR_PRED, f"{city}_effcn_pred.png")
    unet_path = os.path.join(DIR_PRED, f"{city}_unet_pred.png")
    ours_path = os.path.join(DIR_PRED, f"{city}_ours_pred.png")
    
    # 1. Load Ground Truth First to find the best crop location!
    if os.path.exists(gt_path):
        full_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        cy, cx = find_best_crop_center(full_gt)
    else:
        cy, cx = 500, 500 # Arbitrary fallback
        print(f"Could not find GT for {city} to calculate crop. Using default center.")

    # 2. Crop all 6 images using that exact same center point
    t1_crop   = load_and_crop(t1_path, cy, cx, is_mask=False)
    t2_crop   = load_and_crop(t2_path, cy, cx, is_mask=False)
    gt_crop   = load_and_crop(gt_path, cy, cx, is_mask=True)
    ef_crop   = load_and_crop(ef_path, cy, cx, is_mask=True)
    unet_crop = load_and_crop(unet_path, cy, cx, is_mask=True)
    ours_crop = load_and_crop(ours_path, cy, cx, is_mask=True)
    
    row_images = [t1_crop, t2_crop, gt_crop, ef_crop, unet_crop, ours_crop]
    
    # 3. Plot them onto the grid
    for j, img in enumerate(row_images):
        ax = axes[i, j]
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
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

# ==========================================
# 4. SAVE FINAL PUBLICATION IMAGE
# ==========================================
output_filename = "ieee_advanced_comparison_grid.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS! High-Res IEEE Paper image saved as: {output_filename}")