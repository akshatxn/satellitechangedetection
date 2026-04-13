import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Configuration
RAW_DATA_DIR = "raw_data"
OUTPUT_DIR = "aligned_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def enhance_contrast(img):
    """
    Makes the image clearer so SIFT can find features.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(img)

def load_band_image(folder_path, band_name="B04.tif"):
    """
    Robust image loader. Tries B04 (Red), then B02 (Blue), then ANY tif.
    """
    # Try different bands if one fails
    for b in [band_name, "B02.tif", "B03.tif", "*.tif"]:
        search_path = os.path.join(folder_path, f"*{b}") if "*" not in b else os.path.join(folder_path, b)
        files = glob.glob(search_path)
        if files:
            break
    
    if not files:
        return None

    # Read TIF image
    path = files[0]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image is loaded properly
    if img is None:
        return None
        
    # Normalize (Fix dark images)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype('uint8')
    
    return img

def align_images(img_ref, img_target):
    """ Aligns img_target to img_ref using SIFT with Enhancement """
    
    # 1. Enhance Contrast (Crucial for Dubai/Desert images)
    img_ref_enhanced = enhance_contrast(img_ref)
    img_target_enhanced = enhance_contrast(img_target)

    # 2. Detect Features (SIFT)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_ref_enhanced, None)
    kp2, des2 = sift.detectAndCompute(img_target_enhanced, None)

    if des1 is None or des2 is None:
        print("   ⚠️ No descriptors found.")
        return None, None

    # 3. Match Features
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # 4. Filter Good Matches (Relaxed ratio to 0.8)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    print(f"   > Found {len(good)} good matches.")

    if len(good) < 10:
        print("   ⚠️ Not enough matches to align safely.")
        return None, None

    # 5. Calculate Transformation
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    # 6. Apply Alignment to the ORIGINAL (not enhanced) image
    h, w = img_ref.shape
    aligned_img = cv2.warpPerspective(img_target, H, (w, h))
    
    # Visualization
    match_viz = cv2.drawMatches(img_ref_enhanced, kp1, img_target_enhanced, kp2, good[:50], None, flags=2)

    return aligned_img, match_viz

# --- MAIN EXECUTION ---
cities = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

for city in cities:
    print(f"\nProcessing {city}...")
    
    path_t1 = os.path.join(RAW_DATA_DIR, city, "imgs_1_rect")
    path_t2 = os.path.join(RAW_DATA_DIR, city, "imgs_2_rect")

    img1 = load_band_image(path_t1)
    img2 = load_band_image(path_t2)

    if img1 is None or img2 is None:
        print(f"   ❌ Could not load images for {city}")
        continue

    # Run Alignment
    aligned_img, match_viz = align_images(img1, img2)

    # Only save/show if alignment succeeded
    if aligned_img is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{city}_t1.jpg"), img1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{city}_t2_aligned.jpg"), aligned_img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{city}_matches.jpg"), match_viz)
        
        print(f"   ✅ Success! Results in '{OUTPUT_DIR}'")

        # Show Plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title(f"{city} Before")
        plt.imshow(img1, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title(f"{city} Aligned")
        plt.imshow(aligned_img, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title("Matches Found")
        plt.imshow(match_viz)
        plt.tight_layout()
        plt.show()
    else:
        print(f"   ❌ Skipping {city} due to alignment failure.")