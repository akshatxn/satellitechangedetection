import os
import cv2
import glob

# Config
SOURCE_RAW_DATA = "../Stage1_Data_Alignment/raw_data"
DEST_DIR = "dataset"

# Create clean folders
for d in ["time1", "time2", "masks"]:
    os.makedirs(os.path.join(DEST_DIR, d), exist_ok=True)

print("🚀 Preparing Data & Converting to PNG...")

# Get list of cities
cities = [d for d in os.listdir(SOURCE_RAW_DATA) 
          if os.path.isdir(os.path.join(SOURCE_RAW_DATA, d)) and d != "train_labels"]

count = 0
for city in cities:
    city_path = os.path.join(SOURCE_RAW_DATA, city)
    
    # 1. FIND FILES
    t1_files = glob.glob(os.path.join(city_path, "imgs_1_rect", "*"))
    t1_files = [f for f in t1_files if f.lower().endswith(('.tif', '.png', '.jpg'))]
    
    t2_files = glob.glob(os.path.join(city_path, "imgs_2_rect", "*"))
    t2_files = [f for f in t2_files if f.lower().endswith(('.tif', '.png', '.jpg'))]

    # Search for mask in city folder OR train_labels folder
    mask_files = glob.glob(os.path.join(city_path, "cm", "*"))
    if not mask_files:
        mask_files = glob.glob(os.path.join(SOURCE_RAW_DATA, "train_labels", city, "cm", "*"))

    # 2. PROCESS & CONVERT
    if t1_files and t2_files and mask_files:
        # Pick files (Prioritize B04/Red band if available)
        src_t1 = next((f for f in t1_files if "B04" in f), t1_files[0])
        src_t2 = next((f for f in t2_files if "B04" in f), t2_files[0])
        src_mask = mask_files[0]
        
        # Read Images
        img_t1 = cv2.imread(src_t1)
        img_t2 = cv2.imread(src_t2)
        img_mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE) # Masks are grayscale

        if img_t1 is None or img_t2 is None or img_mask is None:
            print(f"❌ Error reading files for {city}. Skipping.")
            continue

        # Save everything as PNG (Standardization)
        cv2.imwrite(os.path.join(DEST_DIR, "time1", f"{city}.png"), img_t1)
        cv2.imwrite(os.path.join(DEST_DIR, "time2", f"{city}.png"), img_t2)
        cv2.imwrite(os.path.join(DEST_DIR, "masks", f"{city}.png"), img_mask)
        
        print(f"✅ Processed & Converted: {city}")
        count += 1
    else:
        print(f"⚠️ Skipping {city}: Missing files")

print(f"\n🎉 Done! {count} pairs converted to PNG in 'dataset/' folder.")