import os
import shutil
import zipfile

def organize_oscd():
    base_dir = "OSCD_Dataset"
    
    # Create the master structure
    folders = [
        f"{base_dir}/Images",
        f"{base_dir}/Train_Labels",
        f"{base_dir}/Test_Labels"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("--- Starting OSCD Organization ---")

    # 1. Unzip Images if they are still in zip format
    for item in os.listdir('.'):
        if item.endswith('.zip') and 'Images' in item:
            print(f"Extracting {item}...")
            with zipfile.ZipFile(item, 'r') as zip_ref:
                zip_ref.extractall(f"{base_dir}/Images")

    # 2. Restructure Image Folders (Rename imgs_1 -> content, imgs_2 -> content_target)
    img_root = f"{base_dir}/Images"
    # Sometimes zips extract into a nested folder, let's find the actual city folders
    for root, dirs, files in os.walk(img_root):
        if 'imgs_1' in dirs and 'imgs_2' in dirs:
            city_path = root
            # Rename imgs_1 to content
            os.rename(os.path.join(city_path, 'imgs_1'), os.path.join(city_path, 'content'))
            # Rename imgs_2 to content_target
            os.rename(os.path.join(city_path, 'imgs_2'), os.path.join(city_path, 'content_target'))
            print(f"Processed city: {os.path.basename(city_path)}")

    # 3. Handle Labels
    for item in os.listdir('.'):
        if item.endswith('.zip'):
            target_dir = ""
            if 'Train' in item: target_dir = f"{base_dir}/Train_Labels"
            elif 'Test' in item: target_dir = f"{base_dir}/Test_Labels"
            
            if target_dir:
                print(f"Extracting {item} to {target_dir}...")
                with zipfile.ZipFile(item, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)

    print("\n--- Done! Your OSCD_Dataset folder is ready for Review 2 ---")

if __name__ == "__main__":
    organize_oscd()