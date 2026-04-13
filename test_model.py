import os
import torch
import scipy.ndimage as ndimage
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

# ==========================================
# 1. YOUR AI BRAIN (3-Channel Version)
# ==========================================
class SiameseUNet(nn.Module):
    # Changed default n_channels from 2 to 3
    def __init__(self, n_channels=3, n_classes=1):
        super(SiameseUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, n_classes, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, t1, t2):
        feat1 = self.encoder(t1)
        feat2 = self.encoder(t2)
        combined = torch.cat([feat1, feat2], dim=1)
        out = self.decoder(combined)
        return out

# ==========================================
# 2. YOUR EYES (3-Band NIR Loading)
# ==========================================
class OSCDDataset(Dataset):
    def __init__(self, root_dir='.', split='Test'): 
        self.img_dir = os.path.join(root_dir, 'Onera Satellite Change Detection dataset - Images')
        self.label_dir = os.path.join(root_dir, f'Onera Satellite Change Detection dataset - {split} Labels')
        if os.path.exists(self.label_dir):
            self.cities = [f for f in os.listdir(self.label_dir) if os.path.isdir(os.path.join(self.label_dir, f))]
        else:
            raise FileNotFoundError(f"[ERROR] Could not find the folder: {self.label_dir}")

    def normalize(self, img):
        img = img.astype(np.float32)
        low, high = np.percentile(img, 2), np.percentile(img, 98)
        return np.clip((img - low) / (high - low + 1e-6), 0, 1)

    def __len__(self):
        return len(self.cities)

    def _get_image_path(self, city, possible_folders, band):
        for folder in possible_folders:
            path = os.path.join(self.img_dir, city, folder, band)
            if os.path.exists(path): return path
        raise FileNotFoundError(f"Could not find {band} for {city}")

    def __getitem__(self, idx):
        city = self.cities[idx]
        
        # --- NEW: 3-BAND LOADING LOGIC (Red, Green, Near-Infrared) ---
        bands = ['B04.tif', 'B03.tif', 'B08.tif']
        
        t1_paths = [self._get_image_path(city, ['content', 'imgs_1_rect', 'imgs_1'], b) for b in bands]
        t2_paths = [self._get_image_path(city, ['content_target', 'imgs_2_rect', 'imgs_2'], b) for b in bands]
        
        t1_data = [self.normalize(np.array(Image.open(p))) for p in t1_paths]
        t2_data = [self.normalize(np.array(Image.open(p))) for p in t2_paths]
        
        t1 = np.stack(t1_data, axis=0)
        t2 = np.stack(t2_data, axis=0)
        # -------------------------------------------------------------
        
        label_path = os.path.join(self.label_dir, city, 'cm', 'cm.png')
        if not os.path.exists(label_path):
            label_path = os.path.join(self.label_dir, city, 'cm.png')
            
        label = np.array(Image.open(label_path).convert('L'))
        label = (label > 0).astype(np.float32)
        label = np.expand_dims(label, axis=0)

        crop_size = 256
        _, h, w = t1.shape
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        if pad_h > 0 or pad_w > 0:
            t1 = np.pad(t1, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            t2 = np.pad(t2, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            _, h, w = t1.shape

        start_y = np.random.randint(0, h - crop_size + 1)
        start_x = np.random.randint(0, w - crop_size + 1)
        t1 = t1[:, start_y:start_y+crop_size, start_x:start_x+crop_size]
        t2 = t2[:, start_y:start_y+crop_size, start_x:start_x+crop_size]
        label = label[:, start_y:start_y+crop_size, start_x:start_x+crop_size]

        return torch.tensor(t1), torch.tensor(t2), torch.tensor(label), city

# ==========================================
# 3. THE EVALUATION ENGINE
# ==========================================
def run_evaluation(num_cities_to_view=10): 
    print("Loading Test Dataset...")
    test_dataset = OSCDDataset(root_dir='.', split='Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure the tester knows we are using 3 channels now
    model = SiameseUNet(n_channels=3, n_classes=1).to(device)
    
    if os.path.exists('siamese_unet_weights.pth'):
        model.load_state_dict(torch.load('siamese_unet_weights.pth', map_location=device))
        print("Model weights loaded successfully!\n")
    else:
        print("ERROR: Could not find siamese_unet_weights.pth!")
        return
        
    model.eval() 
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # --- AUTO-DELETE OLD IMAGES ---
    print("Sweeping out old images from the results folder...")
    for file in os.listdir(save_dir):
        if file.endswith(".png"):
            os.remove(os.path.join(save_dir, file))
            
    print(f"📁 New images will be saved to '{save_dir}'.\n")
    
    total_iou = 0; total_precision = 0; total_recall = 0
    
    for step in range(num_cities_to_view):
        idx = np.random.randint(len(test_dataset))
        t1, t2, label, city_name = test_dataset[idx]
        
        t1_batch = t1.unsqueeze(0).to(device)
        t2_batch = t2.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(t1_batch, t2_batch)
            
        # Slicing [0] to only plot the first band (Red) for the grayscale visualization
        t1_vis = t1[0].cpu().numpy() 
        t2_vis = t2[0].cpu().numpy()
        
        true_mask = label[0].cpu().numpy()
        pred_mask = prediction[0][0].cpu().numpy()
        
        # --- TRICK #2: THRESHOLD TUNING ---
        CONFIDENCE_THRESHOLD = 0.50 
        raw_binary = (pred_mask > CONFIDENCE_THRESHOLD)
        
        # Morphological Opening to remove salt-and-pepper noise
        pred_binary = ndimage.binary_opening(raw_binary, structure=np.ones((2, 2))).astype(np.float32)
        true_binary = (true_mask > 0.5).astype(np.float32)
        
        TP = np.sum((pred_binary == 1) & (true_binary == 1))
        FP = np.sum((pred_binary == 1) & (true_binary == 0))
        FN = np.sum((pred_binary == 0) & (true_binary == 1))
        
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        iou = TP / (TP + FP + FN + 1e-6)
        
        total_precision += precision
        total_recall += recall
        total_iou += iou
        
        print(f"[{step+1}/{num_cities_to_view}] {city_name} | IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(t1_vis, cmap='gray'); axes[0].set_title(f'2015 Image ({city_name})'); axes[0].axis('off')
        axes[1].imshow(t2_vis, cmap='gray'); axes[1].set_title('2018 Image'); axes[1].axis('off')
        axes[2].imshow(true_mask, cmap='gray'); axes[2].set_title('Actual Change'); axes[2].axis('off')
        axes[3].imshow(pred_binary, cmap='hot', vmin=0, vmax=1); axes[3].set_title('AI Prediction (NIR Boosted)'); axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"prediction_{city_name}_{step+1}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig) 

    print("\n" + "="*40)
    print("🏆 FINAL AI REPORT CARD (MULTISPECTRAL MODE) 🏆")
    print("="*40)
    print(f"Average Precision : {total_precision / num_cities_to_view:.4f}")
    print(f"Average Recall    : {total_recall / num_cities_to_view:.4f}")
    print(f"Average IoU       : {total_iou / num_cities_to_view:.4f}")
    print("="*40)

    # Save to CSV for the Streamlit Dashboard
    import pandas as pd
    report_data = {
        "Metric": ["Precision", "Recall", "IoU"],
        "Score": [total_precision/num_cities_to_view, total_recall/num_cities_to_view, total_iou/num_cities_to_view]
    }
    pd.DataFrame(report_data).to_csv("final_report.csv", index=False)

if __name__ == "__main__":
    run_evaluation(num_cities_to_view=10)