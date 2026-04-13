import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ==========================================
# 1. YOUR AI BRAIN (Upgraded for 3 Channels)
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
# PURE DICE LOSS (The Best Teacher for this Task)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth 

    def forward(self, inputs, targets):
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice

# ==========================================
# 2. YOUR EYES (With TRICK #1, 3-Band NIR, & Data Intake Fix)
# ==========================================
class OSCDDataset(Dataset):
    def __init__(self, root_dir='.', split='Train'): 
        self.img_dir = os.path.join(root_dir, 'Onera Satellite Change Detection dataset - Images')
        self.label_dir = os.path.join(root_dir, f'Onera Satellite Change Detection dataset - {split} Labels')
        self.split = split
        
        if os.path.exists(self.label_dir):
            self.cities = [f for f in os.listdir(self.label_dir) if os.path.isdir(os.path.join(self.label_dir, f))]
            print(f"[DEBUG] Successfully found {len(self.cities)} cities for {split}ing!")
        else:
            raise FileNotFoundError(f"[ERROR] Could not find the folder: {self.label_dir}")

    def normalize(self, img):
        img = img.astype(np.float32)
        low, high = np.percentile(img, 2), np.percentile(img, 98)
        return np.clip((img - low) / (high - low + 1e-6), 0, 1)

    # --- THE DATA INTAKE FIX ---
    def __len__(self):
        if self.split == 'Train':
            return 500  # Pull 500 random crops per epoch instead of 14!
        else:
            return len(self.cities)

    # --- THE MISSING FUNCTION HAS BEEN RESTORED ---
    def _get_image_path(self, city, possible_folders, band):
        for folder in possible_folders:
            path = os.path.join(self.img_dir, city, folder, band)
            if os.path.exists(path): return path
        raise FileNotFoundError(f"Could not find {band} for {city}")

    def __getitem__(self, idx):
        # 2. Pick a random city for each of the 500 steps
        if self.split == 'Train':
            city = np.random.choice(self.cities)
        else:
            city = self.cities[idx]
        
        # --- NEW: 3-BAND LOADING LOGIC (Red, Green, Near-Infrared) ---
        bands = ['B04.tif', 'B03.tif', 'B08.tif']
        
        t1_paths = [self._get_image_path(city, ['content', 'imgs_1_rect', 'imgs_1'], b) for b in bands]
        t2_paths = [self._get_image_path(city, ['content_target', 'imgs_2_rect', 'imgs_2'], b) for b in bands]
        
        t1_data = [self.normalize(np.array(Image.open(p))) for p in t1_paths]
        t2_data = [self.normalize(np.array(Image.open(p))) for p in t2_paths]
        
        # Stack into 3D tensors: (3, H, W)
        t1 = np.stack(t1_data, axis=0)
        t2 = np.stack(t2_data, axis=0)
        # -------------------------------------------------------------
        
        label_path = os.path.join(self.label_dir, city, 'cm', 'cm.png')
        if not os.path.exists(label_path):
            label_path = os.path.join(self.label_dir, city, 'cm.png')
            
        label = np.array(Image.open(label_path).convert('L'))
        label = (label > 0).astype(np.float32)
        label = np.expand_dims(label, axis=0)

        # Padding (Handles the 3-channel depth perfectly)
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

        # ---------------------------------------------------------
        # TRICK #1: ADVANCED DATA AUGMENTATION (Flips + Rotations)
        # ---------------------------------------------------------
        if self.split == 'Train':
            # 1. Horizontal Flip
            if np.random.rand() > 0.5:
                t1 = np.flip(t1, axis=2).copy()
                t2 = np.flip(t2, axis=2).copy()
                label = np.flip(label, axis=2).copy()
            # 2. Vertical Flip
            if np.random.rand() > 0.5:
                t1 = np.flip(t1, axis=1).copy()
                t2 = np.flip(t2, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            # 3. Random 90-degree Rotations
            k = np.random.randint(0, 4) 
            if k > 0:
                t1 = np.rot90(t1, k, axes=(1, 2)).copy()
                t2 = np.rot90(t2, k, axes=(1, 2)).copy()
                label = np.rot90(label, k, axes=(1, 2)).copy()

        return torch.tensor(t1), torch.tensor(t2), torch.tensor(label)

# ==========================================
# 3. THE TRAINING ENGINE
# ==========================================
def train_model():
    dataset = OSCDDataset(root_dir='.', split='Train')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    print("Beginning 120-Epoch Deep Training (500 crops per epoch)...")
    
    model = SiameseUNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 120 Epochs for better generalization
    epochs = 120
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (t1, t2, labels) in enumerate(dataloader):
            t1, t2, labels = t1.to(device), t2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(t1, t2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {epoch_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), 'siamese_unet_weights.pth')
    print("\n✅ Training Complete! Model saved as 'siamese_unet_weights.pth'")

if __name__ == "__main__":
    train_model()