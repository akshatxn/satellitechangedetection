import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from model import UNet

# --- CONFIGURATION ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 4       # Increased slightly for stability
LEARNING_RATE = 0.001
NUM_EPOCHS = 25      # Increased to ensure it learns
DATA_DIR = "dataset" # Ensure this folder exists in Stage2

class ChangeDetectionDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        # Safety check for empty folders
        if not os.path.exists(os.path.join(dir_path, "time1")):
            print(f"❌ CRITICAL ERROR: Could not find dataset at {dir_path}")
            self.images = []
        else:
            self.images = [f for f in os.listdir(os.path.join(dir_path, "time1")) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        
        # Load Images
        t1 = cv2.imread(os.path.join(self.dir_path, "time1", img_name), cv2.IMREAD_GRAYSCALE)
        t2 = cv2.imread(os.path.join(self.dir_path, "time2", img_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.dir_path, "masks", img_name), cv2.IMREAD_GRAYSCALE)

        # Resize to standard size
        t1 = cv2.resize(t1, (IMG_WIDTH, IMG_HEIGHT))
        t2 = cv2.resize(t2, (IMG_WIDTH, IMG_HEIGHT))
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

        # Normalize (0 to 1)
        t1 = t1 / 255.0
        t2 = t2 / 255.0
        mask = mask / 255.0
        
        # Binarize Mask (Make sure it is strictly 0 or 1)
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0

        # Convert to Tensor [Channel, H, W]
        t1_t = torch.tensor(t1, dtype=torch.float32).unsqueeze(0)
        t2_t = torch.tensor(t2, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return t1_t, t2_t, mask_t

def train():
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}")

    # 2. Initialize Model
    model = UNet().to(device)
    
    # 3. CRITICAL FIX: Aggressive Weighted Loss
    # We set weight to 30.0 to FORCE the model to predict changes.
    # If the result is too messy (too much white), lower this to 10.0 or 5.0.
    pos_weight = torch.tensor([30.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Load Data
    print("📂 Loading Dataset...")
    train_ds = ChangeDetectionDataset(DATA_DIR)
    
    if len(train_ds) == 0:
        print("❌ Error: Dataset is empty. Run 'prepare_data.py' first!")
        return

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ Found {len(train_ds)} images. Starting training for {NUM_EPOCHS} epochs...")

    # 5. Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        
        for t1, t2, mask in loop:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)

            # Forward Pass
            predictions = model(t1, t2)
            loss = loss_fn(predictions, mask)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"   >>> Avg Loss: {epoch_loss/len(loader):.4f}")

    # 6. Save Final Model
    save_path = "my_unet_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 SUCCESS! Model saved to: {os.path.abspath(save_path)}")
    print("👉 ACTION REQUIRED: Copy 'my_unet_model.pth' to your Stage4_Dashboard folder now.")

if __name__ == "__main__":
    train()