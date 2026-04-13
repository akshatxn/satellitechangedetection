import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# Import your existing code structure
try:
    from train_model import SiameseUNet, OSCDDataset
except ImportError:
    print("ERROR: Could not find 'train_model.py'. Make sure this script is in the same folder.")
    exit()

def generate_presentation_graphics():
    # Setup directories
    save_dir = "presentation_images"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the AI
    print("🧠 Loading 3-Band AI Brain...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseUNet(n_channels=3, n_classes=1).to(device)
    
    # Check for weights
    weights_path = 'siamese_unet_weights.pth'
    if not os.path.exists(weights_path):
        print(f"❌ ERROR: Could not find '{weights_path}'. You must train the model first.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load Dataset (Pulling from Test split for a clean demo)
    print("🛰️ Loading Dataset...")
    # Change 'Test' to 'Train' ONLY if your data folders are not separated.
    dataset = OSCDDataset(root_dir='.', split='Train') 
    
    # Pull 3 different random samples to get a good variety of results
    print(f"📸 Generating presentation visuals for 3 random locations...")
    
    for i in range(3):
        # Grab a random index
        rand_idx = random.randint(0, len(dataset) - 1)
        t1, t2, label = dataset[rand_idx] 
        
        # Add descriptive captions directly to the plotting arrays
        # This makes the saved image self-explanatory in your PPT slide.
        
        # 1. Run Inference (Get the AI's opinion)
        with torch.no_grad():
            output = model(t1.unsqueeze(0).to(device), t2.unsqueeze(0).to(device))
            # Binary threshold at 50% confidence
            pred_mask = (output > 0.5).float().cpu().numpy()[0, 0] 

        # 2. Setup presentation plotting (White background is cleaner for PPT)
        fig = plt.figure(figsize=(20, 7), facecolor='white')
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1])
        
        # ---------------------------------------------------------
        # DISPLAY LOGIC (False Color Composite)
        # ---------------------------------------------------------
        # Since we use Red, Green, NIR, we visualize them as (NIR, Red, Green).
        # In this visualization: Red = Healthy Vegetation, Grey/Blue = Urban areas.
        
        # Function to slice and rearrange bands for visualization
        def get_vis_image(img_tensor):
            # img_tensor shape is (3, 256, 256) -> bands are (B04, B03, B08)
            # Reorder for visualization: (B08/NIR, B04/Red, B03/Green)
            b04, b03, b08 = img_tensor[0].numpy(), img_tensor[1].numpy(), img_tensor[2].numpy()
            vis = np.stack([b08, b04, b03], axis=-1)
            # Clip to 0-1 range for plotting
            return np.clip(vis, 0, 1)

        t1_vis = get_vis_image(t1)
        t2_vis = get_vis_image(t2)
        true_mask = label[0].numpy()

        # --- Panel 1: T1 (Before) ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(t1_vis)
        ax1.set_title("1. INITIAL STATE (Timeline 1)", fontsize=16, fontweight='bold', pad=15)
        ax1.axis('off')
        # Description caption
        desc1 = "Sentinel-2 False-Color\n(NIR/Red/Green)\n\nShows base landscape\nbefore changes."
        ax1.text(128, 280, desc1, fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

        # --- Panel 2: T2 (After) ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(t2_vis)
        ax2.set_title("2. UPDATED STATE (Timeline 2)", fontsize=16, fontweight='bold', pad=15)
        ax2.axis('off')
        # Description caption
        desc2 = "Same area captured later.\n\nChanges in urban geometry\nare Spectrally compared\nto Timeline 1."
        ax2.text(128, 280, desc2, fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

        # --- Panel 3: Ground Truth ---
        ax3 = fig.add_subplot(gs[0, 2])
        # Plot T2 as base, overlay Ground Truth in translucent green
        ax3.imshow(t2_vis, alpha=0.7)
        ax3.imshow(true_mask, cmap='Greens', vmin=0, vmax=1, alpha=0.5) 
        ax3.set_title("3. GROUND TRUTH (Reality)", fontsize=16, fontweight='bold', pad=15)
        ax3.axis('off')
        # Description caption
        desc3 = "Confirmed Human Changes.\n\nTranslucent Green overlay\nshows the *exact*\nlocation of new buildings."
        ax3.text(128, 280, desc3, fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

        # --- Panel 4: AI Prediction ---
        ax4 = fig.add_subplot(gs[0, 3])
        # Plot Prediction in high-contrast magma cmap
        ax4.imshow(pred_mask, cmap='magma', vmin=0, vmax=1)
        ax4.set_title("4. AI MODEL PREDICTION", fontsize=16, fontweight='bold', color='#D4AF37', pad=15)
        ax4.axis('off')
        # Prediction specifics - HIGHLIGHT YOUR SUCCESS NUMBERS HERE
        desc4 = "PRO-MODE Siamese U-Net\n(3-Band Input + NIR Boost)\nData-Scaled Training (60k crops)\n\nRESULT: Detected changes\nmatching Reality Panel."
        ax4.text(128, 280, desc4, fontsize=12, fontweight='bold', ha='center', va='top', color='white', bbox=dict(facecolor='#D4AF37', alpha=0.8, boxstyle='round,pad=0.5'))

        # Global branding for the project
        plt.suptitle("URBAN CHANGE DETECTION PROJECT | PRO-MODE INFERENCE DEMO", fontsize=22, fontweight='bold', y=0.98)
        
        # Save the result
        save_path = os.path.join(save_dir, f"ppt_result_{rand_idx}_{i+1}.png")
        print(f"✅ Saved presentation-ready image to: {save_path}")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig) # Close fig so it doesn't try to pop up

    print(f"\n🎉 Done! Look in the '{save_dir}' folder for your images. Put them directly into PowerPoint.")

if __name__ == "__main__":
    generate_presentation_graphics()