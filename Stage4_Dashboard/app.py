import streamlit as st
import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import UNet

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Satellite AI Pipeline", layout="wide")

st.title("🛰️ End-to-End Satellite Change Detection")

# --- SIDEBAR & CONFIG ---
st.sidebar.header("🕹️ Project Controls")
DATA_DIR = "dataset"
available_cities = [f.split('.')[0] for f in os.listdir(os.path.join(DATA_DIR, "time1")) if f.endswith('.png')]
selected_city = st.sidebar.selectbox("Select Region:", available_cities)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    device = "cpu"
    model = UNet()
    try:
        model.load_state_dict(torch.load("my_unet_model.pth", map_location=device))
    except:
        st.error("Model file not found!")
    model.eval()
    return model

def load_images(city):
    t1_path = os.path.join(DATA_DIR, "time1", f"{city}.png")
    t2_path = os.path.join(DATA_DIR, "time2", f"{city}.png")
    mask_path = os.path.join(DATA_DIR, "masks", f"{city}.png")
    
    img1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    return img1, img2, gt_mask

def predict_unet(model, img1, img2, threshold=0.5):
    img1_r = cv2.resize(img1, (256, 256)) / 255.0
    img2_r = cv2.resize(img2, (256, 256)) / 255.0
    
    t1 = torch.tensor(img1_r, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t2 = torch.tensor(img2_r, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        out = model(t1, t2)
        prob_map = torch.sigmoid(out).squeeze().numpy() # Raw probability (0.0 to 1.0)
        
    mask_pred = (prob_map > threshold).astype(np.uint8) * 255
    
    # Resize back
    mask_resized = cv2.resize(mask_pred, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
    prob_resized = cv2.resize(prob_map, (img1.shape[1], img1.shape[0]))
    
    return mask_resized, prob_resized

# --- MAIN LOGIC ---
model = load_model()
img1, img2, gt_mask = load_images(selected_city)

tab1, tab2, tab3 = st.tabs(["1️⃣ Stage 1: Alignment", "2️⃣ Stage 2: AI Detection", "3️⃣ Stage 3: Analytics"])

# === TAB 2: AI DETECTION (Updated) ===
with tab2:
    st.header("Step 2: Change Detection (U-Net)")
    
    # --- NEW: SENSITIVITY SLIDER ---
    st.write("### 🎛️ Model Sensitivity")
    threshold = st.slider("Detection Threshold (Lower = More Sensitive)", 0.01, 0.99, 0.50, 0.01)
    
    pred_mask, prob_map = predict_unet(model, img1, img2, threshold)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img1, caption="Time 1", use_container_width=True)
    with col2:
        st.image(img2, caption="Time 2", use_container_width=True)
    with col3:
        # Show the raw probability map to see if it's learning ANYTHING
        st.image(prob_map, caption="Raw Probability Heatmap", use_container_width=True, clamp=True)

    st.subheader("Final Prediction Overlay")
    alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.4)
    
    # Red Overlay Logic
    overlay = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    overlay[pred_mask == 255] = [255, 0, 0] # Red
    blended = cv2.addWeighted(overlay, alpha, cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB), 1 - alpha, 0)
    
    st.image(blended, caption=f"Construction Detected (Threshold: {threshold})", use_container_width=True)

# === TAB 3: ANALYTICS (Updated) ===
with tab3:
    st.header("Step 3: Quantitative Analysis")
    
    gt_flat = (cv2.resize(gt_mask, (256,256)) > 128).flatten()
    pred_flat = (cv2.resize(pred_mask, (256,256)) > 128).flatten()
    
    total = len(gt_flat)
    correct = np.sum(gt_flat == pred_flat)
    acc = (correct / total) * 100
    
    growth_pixels = np.sum(pred_flat == 1)
    growth_pct = (growth_pixels / total) * 100
    
    c1, c2 = st.columns(2)
    c1.metric("Pixel Accuracy", f"{acc:.2f}%")
    c2.metric("Urban Growth", f"{growth_pct:.2f}%")
    
    if growth_pct == 0:
        st.warning("⚠️ Growth is 0%. Try lowering the threshold slider in Tab 2!")