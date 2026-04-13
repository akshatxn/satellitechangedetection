# 🛰️ An Optimized Siamese U-Net Framework for Urban Change Detection using MultiSpectral Sentinel-2 Imagery and Dynamic Spatial-Temporal Sampling

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Paper-Under%20Review%20@%20IEEE-blue?style=for-the-badge)

*VIT Chennai — School of Electrical Engineering — B.Tech ECE Capstone Project, April 2025*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Pipeline](#-end-to-end-pipeline)
- [Dashboard Features](#-dashboard-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance Metrics](#-performance-metrics)
- [Environmental Impact Engine](#-environmental-impact-engine)
- [Future Scope](#-future-scope)
- [Publication](#-publication)
- [Team](#-team)

---

## 🌍 Overview

Rapid urbanization is one of the defining challenges of our time — yet traditional monitoring methods are too slow, too expensive, and too inaccurate for the scale of global growth. **GeoAI Urban Tracker** solves this with a production-grade deep learning system that:

- **Detects urban expansion** from bi-temporal Sentinel-2 satellite imagery with high precision
- **Filters out false positives** caused by seasonal vegetation changes using Near-Infrared (NIR) spectral bands
- **Quantifies environmental impact** — sprawl area, vegetation loss, temperature increase, and runoff risk — automatically
- **Scales globally** across diverse climates and geographies, validated across 11 cities on 4 continents

> *"When the model flags a change, it is highly accurate."* — Precision of **51.4%** achieved, doubling the baseline through pure data engineering without modifying the neural architecture.

---

## 🏆 Key Results

| Metric | Baseline (14 Crops) | Optimized (500 Crops) | Improvement |
|---|---|---|---|
| **Precision** | 25.8% | **51.4%** | ⬆️ 2× |
| **Recall** | 18.3% | **36.6%** | ⬆️ 2× |
| **Average IoU** | 13.3% | **25.8%** | ⬆️ 2× |
| **F1-Score** | ~21% | **~42.8%** | ⬆️ 2× |
| **Dubai Overlap Score** | — | **95.5%** | 🎯 |

> All improvements achieved through **data engineering alone** — no changes to the neural network architecture.

---

## 🧠 Architecture

### The Siamese U-Net — "Twin Brain" Neural Network

```
         Input T1 (Baseline)      Input T2 (Follow-up)
               │                         │
        ┌──────▼──────┐           ┌──────▼──────┐
        │  Encoder A  │◄──── W ───►│  Encoder B  │   (Shared Weights)
        │  (U-Net)    │           │  (U-Net)    │
        └──────┬──────┘           └──────┬──────┘
               │   feat1           feat2 │
               └──────────┬────────────┘
                           │
                   torch.cat(feat1, feat2)
                           │
                    ┌──────▼──────┐
                    │   Decoder   │
                    └──────┬──────┘
                           │
                  Binary Change Mask
                  (threshold > 0.5)
```

**Why Siamese?** Standard U-Nets analyze a single image. A Siamese network runs two identical weight-sharing encoders in parallel — one per time period — so the model learns *geometric differences* between timelines rather than memorizing static shapes.

**Why NIR?** Natural vegetation reflects Near-Infrared light strongly; concrete and asphalt do not. By incorporating the B08 (NIR) Sentinel-2 band, the model mathematically separates seasonal vegetation noise from genuine construction signatures.

---

## 🔄 End-to-End Pipeline

```
Phase 1: Data Engineering          Phase 2: Model Training
┌─────────────────────────┐        ┌─────────────────────────┐
│ Acquire Sentinel-2 Data │──────►│ Initialize Siamese U-Net │
│ Import T1 + T2 + Masks  │        │ BCEWithLogitsLoss + Adam  │
│ Spatial Co-registration │        │ LR = 0.001               │
│ Patch Gen (256×256 px)  │        │ 500 crops/epoch → 60K    │
│ Data Augmentation       │        │ patches total             │
│ Train/Val/Test Split    │        │ Validation + Tuning Loop  │
└─────────────────────────┘        └─────────────────────────┘
                                              │
Phase 3: Dashboard Init            Phase 4: Live Inference
┌─────────────────────────┐        ┌─────────────────────────┐
│ Streamlit Web App       │        │ User uploads T1 + T2    │
│ Load Model Weights      │        │ Resize + Normalize       │
│ Render UI Components    │        │ Siamese U-Net Forward    │
└─────────────────────────┘        │ Threshold > 0.5 → Mask  │
                                   └─────────────────────────┘
                                              │
                               Phase 5: Environmental Impact
                               ┌─────────────────────────────┐
                               │ Changed px / Total px = %   │
                               │ Extrapolate: Vegetation Loss │
                               │ Temp Increase, Runoff Risk   │
                               │ Categorize: Critical /       │
                               │ Moderate / Stable Sprawl     │
                               └─────────────────────────────┘
```

---

## 📊 Dashboard Features

The **GeoAI Urban Tracker** is a three-view Streamlit dashboard:

### 1️⃣ Global Analytics & Model Performance
- Interactive bar chart benchmarking actual urban growth across **11 global cities**
- Cities ranked by ground-truth sprawl intensity (Abu Dhabi 5.21% → Milano 0.33%)
- Reveals dataset class imbalance, justifying the dynamic dataloader design

### 2️⃣ City-Level Predictions (Test Set)
- Dropdown to select any validation city
- Four synchronized panels: T1 baseline · T2 follow-up · Ground Truth mask · AI Prediction (NIR Boosted)
- Per-city accuracy metrics: ground truth %, AI predicted %, margin of error, overlap score
- **Dubai 9 showcase**: 8.9% human-detected vs. 8.5% AI-predicted — **95.5% overlap score**

### 3️⃣ Live Inference Engine (Custom Upload)
- Upload any T1/T2 satellite image pair (PNG, JPG, TIF, TIFF — up to 200MB)
- Interactive temporal comparison slider (drag to reveal before/after)
- Real-time AI sprawl detection mask overlaid in red
- Automated **Comprehensive Environmental Impact Report** generated instantly

---

## 📡 Dataset

| Property | Details |
|---|---|
| **Source** | OSCD — Onera Satellite Change Detection Dataset |
| **Satellite** | ESA Sentinel-2 |
| **Spectral Bands Used** | Red (B04), Green (B03), Near-Infrared (B08) |
| **Image Type** | Bi-temporal multispectral GeoTIFF |
| **Training Cities** | 11 global cities across 4 continents |
| **Patch Size** | 256 × 256 pixels |
| **Patches Per Epoch** | 500 crops → ~60,000 unique spatial views |
| **Labels** | Binary change masks (human-verified ground truth) |

**Cities in Dataset:** Abu Dhabi · Dubai · Berlin Outskirts · Madrid · Paris Suburb · Prague · Rome · Riyadh · São Paulo · Valencia · Milano

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/geoai-urban-tracker.git
cd geoai-urban-tracker

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
torchvision
numpy
rasterio
opencv-python
Pillow
streamlit
matplotlib
seaborn
tqdm
```

### GPU Acceleration (Recommended)
```bash
# Install CUDA-enabled PyTorch (check pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage

### Train the Model
```bash
python train_model.py
```

Key training parameters (configurable in `train_model.py`):
```python
# Data scaling — the core innovation
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
epochs = 10          # Optimized micro-training cycle

# Optimizer & loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    for t1, t2, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(t1, t2)   # Siamese forward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Run Inference on a City
```bash
python run_inference.py
# Enter city name to test (e.g., abudhabi):
```

### Launch the Dashboard
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

---

## 📈 Performance Metrics

### Urban Growth by City (Ground Truth)

```
Abu Dhabi      ████████████████████████████  5.21%
Dubai          ████████████████████████      4.15%
Berlin         ███████████████████████       3.88%
Madrid         ██████████████████            3.11%
Paris Suburb   ████████████████              2.72%
Prague         ███████████                   1.90%
Rome           ████████████                  1.97%
Riyadh         █████████                     1.64%
São Paulo      ███                           0.49%
Milano         ██                            0.45%
Valencia       ██                            0.33%
```

### Model Architecture Performance
```python
# Core Siamese U-Net forward pass
class SiameseUNet(nn.Module):
    def forward(self, x1, x2):
        # 1. Parallel Feature Extraction (Shared Weights)
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)

        # 2. Mathematical Concatenation of Timelines
        combined_features = torch.cat((feat1, feat2), dim=1)

        # 3. Decode into Binary Segmentation Mask
        output_mask = self.decoder(combined_features)
        return output_mask
```

---

## 🌱 Environmental Impact Engine

When sprawl is detected, the system automatically computes environmental consequences:

| Detected Change | Environmental Metric | Calculation Basis |
|---|---|---|
| Sprawl Area % | Land converted to impervious surface | Changed px / Total px |
| Vegetation Loss | Canopy reduction | Correlated to sprawl footprint |
| Temperature Increase | Urban Heat Island (UHI) effect | +°C per % impervious cover |
| Runoff Risk | Drainage strain | % increase in stormwater volume |

### Sprawl Severity Classification
```
> 5.0%  → 🔴 CRITICAL ALERT    — High ecological fragmentation risk
1.5–5%  → 🟡 MODERATE WARNING  — Monitor and plan green infrastructure
< 1.5%  → 🟢 STABLE            — Within acceptable development bounds
```

**Example Output (Live Inference):**
```
🚨 Critical Sprawl Detected: 42.62% change
   Vegetation Loss:      25.57%  (Canopy Reduction)
   Temp Increase:        +2.13°C (Microclimate UHI)
   Runoff Risk:          +179.0% (Drainage Strain)
```

---

## 🔬 Key Engineering Decisions

### Challenge 1: Data Starvation & Overfitting
**Problem:** The baseline model trained on just 14 full-city images rapidly memorized the dataset, failing to generalize.

**Solution:** Engineered a **dynamic 256×256 cropping algorithm** scaling epoch intake from 14 to 500 random patches — exposing the model to 60,000+ unique spatial views per training run.

### Challenge 2: Spectral Confusion (False Positives)
**Problem:** Seasonal vegetation changes were flagged as new construction, destroying precision.

**Solution:** Rewrote the data pipeline to ingest the **Near-Infrared B08 band** alongside Red and Green. Since chlorophyll strongly reflects NIR while concrete does not, the model learned to mathematically filter vegetation noise.

### Challenge 3: Class Imbalance
**Problem:** Unchanged pixels vastly outnumber changed pixels (~97% vs ~3%), causing the model to trivially predict "no change."

**Solution:** Combined **BCEWithLogitsLoss + Dice Loss** (hybrid loss function) to upweight the minority change class during training.

---

## 🔭 Future Scope

- [ ] **ResNet Backbone** — Swap the encoder for ResNet-50/101 to improve recall on small residential structures
- [ ] **Transformer Integration** — Explore Swin-Transformer encoders for capturing long-range spatial dependencies
- [ ] **Additional Spectral Bands** — Incorporate SWIR bands for improved material classification
- [ ] **Real-Time Satellite Feed** — Auto-process new Sentinel-2 acquisitions as they become available
- [ ] **Global Deployment** — Web-accessible API for urban planners and environmental agencies

---

## 📄 Publication

> **Akshat Pal, Nipun Varshneya, Aastha Singh.**
> *"An Optimized Siamese U-Net Framework for Urban Change Detection using Multi-Spectral Sentinel-2 Imagery and Dynamic Spatial-Temporal Sampling"*
> **IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing** — *Under Review*

---

## 👥 Team

| Name | Registration | Role |
|---|---|---|
| **Akshat Pal** | 22BLC1127 | Model Architecture & Training Pipeline |
| **Nipun Varshneya** | 22BLC1176 | Data Engineering & Dashboard |
| **Aastha Singh** | 22BLC1286 | Evaluation & Environmental Impact Engine |

**Guide:** Dr. Gnana Swathika OV
**Institution:** School of Electrical Engineering, VIT Chennai
**Degree:** B.Tech in Electronics and Computer Engineering, April 2025

---

## 📚 References

Key papers that informed this work:

1. Daudt et al. (2018) — *Fully Convolutional Siamese Networks for Change Detection* — IEEE ICIP
2. Chen et al. (2022) — *A Siamese Network Based U-Net for Change Detection* — IEEE J-STARS
3. Hafner et al. (2023) — *Semi-Supervised Urban Change Detection Using Multi-Modal Data* — MDPI Remote Sensing
4. Wang et al. (2024) — *Multi-Scale Fusion Siamese Network Based on Attention Mechanism* — MDPI Remote Sensing
5. Ronneberger et al. (2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation* — MICCAI

Full reference list available in [`Final_report1.pdf`](./Final_report1.pdf).

---

## 📜 License

This project is released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

---

<div align="center">

*Built with 🛰️ satellite imagery, 🧠 deep learning, and ☕ a lot of coffee — VIT Chennai, 2025*

⭐ Star this repo if you found it useful!

</div>
