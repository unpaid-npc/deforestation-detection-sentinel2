# TerraSentinel 🌍 — AI-Based Deforestation Detection & Forest Monitoring

## 1. Project Overview & Objectives
TerraSentinel is an end-to-end geospatial artificial intelligence pipeline designed to monitor, isolate, and quantify deforestation over time using open-source multi-spectral satellite imagery. The platform automates data acquisition from the Microsoft Planetary Computer, performs strict radiometric and cloud-masking preprocessing, trains a custom 6-band PyTorch U-Net segmentation model, and serves verified change-detection metrics via an executive Streamlit dashboard.

## 2. Technologies & Stack
*   **Deep Learning:** PyTorch (U-Net architecture with Batch Normalization and Dropout).
*   **Geospatial Processing:** Rasterio, Shapely, Rasterio-Warp (UTM ↔ WGS84 reprojection).
*   **Data API:** Planetary Computer STAC Client (`pystac-client`).
*   **Web Dashboard:** Streamlit, Folium, Streamlit-Folium.
*   **Language:** Python 3.12+

## 3. Core Architecture & Pipeline
1.  **Data Acquisition (`src/data/download.py`):** Targeted STAC queries fetch 10m bands (B02, B03, B04, B08), 20m bands (B11, B12, SCL), and ESA WorldCover baseline layers with boundless pixel-padding.
2.  **Preprocessing (`src/data/preprocess.py`):** Resamples 20m bands to 10m, reprojects ESA WorldCover, extracts authoritative radiometric offsets from raster file tags, and masks invalid Scene Classification Layer (SCL) pixels.
3.  **Spatial Blocking (`src/data/dataset.py`):** Partitions the Area of Interest into non-overlapping geographic quadrants (NW, NE, SW, SE) separated by buffer zones to completely eliminate spatial data leakage.
4.  **Model Training (`src/models/train.py` & `unet.py`):** Trains a U-Net using a custom **Masked Focal Loss** function to handle heavy class imbalance while ignoring cloud-obscured pixels.
5.  **Change Detection & Inference (`src/inference/predict.py`):** Sliding-window inference generates binary forest masks for Date A and Date B, subtracting them to isolate verified canopy loss and calculating total impact in **Hectares**.
6.  **Interactive Dashboard (`src/app/app.py`):** An executive dark-mode Streamlit app featuring metric cards and an Esri Satellite / Folium geospatial overlay.

## 4. Verified Results (Rondonia AOI MVP)
*   **Total Deforested Area Detected:** 1,174.83 Hectares (117,483 pixels at 10m resolution).
*   **Model Performance:** Converged validation loss down to `0.0094` using Masked Focal Loss.

## 5. Quick Start & Execution
1.  **Install Dependencies:** `pip install -r requirements.txt`
2.  **Download STAC Data:** `python -m src.data.download --config config/config.yaml`
3.  **Run Preprocessing ARD:** `python -m src.data.preprocess --config config/config.yaml`
4.  **Train Model:** `python -m src.models.train --config config/config.yaml`
5.  **Run Inference:** `python -m src.inference.predict --config config/config.yaml`
6.  **Launch Dashboard:** `streamlit run src/app/app.py`