# AI-Based Deforestation Detection Using Sentinel-2 Satellite Imagery

## 1. Project Overview & Objectives
This project implements an automated, deep-learning pipeline to monitor and detect deforestation (forest loss) over time. It leverages free, open-source multi-spectral satellite imagery (Sentinel-2 Level-2A) and global land-cover datasets (ESA WorldCover) to perform binary semantic segmentation (Forest vs. Non-Forest). 

The ultimate goal is to process two satellite images of the same area from different dates (Date A and Date B), identify forested pixels in both, and highlight areas where forest cover existed in Date A but disappeared in Date B.

## 2. Technologies and Tools Used
*   **Python 3.12+**: Core programming language.
*   **PyTorch**: Deep learning framework used to build and train the U-Net segmentation model.
*   **Rasterio & Shapely**: Geospatial libraries for handling, cropping, and projecting GeoTIFF satellite files and coordinate reference systems (CRS).
*   **Planetary Computer & PySTAC**: APIs used to programmatically query and download cloud-filtered Sentinel-2 and ESA WorldCover data.
*   **NumPy**: Array manipulation for spectral band stacking and cloud masking.
*   **Pytest**: Unit testing framework to guarantee pipeline stability.

## 3. Current Implemented Features (Verified)
*   **STAC Feasibility Probe (`stac_probe.py`)**: Can query the Microsoft Planetary Computer, validate Area of Interest (AOI) bounding boxes, and assess cloud cover and ESA WorldCover availability.
*   **Quicklook Visualization (`quicklook.py`)**: Generates RGB true-color composites to visually verify that the downloaded Date A and Date B images are radiometrically aligned.
*   **Geospatial Downloader (`download.py`)**: Safely downloads 10m and 20m Sentinel-2 bands, enforcing strict mathematical pixel snapping and `boundless=True` reading to avoid edge-crashes.
*   **Analysis-Ready Data Preprocessing (`preprocess.py`)**: Resamples 20m bands to 10m, aligns the ESA WorldCover labels to the UTM projection, extracts authoritative radiometric offsets directly from file tags, and strictly masks clouds/shadows using the Scene Classification Layer (SCL).
*   **Deep Learning Dataset Generator (`dataset.py`)**: Implements strict spatial blocking (splitting the AOI into physically separated NW, NE, SW, SE quadrants) to prevent spatial data leakage between Train, Val, and Test splits. Safely normalizes reflectance values.
*   **U-Net Architecture (`unet.py`)**: A standard, compiled PyTorch U-Net accepting 6-channel input and producing 1-channel binary segmentation logits.

## 4. Planned / Pending Features
*   **Model Training Loop (`train.py`)**: Implementing the PyTorch optimization loop, focal loss functions, learning rate schedulers, and model checkpointing.
*   **Inference & Multi-Date Segmentation (`predict.py`)**: Running the trained U-Net over Date A and Date B to generate the final prediction masks.
*   **Change Detection & Area Calculation (`change_detection.py`)**: Subtracting masks to flag deforestation and converting projected pixel counts into physical hectares.
*   **Interactive Dashboard (`app.py`)**: A Streamlit and Folium web application to upload coordinates and visualize the detected forest loss on an interactive map.

## 5. Project Architecture
The project strictly separates data acquisition, geospatial preprocessing, machine learning, and visualization into distinct modules.

1.  **Data Acquisition (`src/data/download.py`)**: Raw STAC fetching to `data/raw/`.
2.  **Preprocessing (`src/data/preprocess.py`)**: Grid alignment, cloud masking, and radiometric scaling to `data/processed/`.
3.  **Dataset Construction (`src/data/dataset.py`)**: Spatial blocking and PyTorch `DataLoader` generation.
4.  **Modeling (`src/models/`)**: PyTorch network definitions and training logic.
5.  **Inference (`src/inference/`)**: *Pending*.
6.  **Dashboard (`src/app/`)**: *Pending*.

## 6. Installation & Execution
*(Note: Complete installation instructions pending finalization of the environment requirements)*

**To run the existing verified data pipeline:**
1. Configure your AOI in `config/config.yaml`.
2. Probe the API: `python -m src.data.stac_probe --config config/config.yaml`
3. Download data: `python -m src.data.download --config config/config.yaml`
4. Preprocess ARD: `python -m src.data.preprocess --config config/config.yaml`
5. Test dataloaders: `python -m src.data.dataset --config config/config.yaml`

## 7. Datasets
*   **Sentinel-2 Level-2A**: 10m bands (B02, B03, B04, B08) and 20m bands (B11, B12, SCL).
*   **ESA WorldCover (2020/2021)**: 10m global land cover product used as the ground truth training label (Class 10 = Tree Cover).

## 8. Limitations & Scientific Constraints
*   **ESA WorldCover is not perfect ground truth**: It represents a specific epoch. Training on imagery from years far removed from the WorldCover release date without accounting for natural changes will confuse the model.
*   **Cloud Contamination**: Sentinel-2's SCL mask is imperfect. Undetected thin cirrus clouds or cloud shadows in Date B can cause the model to predict "Non-Forest", leading to false-positive deforestation alerts.
*   **Phenology**: Seasonal leaf-off conditions can mimic deforestation. Date A and Date B must ideally be drawn from the same season/month.