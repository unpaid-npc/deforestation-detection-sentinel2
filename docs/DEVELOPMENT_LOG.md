# Development Log

## Status: ACTIVE
**Overall Completion:** ~50%
**Current Phase:** Transitioning from Data Engineering to Deep Learning Training.

### Work Completed So Far
1. **Architecture & Scoping:** Established a strict 8-step pipeline focused on a reproducible Minimum Viable Product (MVP) over a single Area of Interest (Rondonia).
2. **Data Acquisition (`src/data/stac_probe.py`, `src/data/download.py`, `src/data/quicklook.py`):** 
   - Connected to Microsoft Planetary Computer via STAC.
   - Successfully downloaded Sentinel-2 granules and ESA WorldCover baseline.
3. **Geospatial Preprocessing (`src/data/preprocess.py`, `src/utils/geo_2.py`):**
   - Successfully snapped 20m bands to 10m grids.
   - Handled Sentinel-2 radiometric baseline offset (-0.1 reflectance shift).
4. **Deep Learning Datasets (`src/data/dataset.py`):**
   - Implemented Spatial Blocking to prevent data leakage.
   - Calculated numerically stable standard deviation and mean for normalization.
5. **Model Architecture (`src/models/unet.py`):**
   - Created a standard PyTorch U-Net for 6-channel input.

### Errors Encountered & Resolved
1. **Rasterio Boundary Crashing:** `from_bounds` calculated edge windows that threw `ValueError` when extending 1 pixel outside the granule. 
   * *Resolution:* Implemented `boundless=True` reading with nodata fill in `download.py` and `geo_2.py`.
2. **Boolean Masking Broadcast Error:** Attempting to assign `np.nan` to a 3D tensor using a 2D spatial boolean mask crashed NumPy.
   * *Resolution:* Replaced indexing with `np.where(valid, reflectance, np.nan)` in `preprocess.py`.
3. **Catastrophic Cancellation in Variance:** `total_sq / count - mean**2` produced negative numbers due to float precision, yielding `NaN` when squared.
   * *Resolution:* Switched to `np.std(all_valid, axis=1)` over a concatenated array in `dataset.py`.

### Remaining Tasks and Milestones
- [ ] **Milestone 1:** Build `src/models/train.py` and execute the training loop.
- [ ] **Milestone 2:** Build `src/inference/predict.py` to generate masks for Date A and Date B.
- [ ] **Milestone 3:** Build `src/change_detection/compare.py` to flag deforestation.
- [ ] **Milestone 4:** Build `src/app/app.py` Streamlit dashboard.