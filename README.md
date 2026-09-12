# ALGAEVERSE — Bio-Carbon Intelligence & Satellite Digital Twin 🌿🛰️

**ByteX HackOut '26 Project**

ALGAEVERSE is a multimodal ecological monitoring and digital twin platform designed for real-time algae bloom surveillance, carbon sequestration estimation, and automated blue carbon credit accounting (dMRV).

---

## 🌟 Key Features

- **Page 1 — IoT & Satellite Simulator**:
  - **Catchment Selector**: Interactive monitoring of water bodies (Chilika Lagoon, Lake Erie, Lake Taihu, Bellandur Lake, Lake Tahoe).
  - **Quick Scenario Presets**: 1-click biogeochemical calibrations (*Optimal Summer Harvest*, *Severe Anoxia Emergency*, *Pristine Oligotrophic*, *Rapid Eutrophication*).
  - **In-Situ IoT Sensor Array**: Synchronized controls for Water Temperature, Dissolved Oxygen, pH, Solar PAR, and Catchment Area.
  - **Multispectral Satellite Studio**: Select Sentinel-2 MSI band composites or upload custom drone/satellite imagery.
  - **Real-Time LoRaWAN Uplink & Firebase Sync**: Encrypted telemetry packet transmission committed directly to Firebase Firestore.

- **Page 2 — AI Command Center & Temporal Timeline**:
  - **Ecosystem Health Verdict & Banner**: Live ecological alerts with autonomous dMRV recommendations.
  - **4-Card KPI Bento Grid**: Algae Bloom Area (ha), Daily CO₂ Sequestered (kg), Carbon Credit Value ($ USD/day), and Dissolved Oxygen (mg/L) with historical % deltas.
  - **Side-by-Side Temporal Comparison Panel**: Compares **Previous Checkpoint (Firestore)** vs **Latest Broadcast (Real-time)** with coverage, area, CO₂, and credit deltas.
  - **Firestore Timeline Strip**: Chronological interactive strip of all past scans saved in Firebase Firestore. Clicking any card inspects that historical checkpoint in the comparison panel.
  - **Dual ML Pipeline**:
    - **PyTorch ResNet-18 Vision Classifier**: Predicts bloom severity class and coverage percentage with class confidence breakdowns.
    - **XGBoost Carbon Regressor**: Calculates daily biomass, daily CO₂ fixation, monthly offset, and annualized carbon revenue ($35/tonne offset rate).
  - **Interactive Chart.js Visualizations**: Historical trends for Bloom Coverage (%) and Daily CO₂ Fixation (kg).

---

## 🏗️ Architecture & Technology Stack

- **Backend**: FastAPI (Python 3.12)
- **Computer Vision Model**: PyTorch (ResNet-18 Fine-Tuned Classifier)
- **Carbon Accounting Engine**: XGBoost Regressor + Scikit-Learn Scaler
- **Cloud Database**: Firebase Firestore REST API (`hackout26-d49d5`) + Persistent Local Cache
- **Frontend**: HTML5, Vanilla CSS3 (Custom Luminous Design System), JavaScript (ES6+), Chart.js v4.4

---

## 🚀 Quickstart & Local Setup

### Prerequisites
- Python 3.10+
- Virtual environment (`.venv`)

### Installation & Execution

```bash
# Clone the repository
git clone https://github.com/harshitmakhijani/ByteX_Hackout-26.git
cd ByteX_Hackout-26

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn torch torchvision pillow xgboost scikit-learn requests pydantic

# Run FastAPI Server
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open your browser at **`http://localhost:8000`**.

---

## 📊 API Endpoints

- `GET /` — Serves the ALGAEVERSE frontend application
- `GET /api/health` — Returns system status, loaded ML models, and Firestore database state
- `POST /api/simulate` — Multimodal prediction endpoint (accepts IoT sensors + satellite scene, executes PyTorch & XGBoost models, computes deltas, and commits to Firebase Firestore)
- `GET /api/timeline/{location_id}` — Returns chronological Firestore scan records for a location
- `GET /api/comparison/{location_id}` — Returns side-by-side latest vs previous scan comparison
