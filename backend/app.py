"""
FastAPI Main Application
Serves the EcoSense Algae Monitoring & Carbon Sequestration Digital Twin platform.
Endpoints for ML inference (ResNet-18 + XGBoost), Firebase Firestore sync, and telemetry history.
"""

import os
import time
import uuid
import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from backend.ml_pipeline import get_ml_pipeline
from backend.firestore_service import get_firestore_service

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "backend", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(
    title="ALGAEVERSE — Bio-Carbon Intelligence & Satellite Digital Twin",
    description="Multimodal IoT Telemetry Simulator & Live Ecological Monitoring Dashboard",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/css", StaticFiles(directory=os.path.join(STATIC_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(STATIC_DIR, "js")), name="js")
app.mount("/images", StaticFiles(directory=os.path.join(STATIC_DIR, "images")), name="images")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>EcoSense Simulator — UI Initializing...</h1>"


@app.get("/api/locations")
async def get_locations():
    """Returns supported water bodies and monitoring stations."""
    return [
        {
            "id": "chilika_lake",
            "name": "Chilika Lagoon",
            "region": "Odisha, India",
            "coords": "19.71° N, 85.32° E",
            "type": "Brackish Coastal Lagoon",
            "area_ha": 116500,
            "default_bloom_ha": 35.0,
            "sample_image": "satellite_dense_bloom.jpg",
            "description": "Asia's largest brackish water lagoon, highly prone to seasonal cyanobacterial blooms and vital for blue carbon."
        },
        {
            "id": "lake_erie",
            "name": "Lake Erie (Western Basin)",
            "region": "Ohio / Ontario, USA & Canada",
            "coords": "41.83° N, 82.50° W",
            "type": "Freshwater Great Lake",
            "area_ha": 2566700,
            "default_bloom_ha": 65.0,
            "sample_image": "sample_lake_erie.png",
            "description": "Epicenter of recurring Microcystis toxic algal blooms driven by agricultural runoff in Sandusky Bay."
        },
        {
            "id": "hussain_sagar",
            "name": "Hussain Sagar Lake",
            "region": "Hyderabad, Telangana, India",
            "coords": "17.42° N, 78.47° E",
            "type": "Urban Eutrophic Reservoir",
            "area_ha": 570,
            "default_bloom_ha": 18.0,
            "sample_image": "satellite_dense_bloom.jpg",
            "description": "Historic urban water body with dense nutrient loading and active bioremediation projects."
        },
        {
            "id": "bellandur_lake",
            "name": "Bellandur Lake",
            "region": "Bengaluru, Karnataka, India",
            "coords": "12.93° N, 77.67° E",
            "type": "Hyper-Eutrophic Urban Lake",
            "area_ha": 360,
            "default_bloom_ha": 22.0,
            "sample_image": "sample_lake_erie.png",
            "description": "High-profile urban catchment with extreme algal biomass and flammable froth incidents."
        },
        {
            "id": "lake_tahoe",
            "name": "Lake Tahoe (Pristine Reference)",
            "region": "California / Nevada, USA",
            "coords": "39.09° N, 120.03° W",
            "type": "Oligotrophic Alpine Lake",
            "area_ha": 49000,
            "default_bloom_ha": 2.0,
            "sample_image": "satellite_pristine_water.jpg",
            "description": "Benchmark oligotrophic lake with ultra-high water clarity (>20m Secchi depth) and minimal chlorophyll."
        }
    ]


@app.get("/api/presets")
async def get_presets():
    """Returns 1-click scenario presets for quick pitching and testing."""
    return [
        {
            "id": "optimal_harvest",
            "name": "Peak Harvest Window",
            "badge": "Maximum Carbon Credit",
            "description": "Optimal photosynthesis parameters. Dense bloom without oxygen depletion. High carbon monetization.",
            "values": {
                "water_temp_c": 26.8,
                "ph_level": 8.2,
                "dissolved_oxygen_mg_l": 8.6,
                "solar_irradiance_w_m2": 820.0,
                "bloom_area_ha": 45.0,
                "coverage_pct": 78.0,
                "sample_image": "satellite_dense_bloom.jpg"
            }
        },
        {
            "id": "anoxia_emergency",
            "name": "Severe Anoxia Emergency",
            "badge": "Aeration Required",
            "description": "Critical oxygen crash (DO < 2.0 mg/L) caused by nocturnal respiration. Urgent fish-kill risk!",
            "values": {
                "water_temp_c": 32.5,
                "ph_level": 8.9,
                "dissolved_oxygen_mg_l": 1.6,
                "solar_irradiance_w_m2": 950.0,
                "bloom_area_ha": 80.0,
                "coverage_pct": 92.0,
                "sample_image": "satellite_dense_bloom.jpg"
            }
        },
        {
            "id": "pristine_baseline",
            "name": "Pristine Clean Water",
            "badge": "Zero Bloom Baseline",
            "description": "Clear oligotrophic water. High dissolved oxygen, low biomass, baseline carbon fixation.",
            "values": {
                "water_temp_c": 18.5,
                "ph_level": 7.4,
                "dissolved_oxygen_mg_l": 10.2,
                "solar_irradiance_w_m2": 450.0,
                "bloom_area_ha": 1.5,
                "coverage_pct": 6.0,
                "sample_image": "satellite_pristine_water.jpg"
            }
        },
        {
            "id": "eutrophication_spurt",
            "name": "Rapid Eutrophication Spurt",
            "badge": "Early Warning",
            "description": "Early stage algal bloom expansion following agricultural fertilizer runoff.",
            "values": {
                "water_temp_c": 24.5,
                "ph_level": 7.9,
                "dissolved_oxygen_mg_l": 6.8,
                "solar_irradiance_w_m2": 680.0,
                "bloom_area_ha": 28.0,
                "coverage_pct": 45.0,
                "sample_image": "sample_lake_erie.png"
            }
        }
    ]


@app.post("/api/simulate")
async def run_simulation(
    location_id: str = Form("chilika_lake"),
    location_name: str = Form("Chilika Lagoon"),
    water_temp_c: float = Form(26.5),
    ph_level: float = Form(8.0),
    dissolved_oxygen_mg_l: float = Form(7.5),
    solar_irradiance_w_m2: float = Form(650.0),
    bloom_area_ha: float = Form(25.0),
    coverage_pct: float = Form(60.0),
    sample_image: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None)
):
    """
    Main Simulation & Telemetry Endpoint:
    1. Loads Satellite Image (Uploaded file OR Sample image)
    2. Runs Model 1 (PyTorch ResNet-18) to predict Algae Severity & Coverage
    3. Feeds features to Model 2 (XGBoost) to calculate Carbon Sequestration & Credits
    4. Computes Deltas against historical records in Firestore/Local DB
    5. Returns full environmental intelligence payload
    """
    start_time = time.time()
    pipeline = get_ml_pipeline()
    db_service = get_firestore_service()

    # Determine image source
    pil_image = None
    image_url = None

    if image_file and image_file.filename:
        try:
            content = await image_file.read()
            pil_image = Image.open(io.BytesIO(content))
            filename = f"upload_{uuid.uuid4().hex[:8]}_{image_file.filename}"
            save_path = os.path.join(UPLOADS_DIR, filename)
            pil_image.save(save_path)
            image_url = f"/uploads/{filename}"
        except Exception as e:
            print(f"Error reading uploaded image: {e}")

    if pil_image is None:
        # Fall back to sample image from static/images
        img_name = sample_image if sample_image else "satellite_dense_bloom.jpg"
        sample_path = os.path.join(STATIC_DIR, "images", img_name)
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, "sample_test_image.png")
            img_name = "sample_test_image.png"

        try:
            pil_image = Image.open(sample_path)
            image_url = f"/static/images/{img_name}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load image: {e}")

    # Build sensor telemetry dictionary
    sensors = {
        "water_temp_c": water_temp_c,
        "ph_level": ph_level,
        "dissolved_oxygen_mg_l": dissolved_oxygen_mg_l,
        "solar_irradiance_w_m2": solar_irradiance_w_m2,
        "bloom_area_ha": bloom_area_ha,
        "coverage_pct": coverage_pct
    }

    # Execute ML Pipeline
    ai_results = pipeline.run_full_pipeline(pil_image, sensors)
    execution_time_ms = round((time.time() - start_time) * 1000, 1)

    # Prepare database record
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record_id = f"telemetry_{int(time.time())}_{uuid.uuid4().hex[:4]}"

    record_data = {
        "id": record_id,
        "timestamp": now_iso,
        "location_id": location_id,
        "location_name": location_name,
        "image_url": image_url,
        "water_temp_c": water_temp_c,
        "ph_level": ph_level,
        "dissolved_oxygen_mg_l": dissolved_oxygen_mg_l,
        "solar_irradiance_w_m2": solar_irradiance_w_m2,
        "coverage_pct": ai_results["model1_vision"]["coverage_pct"],
        "bloom_area_ha": ai_results["model1_vision"]["bloom_area_ha"],
        "severity": ai_results["model1_vision"]["severity"],
        "confidence_pct": ai_results["model1_vision"]["confidence_pct"],
        "daily_biomass_kg": ai_results["model2_carbon"]["daily_biomass_kg"],
        "daily_co2_kg": ai_results["model2_carbon"]["daily_co2_kg"],
        "monthly_co2_tons": ai_results["model2_carbon"]["monthly_co2_tons"],
        "carbon_credit_usd": ai_results["model2_carbon"]["carbon_credit_usd"],
        "ecosystem_status": ai_results["model2_carbon"]["ecosystem_status"],
        "ecosystem_code": ai_results["model2_carbon"]["ecosystem_code"],
        "ecosystem_severity": ai_results["model2_carbon"]["ecosystem_severity"],
        "recommendation": ai_results["model2_carbon"]["recommendation"],
        "execution_time_ms": execution_time_ms
    }

    # Record in telemetry service (Firestore sync + Delta calculation)
    enriched_record = db_service.record_telemetry(record_data)

    return {
        "success": True,
        "record": enriched_record,
        "model1_vision": ai_results["model1_vision"],
        "model2_carbon": ai_results["model2_carbon"],
        "deltas": enriched_record.get("deltas", {}),
        "execution_time_ms": execution_time_ms,
        "firestore_synced": enriched_record.get("firestore_synced", False)
    }


@app.get("/api/history/{location_id}")
async def get_location_history(location_id: str):
    """Returns chronological observation records for trend chart visualization."""
    db_service = get_firestore_service()
    history = db_service.get_history(location_id, limit=12)
    return {
        "location_id": location_id,
        "total_records": len(history),
        "history": history
    }


@app.get("/api/timeline/{location_id}")
async def get_location_timeline(location_id: str):
    """Returns full timeline of scans for a location, enriched for dashboard display."""
    db_service = get_firestore_service()
    timeline = db_service.get_timeline(location_id, limit=20)
    return {
        "location_id": location_id,
        "total_scans": len(timeline),
        "timeline": timeline
    }


@app.get("/api/comparison/{location_id}")
async def get_location_comparison(location_id: str):
    """Returns latest vs previous scan for side-by-side comparison."""
    db_service = get_firestore_service()
    comparison = db_service.get_comparison(location_id)
    return comparison


@app.api_route("/api/clear-database", methods=["GET", "POST"])
async def clear_database():
    """Purges all dummy and live data from both local cache and cloud Firebase Firestore."""
    db_service = get_firestore_service()
    success = db_service.clear_all()
    return {
        "status": "success",
        "message": "All database records purged from local database and Firebase Firestore.",
        "cleared": success
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "model1_vision": "ResNet-18 Algae Classifier [Loaded]",
            "model2_carbon": "XGBoost Carbon Regressor [Loaded]"
        },
        "database": "Firebase Firestore (hackout26-d49d5) + Local Persistent Cache",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
