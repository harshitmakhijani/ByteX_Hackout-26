"""
Machine Learning Pipeline Service
Loads Model 1 (PyTorch ResNet-18 Algae Classifier & Segmentation)
and Model 2 (XGBoost Carbon Sequestration Regressor + Scaler)
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import joblib
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model 1 architecture
class AlgaeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class MLPipeline:
    def __init__(self):
        self.m1_path = os.path.join(BASE_DIR, "model1_algae_classifier.pt")
        self.m1_config_path = os.path.join(BASE_DIR, "model1_config.json")
        self.m2_path = os.path.join(BASE_DIR, "model2_carbon_regressor.pkl")
        self.m2_scaler_path = os.path.join(BASE_DIR, "model2_scaler.pkl")
        self.m2_config_path = os.path.join(BASE_DIR, "model2_config.json")

        self._load_configs()
        self._load_model1()
        self._load_model2()

    def _load_configs(self):
        with open(self.m1_config_path, "r") as f:
            self.m1_config = json.load(f)
        with open(self.m2_config_path, "r") as f:
            self.m2_config = json.load(f)

        self.class_names = self.m1_config.get("class_names", ["Low", "Moderate", "High"])
        self.img_size = self.m1_config.get("img_size", 224)

    def _load_model1(self):
        print("Loading Model 1 (ResNet-18 Algae Classifier)...")
        self.model1 = AlgaeClassifier(num_classes=len(self.class_names)).to(DEVICE)
        ckpt = torch.load(self.m1_path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model1.load_state_dict(state_dict)
        self.model1.eval()
        print("✅ Model 1 successfully initialized on", DEVICE)

    def _load_model2(self):
        print("Loading Model 2 (XGBoost Carbon Regressor)...")
        self.model2 = joblib.load(self.m2_path)
        self.scaler = joblib.load(self.m2_scaler_path)
        print("✅ Model 2 and Scaler successfully initialized")

    def preprocess_image(self, pil_image):
        """Converts PIL image to normalized tensor for ResNet-18."""
        rgb_img = pil_image.convert("RGB")
        tfm = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return tfm(rgb_img).unsqueeze(0).to(DEVICE), rgb_img

    def segment_algae(self, rgb_pil, farm_area_ha=25.0):
        """Calculates chlorophyll ExG index & coverage percentage."""
        img_np = np.array(rgb_pil, dtype=np.float32) / 255.0
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        exg = 2.0 * g - r - b
        exg_range = exg.max() - exg.min()
        exg_norm = (exg - exg.min()) / exg_range if exg_range > 1e-6 else np.zeros_like(exg)
        is_water = ((r + g + b) / 3.0) < 0.75

        algae_binary = (exg_norm > 0.45) & is_water
        water_px = int(is_water.sum())
        algae_px = int(algae_binary.sum())
        coverage = (algae_px / water_px * 100.0) if water_px > 0 else 0.0
        algae_ha = farm_area_ha * coverage / 100.0

        return {
            "coverage_pct": round(float(coverage), 1),
            "bloom_area_ha": round(float(algae_ha), 1),
            "water_pixels": water_px,
            "algae_pixels": algae_px
        }

    def predict_model1(self, pil_image, farm_area_ha=25.0):
        """Runs inference on satellite image using Model 1."""
        tensor, rgb_pil = self.preprocess_image(pil_image)
        with torch.no_grad():
            logits = self.model1(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

        seg = self.segment_algae(rgb_pil, farm_area_ha)

        return {
            "severity": self.class_names[pred_idx],
            "severity_index": pred_idx,
            "confidence_pct": round(confidence * 100, 1),
            "coverage_pct": seg["coverage_pct"],
            "bloom_area_ha": seg["bloom_area_ha"],
            "class_probabilities": {
                name: round(float(p) * 100, 1) for name, p in zip(self.class_names, probs)
            }
        }

    def classify_ecosystem_status(self, dissolved_oxygen, daily_co2, coverage):
        """Biological alert classification matching the trained model."""
        if dissolved_oxygen < 2.0:
            return {
                "status": "⚠️ Anoxia Risk — Aerate Immediately",
                "code": "CRITICAL_ANOXIA",
                "severity": "danger",
                "recommendation": "Emergency: Dissolved oxygen < 2.0 mg/L causes catastrophic fish kill and anaerobic decay. Activate surface aerators and micro-bubble sparging immediately."
            }
        elif daily_co2 > 500 and coverage > 60:
            return {
                "status": "🌿 Optimal Harvest Window",
                "code": "OPTIMAL_HARVEST",
                "severity": "success",
                "recommendation": "Peak biomass density achieved with maximum carbon fixation. Harvest now to prevent self-shading, nutrient exhaustion, and nocturnal oxygen crashes."
            }
        elif daily_co2 > 50:
            return {
                "status": "✅ Active Carbon Sequestration",
                "code": "ACTIVE_SEQUESTRATION",
                "severity": "optimal",
                "recommendation": "Algae culture actively photosynthesizing. Water parameters are well-balanced for continuous carbon fixation."
            }
        else:
            return {
                "status": "💤 Minimal Sequestration Activity",
                "code": "MINIMAL_ACTIVITY",
                "severity": "neutral",
                "recommendation": "Low biological fixation. Check solar irradiance, water temperature, or nutrient levels to stimulate growth."
            }

    def predict_model2(self, bloom_area_ha, coverage_pct, severity_index,
                       water_temp_c, ph_level, dissolved_oxygen_mg_l, solar_irradiance_w_m2):
        """Runs Model 2 Carbon Sequestration Regressor."""
        features = np.array([[
            float(bloom_area_ha),
            float(coverage_pct),
            float(severity_index),
            float(water_temp_c),
            float(ph_level),
            float(dissolved_oxygen_mg_l),
            float(solar_irradiance_w_m2)
        ]])

        scaled_features = self.scaler.transform(features)
        preds = self.model2.predict(scaled_features)[0]

        daily_biomass_kg = max(0.0, float(preds[0]))
        daily_co2_kg = max(0.0, float(preds[1]))
        monthly_co2_tons = max(0.0, float(preds[2]))
        carbon_credit_usd = max(0.0, float(preds[3]))

        eco = self.classify_ecosystem_status(dissolved_oxygen_mg_l, daily_co2_kg, coverage_pct)

        return {
            "daily_biomass_kg": round(daily_biomass_kg, 1),
            "daily_co2_kg": round(daily_co2_kg, 1),
            "monthly_co2_tons": round(monthly_co2_tons, 2),
            "carbon_credit_usd": round(carbon_credit_usd, 2),
            "ecosystem_status": eco["status"],
            "ecosystem_code": eco["code"],
            "ecosystem_severity": eco["severity"],
            "recommendation": eco["recommendation"],
            "annual_co2_tons_projected": round(monthly_co2_tons * 12.0, 1),
            "annual_carbon_credit_usd": round(carbon_credit_usd * 12.0, 2)
        }

    def run_full_pipeline(self, pil_image, sensors_dict):
        """
        Unified Multimodal Pipeline:
        Model 1 Vision -> Model 2 Carbon Engine -> Ecosystem Risk
        """
        farm_area_ha = float(sensors_dict.get("bloom_area_ha", 25.0))
        
        # Step 1: Model 1 Vision Analysis
        m1_res = self.predict_model1(pil_image, farm_area_ha)

        # Override or blend coverage if user provided specific field measurement
        coverage = float(sensors_dict.get("coverage_pct", m1_res["coverage_pct"]))
        bloom_area = float(sensors_dict.get("bloom_area_ha", m1_res["bloom_area_ha"]))
        severity_idx = m1_res["severity_index"]

        # Step 2: Model 2 Carbon & Economics
        m2_res = self.predict_model2(
            bloom_area_ha=bloom_area,
            coverage_pct=coverage,
            severity_index=severity_idx,
            water_temp_c=float(sensors_dict.get("water_temp_c", 26.5)),
            ph_level=float(sensors_dict.get("ph_level", 8.0)),
            dissolved_oxygen_mg_l=float(sensors_dict.get("dissolved_oxygen_mg_l", 7.5)),
            solar_irradiance_w_m2=float(sensors_dict.get("solar_irradiance_w_m2", 650.0))
        )

        return {
            "model1_vision": m1_res,
            "model2_carbon": m2_res,
            "inputs_used": {
                "bloom_area_ha": bloom_area,
                "coverage_pct": coverage,
                "severity_index": severity_idx,
                "water_temp_c": float(sensors_dict.get("water_temp_c", 26.5)),
                "ph_level": float(sensors_dict.get("ph_level", 8.0)),
                "dissolved_oxygen_mg_l": float(sensors_dict.get("dissolved_oxygen_mg_l", 7.5)),
                "solar_irradiance_w_m2": float(sensors_dict.get("solar_irradiance_w_m2", 650.0))
            }
        }


# Singleton instance
_pipeline_instance = None

def get_ml_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MLPipeline()
    return _pipeline_instance
