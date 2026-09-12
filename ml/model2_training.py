"""
==========================================================================
 MODEL 2: Carbon Sequestration Regression Engine
 Algae-Based Carbon Sequestration Monitoring Platform — HackOut '26
==========================================================================

 PURPOSE:
   Train a regression model that takes algae bloom visual metrics
   (from Model 1) + environmental water sensor data and predicts:
     1. Daily biomass yield (kg / day)
     2. Daily CO₂ sequestered (kg CO₂ / day)
     3. Monthly CO₂ projection (metric tons CO₂ / month)
     4. Carbon credit value ($ USD / month)
     5. Ecosystem health status (Active / Harvest / Anoxia Risk)

 SCIENTIFIC BASIS:
   Microalgae photosynthesis:
     6CO₂ + 6H₂O + hν → C₆H₁₂O₆ + 6O₂

   Typical productivity: 10–50 g dry weight / m² / day
   Carbon content of dry algae: ~50% by weight
   CO₂-to-Carbon ratio: 44/12 = 3.67
   ∴ 1 kg dry algae ≈ 0.5 kg C ≈ 1.83 kg CO₂ absorbed

 TRAINING DATA:
   Synthetically generated from peer-reviewed biological models:
     - Temperature response: Eppley (1972) curve
     - Light saturation: Monod/Michaelis-Menten kinetics
     - pH effect: Gaussian around optimal 7.5–8.5
     - DO correlation: Photosynthetic oxygen proxy

 HOW TO RUN:
   1. Open Google Colab (CPU is fine, no GPU needed)
   2. Copy-paste this entire script into cells (split by sections)
   3. Run all cells (~30 seconds total)
   4. Download exported model files
==========================================================================
"""

# =========================================================================
# SECTION 1: Install Dependencies (uncomment in Colab)
# =========================================================================
# !pip install -q scikit-learn xgboost matplotlib pandas seaborn joblib

import os
import json
import time
import warnings
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# scipy not needed for core pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.multioutput import MultiOutputRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
    print("✅ XGBoost available")
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed, using GradientBoosting fallback")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

warnings.filterwarnings("ignore")
np.random.seed(42)


# =========================================================================
# SECTION 2: Configuration
# =========================================================================

class Config:
    """All hyperparameters and constants for Model 2."""

    # --- Synthetic Dataset ---
    NUM_SAMPLES = 15000           # Training samples to generate
    TRAIN_SPLIT = 0.80
    RANDOM_STATE = 42

    # --- Input Feature Ranges (realistic environmental bounds) ---
    # From Model 1 outputs
    AREA_RANGE = (0.1, 500.0)         # Algae bloom area in hectares
    COVERAGE_RANGE = (1.0, 100.0)     # Coverage percentage of water body
    SEVERITY_LEVELS = [0, 1, 2]       # 0=Low, 1=Moderate, 2=High

    # From IoT / Weather sensors
    TEMP_RANGE = (5.0, 40.0)          # Water temperature °C
    PH_RANGE = (5.5, 10.5)           # pH level
    DO_RANGE = (0.5, 18.0)           # Dissolved Oxygen mg/L
    LIGHT_RANGE = (50.0, 1200.0)     # Solar irradiance W/m² (PAR proxy)

    # --- Biological Constants (peer-reviewed) ---
    # Maximum specific growth rate at optimal conditions
    MAX_PRODUCTIVITY_G_M2_DAY = 35.0   # g dry weight / m² / day (Chisti 2007)

    # Carbon content of microalgae dry biomass
    CARBON_FRACTION = 0.50              # 50% of dry weight is carbon

    # CO₂ absorbed per kg of carbon fixed
    CO2_PER_CARBON = 44.0 / 12.0       # 3.667 kg CO₂ per kg C

    # Therefore: 1 kg dry algae → 0.5 kg C → 1.833 kg CO₂
    CO2_PER_KG_BIOMASS = CARBON_FRACTION * CO2_PER_CARBON  # ≈ 1.833

    # Temperature response (Eppley 1972)
    TEMP_OPTIMAL = 27.0                 # °C
    TEMP_SIGMA = 8.0                    # Standard deviation of Gaussian response

    # Light saturation (Monod kinetics)
    LIGHT_HALF_SAT = 250.0             # W/m² at half-max productivity (Kₛ)

    # pH response
    PH_OPTIMAL = 8.0
    PH_SIGMA = 1.2

    # DO correlation factor
    DO_OPTIMAL = 10.0                  # mg/L
    DO_SIGMA = 4.0

    # Carbon credit pricing (voluntary market, 2024)
    CARBON_CREDIT_USD_PER_TON = 35.0   # $ per metric ton CO₂

    # --- Model Config ---
    N_ESTIMATORS = 300
    MAX_DEPTH = 8
    LEARNING_RATE_XGB = 0.08

    # --- Export ---
    EXPORT_DIR = "./model2_export"
    MODEL_FILENAME = "model2_carbon_regressor.pkl"
    SCALER_FILENAME = "model2_scaler.pkl"
    CONFIG_FILENAME = "model2_config.json"
    METRICS_FILENAME = "model2_metrics.json"

    # Feature and target names
    FEATURE_NAMES = [
        "bloom_area_ha",
        "coverage_pct",
        "severity_index",
        "water_temp_c",
        "ph_level",
        "dissolved_oxygen_mg_l",
        "solar_irradiance_w_m2",
    ]

    TARGET_NAMES = [
        "daily_biomass_kg",
        "daily_co2_kg",
        "monthly_co2_tons",
        "carbon_credit_usd",
    ]

    SEVERITY_NAMES = ["Low", "Moderate", "High"]


config = Config()
os.makedirs(config.EXPORT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("MODEL 2: CARBON SEQUESTRATION REGRESSION ENGINE")
print("=" * 60)
print(f"Samples to generate : {config.NUM_SAMPLES}")
print(f"Train/Val split     : {config.TRAIN_SPLIT * 100:.0f}% / {(1 - config.TRAIN_SPLIT) * 100:.0f}%")
print(f"CO₂ per kg biomass  : {config.CO2_PER_KG_BIOMASS:.3f} kg CO₂")
print(f"Carbon credit rate  : ${config.CARBON_CREDIT_USD_PER_TON}/ton CO₂")


# =========================================================================
# SECTION 3: Physics-Based Synthetic Data Generation
# =========================================================================

print("\n" + "=" * 60)
print("GENERATING PHYSICS-BASED TRAINING DATA")
print("=" * 60)
print("""
  Using peer-reviewed biological models:
    • Eppley (1972): Temperature-dependent growth rate
    • Monod kinetics: Light saturation response
    • Gaussian pH response curve
    • Photosynthetic oxygen correlation
    • Stoichiometric CO₂ fixation: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂
""")

start_time = time.time()


def temperature_response(temp):
    """
    Eppley (1972) temperature-growth curve.
    Gaussian centered at optimal temp with exponential tail.
    Returns a factor between 0.0 and 1.0.
    """
    return np.exp(-0.5 * ((temp - config.TEMP_OPTIMAL) / config.TEMP_SIGMA) ** 2)


def light_response(irradiance):
    """
    Monod / Michaelis-Menten light saturation kinetics.
    P = Pmax * I / (Ks + I)
    Returns a factor between 0.0 and 1.0.
    """
    return irradiance / (config.LIGHT_HALF_SAT + irradiance)


def ph_response(ph):
    """
    Gaussian pH response. Algae grow best at pH 7.5–8.5.
    Strongly inhibited below 6.0 or above 10.0.
    Returns a factor between 0.0 and 1.0.
    """
    return np.exp(-0.5 * ((ph - config.PH_OPTIMAL) / config.PH_SIGMA) ** 2)


def do_factor(do):
    """
    Dissolved oxygen as a proxy for photosynthetic activity.
    High DO indicates active CO₂ fixation. Very low DO indicates
    anaerobic decay (net carbon release, not sequestration).
    Returns a factor between -0.2 and 1.0.
    """
    # Positive contribution from moderate-high DO
    factor = np.exp(-0.5 * ((do - config.DO_OPTIMAL) / config.DO_SIGMA) ** 2)
    # Penalize very low DO (anoxic conditions = decomposition, not growth)
    anoxia_penalty = np.where(do < 2.0, -0.15 * (2.0 - do), 0.0)
    return np.clip(factor + anoxia_penalty, -0.2, 1.0)


def severity_density_multiplier(severity):
    """
    Map severity tier to a cellular density multiplier.
    Higher severity = denser bloom = more biomass per unit area.
      Low (0)     : thin surface film → 0.3x
      Moderate (1): established bloom → 0.7x
      High (2)    : dense pea-soup scum → 1.0x
    """
    mapping = {0: 0.30, 1: 0.70, 2: 1.00}
    return np.array([mapping[int(s)] for s in severity])


def generate_carbon_dataset(n_samples):
    """
    Generate a physics-informed synthetic dataset for carbon
    sequestration regression.

    Each sample represents one daily measurement of a water body
    with an algal bloom, combining visual metrics from Model 1
    with environmental sensor readings.

    Returns:
        X : numpy array (n_samples, 7) — input features
        y : numpy array (n_samples, 4) — target outputs
        df: pandas DataFrame with all columns
    """
    rng = np.random.default_rng(config.RANDOM_STATE)

    # --- Generate Input Features ---
    # Use realistic distributions (not just uniform)

    # Bloom area: log-normal (many small blooms, few giant ones)
    bloom_area = np.clip(
        rng.lognormal(mean=2.5, sigma=1.2, size=n_samples),
        config.AREA_RANGE[0], config.AREA_RANGE[1]
    )

    # Coverage: beta distribution (often partial coverage, rarely 100%)
    coverage = np.clip(
        rng.beta(2.5, 2.0, size=n_samples) * 100.0,
        config.COVERAGE_RANGE[0], config.COVERAGE_RANGE[1]
    )

    # Severity: weighted random (more moderate/high in bloom scenarios)
    severity = rng.choice(
        config.SEVERITY_LEVELS,
        size=n_samples,
        p=[0.25, 0.40, 0.35]
    ).astype(float)

    # Correlate severity with coverage (higher coverage → higher severity)
    # Add a soft correlation: if coverage > 60%, bump severity up
    for i in range(n_samples):
        if coverage[i] > 70 and severity[i] == 0:
            severity[i] = rng.choice([1, 2], p=[0.6, 0.4])
        elif coverage[i] < 20 and severity[i] == 2:
            severity[i] = rng.choice([0, 1], p=[0.5, 0.5])

    # Temperature: seasonal Gaussian (centered at 22°C with wide spread)
    water_temp = np.clip(
        rng.normal(22.0, 7.0, size=n_samples),
        config.TEMP_RANGE[0], config.TEMP_RANGE[1]
    )

    # pH: normal around 7.8 (slightly alkaline freshwater)
    ph = np.clip(
        rng.normal(7.8, 0.9, size=n_samples),
        config.PH_RANGE[0], config.PH_RANGE[1]
    )

    # DO: correlated with temperature (cold water holds more O₂)
    base_do = 14.6 - 0.3 * water_temp + rng.normal(0, 1.5, size=n_samples)
    dissolved_oxygen = np.clip(base_do, config.DO_RANGE[0], config.DO_RANGE[1])

    # Solar irradiance: skewed toward daytime measurements
    solar = np.clip(
        rng.gamma(4.0, 150.0, size=n_samples),
        config.LIGHT_RANGE[0], config.LIGHT_RANGE[1]
    )

    # --- Compute Target Variables Using Biological Models ---

    # Step 1: Base productivity (g / m² / day)
    # Combines all environmental response factors multiplicatively
    f_temp = temperature_response(water_temp)
    f_light = light_response(solar)
    f_ph = ph_response(ph)
    f_do = do_factor(dissolved_oxygen)
    f_severity = severity_density_multiplier(severity)

    # Combined growth factor (0 to 1)
    growth_factor = f_temp * f_light * f_ph * np.clip(f_do, 0.01, 1.0) * f_severity

    # Add biological noise (±15% stochasticity)
    noise = 1.0 + rng.normal(0, 0.15, size=n_samples)
    noise = np.clip(noise, 0.5, 1.5)

    # Productivity: g dry weight / m² / day
    productivity = config.MAX_PRODUCTIVITY_G_M2_DAY * growth_factor * noise

    # Step 2: Scale to entire bloom area
    # Effective algae area = bloom_area × (coverage / 100)
    effective_area_m2 = bloom_area * 10000.0 * (coverage / 100.0)  # ha → m²

    # Daily biomass yield (kg / day)
    daily_biomass_kg = (productivity * effective_area_m2) / 1000.0  # g → kg

    # Step 3: CO₂ sequestration
    # 1 kg dry algae → 0.5 kg C → 1.833 kg CO₂
    daily_co2_kg = daily_biomass_kg * config.CO2_PER_KG_BIOMASS

    # Step 4: Monthly projection (30 days, with 10% degradation factor)
    monthly_co2_kg = daily_co2_kg * 30.0 * 0.90  # 10% loss for variable conditions
    monthly_co2_tons = monthly_co2_kg / 1000.0    # kg → metric tons

    # Step 5: Carbon credit valuation
    carbon_credit_usd = monthly_co2_tons * config.CARBON_CREDIT_USD_PER_TON

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        # Inputs
        "bloom_area_ha": np.round(bloom_area, 2),
        "coverage_pct": np.round(coverage, 2),
        "severity_index": severity.astype(int),
        "water_temp_c": np.round(water_temp, 1),
        "ph_level": np.round(ph, 2),
        "dissolved_oxygen_mg_l": np.round(dissolved_oxygen, 1),
        "solar_irradiance_w_m2": np.round(solar, 1),

        # Intermediate factors (for analysis)
        "f_temperature": np.round(f_temp, 4),
        "f_light": np.round(f_light, 4),
        "f_ph": np.round(f_ph, 4),
        "f_do": np.round(f_do, 4),
        "f_severity": np.round(f_severity, 4),
        "growth_factor": np.round(growth_factor, 4),
        "productivity_g_m2_day": np.round(productivity, 2),

        # Targets
        "daily_biomass_kg": np.round(daily_biomass_kg, 2),
        "daily_co2_kg": np.round(daily_co2_kg, 2),
        "monthly_co2_tons": np.round(monthly_co2_tons, 4),
        "carbon_credit_usd": np.round(carbon_credit_usd, 2),
    })

    X = df[config.FEATURE_NAMES].values
    y = df[config.TARGET_NAMES].values

    return X, y, df


# Generate the dataset
X_all, y_all, df = generate_carbon_dataset(config.NUM_SAMPLES)

gen_time = time.time() - start_time
print(f"Generated {len(df)} samples in {gen_time:.1f}s")
print(f"\nDataset shape: X={X_all.shape}, y={y_all.shape}")
print(f"\nFeature columns: {config.FEATURE_NAMES}")
print(f"Target columns:  {config.TARGET_NAMES}")

# Quick stats
print("\n--- Input Feature Statistics ---")
print(df[config.FEATURE_NAMES].describe().round(2).to_string())

print("\n--- Target Variable Statistics ---")
print(df[config.TARGET_NAMES].describe().round(2).to_string())


# =========================================================================
# SECTION 4: Data Exploration & Visualization
# =========================================================================

print("\n" + "=" * 60)
print("DATA EXPLORATION & VISUALIZATION")
print("=" * 60)

# 4A: Distribution of input features
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
feature_labels = [
    "Bloom Area (ha)", "Coverage (%)", "Severity Index",
    "Water Temp (°C)", "pH Level", "Dissolved O₂ (mg/L)",
    "Solar Irradiance (W/m²)", ""
]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#FFC107", "#607D8B"]

for i, ax in enumerate(axes.flatten()):
    if i < len(config.FEATURE_NAMES):
        name = config.FEATURE_NAMES[i]
        ax.hist(df[name], bins=50, color=colors[i], alpha=0.8, edgecolor="white")
        ax.set_title(feature_labels[i], fontsize=11, fontweight="bold")
        ax.set_xlabel(name, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.axis("off")

# Last subplot already hidden in loop above
plt.suptitle("Distribution of Input Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "feature_distributions.png"), dpi=120)
plt.show()

# 4B: Distribution of target variables
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
target_labels = [
    "Daily Biomass (kg/day)", "Daily CO₂ (kg/day)",
    "Monthly CO₂ (tons/month)", "Carbon Credits ($/month)"
]
target_colors = ["#1B5E20", "#0D47A1", "#4A148C", "#E65100"]

for ax, name, label, color in zip(axes, config.TARGET_NAMES, target_labels, target_colors):
    ax.hist(df[name], bins=50, color=color, alpha=0.8, edgecolor="white")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Distribution of Target Variables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "target_distributions.png"), dpi=120)
plt.show()

# 4C: Environmental response curves
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# Temperature response
temps = np.linspace(5, 40, 200)
axes[0].plot(temps, temperature_response(temps), color="#E91E63", linewidth=2)
axes[0].axvline(config.TEMP_OPTIMAL, ls="--", color="gray", alpha=0.7, label=f"Optimal={config.TEMP_OPTIMAL}°C")
axes[0].set_xlabel("Water Temperature (°C)")
axes[0].set_ylabel("Growth Factor")
axes[0].set_title("Temperature Response\n(Eppley 1972)", fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Light response
lights = np.linspace(50, 1200, 200)
axes[1].plot(lights, light_response(lights), color="#FF9800", linewidth=2)
axes[1].axvline(config.LIGHT_HALF_SAT, ls="--", color="gray", alpha=0.7, label=f"Kₛ={config.LIGHT_HALF_SAT} W/m²")
axes[1].set_xlabel("Solar Irradiance (W/m²)")
axes[1].set_ylabel("Growth Factor")
axes[1].set_title("Light Saturation\n(Monod Kinetics)", fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# pH response
phs = np.linspace(5.5, 10.5, 200)
axes[2].plot(phs, ph_response(phs), color="#9C27B0", linewidth=2)
axes[2].axvline(config.PH_OPTIMAL, ls="--", color="gray", alpha=0.7, label=f"Optimal pH={config.PH_OPTIMAL}")
axes[2].set_xlabel("pH Level")
axes[2].set_ylabel("Growth Factor")
axes[2].set_title("pH Response\n(Gaussian)", fontweight="bold")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

# DO response
dos = np.linspace(0.5, 18, 200)
axes[3].plot(dos, do_factor(dos), color="#00BCD4", linewidth=2)
axes[3].axvline(config.DO_OPTIMAL, ls="--", color="gray", alpha=0.7, label=f"Optimal DO={config.DO_OPTIMAL} mg/L")
axes[3].axhline(0, ls=":", color="red", alpha=0.5)
axes[3].set_xlabel("Dissolved Oxygen (mg/L)")
axes[3].set_ylabel("Growth Factor")
axes[3].set_title("DO Response\n(Photosynthesis Proxy)", fontweight="bold")
axes[3].legend(fontsize=9)
axes[3].grid(alpha=0.3)

plt.suptitle("Biological Response Curves (Scientific Basis for Model 2)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "biological_response_curves.png"), dpi=150)
plt.show()

# 4D: Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr_cols = config.FEATURE_NAMES + config.TARGET_NAMES
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, ax=ax,
    linewidths=0.5, square=True
)
ax.set_title("Feature-Target Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "correlation_matrix.png"), dpi=120)
plt.show()

print("Visualizations saved to:", config.EXPORT_DIR)


# =========================================================================
# SECTION 5: Feature Engineering & Train/Val Split
# =========================================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING & TRAIN/VAL SPLIT")
print("=" * 60)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=1 - config.TRAIN_SPLIT,
    random_state=config.RANDOM_STATE,
)

print(f"Training set  : {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# Scale features (important for gradient-based models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"\nFeature scaling applied (StandardScaler)")
print(f"  Mean: {scaler.mean_.round(2)}")
print(f"  Std:  {scaler.scale_.round(2)}")


# =========================================================================
# SECTION 6: Model Training — XGBoost + Random Forest
# =========================================================================

print("\n" + "=" * 60)
print("⚡ TRAINING REGRESSION MODELS")
print("=" * 60)

models = {}
results = {}

# --- Model A: XGBoost (or GradientBoosting fallback) ---
print("\n--- Training Model A: XGBoost Regressor ---")
train_start = time.time()

if HAS_XGBOOST:
    base_model_a = xgb.XGBRegressor(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        learning_rate=config.LEARNING_RATE_XGB,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=config.RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
else:
    base_model_a = GradientBoostingRegressor(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        learning_rate=config.LEARNING_RATE_XGB,
        subsample=0.8,
        random_state=config.RANDOM_STATE,
    )

# Multi-output wrapper for 4 targets
model_a = MultiOutputRegressor(base_model_a)
model_a.fit(X_train_scaled, y_train)
train_time_a = time.time() - train_start

y_pred_a = model_a.predict(X_val_scaled)
print(f"  Training time: {train_time_a:.1f}s")

# Evaluate per-target
print("\n  Per-Target Metrics (XGBoost):")
for i, name in enumerate(config.TARGET_NAMES):
    r2 = r2_score(y_val[:, i], y_pred_a[:, i])
    mae = mean_absolute_error(y_val[:, i], y_pred_a[:, i])
    rmse = np.sqrt(mean_squared_error(y_val[:, i], y_pred_a[:, i]))
    print(f"    {name:25s} | R²={r2:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f}")

models["xgboost"] = model_a
results["xgboost"] = {
    "r2": [r2_score(y_val[:, i], y_pred_a[:, i]) for i in range(4)],
    "mae": [mean_absolute_error(y_val[:, i], y_pred_a[:, i]) for i in range(4)],
    "train_time": train_time_a,
}


# --- Model B: Random Forest ---
print("\n--- Training Model B: Random Forest Regressor ---")
train_start = time.time()

model_b = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH + 2,   # RF can go slightly deeper
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
)
model_b.fit(X_train_scaled, y_train)
train_time_b = time.time() - train_start

y_pred_b = model_b.predict(X_val_scaled)
print(f"  Training time: {train_time_b:.1f}s")

print("\n  Per-Target Metrics (Random Forest):")
for i, name in enumerate(config.TARGET_NAMES):
    r2 = r2_score(y_val[:, i], y_pred_b[:, i])
    mae = mean_absolute_error(y_val[:, i], y_pred_b[:, i])
    rmse = np.sqrt(mean_squared_error(y_val[:, i], y_pred_b[:, i]))
    print(f"    {name:25s} | R²={r2:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f}")

models["random_forest"] = model_b
results["random_forest"] = {
    "r2": [r2_score(y_val[:, i], y_pred_b[:, i]) for i in range(4)],
    "mae": [mean_absolute_error(y_val[:, i], y_pred_b[:, i]) for i in range(4)],
    "train_time": train_time_b,
}


# --- Pick Best Model ---
avg_r2_a = np.mean(results["xgboost"]["r2"])
avg_r2_b = np.mean(results["random_forest"]["r2"])

best_name = "xgboost" if avg_r2_a >= avg_r2_b else "random_forest"
best_model = models[best_name]
best_r2 = max(avg_r2_a, avg_r2_b)
y_pred_best = y_pred_a if best_name == "xgboost" else y_pred_b

print(f"\n🏆 Best Model: {best_name.upper()}")
print(f"   Average R²: {best_r2:.4f}")
print(f"   Per-target R²: {[round(r, 4) for r in results[best_name]['r2']]}")


# =========================================================================
# SECTION 7: Evaluation & Visualization
# =========================================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION & VISUALIZATION")
print("=" * 60)

# 7A: Predicted vs Actual scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
target_labels = [
    "Daily Biomass (kg/day)", "Daily CO₂ (kg CO₂/day)",
    "Monthly CO₂ (tons/month)", "Carbon Credits ($/month)"
]
scatter_colors = ["#1B5E20", "#0D47A1", "#4A148C", "#E65100"]

for ax, i, label, color in zip(axes.flatten(), range(4), target_labels, scatter_colors):
    actual = y_val[:, i]
    predicted = y_pred_best[:, i]
    r2 = r2_score(actual, predicted)

    ax.scatter(actual, predicted, alpha=0.15, s=8, color=color)

    # Perfect prediction line
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="Perfect Prediction")

    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(f"{label}\nR² = {r2:.4f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle(f"Model 2 ({best_name.upper()}) — Predicted vs Actual", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "predicted_vs_actual.png"), dpi=150)
plt.show()

# 7B: Residual distribution
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, i, label in zip(axes, range(4), target_labels):
    residuals = y_val[:, i] - y_pred_best[:, i]
    ax.hist(residuals, bins=50, color=scatter_colors[i], alpha=0.8, edgecolor="white")
    ax.axvline(0, ls="--", color="red", linewidth=1.5)
    ax.set_title(f"Residuals: {label}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Residual Distributions (Should be centered at 0)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "residual_distributions.png"), dpi=120)
plt.show()

# 7C: Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(config.TARGET_NAMES))
width = 0.35
short_names = ["Biomass", "Daily CO₂", "Monthly CO₂", "Credits"]

# R² comparison
bars1 = axes[0].bar(x - width/2, results["xgboost"]["r2"], width, label="XGBoost", color="#2196F3", alpha=0.85)
bars2 = axes[0].bar(x + width/2, results["random_forest"]["r2"], width, label="Random Forest", color="#4CAF50", alpha=0.85)
axes[0].set_ylabel("R² Score")
axes[0].set_title("R² Score Comparison", fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names)
axes[0].legend()
all_r2 = results["xgboost"]["r2"] + results["random_forest"]["r2"]
axes[0].set_ylim(max(0, min(all_r2) - 0.05), 1.02)
axes[0].grid(axis="y", alpha=0.3)

# MAE comparison
bars3 = axes[1].bar(x - width/2, results["xgboost"]["mae"], width, label="XGBoost", color="#2196F3", alpha=0.85)
bars4 = axes[1].bar(x + width/2, results["random_forest"]["mae"], width, label="Random Forest", color="#4CAF50", alpha=0.85)
axes[1].set_ylabel("Mean Absolute Error")
axes[1].set_title("MAE Comparison (Lower is Better)", fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(short_names)
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("XGBoost vs Random Forest — Model Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "model_comparison.png"), dpi=120)
plt.show()


# =========================================================================
# SECTION 8: Feature Importance Analysis
# =========================================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature importance from each sub-estimator (one per target)
importances = np.zeros(len(config.FEATURE_NAMES))
for estimator in best_model.estimators_:
    importances += estimator.feature_importances_
importances /= len(best_model.estimators_)

# Sort by importance
sorted_idx = np.argsort(importances)
sorted_names = [config.FEATURE_NAMES[i] for i in sorted_idx]
sorted_vals = importances[sorted_idx]

print("\nFeature Importance Ranking:")
for name, val in zip(reversed(sorted_names), reversed(sorted_vals)):
    bar = "█" * int(val * 100)
    print(f"  {name:30s} : {val:.4f}  {bar}")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_names)))
ax.barh(sorted_names, sorted_vals, color=colors_bar, edgecolor="white")
ax.set_xlabel("Mean Feature Importa {best_name.upper()}", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.EXPORT_DIR, "feature_importance.png"), dpi=120)
plt.show()


# =========================================================================
# SECTION 9: Full Inference Function
# =========================================================================

print("\n" + "=" * 60)
print("INFERENCE FUNCTION")
print("=" * 60)


def classify_ecosystem_status(daily_co2, dissolved_oxygen, coverage):
    """
    Classify the ecosystem's carbon sequestration status.

    Returns one of:
      - "Active Carbon Sequestration" : Healthy photosynthesis
      - "Optimal Harvest Window"     : Peak biomass, ready for carbon capture/harvest
      - "Anoxia Risk — Aerate"       : Oxygen depleted, decomposition may release CO₂
      - "Minimal Activity"           : Very low bloom / dormant
    """
    if dissolved_oxygen < 2.0:
        return "⚠️ Anoxia Risk — Aerate Immediately"
    elif daily_co2 > 500 and coverage > 60:
        return "🌿 Optimal Harvest Window"
    elif daily_co2 > 50:
        return "✅ Active Carbon Sequestration"
    else:
        return "💤 Minimal Sequestration Activity"


def run_model2_inference(
    bloom_area_ha,
    coverage_pct,
    severity_index,
    water_temp_c,
    ph_level,
    dissolved_oxygen_mg_l,
    solar_irradiance_w_m2,
):
    """
    Complete Model 2 inference.

    INPUT:
      bloom_area_ha           : float — Algae bloom surface area in hectares (from Model 1)
      coverage_pct            : float — Bloom coverage % of water body (from Model 1)
      severity_index          : int   — 0=Low, 1=Moderate, 2=High (from Model 1)
      water_temp_c            : float — Water temperature in °C
      ph_level                : float — Water pH
      dissolved_oxygen_mg_l   : float — Dissolved oxygen in mg/L
      solar_irradiance_w_m2   : float — Solar irradiance in W/m²

    OUTPUT:
      dict with carbon sequestration metrics:
        daily_biomass_kg      : float — Dry biomass produced per day (kg)
        daily_co2_kg          : float — CO₂ sequestered per day (kg)
        monthly_co2_tons      : float — Projected monthly CO₂ (metric tons)
        carbon_credit_usd     : float — Estimated monthly carbon credit value ($)
        ecosystem_status      : str   — Current status label
        environmental_factors : dict  — Individual growth factor contributions
    """
    # Prepare feature vector
    X_input = np.array([[
        bloom_area_ha,
        coverage_pct,
        severity_index,
        water_temp_c,
        ph_level,
        dissolved_oxygen_mg_l,
        solar_irradiance_w_m2,
    ]])

    # Scale
    X_scaled = scaler.transform(X_input)

    # Predict
    prediction = best_model.predict(X_scaled)[0]

    daily_biomass = max(0, prediction[0])
    daily_co2 = max(0, prediction[1])
    monthly_tons = max(0, prediction[2])
    credit_usd = max(0, prediction[3])

    # Ecosystem status
    status = classify_ecosystem_status(daily_co2, dissolved_oxygen_mg_l, coverage_pct)

    # Individual factor contributions (for explainability)
    factors = {
        "temperature": round(float(temperature_response(water_temp_c)), 4),
        "light": round(float(light_response(solar_irradiance_w_m2)), 4),
        "ph": round(float(ph_response(ph_level)), 4),
        "dissolved_oxygen": round(float(do_factor(dissolved_oxygen_mg_l)), 4),
    }

    return {
        "daily_biomass_kg": round(float(daily_biomass), 2),
        "daily_co2_kg": round(float(daily_co2), 2),
        "monthly_co2_tons": round(float(monthly_tons), 4),
        "carbon_credit_usd": round(float(credit_usd), 2),
        "ecosystem_status": status,
        "environmental_factors": factors,
        "severity_label": config.SEVERITY_NAMES[int(severity_index)],
        "model_name": f"CarbonRegressor-{best_name}-v1.0",
    }


print("✅ run_model2_inference() ready")


# =========================================================================
# SECTION 10: Demo — Run Full Pipeline
# =========================================================================

print("\n" + "=" * 60)
print("DEMO: CARBON SEQUESTRATION PREDICTIONS")
print("=" * 60)

# Test scenarios
scenarios = [
    {
        "name": "🏖️ Small Pond — Low Bloom",
        "inputs": {
            "bloom_area_ha": 2.0,
            "coverage_pct": 15.0,
            "severity_index": 0,
            "water_temp_c": 20.0,
            "ph_level": 7.5,
            "dissolved_oxygen_mg_l": 9.0,
            "solar_irradiance_w_m2": 500.0,
        },
    },
    {
        "name": "🌊 Medium Lake — Moderate Bloom",
        "inputs": {
            "bloom_area_ha": 25.0,
            "coverage_pct": 48.0,
            "severity_index": 1,
            "water_temp_c": 26.0,
            "ph_level": 8.2,
            "dissolved_oxygen_mg_l": 10.5,
            "solar_irradiance_w_m2": 700.0,
        },
    },
    {
        "name": "🟢 Large Reservoir — High Bloom",
        "inputs": {
            "bloom_area_ha": 120.0,
            "coverage_pct": 72.0,
            "severity_index": 2,
            "water_temp_c": 28.0,
            "ph_level": 8.0,
            "dissolved_oxygen_mg_l": 11.0,
            "solar_irradiance_w_m2": 850.0,
        },
    },
    {
        "name": "⚠️ Stagnant Inlet — Anoxia Risk",
        "inputs": {
            "bloom_area_ha": 5.0,
            "coverage_pct": 90.0,
            "severity_index": 2,
            "water_temp_c": 35.0,
            "ph_level": 9.5,
            "dissolved_oxygen_mg_l": 1.2,
            "solar_irradiance_w_m2": 300.0,
        },
    },
]

for scenario in scenarios:
    result = run_model2_inference(**scenario["inputs"])
    print(f"\n{'─' * 50}")
    print(f"  {scenario['name']}")
    print(f"{'─' * 50}")
    print(f"  Bloom Area    : {scenario['inputs']['bloom_area_ha']} ha @ {scenario['inputs']['coverage_pct']}% coverage")
    print(f"  Severity      : {result['severity_label']}")
    print(f"  Water Temp    : {scenario['inputs']['water_temp_c']}°C | pH: {scenario['inputs']['ph_level']} | DO: {scenario['inputs']['dissolved_oxygen_mg_l']} mg/L")
    print(f"  ──────────────────────────────────")
    print(f"  📊 Daily Biomass     : {result['daily_biomass_kg']:,.1f} kg/day")
    print(f"  🌍 Daily CO₂ Captured: {result['daily_co2_kg']:,.1f} kg CO₂/day")
    print(f"  📅 Monthly CO₂       : {result['monthly_co2_tons']:,.2f} metric tons")
    print(f"  💰 Carbon Credits    : ${result['carbon_credit_usd']:,.2f}/month")
    print(f"  🏷️  Status            : {result['ecosystem_status']}")
    print(f"  📈 Growth Factors    : {result['environmental_factors']}")


# =========================================================================
# SECTION 11: Export Model, Scaler & Config
# =========================================================================

print("\n" + "=" * 60)
print("EXPORTING MODEL ARTIFACTS")
print("=" * 60)

# 11A: Save the trained model
model_path = os.path.join(config.EXPORT_DIR, config.MODEL_FILENAME)
if HAS_JOBLIB:
    joblib.dump(best_model, model_path)
else:
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
print(f"✅ Model saved: {model_path}")

# 11B: Save the scaler
scaler_path = os.path.join(config.EXPORT_DIR, config.SCALER_FILENAME)
if HAS_JOBLIB:
    joblib.dump(scaler, scaler_path)
else:
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
print(f"✅ Scaler saved: {scaler_path}")

# 11C: Save config
config_export = {
    "model_name": f"CarbonRegressor-{best_name}-v1.0",
    "model_type": best_name,
    "feature_names": config.FEATURE_NAMES,
    "target_names": config.TARGET_NAMES,
    "severity_names": config.SEVERITY_NAMES,
    "n_estimators": config.N_ESTIMATORS,
    "max_depth": config.MAX_DEPTH,
    "co2_per_kg_biomass": config.CO2_PER_KG_BIOMASS,
    "carbon_credit_usd_per_ton": config.CARBON_CREDIT_USD_PER_TON,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "input_ranges": {
        "bloom_area_ha": list(config.AREA_RANGE),
        "coverage_pct": list(config.COVERAGE_RANGE),
        "severity_index": config.SEVERITY_LEVELS,
        "water_temp_c": list(config.TEMP_RANGE),
        "ph_level": list(config.PH_RANGE),
        "dissolved_oxygen_mg_l": list(config.DO_RANGE),
        "solar_irradiance_w_m2": list(config.LIGHT_RANGE),
    },
    "biological_constants": {
        "max_productivity_g_m2_day": config.MAX_PRODUCTIVITY_G_M2_DAY,
        "carbon_fraction": config.CARBON_FRACTION,
        "co2_per_carbon_ratio": config.CO2_PER_CARBON,
        "temp_optimal_c": config.TEMP_OPTIMAL,
        "ph_optimal": config.PH_OPTIMAL,
        "light_half_saturation_w_m2": config.LIGHT_HALF_SAT,
        "do_optimal_mg_l": config.DO_OPTIMAL,
    },
    "created_at": datetime.now().isoformat(),
}

config_path = os.path.join(config.EXPORT_DIR, config.CONFIG_FILENAME)
with open(config_path, "w") as f:
    json.dump(config_export, f, indent=2)
print(f"✅ Config saved: {config_path}")

# 11D: Save metrics
metrics_export = {
    "best_model": best_name,
    "training_samples": int(X_train.shape[0]),
    "validation_samples": int(X_val.shape[0]),
    "models": {},
}

for model_name, res in results.items():
    metrics_export["models"][model_name] = {
        "avg_r2": round(np.mean(res["r2"]), 4),
        "per_target_r2": {
            name: round(r2, 4) for name, r2 in zip(config.TARGET_NAMES, res["r2"])
        },
        "per_target_mae": {
            name: round(mae, 4) for name, mae in zip(config.TARGET_NAMES, res["mae"])
        },
        "train_time_seconds": round(res["train_time"], 2),
    }

metrics_path = os.path.join(config.EXPORT_DIR, config.METRICS_FILENAME)
with open(metrics_path, "w") as f:
    json.dump(metrics_export, f, indent=2)
print(f"✅ Metrics saved: {metrics_path}")

# 11E: Copy to project root for easy download
import shutil
for filename in [config.MODEL_FILENAME, config.SCALER_FILENAME, config.CONFIG_FILENAME, config.METRICS_FILENAME]:
    src = os.path.join(config.EXPORT_DIR, filename)
    dst = filename
    shutil.copy2(src, dst)
    print(f"  📦 Copied to root: {dst}")


print("\n" + "=" * 60)
print("✅ MODEL 2 TRAINING COMPLETE")
print("=" * 60)
print(f"""
Summary:
  • Best Model       : {best_name.upper()}
  • Average R²       : {best_r2:.4f}
  • Training Samples  : {X_train.shape[0]}
  • Validation Samples: {X_val.shape[0]}

Exported Files:
  • {config.MODEL_FILENAME}   — Trained regressor weights
  • {config.SCALER_FILENAME}  — Feature scaler
  • {config.CONFIG_FILENAME}  — Model configuration & constants
  • {config.METRICS_FILENAME} — Evaluation metrics

Next Steps:
  → Download all 4 files from Colab
  → Place them in your project: ml/weights/
  → Build FastAPI backend to serve both Model 1 + Model 2
""")
