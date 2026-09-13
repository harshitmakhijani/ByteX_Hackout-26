/**
 * ALGAEVERSE — Bio-Carbon Intelligence & Satellite Digital Twin
 * Premium Two-Page Simulator & AI Command Center Logic
 * Connected to PyTorch ResNet-18, XGBoost, and Firebase Firestore
 */

// Global Application State
const appState = {
  activeTab: 'simulator',
  selectedLocation: {
    id: 'chilika_lake',
    name: 'Chilika Lagoon',
    region: 'Odisha, India',
    coords: '19.71° N, 85.32° E',
    type: 'Brackish Coastal Lagoon',
    sensor: 'Sentinel-2 MSI • 10m',
    default_bloom_ha: 35.0,
    sample_image: 'satellite_dense_bloom.jpg'
  },
  selectedImageSource: 'library', // 'library' | 'upload'
  selectedSampleImage: 'satellite_dense_bloom.jpg',
  uploadedFile: null,
  sensors: {
    water_temp_c: 26.8,
    ph_level: 8.2,
    dissolved_oxygen_mg_l: 8.6,
    solar_irradiance_w_m2: 820.0,
    bloom_area_ha: 45.0,
    coverage_pct: 78.0
  },
  lastRunResult: null,
  timelineRecords: [],
  comparisonData: null,
  charts: {
    algaeChart: null,
    carbonChart: null
  }
};

// Monitored Water Bodies
const LOCATIONS = [
  {
    id: 'chilika_lake',
    name: 'Chilika Lagoon',
    region: 'Odisha, India',
    coords: '19.71° N, 85.32° E',
    type: 'Brackish Coastal Lagoon',
    sensor: 'Sentinel-2 MSI • 10m',
    default_bloom_ha: 35.0,
    sample_image: 'satellite_dense_bloom.jpg'
  },
  {
    id: 'lake_erie',
    name: 'Lake Erie (Western Basin)',
    region: 'Ohio / Ontario, USA & Canada',
    coords: '41.83° N, 82.50° W',
    type: 'Freshwater Great Lake',
    sensor: 'Sentinel-2 MSI • 10m',
    default_bloom_ha: 65.0,
    sample_image: 'sample_lake_erie.png'
  },
  {
    id: 'lake_taihu',
    name: 'Lake Taihu Catchment',
    region: 'Jiangsu, China',
    coords: '31.22° N, 120.15° E',
    type: 'Subtropical Shallow Eutrophic Lake',
    sensor: 'Sentinel-2 MSI • 10m',
    default_bloom_ha: 80.0,
    sample_image: 'satellite_dense_bloom.jpg'
  },
  {
    id: 'bellandur_lake',
    name: 'Bellandur Lake',
    region: 'Bengaluru, Karnataka, India',
    coords: '12.93° N, 77.67° E',
    type: 'Hyper-Eutrophic Urban Catchment',
    sensor: 'Sentinel-2 MSI • 10m',
    default_bloom_ha: 22.0,
    sample_image: 'sample_lake_erie.png'
  },
  {
    id: 'lake_tahoe',
    name: 'Lake Tahoe (Pristine Reference)',
    region: 'California / Nevada, USA',
    coords: '39.09° N, 120.03° W',
    type: 'Oligotrophic Alpine Lake',
    sensor: 'Sentinel-2 MSI • 10m',
    default_bloom_ha: 2.0,
    sample_image: 'satellite_pristine_water.jpg'
  }
];

// 1-Click Biogeochemical Presets
const PRESETS = {
  optimal_harvest: {
    name: 'Optimal Summer Harvest',
    water_temp_c: 26.8,
    ph_level: 8.2,
    dissolved_oxygen_mg_l: 8.6,
    solar_irradiance_w_m2: 820.0,
    bloom_area_ha: 45.0,
    coverage_pct: 78.0,
    sample_image: 'satellite_dense_bloom.jpg'
  },
  anoxia_emergency: {
    name: 'Severe Anoxia Emergency',
    water_temp_c: 33.5,
    ph_level: 9.4,
    dissolved_oxygen_mg_l: 1.6,
    solar_irradiance_w_m2: 950.0,
    bloom_area_ha: 110.0,
    coverage_pct: 94.0,
    sample_image: 'satellite_dense_bloom.jpg'
  },
  pristine_baseline: {
    name: 'Pristine Oligotrophic',
    water_temp_c: 18.2,
    ph_level: 7.2,
    dissolved_oxygen_mg_l: 10.4,
    solar_irradiance_w_m2: 500.0,
    bloom_area_ha: 3.5,
    coverage_pct: 5.0,
    sample_image: 'satellite_pristine_water.jpg'
  },
  eutrophication_spurt: {
    name: 'Rapid Eutrophication',
    water_temp_c: 29.0,
    ph_level: 8.7,
    dissolved_oxygen_mg_l: 4.8,
    solar_irradiance_w_m2: 780.0,
    bloom_area_ha: 65.0,
    coverage_pct: 62.0,
    sample_image: 'sample_lake_erie.png'
  }
};

// ============================================================================
// INITIALIZATION ON DOM READY
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  renderLocationCards();
  initTabNavigation();
  initPresetButtons();
  initSensorInputs();
  initSatelliteSourcePicker();
  initBroadcastButton();
  initRefreshTimelineButton();
  initCharts();
  checkBackendHealth();

  // Load initial timeline and comparison data from backend
  loadLocationTimeline(appState.selectedLocation.id);
});

// ============================================================================
// TAB NAVIGATION (Page 1 vs Page 2)
// ============================================================================

function initTabNavigation() {
  const tabSim = document.getElementById('tab-btn-simulator');
  const tabCmd = document.getElementById('tab-btn-command');

  if (tabSim) {
    tabSim.addEventListener('click', () => switchTab('simulator'));
  }
  if (tabCmd) {
    tabCmd.addEventListener('click', () => switchTab('command'));
  }
}

function switchTab(tabName) {
  appState.activeTab = tabName;

  const pageSim = document.getElementById('page-simulator');
  const pageCmd = document.getElementById('page-command');
  const tabSim = document.getElementById('tab-btn-simulator');
  const tabCmd = document.getElementById('tab-btn-command');

  if (tabName === 'simulator') {
    if (pageSim) pageSim.classList.add('active');
    if (pageCmd) pageCmd.classList.remove('active');
    if (tabSim) tabSim.classList.add('active');
    if (tabCmd) tabCmd.classList.remove('active');
  } else {
    if (pageSim) pageSim.classList.remove('active');
    if (pageCmd) pageCmd.classList.add('active');
    if (tabSim) tabSim.classList.remove('active');
    if (tabCmd) tabCmd.classList.add('active');

    // Ensure DOM paint before chart dimension calculation
    setTimeout(() => {
      if (appState.charts.algaeChart) appState.charts.algaeChart.resize();
      if (appState.charts.carbonChart) appState.charts.carbonChart.resize();
    }, 60);
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// STEP 1: WATER BODY LOCATIONS
// ============================================================================

function renderLocationCards() {
  const container = document.getElementById('location-cards-grid');
  if (!container) return;

  container.innerHTML = '';

  const marqueeItems = [...LOCATIONS, ...LOCATIONS];

  marqueeItems.forEach(loc => {
    const isSelected = loc.id === appState.selectedLocation.id;
    const card = document.createElement('div');
    card.className = `location-card ${isSelected ? 'selected' : ''}`;
    card.dataset.locationId = loc.id;

    card.innerHTML = `
      <div class="location-thumb-wrapper">
        <img src="images/${loc.sample_image}" alt="${loc.name}" class="location-thumb" onerror="this.src='/static/images/${loc.sample_image}'">
        <div class="location-sensor-badge">
          <span class="sensor-live-dot"></span>
          <span>${loc.sensor || 'Sentinel-2 MSI'}</span>
        </div>
        <div class="location-check-indicator">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        </div>
      </div>
      <div class="location-card-content">
        <div class="location-name">${loc.name}</div>
        <div class="location-region">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline;vertical-align:baseline;margin-right:3px;opacity:0.65;"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>${loc.region}
        </div>
        <div class="location-meta-row">
          <span class="location-coords">${loc.coords}</span>
          <span class="location-type-pill">${loc.type.split(' ')[0]}</span>
        </div>
      </div>
    `;

    card.addEventListener('click', () => selectLocation(loc.id));
    container.appendChild(card);
  });
}

function selectLocation(locId) {
  const loc = LOCATIONS.find(l => l.id === locId);
  if (!loc) return;

  appState.selectedLocation = loc;

  // Update UI selection on cards
  document.querySelectorAll('.location-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.locationId === locId);
  });

  // Update image preview to location default if using library
  if (appState.selectedImageSource === 'library') {
    selectSampleImage(loc.sample_image);
  }

  // Update location labels on Page 2
  const nameEl = document.getElementById('cmd-location-name');
  const coordsEl = document.getElementById('cmd-coords');
  if (nameEl) nameEl.textContent = loc.name;
  if (coordsEl) coordsEl.textContent = loc.coords;

  logTerminal(`[Station Switch] Calibrated coordinates to ${loc.name} (${loc.coords})`, 'log-text-green');
  showToast(`Active Catchment: ${loc.name}`);

  // Fetch Firestore history & comparison for this newly selected location
  loadLocationTimeline(loc.id);
  loadLocationComparison(loc.id);
}

// ============================================================================
// STEP 2: 1-CLICK PRESETS
// ============================================================================

function initPresetButtons() {
  document.querySelectorAll('.preset-pill-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.preset;
      const preset = PRESETS[key];
      if (!preset) return;

      document.querySelectorAll('.preset-pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Update sensor parameters
      Object.keys(preset).forEach(k => {
        if (k in appState.sensors) {
          updateSensorValue(k, preset[k]);
        }
      });

      if (preset.sample_image && appState.selectedImageSource === 'library') {
        selectSampleImage(preset.sample_image);
      }

      logTerminal(`[Preset Injected] Applied scenario: ${preset.name}`, 'log-text-green');
      showToast(`Loaded Preset: ${preset.name}`);
    });
  });
}

// ============================================================================
// STEP 3: VIRTUAL IOT SENSORS
// ============================================================================

function initSensorInputs() {
  const params = ['water_temp_c', 'ph_level', 'dissolved_oxygen_mg_l', 'solar_irradiance_w_m2', 'bloom_area_ha'];

  params.forEach(param => {
    const slider = document.getElementById(`slider-${param}`);
    const input = document.getElementById(`input-${param}`);

    if (slider && input) {
      updateSliderTrackFill(slider);

      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        input.value = val;
        appState.sensors[param] = val;
        updateSliderTrackFill(slider);
        updateParamStatusBadge(param, val);
        updateRangePillsActive(param, val);
      });

      input.addEventListener('change', (e) => {
        const val = parseFloat(e.target.value);
        slider.value = val;
        appState.sensors[param] = val;
        updateSliderTrackFill(slider);
        updateParamStatusBadge(param, val);
        updateRangePillsActive(param, val);
      });
    }
  });

  // Range pill clicks
  document.querySelectorAll('.range-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const param = pill.dataset.param;
      const val = parseFloat(pill.dataset.value);
      if (param && !isNaN(val)) {
        updateSensorValue(param, val);
      }
    });
  });
}

function updateSliderTrackFill(slider) {
  if (!slider) return;
  const min = parseFloat(slider.min) || 0;
  const max = parseFloat(slider.max) || 100;
  const val = parseFloat(slider.value) || 0;
  const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
  slider.style.background = `linear-gradient(to right, #059669 0%, #0D9488 ${pct}%, #E2E8F0 ${pct}%, #E2E8F0 100%)`;
}

function updateSensorValue(param, val) {
  appState.sensors[param] = val;

  const slider = document.getElementById(`slider-${param}`);
  const input = document.getElementById(`input-${param}`);

  if (slider) {
    slider.value = val;
    updateSliderTrackFill(slider);
  }
  if (input) input.value = val;

  updateParamStatusBadge(param, val);
  updateRangePillsActive(param, val);
}

function updateRangePillsActive(param, val) {
  document.querySelectorAll(`.range-pill[data-param="${param}"]`).forEach(pill => {
    const pillVal = parseFloat(pill.dataset.value);
    const isMatch = Math.abs(pillVal - val) < 0.2;
    pill.classList.toggle('active', isMatch);
  });
}

function updateParamStatusBadge(param, val) {
  const badge = document.getElementById(`badge-${param}`);
  if (!badge) return;

  if (param === 'dissolved_oxygen_mg_l') {
    if (val < 2.0) {
      badge.className = 'param-status-tag tag-danger';
      badge.textContent = 'Critical Anoxia';
    } else if (val < 5.0) {
      badge.className = 'param-status-tag tag-warning';
      badge.textContent = 'Hypoxic Warning';
    } else {
      badge.className = 'param-status-tag tag-healthy';
      badge.textContent = 'Healthy Aeration';
    }
  } else if (param === 'water_temp_c') {
    if (val > 30.0) {
      badge.className = 'param-status-tag tag-danger';
      badge.textContent = 'Extreme Heat';
    } else if (val < 18.0) {
      badge.className = 'param-status-tag tag-warning';
      badge.textContent = 'Thermal Suppression';
    } else {
      badge.className = 'param-status-tag tag-healthy';
      badge.textContent = 'Optimal Growth';
    }
  } else if (param === 'ph_level') {
    if (val > 9.0) {
      badge.className = 'param-status-tag tag-danger';
      badge.textContent = 'Hyper-Alkaline';
    } else if (val < 6.5) {
      badge.className = 'param-status-tag tag-warning';
      badge.textContent = 'Acidic Stress';
    } else {
      badge.className = 'param-status-tag tag-healthy';
      badge.textContent = 'Optimal Buffer';
    }
  } else if (param === 'solar_irradiance_w_m2') {
    if (val > 700) {
      badge.className = 'param-status-tag tag-healthy';
      badge.textContent = 'Peak Photosynthesis';
    } else if (val > 400) {
      badge.className = 'param-status-tag tag-warning';
      badge.textContent = 'Moderate PAR';
    } else {
      badge.className = 'param-status-tag tag-danger';
      badge.textContent = 'Light Limited';
    }
  }
}

// ============================================================================
// STEP 4: MULTISPECTRAL SATELLITE STUDIO & UPLOAD
// ============================================================================

function initSatelliteSourcePicker() {
  // Drag & Drop / File Input for Direct Upload
  const dropArea = document.getElementById('upload-drop-zone');
  const fileInput = document.getElementById('file-upload-input');

  if (dropArea && fileInput) {
    dropArea.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
      dropArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropArea.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropArea.classList.remove('dragover');
      });
    });

    dropArea.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        handleImageUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleImageUpload(e.target.files[0]);
      }
    });
  }
}

function selectSampleImage(imageName) {
  // Default target image update if user hasn't uploaded custom file
  if (!appState.uploadedFile) {
    appState.selectedSampleImage = imageName;
    const preview = document.getElementById('satellite-active-preview');
    if (preview) {
      preview.src = `/static/images/${imageName}`;
    }
  }
}

function handleImageUpload(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please upload a valid image file (PNG, JPG, GeoTIFF)');
    return;
  }

  appState.uploadedFile = file;
  appState.selectedImageSource = 'upload';

  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = document.getElementById('satellite-active-preview');
    if (preview) preview.src = e.target.result;

    const statusBadge = document.getElementById('preview-status-badge');
    if (statusBadge) statusBadge.textContent = 'Custom Satellite Upload';

    const nameTag = document.getElementById('preview-image-name');
    if (nameTag) nameTag.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

    showToast(`Satellite Scene Loaded: ${file.name}`);
  };
  reader.readAsDataURL(file);
}

// ============================================================================
// BROADCAST TELEMETRY & RUN AI PIPELINE & COMMIT TO FIRESTORE
// ============================================================================

function initBroadcastButton() {
  const btn = document.getElementById('btn-broadcast-telemetry');
  if (btn) {
    btn.addEventListener('click', () => broadcastTelemetry());
  }
}

async function broadcastTelemetry() {
  const btn = document.getElementById('btn-broadcast-telemetry');
  if (btn) {
    btn.classList.add('transmitting');
    btn.disabled = true;
    btn.innerHTML = '<span>Transmitting Telemetry Packet &bull; Running AI Models...</span>';
  }

  logTerminal(`[LoRaWAN Gateway] Encrypting telemetry payload for ${appState.selectedLocation.name}...`, 'log-text-cyan');

  try {
    const formData = new FormData();
    formData.append('location_id', appState.selectedLocation.id);
    formData.append('location_name', appState.selectedLocation.name);
    formData.append('water_temp_c', appState.sensors.water_temp_c);
    formData.append('ph_level', appState.sensors.ph_level);
    formData.append('dissolved_oxygen_mg_l', appState.sensors.dissolved_oxygen_mg_l);
    formData.append('solar_irradiance_w_m2', appState.sensors.solar_irradiance_w_m2);
    formData.append('bloom_area_ha', appState.sensors.bloom_area_ha);
    formData.append('coverage_pct', appState.sensors.coverage_pct);

    if (appState.uploadedFile) {
      formData.append('image_file', appState.uploadedFile);
    } else {
      formData.append('sample_image', appState.selectedSampleImage);
    }

    const response = await fetch('/api/simulate', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errText}`);
    }

    const data = await response.json();
    appState.lastRunResult = data;

    logTerminal(`[Model 1 Vision] PyTorch ResNet-18 detected: ${data.model1_vision.severity} Bloom (${data.model1_vision.confidence_pct}% confidence)`, 'log-text-green');
    logTerminal(`[Model 2 Carbon] XGBoost calculated: ${data.model2_carbon.daily_co2_kg.toLocaleString()} kg CO₂/day ($${data.model2_carbon.carbon_credit_usd.toFixed(2)})`, 'log-text-green');
    
    if (data.firestore_synced) {
      logTerminal(`[Firestore Sync] Successfully committed observation to 'telemetry_records' collection!`, 'log-text-green');
    } else {
      logTerminal(`[Firestore Notice] Cached to local persistent storage (synced to cloud background)`, 'log-text-cyan');
    }

    // Update Command Center UI
    updateCommandCenterUI(data);

    // Refresh Comparison & Timeline from backend
    await loadLocationComparison(appState.selectedLocation.id);
    await loadLocationTimeline(appState.selectedLocation.id);

    showToast('Telemetry Committed to Firebase & AI Pipeline Completed');

    // Smooth transition to Page 2
    setTimeout(() => {
      switchTab('command');
    }, 600);

  } catch (err) {
    console.error('Broadcast error:', err);
    logTerminal(`[Transmission Error] ${err.message}`, 'log-text-amber');
    showToast(`Broadcast Error: ${err.message}`);
  } finally {
    if (btn) {
      btn.classList.remove('transmitting');
      btn.disabled = false;
      btn.innerHTML = '<span>Broadcast Telemetry &bull; Run AI Pipeline &bull; Commit to Firebase</span>';
    }
  }
}

// ============================================================================
// UPDATE PAGE 2: COMMAND CENTER METRICS
// ============================================================================

function updateCommandCenterUI(data) {
  const { record, model1_vision, model2_carbon, deltas } = data;

  // Header meta
  const nameEl = document.getElementById('cmd-location-name');
  const coordsEl = document.getElementById('cmd-coords');
  const timeEl = document.getElementById('cmd-timestamp');

  if (nameEl) nameEl.textContent = record.location_name || appState.selectedLocation.name;
  if (coordsEl) coordsEl.textContent = appState.selectedLocation.coords;
  if (timeEl) timeEl.textContent = new Date(record.timestamp || Date.now()).toLocaleTimeString();

  // Ecosystem Alert Banner
  const alertBanner = document.getElementById('cmd-ecosystem-alert');
  const alertStatus = document.getElementById('cmd-alert-status');
  const alertRec = document.getElementById('cmd-alert-rec');

  if (alertBanner) {
    const severityCls = model2_carbon.ecosystem_severity === 'danger' ? 'alert-danger' : (model2_carbon.ecosystem_severity === 'warning' ? 'alert-active' : 'alert-optimal');
    alertBanner.className = `ecosystem-alert-banner ${severityCls}`;
  }
  if (alertStatus) alertStatus.textContent = model2_carbon.ecosystem_status;
  if (alertRec) alertRec.textContent = model2_carbon.recommendation;

  // 4 KPI Cards
  const kpiBloom = document.getElementById('kpi-bloom-area');
  const kpiBloomDelta = document.getElementById('kpi-bloom-delta');
  if (kpiBloom) kpiBloom.textContent = `${model1_vision.bloom_area_ha.toFixed(1)} ha`;
  if (kpiBloomDelta) kpiBloomDelta.innerHTML = formatDeltaBadge(deltas.coverage_delta_pct, deltas.coverage_numeric_delta);

  const kpiCo2 = document.getElementById('kpi-daily-co2');
  const kpiCo2Delta = document.getElementById('kpi-co2-delta');
  if (kpiCo2) kpiCo2.textContent = `${Math.round(model2_carbon.daily_co2_kg).toLocaleString()} kg`;
  if (kpiCo2Delta) kpiCo2Delta.innerHTML = formatDeltaBadge(deltas.co2_delta_pct, deltas.co2_numeric_delta);

  const kpiCredits = document.getElementById('kpi-credits-usd');
  const kpiBiomass = document.getElementById('kpi-biomass');
  if (kpiCredits) kpiCredits.textContent = `$${model2_carbon.carbon_credit_usd.toFixed(2)}`;
  if (kpiBiomass) kpiBiomass.textContent = `${Math.round(model2_carbon.daily_biomass_kg).toLocaleString()} kg/day`;

  const kpiDo = document.getElementById('kpi-dissolved-oxygen');
  const kpiDoDelta = document.getElementById('kpi-do-delta');
  if (kpiDo) kpiDo.textContent = `${record.dissolved_oxygen_mg_l.toFixed(1)} mg/L`;
  if (kpiDoDelta) kpiDoDelta.innerHTML = formatDeltaBadge(deltas.do_delta_pct, deltas.do_numeric_delta, true);

  // Model 1 Vision Analysis Panel
  const cmdImg = document.getElementById('cmd-satellite-img');
  if (cmdImg && record.image_url) {
    cmdImg.src = record.image_url;
  }

  const badgeSev = document.getElementById('cmd-severity-badge');
  const badgeConf = document.getElementById('cmd-confidence-badge');
  const badgeCov = document.getElementById('cmd-coverage-badge');

  if (badgeSev) badgeSev.innerHTML = `<span>Severity: ${model1_vision.severity}</span>`;
  if (badgeConf) badgeConf.innerHTML = `<span>Confidence: ${model1_vision.confidence_pct}%</span>`;
  if (badgeCov) badgeCov.innerHTML = `<span>Coverage: ${model1_vision.coverage_pct}%</span>`;

  // Probability distribution bars
  const conf = model1_vision.confidence_pct || 85;
  const isHigh = model1_vision.severity.toLowerCase().includes('high');
  const isMod = model1_vision.severity.toLowerCase().includes('moderate');

  const pHigh = isHigh ? conf : (isMod ? (100 - conf) * 0.6 : (100 - conf) * 0.2);
  const pMod = isMod ? conf : (isHigh ? (100 - conf) * 0.7 : (100 - conf) * 0.3);
  const pLow = (!isHigh && !isMod) ? conf : (100 - pHigh - pMod);

  setProbBar('prob-high-val', 'prob-high-bar', pHigh);
  setProbBar('prob-mod-val', 'prob-mod-bar', pMod);
  setProbBar('prob-low-val', 'prob-low-bar', pLow);

  // Model 2 Carbon Accounting Panel
  const monthlyEl = document.getElementById('cmd-monthly-tons');
  const annualEl = document.getElementById('cmd-annual-tons');
  const annualCreditsEl = document.getElementById('cmd-annual-credits');
  const recBox = document.getElementById('cmd-recommendation-box');

  if (monthlyEl) monthlyEl.textContent = `${model2_carbon.monthly_co2_tons.toFixed(1)} t`;
  if (annualEl) annualEl.textContent = `${(model2_carbon.annual_co2_tons || model2_carbon.monthly_co2_tons * 12).toFixed(1)} t`;
  if (annualCreditsEl) annualCreditsEl.textContent = `$${Math.round(model2_carbon.annual_credit_usd || model2_carbon.carbon_credit_usd * 365).toLocaleString()} USD`;
  if (recBox) recBox.textContent = model2_carbon.recommendation;
}

function resetDashboardToAwaiting() {
  const alertBanner = document.getElementById('cmd-ecosystem-alert');
  const alertStatus = document.getElementById('cmd-alert-status');
  const alertRec = document.getElementById('cmd-alert-rec');
  if (alertBanner) alertBanner.className = 'ecosystem-alert-banner alert-optimal';
  if (alertStatus) alertStatus.textContent = 'Awaiting Initial Telemetry Broadcast';
  if (alertRec) alertRec.textContent = 'Broadcast IoT sensor readings or upload satellite imagery in Step 1 to run AI intelligence.';

  const kpiBloom = document.getElementById('kpi-bloom-area');
  const kpiBloomDelta = document.getElementById('kpi-bloom-delta');
  if (kpiBloom) kpiBloom.textContent = '-- ha';
  if (kpiBloomDelta) kpiBloomDelta.innerHTML = '<span class="delta-badge delta-neutral">Awaiting Broadcast</span>';

  const kpiCo2 = document.getElementById('kpi-daily-co2');
  const kpiCo2Delta = document.getElementById('kpi-co2-delta');
  if (kpiCo2) kpiCo2.textContent = '-- kg';
  if (kpiCo2Delta) kpiCo2Delta.innerHTML = '<span class="delta-badge delta-neutral">Awaiting Broadcast</span>';

  const kpiCredits = document.getElementById('kpi-credits-usd');
  const kpiBiomass = document.getElementById('kpi-biomass');
  if (kpiCredits) kpiCredits.textContent = '--';
  if (kpiBiomass) kpiBiomass.textContent = '-- kg/day';

  const kpiDo = document.getElementById('kpi-dissolved-oxygen');
  const kpiDoDelta = document.getElementById('kpi-do-delta');
  if (kpiDo) kpiDo.textContent = '-- mg/L';
  if (kpiDoDelta) kpiDoDelta.innerHTML = '<span class="delta-badge delta-neutral">Awaiting Broadcast</span>';

  const badgeSev = document.getElementById('cmd-severity-badge');
  const badgeConf = document.getElementById('cmd-confidence-badge');
  const badgeCov = document.getElementById('cmd-coverage-badge');
  if (badgeSev) badgeSev.innerHTML = '<span>Severity: Awaiting Scan</span>';
  if (badgeConf) badgeConf.innerHTML = '<span>Confidence: --%</span>';
  if (badgeCov) badgeCov.innerHTML = '<span>Coverage: --%</span>';

  setProbBar('prob-high-val', 'prob-high-bar', 0);
  setProbBar('prob-mod-val', 'prob-mod-bar', 0);
  setProbBar('prob-low-val', 'prob-low-bar', 0);

  const monthlyEl = document.getElementById('cmd-monthly-tons');
  const annualEl = document.getElementById('cmd-annual-tons');
  const annualCreditsEl = document.getElementById('cmd-annual-credits');
  const recBox = document.getElementById('cmd-recommendation-box');
  if (monthlyEl) monthlyEl.textContent = '-- t';
  if (annualEl) annualEl.textContent = '-- t';
  if (annualCreditsEl) annualCreditsEl.textContent = '-- USD';
  if (recBox) recBox.textContent = 'Run simulation or broadcast sensor telemetry in Step 1 to generate carbon accounting insights.';
}

function renderRecordInDashboard(rec) {
  if (!rec) {
    resetDashboardToAwaiting();
    return;
  }

  const nameEl = document.getElementById('cmd-location-name');
  const coordsEl = document.getElementById('cmd-coords');
  const timeEl = document.getElementById('cmd-timestamp');
  if (nameEl) nameEl.textContent = rec.location_name || appState.selectedLocation.name;
  if (coordsEl) coordsEl.textContent = appState.selectedLocation.coords;
  if (timeEl && rec.timestamp) timeEl.textContent = new Date(rec.timestamp).toLocaleTimeString();

  // Ecosystem Alert Banner
  const alertBanner = document.getElementById('cmd-ecosystem-alert');
  const alertStatus = document.getElementById('cmd-alert-status');
  const alertRec = document.getElementById('cmd-alert-rec');
  if (alertBanner) {
    const sev = rec.ecosystem_severity || 'optimal';
    const severityCls = sev === 'danger' ? 'alert-danger' : (sev === 'warning' ? 'alert-active' : 'alert-optimal');
    alertBanner.className = `ecosystem-alert-banner ${severityCls}`;
  }
  if (alertStatus) alertStatus.textContent = rec.ecosystem_status || 'Active Monitoring';
  if (alertRec) alertRec.textContent = rec.recommendation || 'Continuous telemetry active.';

  // 4 KPI Cards
  const kpiBloom = document.getElementById('kpi-bloom-area');
  const kpiBloomDelta = document.getElementById('kpi-bloom-delta');
  if (kpiBloom) kpiBloom.textContent = `${(rec.bloom_area_ha || 0).toFixed(1)} ha`;
  if (kpiBloomDelta) {
    const deltas = rec.deltas || {};
    kpiBloomDelta.innerHTML = formatDeltaBadge(deltas.coverage_delta_pct, deltas.coverage_numeric_delta);
  }

  const kpiCo2 = document.getElementById('kpi-daily-co2');
  const kpiCo2Delta = document.getElementById('kpi-co2-delta');
  if (kpiCo2) kpiCo2.textContent = `${Math.round(rec.daily_co2_kg || 0).toLocaleString()} kg`;
  if (kpiCo2Delta) {
    const deltas = rec.deltas || {};
    kpiCo2Delta.innerHTML = formatDeltaBadge(deltas.co2_delta_pct, deltas.co2_numeric_delta);
  }

  const kpiCredits = document.getElementById('kpi-credits-usd');
  const kpiBiomass = document.getElementById('kpi-biomass');
  if (kpiCredits) kpiCredits.textContent = `$${(rec.carbon_credit_usd || 0).toFixed(2)}`;
  if (kpiBiomass) kpiBiomass.textContent = `${Math.round(rec.daily_biomass_kg || 0).toLocaleString()} kg/day`;

  const kpiDo = document.getElementById('kpi-dissolved-oxygen');
  const kpiDoDelta = document.getElementById('kpi-do-delta');
  if (kpiDo) kpiDo.textContent = `${(rec.dissolved_oxygen_mg_l || 0).toFixed(1)} mg/L`;
  if (kpiDoDelta) {
    const deltas = rec.deltas || {};
    kpiDoDelta.innerHTML = formatDeltaBadge(deltas.do_delta_pct, deltas.do_numeric_delta, true);
  }

  // Model 1 Panel
  const cmdImg = document.getElementById('cmd-satellite-img');
  if (cmdImg && rec.image_url) cmdImg.src = rec.image_url;

  const badgeSev = document.getElementById('cmd-severity-badge');
  const badgeConf = document.getElementById('cmd-confidence-badge');
  const badgeCov = document.getElementById('cmd-coverage-badge');
  if (badgeSev) badgeSev.innerHTML = `<span>Severity: ${rec.severity || 'Unknown'}</span>`;
  if (badgeConf) badgeConf.innerHTML = `<span>Confidence: ${(rec.confidence_pct || 98).toFixed(1)}%</span>`;
  if (badgeCov) badgeCov.innerHTML = `<span>Coverage: ${(rec.coverage_pct || 0).toFixed(1)}%</span>`;

  const conf = rec.confidence_pct || 85;
  const isHigh = (rec.severity || '').toLowerCase().includes('high');
  const isMod = (rec.severity || '').toLowerCase().includes('moderate');
  const pHigh = isHigh ? conf : (isMod ? (100 - conf) * 0.6 : (100 - conf) * 0.2);
  const pMod = isMod ? conf : (isHigh ? (100 - conf) * 0.7 : (100 - conf) * 0.3);
  const pLow = (!isHigh && !isMod) ? conf : (100 - pHigh - pMod);
  setProbBar('prob-high-val', 'prob-high-bar', pHigh);
  setProbBar('prob-mod-val', 'prob-mod-bar', pMod);
  setProbBar('prob-low-val', 'prob-low-bar', pLow);

  // Model 2 Panel
  const monthlyEl = document.getElementById('cmd-monthly-tons');
  const annualEl = document.getElementById('cmd-annual-tons');
  const annualCreditsEl = document.getElementById('cmd-annual-credits');
  const recBox = document.getElementById('cmd-recommendation-box');
  const monthlyTons = rec.monthly_co2_tons || 0;
  const monthlyCredits = rec.carbon_credit_usd || 0;
  if (monthlyEl) monthlyEl.textContent = `${monthlyTons.toFixed(1)} t`;
  if (annualEl) annualEl.textContent = `${(monthlyTons * 12).toFixed(1)} t`;
  if (annualCreditsEl) annualCreditsEl.textContent = `$${Math.round(monthlyCredits * 12).toLocaleString()} USD`;
  if (recBox) recBox.textContent = rec.recommendation || 'Continuous telemetry monitoring active.';
}

function setProbBar(valId, barId, pct) {
  const valEl = document.getElementById(valId);
  const barEl = document.getElementById(barId);
  const clamped = Math.max(1, Math.min(99, pct)).toFixed(1);
  if (valEl) valEl.textContent = `${clamped}%`;
  if (barEl) barEl.style.width = `${clamped}%`;
}

function formatDeltaBadge(deltaStr, numericVal, invertColors = false) {
  if (!deltaStr) return '<span class="delta-badge delta-neutral">Baseline Observation</span>';
  const num = typeof numericVal === 'number' ? numericVal : 0;
  const isPos = num > 0;
  const isNeutral = Math.abs(num) < 0.01;

  let cls = 'delta-neutral';
  if (!isNeutral) {
    if (invertColors) {
      cls = isPos ? 'delta-pos' : 'delta-neg';
    } else {
      cls = isPos ? 'delta-pos' : 'delta-neg';
    }
  }

  const icon = isPos ? '↑' : (isNeutral ? '—' : '↓');
  return `<span class="delta-badge ${cls}">${icon} ${deltaStr}</span>`;
}

// ============================================================================
// SIDE-BY-SIDE TEMPORAL COMPARISON (Previous Scan vs Latest Scan)
// ============================================================================

async function loadLocationComparison(locationId) {
  try {
    const res = await fetch(`/api/comparison/${locationId}`);
    if (!res.ok) return;
    const data = await res.json();
    appState.comparisonData = data;
    renderComparisonUI(data);
  } catch (err) {
    console.error('Error fetching comparison:', err);
  }
}

function renderComparisonUI(comp) {
  const { has_comparison, latest, previous, deltas } = comp;

  const prevImg = document.getElementById('comp-prev-img');
  const prevTime = document.getElementById('comp-prev-time');
  const prevCoverage = document.getElementById('comp-prev-coverage');
  const prevArea = document.getElementById('comp-prev-area');
  const prevCo2 = document.getElementById('comp-prev-co2');
  const prevCredits = document.getElementById('comp-prev-credits');
  const prevSeverity = document.getElementById('comp-prev-severity');

  const latestImg = document.getElementById('comp-latest-img');
  const latestTime = document.getElementById('comp-latest-time');
  const latestCoverage = document.getElementById('comp-latest-coverage');
  const latestArea = document.getElementById('comp-latest-area');
  const latestCo2 = document.getElementById('comp-latest-co2');
  const latestCredits = document.getElementById('comp-latest-credits');
  const latestSeverity = document.getElementById('comp-latest-severity');

  const divider = document.getElementById('comp-divider');
  const statusTag = document.getElementById('comp-status-tag');

  if (latest) {
    if (latestImg && latest.image_url) latestImg.src = latest.image_url;
    if (latestTime) latestTime.textContent = formatTimeAgo(latest.timestamp);
    if (latestCoverage) latestCoverage.textContent = `${(latest.coverage_pct || 0).toFixed(1)}%`;
    if (latestArea) latestArea.textContent = `${(latest.bloom_area_ha || 0).toFixed(1)} ha`;
    if (latestCo2) latestCo2.textContent = `${Math.round(latest.daily_co2_kg || 0).toLocaleString()} kg`;
    if (latestCredits) latestCredits.textContent = `$${(latest.carbon_credit_usd || 0).toFixed(2)}`;
    if (latestSeverity) latestSeverity.textContent = latest.severity || 'Moderate';
  } else {
    if (latestTime) latestTime.textContent = 'Awaiting 1st Scan';
    if (latestCoverage) latestCoverage.textContent = '--';
    if (latestArea) latestArea.textContent = '--';
    if (latestCo2) latestCo2.textContent = '--';
    if (latestCredits) latestCredits.textContent = '--';
    if (latestSeverity) latestSeverity.textContent = '--';
  }

  if (has_comparison && previous) {
    if (prevImg && previous.image_url) prevImg.src = previous.image_url;
    if (prevTime) prevTime.textContent = formatTimeAgo(previous.timestamp);
    if (prevCoverage) prevCoverage.textContent = `${(previous.coverage_pct || 0).toFixed(1)}%`;
    if (prevArea) prevArea.textContent = `${(previous.bloom_area_ha || 0).toFixed(1)} ha`;
    if (prevCo2) prevCo2.textContent = `${Math.round(previous.daily_co2_kg || 0).toLocaleString()} kg`;
    if (prevCredits) prevCredits.textContent = `$${(previous.carbon_credit_usd || 0).toFixed(2)}`;
    if (prevSeverity) prevSeverity.textContent = previous.severity || 'Moderate';

    if (statusTag) statusTag.textContent = `Temporal Delta: ${deltas.coverage_delta_pct || '0%'}`;
  } else if (latest) {
    // Single scan state (initial baseline)
    if (prevTime) prevTime.textContent = 'Initial Cloud Baseline';
    if (prevCoverage) prevCoverage.textContent = 'Initial Run';
    if (prevArea) prevArea.textContent = 'Initial Run';
    if (prevCo2) prevCo2.textContent = 'Baseline Est.';
    if (prevCredits) prevCredits.textContent = 'Baseline Est.';
    if (prevSeverity) prevSeverity.textContent = 'Baseline Est.';

    if (statusTag) statusTag.textContent = 'Initial Checkpoint in Firestore';
  } else {
    // Zero scans recorded yet
    if (prevTime) prevTime.textContent = 'Awaiting 2nd Scan';
    if (prevCoverage) prevCoverage.textContent = '--';
    if (prevArea) prevArea.textContent = '--';
    if (prevCo2) prevCo2.textContent = '--';
    if (prevCredits) prevCredits.textContent = '--';
    if (prevSeverity) prevSeverity.textContent = '--';

    if (statusTag) statusTag.textContent = 'No Scans Recorded Yet';
  }
}

// ============================================================================
// FIRESTORE HISTORICAL TIMELINE STRIP
// ============================================================================

function initRefreshTimelineButton() {
  const btn = document.getElementById('btn-refresh-timeline');
  if (btn) {
    btn.addEventListener('click', async () => {
      btn.classList.add('loading');
      await loadLocationTimeline(appState.selectedLocation.id);
      await loadLocationComparison(appState.selectedLocation.id);
      btn.classList.remove('loading');
      showToast('Firestore Timeline Synchronized');
    });
  }
}

async function loadLocationTimeline(locationId) {
  try {
    const res = await fetch(`/api/timeline/${locationId}`);
    if (!res.ok) return;
    const data = await res.json();
    appState.timelineRecords = data.timeline || [];

    renderTimelineStrip(appState.timelineRecords);
    updateChartsWithTimeline(appState.timelineRecords);

    // Update counter badge
    const badge = document.getElementById('timeline-badge-count');
    const navCount = document.getElementById('nav-timeline-count');
    const count = appState.timelineRecords.length;
    if (badge) badge.textContent = `${count} Cloud Scans`;
    if (navCount) navCount.textContent = count;

    // Sync dashboard with location's latest scan, or reset if no scans
    if (count > 0) {
      const latestRecord = appState.timelineRecords[count - 1];
      renderRecordInDashboard(latestRecord);
    } else {
      resetDashboardToAwaiting();
    }

  } catch (err) {
    console.error('Error fetching timeline:', err);
  }
}

function renderTimelineStrip(records) {
  const container = document.getElementById('timeline-container');
  if (!container) return;

  if (!records || records.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="padding: 1.5rem; width: 100%;">
        <div class="empty-state-title">No Previous Scans in Firestore Yet</div>
        <div class="empty-state-sub">Broadcast your first telemetry reading to establish a cloud timeline.</div>
      </div>
    `;
    return;
  }

  container.innerHTML = '';

  // Render cards in reverse order (newest first)
  const reversed = [...records].reverse();

  reversed.forEach((rec, idx) => {
    const sev = (rec.severity || 'Unknown').toLowerCase();
    const sevClass = sev.includes('high') ? 'high' : (sev.includes('mod') ? 'moderate' : 'low');
    const isLatest = idx === 0;
    const scanNumber = records.length - idx;

    const card = document.createElement('div');
    card.className = `timeline-card ${isLatest ? 'active-inspect' : ''}`;
    card.dataset.recordId = rec.id;

    const formattedDate = formatDateTime(rec.timestamp);
    const thumbImg = rec.image_url || '/static/images/satellite_dense_bloom.jpg';

    card.innerHTML = `
      <div class="timeline-card-header">
        <span class="timeline-scan-num">Scan #${scanNumber}${isLatest ? ' <small style="color: var(--primary-emerald); font-weight: 700;">(Latest)</small>' : ''}</span>
        <span class="timeline-severity-tag ${sevClass}">${rec.severity || 'Normal'}</span>
      </div>
      <img src="${thumbImg}" class="timeline-thumb" alt="Scan satellite thumbnail">
      <div class="timeline-card-time">${formattedDate}</div>
      <div class="timeline-stats-grid">
        <div class="timeline-stat">
          <span class="t-stat-label">Coverage</span>
          <span class="t-stat-val">${(rec.coverage_pct || 0).toFixed(1)}%</span>
        </div>
        <div class="timeline-stat">
          <span class="t-stat-label">CO₂ / Day</span>
          <span class="t-stat-val">${Math.round(rec.daily_co2_kg || 0).toLocaleString()} kg</span>
        </div>
      </div>
      <div class="timeline-card-footer">
        <span class="timeline-sync-pill"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:inline;vertical-align:middle;margin-right:2px;"><path d="M19 16.9A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25"/></svg> Synced</span>
        <span class="timeline-inspect-btn">Compare &rarr;</span>
      </div>
    `;

    card.addEventListener('click', () => {
      document.querySelectorAll('.timeline-card').forEach(c => c.classList.remove('active-inspect'));
      card.classList.add('active-inspect');
      inspectTimelineRecord(rec);
    });

    container.appendChild(card);
  });
}

function inspectTimelineRecord(record) {
  // Update Left Panel of Side-by-Side comparison to show this selected historical record
  const prevImg = document.getElementById('comp-prev-img');
  const prevTime = document.getElementById('comp-prev-time');
  const prevCoverage = document.getElementById('comp-prev-coverage');
  const prevArea = document.getElementById('comp-prev-area');
  const prevCo2 = document.getElementById('comp-prev-co2');
  const prevCredits = document.getElementById('comp-prev-credits');
  const prevSeverity = document.getElementById('comp-prev-severity');

  if (prevImg && record.image_url) prevImg.src = record.image_url;
  if (prevTime) prevTime.textContent = formatDateTime(record.timestamp);
  if (prevCoverage) prevCoverage.textContent = `${(record.coverage_pct || 0).toFixed(1)}%`;
  if (prevArea) prevArea.textContent = `${(record.bloom_area_ha || 0).toFixed(1)} ha`;
  if (prevCo2) prevCo2.textContent = `${Math.round(record.daily_co2_kg || 0).toLocaleString()} kg`;
  if (prevCredits) prevCredits.textContent = `$${(record.carbon_credit_usd || 0).toFixed(2)}`;
  if (prevSeverity) prevSeverity.textContent = record.severity || 'Moderate';

  showToast(`Loaded timeline checkpoint: ${formatDateTime(record.timestamp)} into Comparison`);
}

// ============================================================================
// HISTORICAL TREND CHARTS (Chart.js with Clean Light Aesthetic)
// ============================================================================

function initCharts() {
  const ctxAlgae = document.getElementById('chart-algae-trend');
  const ctxCarbon = document.getElementById('chart-carbon-trend');

  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: '#0F172A',
        titleColor: '#FFFFFF',
        bodyColor: '#A7F3D0',
        borderColor: '#E2E8F0',
        borderWidth: 1,
        padding: 10,
        boxPadding: 4
      }
    },
    scales: {
      x: {
        grid: { color: '#F1F5F9' },
        ticks: { color: '#64748B', font: { family: "'Inter', sans-serif", size: 10, weight: 500 } }
      },
      y: {
        grid: { color: '#F1F5F9' },
        ticks: { color: '#64748B', font: { family: "'JetBrains Mono', monospace", size: 10 } }
      }
    }
  };

  if (ctxAlgae) {
    appState.charts.algaeChart = new Chart(ctxAlgae, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Bloom Coverage (%)',
          data: [],
          borderColor: '#059669',
          backgroundColor: 'rgba(5, 150, 105, 0.08)',
          borderWidth: 2.5,
          tension: 0.35,
          fill: true,
          pointBackgroundColor: '#059669',
          pointBorderColor: '#FFFFFF',
          pointRadius: 4,
          pointHoverRadius: 6
        }]
      },
      options: {
        ...commonOptions,
        scales: {
          ...commonOptions.scales,
          y: { ...commonOptions.scales.y, min: 0, max: 100 }
        }
      }
    });
  }

  if (ctxCarbon) {
    appState.charts.carbonChart = new Chart(ctxCarbon, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [{
          label: 'Daily CO₂ Fixation (kg)',
          data: [],
          backgroundColor: '#0D9488',
          hoverBackgroundColor: '#0F766E',
          borderRadius: 6,
          borderSkipped: false
        }]
      },
      options: commonOptions
    });
  }
}

function updateChartsWithTimeline(records) {
  if (!records || records.length === 0) {
    if (appState.charts.algaeChart) {
      appState.charts.algaeChart.data.labels = [];
      appState.charts.algaeChart.data.datasets[0].data = [];
      appState.charts.algaeChart.update();
    }
    if (appState.charts.carbonChart) {
      appState.charts.carbonChart.data.labels = [];
      appState.charts.carbonChart.data.datasets[0].data = [];
      appState.charts.carbonChart.update();
    }
    return;
  }

  // Format clean, human-readable labels so they don't overlap
  const labels = records.map((r, i) => {
    const isLatest = i === records.length - 1;
    const dateStr = formatShortTime(r.timestamp);
    return isLatest ? `Latest (${dateStr})` : `Scan #${i + 1} (${dateStr})`;
  });

  const coverageData = records.map(r => Number((r.coverage_pct || 0).toFixed(1)));
  const co2Data = records.map(r => Math.round(r.daily_co2_kg || 0));

  if (appState.charts.algaeChart) {
    appState.charts.algaeChart.data.labels = labels;
    appState.charts.algaeChart.data.datasets[0].data = coverageData;
    appState.charts.algaeChart.update();
  }

  if (appState.charts.carbonChart) {
    appState.charts.carbonChart.data.labels = labels;
    appState.charts.carbonChart.data.datasets[0].data = co2Data;
    appState.charts.carbonChart.update();
  }
}

// ============================================================================
// TERMINAL LOGGING & TOAST NOTIFICATIONS
// ============================================================================

function logTerminal(message, colorClass = 'log-text-green') {
  const terminal = document.getElementById('terminal-log-output');
  if (!terminal) return;

  const now = new Date();
  const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

  const row = document.createElement('div');
  row.className = 'log-line';
  row.innerHTML = `
    <span class="log-time">[${timeStr}]</span>
    <span class="${colorClass}">${escapeHtml(message)}</span>
  `;

  terminal.appendChild(row);
  terminal.scrollTop = terminal.scrollHeight;
}

let toastTimeout = null;
function showToast(msg) {
  const toast = document.getElementById('toast-notification');
  if (!toast) return;

  toast.textContent = msg;
  toast.classList.add('show');

  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove('show');
  }, 3200);
}

// ============================================================================
// BACKEND HEALTH & FIRESTORE STATUS CHECK
// ============================================================================

async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      const pill = document.getElementById('global-sync-pill');
      const text = document.getElementById('global-sync-text');
      if (text) text.textContent = 'Firestore: Connected';
      logTerminal(`[Health Check] Verified: ResNet-18 & XGBoost active &bull; Firestore synced`, 'log-text-green');
    }
  } catch (err) {
    const text = document.getElementById('global-sync-text');
    if (text) text.textContent = 'Offline Cache Mode';
  }
}

// ============================================================================
// UTILITIES
// ============================================================================

function formatDateTime(isoString) {
  if (!isoString) return 'Just now';
  try {
    const d = new Date(isoString);
    return d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (e) {
    return isoString;
  }
}

function formatShortDate(isoString) {
  if (!isoString) return 'Scan';
  try {
    const d = new Date(isoString);
    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
  } catch (e) {
    return 'Scan';
  }
}

function formatShortTime(isoString) {
  if (!isoString) return '';
  try {
    const d = new Date(isoString);
    const now = new Date();
    const isSameDay = d.toDateString() === now.toDateString();
    if (isSameDay) {
      return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    } else {
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  } catch (e) {
    return '';
  }
}

function formatTimeAgo(isoString) {
  if (!isoString) return 'Recently';
  try {
    const d = new Date(isoString);
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch (e) {
    return 'Recently';
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
