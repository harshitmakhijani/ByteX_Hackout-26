"""
Firestore & Telemetry History Service
Stores simulation & field sensor logs in Firebase Firestore and calculates
historical deltas (% change in algae coverage, % change in CO2, % change in DO).
Includes local JSON persistence fallback for 100% reliability.
"""

import os
import json
import time
import datetime
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DB_PATH = os.path.join(BASE_DIR, "backend", "telemetry_db.json")

FIREBASE_PROJECT_ID = "hackout26-d49d5"
FIREBASE_API_KEY = "AIzaSyCpJG81mL0M3wZo_gWytANAoCn7CZHnPYc"
FIRESTORE_URL = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/telemetry_records"


class FirestoreTelemetryService:
    def __init__(self):
        self.local_db = self._load_local_db()

    def _load_local_db(self):
        if os.path.exists(LOCAL_DB_PATH):
            try:
                with open(LOCAL_DB_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading local telemetry DB: {e}")
        
        return {}

    def _save_local_db(self):
        try:
            with open(LOCAL_DB_PATH, "w") as f:
                json.dump(self.local_db, f, indent=2)
        except Exception as e:
            print(f"Error saving local DB: {e}")

    def clear_all(self):
        """Purges all in-memory, local JSON, and cloud Firestore records."""
        self.local_db = {}
        self._save_local_db()
        try:
            resp = requests.get(f"{FIRESTORE_URL}?key={FIREBASE_API_KEY}&pageSize=300", timeout=5)
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                for doc in docs:
                    name = doc.get("name")
                    requests.delete(f"https://firestore.googleapis.com/v1/{name}?key={FIREBASE_API_KEY}", timeout=5)
        except Exception as e:
            print(f"Error purging Firestore: {e}")
        return True

    def calculate_deltas(self, current_data, previous_data):
        """Computes % change and trend badges between consecutive scans."""
        if not previous_data:
            return {
                "has_previous": False,
                "coverage_delta_pct": "+0.0%",
                "co2_delta_pct": "+0.0%",
                "do_delta_pct": "+0.0%",
                "summary": "Baseline observation established",
                "trend_badge": "Baseline Monitoring Station",
                "coverage_numeric_delta": 0.0,
                "co2_numeric_delta": 0.0
            }

        prev_cov = float(previous_data.get("coverage_pct", 1.0))
        curr_cov = float(current_data.get("coverage_pct", 1.0))
        cov_delta = ((curr_cov - prev_cov) / max(0.1, prev_cov)) * 100.0

        prev_co2 = float(previous_data.get("daily_co2_kg", 1.0))
        curr_co2 = float(current_data.get("daily_co2_kg", 1.0))
        co2_delta = ((curr_co2 - prev_co2) / max(0.1, prev_co2)) * 100.0

        prev_do = float(previous_data.get("dissolved_oxygen_mg_l", 1.0))
        curr_do = float(current_data.get("dissolved_oxygen_mg_l", 1.0))
        do_delta = ((curr_do - prev_do) / max(0.1, prev_do)) * 100.0

        # Badges and summary logic
        cov_sign = "+" if cov_delta >= 0 else ""
        co2_sign = "+" if co2_delta >= 0 else ""
        do_sign = "+" if do_delta >= 0 else ""

        if cov_delta > 15.0:
            badge = f"{cov_sign}{cov_delta:.1f}% Bloom Expansion"
        elif cov_delta < -15.0:
            badge = f"{cov_delta:.1f}% Bloom Regression"
        else:
            badge = f"Stable Algae Population ({cov_sign}{cov_delta:.1f}%)"

        if do_delta < -20.0 or curr_do < 2.5:
            summary = f"Critical DO drop ({do_sign}{do_delta:.1f}%). High risk of anoxia."
        elif co2_delta > 10.0:
            summary = f"Carbon sequestration surging by {co2_sign}{co2_delta:.1f}%."
        else:
            summary = f"Algae coverage changed by {cov_sign}{cov_delta:.1f}% compared to previous scan."

        return {
            "has_previous": True,
            "coverage_delta_pct": f"{cov_sign}{cov_delta:.1f}%",
            "co2_delta_pct": f"{co2_sign}{co2_delta:.1f}%",
            "do_delta_pct": f"{do_sign}{do_delta:.1f}%",
            "coverage_numeric_delta": round(cov_delta, 1),
            "co2_numeric_delta": round(co2_delta, 1),
            "do_numeric_delta": round(do_delta, 1),
            "summary": summary,
            "trend_badge": badge
        }

    def record_telemetry(self, record_data):
        """
        Saves record to local DB and attempts sync to Firebase Firestore.
        Returns record enriched with historical deltas.
        """
        self.local_db = self._load_local_db()
        location_id = record_data.get("location_id", "custom_location")
        history = self.local_db.setdefault(location_id, [])

        # Get most recent previous record if exists
        previous_record = history[-1] if len(history) > 0 else None

        # Calculate deltas
        deltas = self.calculate_deltas(record_data, previous_record)
        record_data["deltas"] = deltas

        # Append to history
        history.append(record_data)
        self._save_local_db()

        # Sync to Firebase Firestore REST API in background/try-catch
        firestore_synced = self._sync_to_firestore(record_data)
        record_data["firestore_synced"] = firestore_synced

        return record_data

    def _sync_to_firestore(self, record_data):
        """Pushes structured document to Firebase Firestore REST API."""
        try:
            doc_id = record_data.get("id", f"telemetry_{int(time.time())}")
            fields = {
                "location_id": {"stringValue": str(record_data.get("location_id"))},
                "location_name": {"stringValue": str(record_data.get("location_name"))},
                "timestamp": {"stringValue": str(record_data.get("timestamp"))},
                "water_temp_c": {"doubleValue": float(record_data.get("water_temp_c", 0))},
                "ph_level": {"doubleValue": float(record_data.get("ph_level", 0))},
                "dissolved_oxygen_mg_l": {"doubleValue": float(record_data.get("dissolved_oxygen_mg_l", 0))},
                "solar_irradiance_w_m2": {"doubleValue": float(record_data.get("solar_irradiance_w_m2", 0))},
                "coverage_pct": {"doubleValue": float(record_data.get("coverage_pct", 0))},
                "bloom_area_ha": {"doubleValue": float(record_data.get("bloom_area_ha", 0))},
                "severity": {"stringValue": str(record_data.get("severity", "Unknown"))},
                "daily_co2_kg": {"doubleValue": float(record_data.get("daily_co2_kg", 0))},
                "monthly_co2_tons": {"doubleValue": float(record_data.get("monthly_co2_tons", 0))},
                "carbon_credit_usd": {"doubleValue": float(record_data.get("carbon_credit_usd", 0))},
                "ecosystem_status": {"stringValue": str(record_data.get("ecosystem_status", "Active"))}
            }

            url = f"{FIRESTORE_URL}?documentId={doc_id}&key={FIREBASE_API_KEY}"
            resp = requests.post(url, json={"fields": fields}, timeout=3)
            if resp.status_code in [200, 201]:
                print(f"[Firestore] Telemetry record {doc_id} synced to Firebase Firestore")
                return True
            else:
                print(f"Firestore sync response: {resp.status_code} - {resp.text[:100]}")
                return False
        except Exception as e:
            print(f"Firestore sync notice: {e} (saved to local DB successfully)")
            return False

    def get_history(self, location_id, limit=10):
        """Returns historical scans for a location."""
        self.local_db = self._load_local_db()
        history = self.local_db.get(location_id, [])
        return history[-limit:]

    def fetch_from_firestore(self, location_id=None):
        """Reads telemetry records from Firebase Firestore REST API."""
        try:
            url = f"{FIRESTORE_URL}?key={FIREBASE_API_KEY}"
            if location_id:
                url += f'&orderBy=fields.location_id&where={{"fieldFilter":{{"field":{{"fieldPath":"location_id"}},"op":"EQUAL","value":{{"stringValue":"{location_id}"}}}}}}'
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                documents = data.get("documents", [])
                records = []
                for doc in documents:
                    fields = doc.get("fields", {})
                    record = {}
                    for key, val_obj in fields.items():
                        if "stringValue" in val_obj:
                            record[key] = val_obj["stringValue"]
                        elif "doubleValue" in val_obj:
                            record[key] = float(val_obj["doubleValue"])
                        elif "integerValue" in val_obj:
                            record[key] = int(val_obj["integerValue"])
                    records.append(record)
                return records
            return []
        except Exception as e:
            print(f"Firestore fetch notice: {e}")
            return []

    def get_timeline(self, location_id, limit=20):
        """Returns full timeline of scans for a location, enriched for dashboard display."""
        self.local_db = self._load_local_db()
        history = self.local_db.get(location_id, [])
        timeline = []
        for record in history[-limit:]:
            timeline.append({
                "id": record.get("id", ""),
                "timestamp": record.get("timestamp", ""),
                "location_id": record.get("location_id", location_id),
                "location_name": record.get("location_name", ""),
                "image_url": record.get("image_url", ""),
                "severity": record.get("severity", "Unknown"),
                "confidence_pct": record.get("confidence_pct", 0),
                "coverage_pct": record.get("coverage_pct", 0),
                "bloom_area_ha": record.get("bloom_area_ha", 0),
                "daily_co2_kg": record.get("daily_co2_kg", 0),
                "monthly_co2_tons": record.get("monthly_co2_tons", 0),
                "carbon_credit_usd": record.get("carbon_credit_usd", 0),
                "daily_biomass_kg": record.get("daily_biomass_kg", 0),
                "water_temp_c": record.get("water_temp_c", 0),
                "ph_level": record.get("ph_level", 0),
                "dissolved_oxygen_mg_l": record.get("dissolved_oxygen_mg_l", 0),
                "solar_irradiance_w_m2": record.get("solar_irradiance_w_m2", 0),
                "ecosystem_status": record.get("ecosystem_status", ""),
                "ecosystem_severity": record.get("ecosystem_severity", "neutral"),
                "recommendation": record.get("recommendation", ""),
                "firestore_synced": record.get("firestore_synced", False),
            })
        return timeline

    def get_comparison(self, location_id):
        """Returns latest and previous records for side-by-side comparison."""
        self.local_db = self._load_local_db()
        history = self.local_db.get(location_id, [])
        latest = history[-1] if len(history) >= 1 else None
        previous = history[-2] if len(history) >= 2 else None
        
        deltas = self.calculate_deltas(latest, previous) if latest else {}
        
        return {
            "has_comparison": previous is not None,
            "latest": latest,
            "previous": previous,
            "deltas": deltas,
            "total_scans": len(history)
        }


_firestore_service = None

def get_firestore_service():
    global _firestore_service
    if _firestore_service is None:
        _firestore_service = FirestoreTelemetryService()
    return _firestore_service
