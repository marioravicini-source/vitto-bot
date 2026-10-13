"""
garmin_bridge.py — Puente Garmin Connect → Nightscout
=====================================================
Pollea Garmin Connect cada 15 minutos para detectar nuevas actividades
y las registra automáticamente en Nightscout como tratamientos de Exercise.

Requiere:
  - garminconnect (pip install garminconnect)
  - Variables de entorno:
      GARMIN_EMAIL     — email de la cuenta Garmin Connect
      GARMIN_PASSWORD  — contraseña de Garmin Connect

Seguridad: NO calcula ni sugiere dosis. Solo registra ejercicio.
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("garmin_bridge")

# ---------- lazy import (no rompe el bot si no está instalada) ----------
Garmin = None  # type: ignore
try:
    from garminconnect import Garmin as _Garmin
    Garmin = _Garmin
except ImportError:
    logger.warning("garminconnect no instalada — bridge Garmin deshabilitado")

# ---------- config ----------
GARMIN_EMAIL = os.environ.get("GARMIN_EMAIL", "")
GARMIN_PASSWORD = os.environ.get("GARMIN_PASSWORD", "")
POLL_INTERVAL_MIN = 15
TOKEN_COLLECTION = "garmin_tokens"
SYNCED_COLLECTION = "garmin_synced"
DAILY_DATA_COLLECTION = "garmin_daily"

# Mapeo de tipos de actividad Garmin → texto legible
ACTIVITY_TYPE_MAP = {
    "running": "Correr",
    "cycling": "Ciclismo",
    "swimming": "Natación",
    "walking": "Caminata",
    "strength_training": "Pesas/Gym",
    "cardio_training": "Cardio",
    "yoga": "Yoga",
    "hiking": "Senderismo",
    "other": "Ejercicio",
    "ice_skating": "Hockey/Patinaje",
    "inline_skating": "Rollers",
    "hockey": "Hockey",
    "ice_hockey": "Hockey sobre hielo",
    "multi_sport": "Multi-deporte",
    "fitness_equipment": "Gym",
    "elliptical": "Elíptica",
    "stair_climbing": "Escalera",
    "indoor_cardio": "Cardio indoor",
    "indoor_cycling": "Ciclismo indoor",
    "treadmill_running": "Cinta",
    "lap_swimming": "Natación (pileta)",
    "open_water_swimming": "Natación (abierta)",
}


def _get_activity_label(activity: dict) -> str:
    """Devuelve un label legible para el tipo de actividad."""
    type_key = activity.get("activityType", {})
    if isinstance(type_key, dict):
        type_key = type_key.get("typeKey", "other")
    type_key = str(type_key).lower().replace(" ", "_")
    return ACTIVITY_TYPE_MAP.get(type_key, activity.get("activityName", "Ejercicio"))


class GarminBridge:
    """Maneja la conexión a Garmin Connect y sincronización a Nightscout."""

    def __init__(self, db=None):
        self.api = None
        self.db = db
        self._authenticated = False

    def is_configured(self) -> bool:
        """Chequea si las credenciales están configuradas."""
        return bool(Garmin and GARMIN_EMAIL and GARMIN_PASSWORD)

    async def authenticate(self) -> bool:
        """Autentica con Garmin Connect. Intenta reusar tokens guardados."""
        if not self.is_configured():
            return False

        try:
            self.api = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)

            # Intentar cargar tokens guardados de MongoDB
            saved_tokens = None
            if self.db is not None:
                doc = self.db[TOKEN_COLLECTION].find_one({"_id": "garmin_session"})
                if doc and "tokens" in doc:
                    saved_tokens = doc["tokens"]
                    logger.info("Tokens Garmin cargados de MongoDB")

            if saved_tokens:
                try:
                    self.api.login(saved_tokens)
                    self._authenticated = True
                    logger.info("Login Garmin con tokens guardados OK")
                    return True
                except Exception:
                    logger.info("Tokens guardados expirados, re-login...")

            # Login fresco con credenciales
            self.api.login()
            self._authenticated = True
            logger.info("Login Garmin con credenciales OK")

            # Guardar tokens para reusar
            self._save_tokens()
            return True

        except Exception as e:
            logger.error(f"Auth Garmin falló: {type(e).__name__}: {e}")
            self._authenticated = False
            return False

    def _save_tokens(self):
        """Guarda tokens de sesión en MongoDB."""
        if self.db is None or self.api is None:
            return
        try:
            tokens = self.api.garth.dumps()
            self.db[TOKEN_COLLECTION].replace_one(
                {"_id": "garmin_session"},
                {"_id": "garmin_session", "tokens": tokens, "updated": datetime.now(timezone.utc)},
                upsert=True,
            )
            logger.info("Tokens Garmin guardados en MongoDB")
        except Exception as e:
            logger.warning(f"No se pudieron guardar tokens: {e}")

    def _is_already_synced(self, activity_id: int) -> bool:
        """Chequea si una actividad ya fue sincronizada."""
        if self.db is None:
            return False
        return self.db[SYNCED_COLLECTION].find_one({"_id": activity_id}) is not None

    def _mark_synced(self, activity_id: int, activity_name: str):
        """Marca una actividad como sincronizada."""
        if self.db is None:
            return
        self.db[SYNCED_COLLECTION].replace_one(
            {"_id": activity_id},
            {
                "_id": activity_id,
                "name": activity_name,
                "synced_at": datetime.now(timezone.utc),
            },
            upsert=True,
        )

    def get_recent_activities(self, hours: int = 1) -> list:
        """Obtiene actividades de las últimas N horas."""
        if not self._authenticated or self.api is None:
            return []

        try:
            end = datetime.now()
            start = end - timedelta(hours=hours)
            activities = self.api.get_activities_by_date(
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            # Refrescar tokens después de cada llamada exitosa
            self._save_tokens()
            return activities or []
        except Exception as e:
            logger.error(f"Error obteniendo actividades Garmin: {e}")
            # Intentar re-auth si falla
            self._authenticated = False
            return []

    # ─── Sueño ─────────────────────────────────────────────────
    def get_sleep_data(self, date_str: Optional[str] = None) -> Optional[dict]:
        """Obtiene datos de sueño de una fecha (YYYY-MM-DD). Default: hoy."""
        if not self._authenticated or self.api is None:
            return None
        try:
            if date_str is None:
                date_str = datetime.now().strftime("%Y-%m-%d")
            raw = self.api.get_sleep_data(date_str)
            self._save_tokens()
            if not raw:
                return None

            # Extraer métricas clave del sueño
            daily = raw.get("dailySleepDTO", {})
            if not daily:
                return None

            total_sec = daily.get("sleepTimeSeconds", 0)
            deep_sec = daily.get("deepSleepSeconds", 0)
            light_sec = daily.get("lightSleepSeconds", 0)
            rem_sec = daily.get("remSleepSeconds", 0)
            awake_sec = daily.get("awakeSleepSeconds", 0)

            def _to_hm(sec):
                if not sec:
                    return "0h 0m"
                return f"{sec // 3600}h {(sec % 3600) // 60}m"

            # Horarios
            start_ms = daily.get("sleepStartTimestampGMT")
            end_ms = daily.get("sleepEndTimestampGMT")
            start_time = None
            end_time = None
            if start_ms:
                start_time = datetime.fromtimestamp(start_ms / 1000).strftime("%H:%M")
            if end_ms:
                end_time = datetime.fromtimestamp(end_ms / 1000).strftime("%H:%M")

            result = {
                "date": date_str,
                "total": _to_hm(total_sec),
                "total_min": round(total_sec / 60) if total_sec else 0,
                "deep": _to_hm(deep_sec),
                "light": _to_hm(light_sec),
                "rem": _to_hm(rem_sec),
                "awake": _to_hm(awake_sec),
                "start": start_time,
                "end": end_time,
                "score": daily.get("sleepScores", {}).get("overall", {}).get("value"),
            }
            return result

        except Exception as e:
            logger.error(f"Error obteniendo sueño Garmin: {e}")
            return None

    def format_sleep_note(self, sleep: dict) -> str:
        """Formatea datos de sueño como texto legible."""
        parts = [f"😴 Sueño: {sleep['total']}"]
        if sleep.get("start") and sleep.get("end"):
            parts.append(f"({sleep['start']} → {sleep['end']})")
        details = []
        if sleep.get("deep") and sleep["deep"] != "0h 0m":
            details.append(f"profundo {sleep['deep']}")
        if sleep.get("rem") and sleep["rem"] != "0h 0m":
            details.append(f"REM {sleep['rem']}")
        if sleep.get("light") and sleep["light"] != "0h 0m":
            details.append(f"liviano {sleep['light']}")
        if details:
            parts.append("| " + ", ".join(details))
        if sleep.get("score"):
            parts.append(f"| puntuación {sleep['score']}/100")
        return " ".join(parts)

    # ─── Pasos ─────────────────────────────────────────────────
    def get_steps_data(self, date_str: Optional[str] = None) -> Optional[dict]:
        """Obtiene datos de pasos de una fecha (YYYY-MM-DD). Default: hoy."""
        if not self._authenticated or self.api is None:
            return None
        try:
            if date_str is None:
                date_str = datetime.now().strftime("%Y-%m-%d")
            raw = self.api.get_steps_data(date_str)
            self._save_tokens()
            if not raw:
                return None

            # get_steps_data devuelve una lista; tomamos el primer elemento del día
            day = raw[0] if isinstance(raw, list) and raw else raw
            if isinstance(day, dict):
                steps = day.get("totalSteps", day.get("steps", 0))
                distance_m = day.get("totalDistance", day.get("distance", 0))
                calories = day.get("totalCalories", day.get("calories", 0))
            else:
                return None

            return {
                "date": date_str,
                "steps": steps or 0,
                "distance_km": round((distance_m or 0) / 1000, 1),
                "calories": calories or 0,
            }

        except Exception as e:
            logger.error(f"Error obteniendo pasos Garmin: {e}")
            return None

    def format_steps_note(self, steps: dict) -> str:
        """Formatea datos de pasos como texto legible."""
        parts = [f"🚶 Pasos: {steps['steps']:,}".replace(",", ".")]
        if steps.get("distance_km"):
            parts.append(f"({steps['distance_km']} km)")
        if steps.get("calories"):
            parts.append(f"| {steps['calories']} kcal")
        return " ".join(parts)

    # ─── Frecuencia cardíaca y estrés ─────────────────────────
    def get_heart_rate_data(self, date_str: Optional[str] = None) -> Optional[dict]:
        """Obtiene datos de frecuencia cardíaca del día."""
        if not self._authenticated or self.api is None:
            return None
        try:
            if date_str is None:
                date_str = datetime.now().strftime("%Y-%m-%d")
            raw = self.api.get_heart_rates(date_str)
            self._save_tokens()
            if not raw:
                return None

            result = {
                "date": date_str,
                "resting_hr": raw.get("restingHeartRate"),
                "min_hr": raw.get("minHeartRate"),
                "max_hr": raw.get("maxHeartRate"),
            }

            # Zonas de FC (si disponibles)
            zones = raw.get("heartRateTimeInZones", [])
            if zones:
                zone_summary = []
                for z in zones:
                    name = z.get("zoneName", "")
                    mins = round(z.get("secsInZone", 0) / 60)
                    if mins > 0:
                        zone_summary.append({"zone": name, "minutes": mins})
                result["zones"] = zone_summary

            return result

        except Exception as e:
            logger.error(f"Error obteniendo FC Garmin: {e}")
            return None

    def get_stress_data(self, date_str: Optional[str] = None) -> Optional[dict]:
        """Obtiene datos de estrés del día."""
        if not self._authenticated or self.api is None:
            return None
        try:
            if date_str is None:
                date_str = datetime.now().strftime("%Y-%m-%d")
            raw = self.api.get_stress_data(date_str)
            self._save_tokens()
            if not raw:
                return None

            return {
                "date": date_str,
                "avg_stress": raw.get("avgStressLevel"),
                "max_stress": raw.get("maxStressLevel"),
                "high_stress_min": round(raw.get("highStressDuration", 0) / 60),
                "medium_stress_min": round(raw.get("mediumStressDuration", 0) / 60),
                "low_stress_min": round(raw.get("lowStressDuration", 0) / 60),
                "rest_stress_min": round(raw.get("restStressDuration", 0) / 60),
            }

        except Exception as e:
            logger.error(f"Error obteniendo estrés Garmin: {e}")
            return None

    def get_body_battery(self, date_str: Optional[str] = None) -> Optional[dict]:
        """Obtiene datos de Body Battery del día."""
        if not self._authenticated or self.api is None:
            return None
        try:
            if date_str is None:
                date_str = datetime.now().strftime("%Y-%m-%d")
            raw = self.api.get_body_battery(date_str)
            self._save_tokens()
            if not raw:
                return None

            # Body battery viene como lista de lecturas; tomamos min/max/última
            readings = raw if isinstance(raw, list) else raw.get("bodyBatteryValuesArray", [])
            if not readings:
                return None

            values = []
            for r in readings:
                if isinstance(r, dict):
                    v = r.get("bodyBatteryLevel")
                elif isinstance(r, (list, tuple)) and len(r) >= 2:
                    v = r[1]
                else:
                    continue
                if v is not None and v > 0:
                    values.append(v)

            if not values:
                return None

            return {
                "date": date_str,
                "current": values[-1],
                "min": min(values),
                "max": max(values),
            }

        except Exception as e:
            logger.error(f"Error obteniendo Body Battery: {e}")
            return None

    def format_vitals_note(self, hr: Optional[dict], stress: Optional[dict],
                           battery: Optional[dict]) -> str:
        """Formatea datos vitales como texto legible."""
        parts = []
        if hr and hr.get("resting_hr"):
            parts.append(f"❤️ FC reposo: {hr['resting_hr']} bpm")
            if hr.get("max_hr"):
                parts.append(f"(máx {hr['max_hr']})")
        if stress and stress.get("avg_stress"):
            level = "bajo"
            avg = stress["avg_stress"]
            if avg > 50:
                level = "alto"
            elif avg > 35:
                level = "medio"
            parts.append(f"| 🧘 Estrés: {avg} ({level})")
        if battery and battery.get("current"):
            parts.append(f"| 🔋 Body Battery: {battery['current']}/100")
        return " ".join(parts) if parts else ""

    # ─── Datos diarios consolidados ────────────────────────────
    def get_daily_summary(self, date_str: Optional[str] = None) -> dict:
        """Obtiene resumen diario: sueño + pasos + FC + estrés + body battery."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        summary = {"date": date_str}
        summary["sleep"] = self.get_sleep_data(date_str)
        summary["steps"] = self.get_steps_data(date_str)
        summary["heart_rate"] = self.get_heart_rate_data(date_str)
        summary["stress"] = self.get_stress_data(date_str)
        summary["body_battery"] = self.get_body_battery(date_str)

        # Guardar en MongoDB para historial
        if self.db is not None:
            try:
                self.db[DAILY_DATA_COLLECTION].replace_one(
                    {"_id": date_str},
                    {"_id": date_str, **summary, "updated": datetime.now(timezone.utc)},
                    upsert=True,
                )
            except Exception as e:
                logger.warning(f"No pude guardar datos diarios: {e}")

        return summary

    def get_sleep_history(self, days: int = 7) -> list:
        """Obtiene historial de sueño de los últimos N días desde MongoDB."""
        if self.db is None:
            return []
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            docs = list(self.db[DAILY_DATA_COLLECTION].find(
                {"_id": {"$gte": start.strftime("%Y-%m-%d")}},
                sort=[("_id", -1)],
            ))
            return [d.get("sleep") for d in docs if d.get("sleep")]
        except Exception as e:
            logger.error(f"Error leyendo historial sueño: {e}")
            return []

    def format_for_nightscout(self, activity: dict) -> Optional[dict]:
        """Convierte una actividad Garmin en un tratamiento Nightscout."""
        activity_id = activity.get("activityId")
        if not activity_id:
            return None

        # Ya sincronizada?
        if self._is_already_synced(activity_id):
            return None

        label = _get_activity_label(activity)
        duration_min = round(activity.get("duration", 0) / 60)
        if duration_min < 1:
            return None

        # Datos adicionales
        avg_hr = activity.get("averageHR")
        max_hr = activity.get("maxHR")
        calories = activity.get("calories")
        distance_km = None
        if activity.get("distance"):
            distance_km = round(activity["distance"] / 1000, 2)

        # Construir nota descriptiva
        parts = [f"🏃 {label} ({duration_min} min)"]
        if avg_hr:
            parts.append(f"FC avg {avg_hr}")
        if max_hr:
            parts.append(f"max {max_hr}")
        if distance_km:
            parts.append(f"{distance_km} km")
        if calories:
            parts.append(f"{calories} kcal")
        parts.append("[auto-Garmin]")
        notes = " | ".join(parts)

        # Timestamp de inicio de la actividad
        start_time = activity.get("startTimeGMT") or activity.get("startTimeLocal")
        if start_time and not start_time.endswith("Z"):
            # startTimeGMT ya es UTC
            pass

        treatment = {
            "eventType": "Exercise",
            "duration": duration_min,
            "notes": notes,
            "enteredBy": "GarminBridge",
        }
        if start_time:
            treatment["created_at"] = start_time

        return treatment

    def mark_activity_synced(self, activity: dict):
        """Marca la actividad como sincronizada después de postearla."""
        activity_id = activity.get("activityId")
        name = activity.get("activityName", "")
        if activity_id:
            self._mark_synced(activity_id, name)

    def get_status(self) -> dict:
        """Devuelve estado del bridge para diagnóstico."""
        synced_count = 0
        last_sync = None
        if self.db is not None:
            synced_count = self.db[SYNCED_COLLECTION].count_documents({})
            last_doc = self.db[SYNCED_COLLECTION].find_one(
                sort=[("synced_at", -1)]
            )
            if last_doc:
                last_sync = last_doc.get("synced_at")

        return {
            "configured": self.is_configured(),
            "authenticated": self._authenticated,
            "activities_synced": synced_count,
            "last_sync": str(last_sync) if last_sync else "nunca",
        }
