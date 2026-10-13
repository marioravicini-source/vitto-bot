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
