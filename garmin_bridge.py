"""
garmin_bridge.py — Puente Garmin Connect → Nightscout (API personal)
====================================================================
Lee datos de Garmin Connect usando la API PERSONAL oficial (clave API →
token de acceso), en vez de usuario/contraseña. Esto evita el bloqueo de
login web/SSO de las cuentas de menor.

- Registra actividades nuevas en Nightscout como tratamientos de Exercise.
- Recolecta sueño, pasos, FC, estrés y Body Battery para el resumen diario.

Requiere:
  - garmin-connect-personal-api-python (pip)
  - Variable de entorno:
      GARMIN_PERSONAL_API_KEY — clave API personal creada en Garmin Connect
        (Configuración → Clave API personal). Se cambia por un token de acceso.

Seguridad: NO calcula ni sugiere dosis. Solo lee y registra ejercicio/vitales.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("garmin_bridge")

# ---------- lazy import (no rompe el bot si no está instalada) ----------
GarminPersonalApiClient = None  # type: ignore
PersonalApiAuthenticator = None  # type: ignore
try:
    from garmin_personal_api import (  # type: ignore
        GarminPersonalApiClient as _Client,
        PersonalApiAuthenticator as _Auth,
    )
    GarminPersonalApiClient = _Client
    PersonalApiAuthenticator = _Auth
except Exception as _imp_err:  # ImportError u otros
    logger.warning("garmin_personal_api no disponible — bridge Garmin deshabilitado (%s)", _imp_err)

# ---------- config ----------
GARMIN_PERSONAL_API_KEY = os.environ.get("GARMIN_PERSONAL_API_KEY", "")
POLL_INTERVAL_MIN = 15
TZ_AR = timezone(timedelta(hours=-3))  # Argentina (UTC-3): la fecha "de hoy" es la local, no la UTC del servidor
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


def _secs_to_hm(sec) -> str:
    sec = int(sec or 0)
    if sec <= 0:
        return "0h 0m"
    return f"{sec // 3600}h {(sec % 3600) // 60}m"


class GarminBridge:
    """Conexión a la API personal de Garmin Connect y sync a Nightscout."""

    def __init__(self, db=None):
        self.db = db
        self._authenticated = False
        self._access_token: Optional[str] = None
        self._token_expires_at = None
        self._logged_raw: set = set()

    # ---------- configuración / auth ----------
    def is_configured(self) -> bool:
        """Chequea si la librería y la clave API personal están disponibles."""
        return bool(GarminPersonalApiClient and PersonalApiAuthenticator and GARMIN_PERSONAL_API_KEY)

    async def authenticate(self) -> bool:
        """Cambia la clave API personal por un token de acceso."""
        if not self.is_configured():
            return False
        try:
            token = await asyncio.to_thread(self._obtain_token)
            self._access_token = token.access_token
            self._token_expires_at = getattr(token, "expires_at", None)
            self._authenticated = True
            logger.info("Garmin Personal API: token OK (expira: %s)", self._token_expires_at)
            return True
        except Exception as e:
            logger.error("Auth Garmin (API personal) falló: %s: %s", type(e).__name__, e)
            self._authenticated = False
            return False

    def _obtain_token(self):
        """Obtiene el bearer token desde la clave API (bloqueante)."""
        with PersonalApiAuthenticator() as auth:
            return auth.obtain_token(GARMIN_PERSONAL_API_KEY)

    def _client(self):
        """Devuelve un cliente de la API personal con el token actual."""
        return GarminPersonalApiClient(self._access_token)

    def _log_raw_once(self, kind: str, raw) -> None:
        """Loguea (una vez por tipo) la respuesta cruda, para ajustar el parseo."""
        if kind in self._logged_raw:
            return
        self._logged_raw.add(kind)
        try:
            logger.info("GARMIN raw[%s]: %s", kind, str(raw)[:900])
        except Exception:
            pass

    def _note_auth_error(self, e: Exception) -> None:
        """Si el error parece de token vencido/invalidado, forzar re-auth."""
        if "401" in str(e) or "Unauthorized" in str(e) or "token" in str(e).lower():
            self._authenticated = False

    # ---------- dedup de actividades ----------
    def _is_already_synced(self, activity_id) -> bool:
        if self.db is None:
            return False
        return self.db[SYNCED_COLLECTION].find_one({"_id": activity_id}) is not None

    def _mark_synced(self, activity_id, activity_name: str):
        if self.db is None:
            return
        self.db[SYNCED_COLLECTION].replace_one(
            {"_id": activity_id},
            {"_id": activity_id, "name": activity_name, "synced_at": datetime.now(timezone.utc)},
            upsert=True,
        )

    # ---------- actividades ----------
    def get_recent_activities(self, hours: int = 1) -> list:
        """Obtiene actividades recientes (del rango de días que cubre `hours`)."""
        if not self._authenticated or not self._access_token:
            return []
        try:
            end = datetime.now(TZ_AR)
            start = end - timedelta(hours=hours)
            with self._client() as client:
                raw = client.search_activities(
                    start_date=start.strftime("%Y-%m-%d"),
                    end_date=end.strftime("%Y-%m-%d"),
                    limit=20,
                )
            self._log_raw_once("activities", raw)
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict):
                return raw.get("activities") or raw.get("results") or raw.get("activityList") or []
            return []
        except Exception as e:
            logger.error("Error obteniendo actividades Garmin: %s", e)
            self._note_auth_error(e)
            return []

    # ---------- sueño ----------
    def get_sleep_data(self, date_str: Optional[str] = None) -> Optional[dict]:
        if not self._authenticated or not self._access_token:
            return None
        try:
            if date_str is None:
                date_str = datetime.now(TZ_AR).strftime("%Y-%m-%d")
            with self._client() as client:
                raw = client.get_daily_sleep(date_str)
            self._log_raw_once("sleep", raw)
            if not raw:
                return None

            # La respuesta puede venir envuelta en dailySleepDTO o directa.
            daily = raw
            if isinstance(raw, dict) and isinstance(raw.get("dailySleepDTO"), dict):
                daily = raw["dailySleepDTO"]
            if not isinstance(daily, dict):
                return None

            total_sec = daily.get("sleepTimeSeconds") or 0
            deep_sec = daily.get("deepSleepSeconds") or 0
            light_sec = daily.get("lightSleepSeconds") or 0
            rem_sec = daily.get("remSleepSeconds") or 0
            awake_sec = daily.get("awakeSleepSeconds") or daily.get("awakeTimeSeconds") or 0

            start_ms = daily.get("sleepStartTimestampGMT") or daily.get("sleepStartTimestampLocal")
            end_ms = daily.get("sleepEndTimestampGMT") or daily.get("sleepEndTimestampLocal")
            start_time = end_time = None
            try:
                if isinstance(start_ms, (int, float)):
                    start_time = datetime.fromtimestamp(start_ms / 1000).strftime("%H:%M")
                if isinstance(end_ms, (int, float)):
                    end_time = datetime.fromtimestamp(end_ms / 1000).strftime("%H:%M")
            except Exception:
                pass

            score = None
            scores = daily.get("sleepScores")
            if isinstance(scores, dict):
                overall = scores.get("overall")
                if isinstance(overall, dict):
                    score = overall.get("value")
                elif isinstance(overall, (int, float)):
                    score = overall
            if score is None:
                score = daily.get("sleepScore") or daily.get("overallSleepScore")

            if not total_sec and score is None:
                return None

            return {
                "date": date_str,
                "total": _secs_to_hm(total_sec),
                "total_min": round(total_sec / 60) if total_sec else 0,
                "deep": _secs_to_hm(deep_sec),
                "light": _secs_to_hm(light_sec),
                "rem": _secs_to_hm(rem_sec),
                "awake": _secs_to_hm(awake_sec),
                "start": start_time,
                "end": end_time,
                "score": score,
            }
        except Exception as e:
            logger.error("Error obteniendo sueño Garmin: %s", e)
            self._note_auth_error(e)
            return None

    def format_sleep_note(self, sleep: dict) -> str:
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

    # ---------- pasos ----------
    def get_steps_data(self, date_str: Optional[str] = None) -> Optional[dict]:
        if not self._authenticated or not self._access_token:
            return None
        try:
            if date_str is None:
                date_str = datetime.now(TZ_AR).strftime("%Y-%m-%d")
            with self._client() as client:
                raw = client.get_daily_steps(date_str, date_str)
            self._log_raw_once("steps", raw)
            if not raw:
                return None

            day = None
            if isinstance(raw, list) and raw:
                # Elegir la entrada de la fecha pedida si está, si no la última.
                day = next((d for d in raw if isinstance(d, dict)
                            and str(d.get("calendarDate")) == date_str), None)
                if day is None:
                    day = raw[-1] if isinstance(raw[-1], dict) else None
            elif isinstance(raw, dict):
                day = raw
            if not isinstance(day, dict):
                return None

            steps = day.get("totalSteps") or day.get("steps") or 0
            distance_m = day.get("totalDistance") or day.get("distance") or 0
            calories = day.get("totalCalories") or day.get("calories") or 0

            return {
                "date": date_str,
                "steps": steps or 0,
                "distance_km": round((distance_m or 0) / 1000, 1),
                "calories": calories or 0,
            }
        except Exception as e:
            logger.error("Error obteniendo pasos Garmin: %s", e)
            self._note_auth_error(e)
            return None

    def format_steps_note(self, steps: dict) -> str:
        parts = [f"🚶 Pasos: {steps['steps']:,}".replace(",", ".")]
        if steps.get("distance_km"):
            parts.append(f"({steps['distance_km']} km)")
        if steps.get("calories"):
            parts.append(f"| {steps['calories']} kcal")
        return " ".join(parts)

    # ---------- frecuencia cardíaca ----------
    def get_heart_rate_data(self, date_str: Optional[str] = None) -> Optional[dict]:
        if not self._authenticated or not self._access_token:
            return None
        try:
            if date_str is None:
                date_str = datetime.now(TZ_AR).strftime("%Y-%m-%d")
            with self._client() as client:
                raw = client.get_daily_heart_rate(date_str)
            self._log_raw_once("hr", raw)
            if not isinstance(raw, dict):
                return None

            result = {
                "date": date_str,
                "resting_hr": raw.get("restingHeartRate"),
                "min_hr": raw.get("minHeartRate"),
                "max_hr": raw.get("maxHeartRate"),
            }
            zones = raw.get("heartRateTimeInZones", [])
            if isinstance(zones, list) and zones:
                zone_summary = []
                for z in zones:
                    if not isinstance(z, dict):
                        continue
                    mins = round((z.get("secsInZone") or 0) / 60)
                    if mins > 0:
                        zone_summary.append({"zone": z.get("zoneName", ""), "minutes": mins})
                if zone_summary:
                    result["zones"] = zone_summary
            return result
        except Exception as e:
            logger.error("Error obteniendo FC Garmin: %s", e)
            self._note_auth_error(e)
            return None

    # ---------- estrés ----------
    def get_stress_data(self, date_str: Optional[str] = None) -> Optional[dict]:
        if not self._authenticated or not self._access_token:
            return None
        try:
            if date_str is None:
                date_str = datetime.now(TZ_AR).strftime("%Y-%m-%d")
            with self._client() as client:
                raw = client.get_daily_stress(date_str, date_str)
            self._log_raw_once("stress", raw)
            if not raw:
                return None

            day = None
            if isinstance(raw, list) and raw:
                day = next((d for d in raw if isinstance(d, dict)
                            and str(d.get("calendarDate")) == date_str), None)
                if day is None:
                    day = raw[-1] if isinstance(raw[-1], dict) else None
            elif isinstance(raw, dict):
                day = raw
            if not isinstance(day, dict):
                return None

            avg = (day.get("avgStressLevel") or day.get("averageStressLevel")
                   or day.get("overallStressLevel"))
            return {
                "date": date_str,
                "avg_stress": avg if (avg is not None and avg >= 0) else None,
                "max_stress": day.get("maxStressLevel"),
                "high_stress_min": round((day.get("highStressDuration") or 0) / 60),
                "medium_stress_min": round((day.get("mediumStressDuration") or 0) / 60),
                "low_stress_min": round((day.get("lowStressDuration") or 0) / 60),
                "rest_stress_min": round((day.get("restStressDuration") or 0) / 60),
            }
        except Exception as e:
            logger.error("Error obteniendo estrés Garmin: %s", e)
            self._note_auth_error(e)
            return None

    # ---------- body battery ----------
    def get_body_battery(self, date_str: Optional[str] = None) -> Optional[dict]:
        if not self._authenticated or not self._access_token:
            return None
        try:
            if date_str is None:
                date_str = datetime.now(TZ_AR).strftime("%Y-%m-%d")
            with self._client() as client:
                raw = client.get_daily_body_battery(date_str, date_str)
            self._log_raw_once("body_battery", raw)
            if not raw:
                return None

            # Buscar el array de lecturas, cubriendo varias formas posibles.
            container = raw
            if isinstance(raw, list) and raw:
                container = raw[-1] if isinstance(raw[-1], dict) else raw
            readings = None
            if isinstance(container, dict):
                for key in ("bodyBatteryValuesArray", "bodyBatteryValuesList",
                            "bodyBatteryLevels", "values"):
                    if isinstance(container.get(key), list):
                        readings = container[key]
                        break
            elif isinstance(container, list):
                readings = container
            if not readings:
                return None

            values = []
            for r in readings:
                v = None
                if isinstance(r, dict):
                    v = r.get("bodyBatteryLevel") or r.get("level") or r.get("value")
                elif isinstance(r, (list, tuple)) and len(r) >= 2:
                    v = r[1]
                if isinstance(v, (int, float)) and v > 0:
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
            logger.error("Error obteniendo Body Battery: %s", e)
            self._note_auth_error(e)
            return None

    def format_vitals_note(self, hr: Optional[dict], stress: Optional[dict],
                           battery: Optional[dict]) -> str:
        parts = []
        if hr and hr.get("resting_hr"):
            parts.append(f"❤️ FC reposo: {hr['resting_hr']} bpm")
            if hr.get("max_hr"):
                parts.append(f"(máx {hr['max_hr']})")
        if stress and stress.get("avg_stress"):
            avg = stress["avg_stress"]
            level = "bajo"
            if avg > 50:
                level = "alto"
            elif avg > 35:
                level = "medio"
            parts.append(f"| 🧘 Estrés: {avg} ({level})")
        if battery and battery.get("current"):
            parts.append(f"| 🔋 Body Battery: {battery['current']}/100")
        return " ".join(parts) if parts else ""

    # ---------- resumen diario ----------
    def get_daily_summary(self, date_str: Optional[str] = None) -> dict:
        if date_str is None:
            date_str = datetime.now(TZ_AR).strftime("%Y-%m-%d")

        summary = {"date": date_str}
        summary["sleep"] = self.get_sleep_data(date_str)
        summary["steps"] = self.get_steps_data(date_str)
        summary["heart_rate"] = self.get_heart_rate_data(date_str)
        summary["stress"] = self.get_stress_data(date_str)
        summary["body_battery"] = self.get_body_battery(date_str)

        if self.db is not None:
            try:
                self.db[DAILY_DATA_COLLECTION].replace_one(
                    {"_id": date_str},
                    {"_id": date_str, **summary, "updated": datetime.now(timezone.utc)},
                    upsert=True,
                )
            except Exception as e:
                logger.warning("No pude guardar datos diarios: %s", e)
        return summary

    def get_sleep_history(self, days: int = 7) -> list:
        if self.db is None:
            return []
        try:
            start = datetime.now(TZ_AR) - timedelta(days=days)
            docs = list(self.db[DAILY_DATA_COLLECTION].find(
                {"_id": {"$gte": start.strftime("%Y-%m-%d")}},
                sort=[("_id", -1)],
            ))
            return [d.get("sleep") for d in docs if d.get("sleep")]
        except Exception as e:
            logger.error("Error leyendo historial sueño: %s", e)
            return []

    # ---------- actividad → Nightscout ----------
    def format_for_nightscout(self, activity: dict) -> Optional[dict]:
        activity_id = activity.get("activityId")
        if not activity_id:
            return None
        if self._is_already_synced(activity_id):
            return None

        label = _get_activity_label(activity)
        duration_min = round((activity.get("duration") or 0) / 60)
        if duration_min < 1:
            return None

        avg_hr = activity.get("averageHR")
        max_hr = activity.get("maxHR")
        calories = activity.get("calories")
        distance_km = None
        if activity.get("distance"):
            distance_km = round(activity["distance"] / 1000, 2)

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

        start_time = activity.get("startTimeGMT") or activity.get("startTimeLocal")
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
        activity_id = activity.get("activityId")
        name = activity.get("activityName", "")
        if activity_id:
            self._mark_synced(activity_id, name)

    # ---------- estado ----------
    def get_status(self) -> dict:
        synced_count = 0
        last_sync = None
        if self.db is not None:
            try:
                synced_count = self.db[SYNCED_COLLECTION].count_documents({})
                last_doc = self.db[SYNCED_COLLECTION].find_one(sort=[("synced_at", -1)])
                if last_doc:
                    last_sync = last_doc.get("synced_at")
            except Exception:
                pass
        return {
            "configured": self.is_configured(),
            "authenticated": self._authenticated,
            "activities_synced": synced_count,
            "last_sync": str(last_sync) if last_sync else "nunca",
        }
