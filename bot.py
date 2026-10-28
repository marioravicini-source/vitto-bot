se te"""
VittoreD1Bot v5 — Agente conversacional para diabetes tipo 1
============================================================
Lee datos de Nightscout y acompaña por Telegram usando Claude como cerebro.

Cambios v5 (por qué el bot ahora "piensa"):
  1. CEREBRO: Claude (claude-sonnet-4-6) para chat y visión, en lugar de un
     modelo chico. Groq queda solo para transcribir audio (Whisper).
  2. REGISTRO CONFIABLE: en vez de adivinar por palabras clave, Claude usa
     "herramientas" (tools) y decide con qué eventType y datos registrar en
     Nightscout. El registro del día se LEE desde Nightscout (fuente de
     verdad), así no se pierde al reiniciar.
  3. PATRONES: análisis multi-día (tiempo en rango por día, hipoglucemias
     nocturnas, picos post-comida, caídas post-ejercicio) -> /patrones y
     contexto para el chat.
  4. PROACTIVIDAD: alertas predictivas por tendencia + insulina activa,
     anti-spam (solo avisa en cambios de estado), aviso de sensor caído,
     recordatorio post-ejercicio y resumen diario.
  5. MEMORIA: conversación y perfil aprendido persistidos en MongoDB (si hay
     MONGODB_URI); si no, memoria en RAM como antes.

⛔ REGLA DE SEGURIDAD ABSOLUTA (no se negocia):
Este bot NUNCA calcula, sugiere ni recomienda dosis de insulina.
Solo informa, registra, recuerda y alerta. Las dosis las define el
endocrinólogo.
"""

import os
import io
import re
import json
import math
import base64
import asyncio
import logging
import hashlib
import statistics
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Optional, Any
from collections import defaultdict

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ─── Dependencias opcionales ─────────────────────────────────────
try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover
    MongoClient = None

try:
    import ml_model as mlm
except Exception:  # pragma: no cover
    mlm = None

try:
    from garmin_bridge import GarminBridge
except Exception:  # pragma: no cover
    GarminBridge = None


# ─── Configuración ───────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
NIGHTSCOUT_URL = os.environ["NIGHTSCOUT_URL"].rstrip("/")
NIGHTSCOUT_API_SECRET = os.environ.get("NIGHTSCOUT_API_SECRET", "")

# Proveedor del "cerebro": "groq" (gratis) o "anthropic" (Claude, de pago).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Groq (capa gratuita): chat con herramientas + visión + audio (Whisper).
# gpt-oss-120b es la versión grande del que usábamos: razona mucho mejor y es gratis.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
MONGODB_URI = os.environ.get("MONGODB_URI", "")

ALLOWED_USERS = [u.strip() for u in os.environ.get("ALLOWED_USERS", "").split(",") if u.strip()]
# A quién van las alertas automáticas. Por defecto, los mismos usuarios permitidos.
CAREGIVER_CHAT_IDS = [
    c.strip() for c in os.environ.get("CAREGIVER_CHAT_IDS", ",".join(ALLOWED_USERS)).split(",") if c.strip()
]
# Quién puede BORRAR registros por Telegram (por defecto, los usuarios permitidos).
# Conviene setear ADMIN_USERS solo con el chat_id de Mario.
ADMIN_USERS = [
    a.strip() for a in os.environ.get("ADMIN_USERS", ",".join(ALLOWED_USERS)).split(",") if a.strip()
]

# Umbrales de glucosa (mg/dL)
BG_HIGH = int(os.environ.get("BG_HIGH", "180"))
BG_LOW = int(os.environ.get("BG_LOW", "70"))
BG_URGENT_HIGH = int(os.environ.get("BG_URGENT_HIGH", "250"))
BG_URGENT_LOW = int(os.environ.get("BG_URGENT_LOW", "55"))

# ─── Parámetros clínicos (perfil de Vittore) para la predicción ──
# ⚠ Deben coincidir con lo que indica el endocrinólogo. Solo se usan para
# PREDECIR y ALERTAR, nunca para calcular dosis.
ISF = float(os.environ.get("ISF", "30"))            # sensibilidad: mg/dL por unidad
CARB_RATIO = float(os.environ.get("CARB_RATIO", "10"))   # ratio: gramos por unidad (1U:10g)
INSULIN_DIA_MIN = int(os.environ.get("INSULIN_DIA_MIN", "240"))   # duración insulina activa (Apidra 4h)
INSULIN_PEAK_MIN = int(os.environ.get("INSULIN_PEAK_MIN", "65"))  # pico de acción de la Apidra
CARB_ABSORB_MIN = int(os.environ.get("CARB_ABSORB_MIN", "180"))   # tiempo de absorción de carbos
CSF = ISF / CARB_RATIO if CARB_RATIO else 0.0        # sensibilidad a carbos: mg/dL por gramo

# Ventana de predicción (minutos) para el aviso preventivo de baja
PREDICT_MIN = int(os.environ.get("PREDICT_MIN", "20"))
# Minutos sin datos del sensor para avisar "sensor caído"
SENSOR_GAP_MIN = int(os.environ.get("SENSOR_GAP_MIN", "20"))
# Cooldown para re-avisar el mismo estado malo (minutos)
ALERT_COOLDOWN_MIN = int(os.environ.get("ALERT_COOLDOWN_MIN", "30"))

# Zona horaria Argentina
TZ_AR = timezone(timedelta(hours=-3))

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("VittoBot")

# ─── Clientes ────────────────────────────────────────────────────
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if (Anthropic and ANTHROPIC_API_KEY) else None
groq_client = Groq(api_key=GROQ_API_KEY) if (Groq and GROQ_API_KEY) else None


def brain_provider() -> str:
    """Qué proveedor se usa realmente, según config y claves disponibles."""
    if LLM_PROVIDER == "anthropic" and anthropic_client:
        return "anthropic"
    if groq_client:
        return "groq"
    if anthropic_client:
        return "anthropic"
    return "none"


def brain_label() -> str:
    p = brain_provider()
    if p == "anthropic":
        return f"Claude ({ANTHROPIC_MODEL})"
    if p == "groq":
        return f"Groq ({GROQ_CHAT_MODEL})"
    return "SIN cerebro (falta GROQ_API_KEY o ANTHROPIC_API_KEY)"


if brain_provider() == "none":
    logger.warning("⚠ Sin cerebro: configurá GROQ_API_KEY (gratis) o ANTHROPIC_API_KEY en Railway.")


# ─── Persistencia (MongoDB opcional, con fallback en RAM) ────────
class Store:
    """Guarda historial de conversación, perfil aprendido y estado de alertas.

    Usa MongoDB si hay MONGODB_URI; si no, memoria en RAM (se pierde al
    reiniciar, como antes)."""

    def __init__(self, uri: str):
        self._mem_history: dict[int, list[dict]] = defaultdict(list)
        self._mem_profile: dict[int, dict] = defaultdict(dict)
        self._mem_alert: dict = {}
        self.db = None
        if uri and MongoClient:
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="VittoBot")
                client.admin.command("ping")
                self.db = client["vittobot"]
                logger.info("✓ Conectado a MongoDB para persistencia.")
            except Exception as e:
                logger.error(f"No pude conectar a MongoDB, uso memoria en RAM: {e}")
                self.db = None

    # -- Historial de conversación --
    def get_history(self, chat_id: int, limit: int = 20) -> list[dict]:
        if self.db is not None:
            try:
                docs = list(
                    self.db.history.find({"chat_id": chat_id})
                    .sort("ts", -1)
                    .limit(limit)
                )
                docs.reverse()
                return [{"role": d["role"], "content": d["content"]} for d in docs]
            except Exception as e:
                logger.error(f"get_history: {e}")
                return []
        return list(self._mem_history[chat_id])[-limit:]

    def add_history(self, chat_id: int, role: str, content: str):
        if self.db is not None:
            try:
                self.db.history.insert_one({
                    "chat_id": chat_id,
                    "role": role,
                    "content": content,
                    "ts": datetime.now(timezone.utc),
                })
                return
            except Exception as e:
                logger.error(f"add_history: {e}")
        h = self._mem_history[chat_id]
        h.append({"role": role, "content": content})
        if len(h) > 40:
            self._mem_history[chat_id] = h[-40:]

    # -- Perfil aprendido (comidas habituales, rutina, apodo, etc.) --
    def get_profile(self, chat_id: int) -> dict:
        if self.db is not None:
            try:
                doc = self.db.profile.find_one({"chat_id": chat_id})
                return (doc or {}).get("data", {})
            except Exception as e:
                logger.error(f"get_profile: {e}")
                return {}
        return dict(self._mem_profile[chat_id])

    def update_profile(self, chat_id: int, key: str, value: Any):
        if self.db is not None:
            try:
                self.db.profile.update_one(
                    {"chat_id": chat_id},
                    {"$set": {f"data.{key}": value}},
                    upsert=True,
                )
                return
            except Exception as e:
                logger.error(f"update_profile: {e}")
        self._mem_profile[chat_id][key] = value

    # -- Estado de alertas (para anti-spam, sobrevive reinicios) --
    def get_alert_state(self) -> dict:
        if self.db is not None:
            try:
                return self.db.state.find_one({"_id": "alert"}) or {}
            except Exception as e:
                logger.error(f"get_alert_state: {e}")
                return {}
        return dict(self._mem_alert)

    def set_alert_state(self, state: dict):
        state = dict(state)
        if self.db is not None:
            try:
                self.db.state.update_one({"_id": "alert"}, {"$set": state}, upsert=True)
                return
            except Exception as e:
                logger.error(f"set_alert_state: {e}")
        self._mem_alert.update(state)


store = Store(MONGODB_URI)

# ─── Modelo ML (Fase 2) ─────────────────────────────────────────
ml_trained = None  # se carga al arrancar si hay modelo en MongoDB
ML_CONFIG = {
    "isf": ISF,
    "carb_ratio": CARB_RATIO,
    "insulin_dia_min": INSULIN_DIA_MIN,
    "insulin_peak_min": INSULIN_PEAK_MIN,
    "carb_absorb_min": CARB_ABSORB_MIN,
}

if mlm and store.db is not None:
    try:
        ml_trained = mlm.load_model_from_mongo(store.db)
    except Exception as e:
        logger.error(f"No pude cargar modelo ML: {e}")

# ─── Garmin Bridge (Fase 2.5) ────────────────────────────────────
garmin = None
if GarminBridge:
    garmin = GarminBridge(db=store.db)
    if garmin.is_configured():
        logger.info("Garmin Bridge configurado ✓")
    else:
        logger.info("Garmin Bridge: sin credenciales (GARMIN_EMAIL/GARMIN_PASSWORD)")


# ─── Nightscout: headers y lecturas ──────────────────────────────
def _ns_headers() -> dict:
    headers = {}
    if NIGHTSCOUT_API_SECRET:
        headers["api-secret"] = hashlib.sha1(
            NIGHTSCOUT_API_SECRET.encode("utf-8")
        ).hexdigest()
    return headers


async def _ns_get(path: str, params: Optional[dict] = None) -> Any:
    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.get(f"{NIGHTSCOUT_URL}{path}", headers=_ns_headers(), params=params)
        resp.raise_for_status()
        return resp.json()


async def get_current_bg() -> Optional[dict]:
    try:
        data = await _ns_get("/api/v1/entries/current.json")
        if data:
            return data[0]
    except Exception as e:
        logger.error(f"get_current_bg: {e}")
    return None


async def get_recent_entries(hours: int = 3) -> list:
    try:
        return await _ns_get("/api/v1/entries.json", {"count": hours * 12})
    except Exception as e:
        logger.error(f"get_recent_entries: {e}")
    return []


async def get_entries_days(days: int = 14) -> list:
    """Lecturas SGV de los últimos N días para análisis de patrones."""
    try:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        params = {
            "find[dateString][$gte]": since.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "count": days * 288 + 50,  # 288 lecturas/día aprox.
        }
        return await _ns_get("/api/v1/entries/sgv.json", params)
    except Exception as e:
        logger.error(f"get_entries_days: {e}")
    return []


async def get_treatments(hours: int = 6) -> list:
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        params = {
            "find[created_at][$gte]": since.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "count": 100,
        }
        return await _ns_get("/api/v1/treatments.json", params)
    except Exception as e:
        logger.error(f"get_treatments: {e}")
    return []


async def get_treatments_days(days: int = 30) -> list:
    """Tratamientos de los últimos N días (para entrenamiento ML)."""
    try:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        params = {
            "find[created_at][$gte]": since.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "count": days * 50,  # ~50 tratamientos/día max
        }
        return await _ns_get("/api/v1/treatments.json", params)
    except Exception as e:
        logger.error(f"get_treatments_days: {e}")
    return []


async def get_device_status() -> Optional[dict]:
    try:
        data = await _ns_get("/api/v1/devicestatus.json", {"count": 1})
        if data:
            return data[0]
    except Exception as e:
        logger.error(f"get_device_status: {e}")
    return None


async def post_treatment(event_type: str, **kwargs) -> bool:
    try:
        now = datetime.now(timezone.utc)
        treatment = {
            "eventType": event_type,
            "created_at": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "enteredBy": "VittoBot",
        }
        treatment.update({k: v for k, v in kwargs.items() if v is not None})
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.post(
                f"{NIGHTSCOUT_URL}/api/v1/treatments",
                headers=_ns_headers(),
                json=treatment,
            )
            resp.raise_for_status()
            logger.info(f"Treatment NS: {event_type} {kwargs}")
            return True
    except Exception as e:
        logger.error(f"post_treatment: {type(e).__name__}: {e}")
    return False


async def delete_treatment(treatment_id: str) -> bool:
    """Borra un tratamiento de Nightscout por su _id."""
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.delete(
                f"{NIGHTSCOUT_URL}/api/v1/treatments/{treatment_id}",
                headers=_ns_headers(),
            )
            resp.raise_for_status()
            logger.info(f"Treatment borrado NS: {treatment_id}")
            return True
    except Exception as e:
        logger.error(f"delete_treatment: {type(e).__name__}: {e}")
    return False


# ─── Formato y clasificación ─────────────────────────────────────
def format_direction(direction: str) -> str:
    arrows = {
        "DoubleUp": "⬆⬆ subiendo rápido",
        "SingleUp": "⬆ subiendo",
        "FortyFiveUp": "↗ subiendo lento",
        "Flat": "→ estable",
        "FortyFiveDown": "↘ bajando lento",
        "SingleDown": "⬇ bajando",
        "DoubleDown": "⬇⬇ bajando rápido",
        "NOT COMPUTABLE": "? sin tendencia",
        "RATE OUT OF RANGE": "⚠ fuera de rango",
    }
    return arrows.get(direction, direction or "?")


def classify_bg(sgv: int) -> str:
    if sgv <= BG_URGENT_LOW:
        return "urgent_low"
    if sgv < BG_LOW:
        return "low"
    if sgv >= BG_URGENT_HIGH:
        return "urgent_high"
    if sgv > BG_HIGH:
        return "high"
    return "in_range"


def bg_indicator(sgv: int) -> str:
    return {
        "urgent_low": "🔴 URGENTE BAJO",
        "low": "🟡 BAJO",
        "in_range": "🟢 En rango",
        "high": "🟠 ALTO",
        "urgent_high": "🔴 URGENTE ALTO",
    }[classify_bg(sgv)]


def format_bg_message(entry: dict) -> str:
    sgv = entry.get("sgv", 0)
    direction = entry.get("direction", "")
    date_ms = entry.get("date", 0)
    dt = datetime.fromtimestamp(date_ms / 1000, tz=TZ_AR)
    mins_ago = int((datetime.now(TZ_AR) - dt).total_seconds() / 60)
    return (
        f"🩸 Glucosa: {sgv} mg/dL — {bg_indicator(sgv)}\n"
        f"Tendencia: {format_direction(direction)}\n"
        f"Hora: {dt.strftime('%H:%M')} (hace {mins_ago} min)"
    )


def summarize_entries(entries: list) -> str:
    values = [e.get("sgv", 0) for e in entries if e.get("sgv")]
    if not values:
        return "No hay datos de glucosa recientes."
    avg = sum(values) / len(values)
    high = sum(1 for v in values if v > BG_HIGH)
    low = sum(1 for v in values if v < BG_LOW)
    in_range = len(values) - high - low
    pct = in_range / len(values) * 100
    first = datetime.fromtimestamp(entries[-1].get("date", 0) / 1000, tz=TZ_AR)
    last = datetime.fromtimestamp(entries[0].get("date", 0) / 1000, tz=TZ_AR)
    return (
        f"📊 {first.strftime('%H:%M')}–{last.strftime('%H:%M')}: "
        f"prom {avg:.0f} · rango {min(values)}-{max(values)} · {pct:.0f}% en rango\n"
        f"⚠️ {high} altas · {low} bajas ({len(values)} lecturas)"
    )


def format_treatments(treatments: list) -> str:
    if not treatments:
        return "Sin tratamientos recientes."
    lines = []
    for t in treatments:
        created = t.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(TZ_AR)
            time_str = dt.strftime("%H:%M")
        except Exception:
            time_str = created[:16] if created else "?"
        parts = [f"{time_str} — {t.get('eventType', '?')}"]
        if t.get("insulin"):
            parts.append(f"💉 {t['insulin']}U")
        if t.get("carbs"):
            parts.append(f"🍞 {t['carbs']}g")
        if t.get("duration"):
            parts.append(f"⏱ {t['duration']}min")
        if t.get("preBolus"):
            parts.append(f"⏳ pre-bolo {t['preBolus']}min")
        if t.get("glucose"):
            parts.append(f"🩸 {t['glucose']}")
        if t.get("notes"):
            parts.append(f"📝 {t['notes']}")
        lines.append(" · ".join(parts))
    return "\n".join(lines)


def extract_iob_cob(device_status: Optional[dict]) -> tuple[Optional[float], Optional[float]]:
    """Devuelve (IOB, COB) desde devicestatus si existen."""
    if not device_status:
        return None, None
    iob = cob = None
    for key in ("openaps", "loop"):
        section = device_status.get(key, {})
        if isinstance(section, dict):
            iob_data = section.get("iob", {})
            if isinstance(iob_data, dict):
                iob = iob_data.get("iob", iob_data.get("IOB"))
            elif isinstance(iob_data, (int, float)):
                iob = iob_data
            enacted = section.get("enacted", section.get("suggested", {}))
            if isinstance(enacted, dict) and cob is None:
                cob = enacted.get("COB")
    pump = device_status.get("pump", {})
    if isinstance(pump, dict) and iob is None:
        pump_iob = pump.get("iob", {})
        if isinstance(pump_iob, dict):
            iob = pump_iob.get("iob", pump_iob.get("bolusiob"))
        elif isinstance(pump_iob, (int, float)):
            iob = pump_iob
    return (float(iob) if isinstance(iob, (int, float)) else None,
            float(cob) if isinstance(cob, (int, float)) else None)


# ─── Analítica: predicción y patrones ────────────────────────────
def bg_slope(entries: list) -> Optional[float]:
    """Pendiente reciente en mg/dL por minuto (negativa = bajando).

    entries: lista de Nightscout (más nuevo primero)."""
    pts = []
    for e in entries[:4]:
        sgv = e.get("sgv")
        date_ms = e.get("date")
        if sgv and date_ms:
            pts.append((date_ms / 1000.0, sgv))
    if len(pts) < 2:
        return None
    pts.sort()  # más viejo primero
    (t0, v0), (t1, v1) = pts[0], pts[-1]
    dt_min = (t1 - t0) / 60.0
    if dt_min <= 0:
        return None
    return (v1 - v0) / dt_min


def predict_bg(current_sgv: int, slope: Optional[float], minutes: int) -> Optional[int]:
    if slope is None:
        return None
    return int(current_sgv + slope * minutes)


def analyze_patterns(entries: list, treatments: list, days: int = 14) -> dict:
    """Calcula métricas de patrones sobre los últimos N días."""
    by_day: dict[str, list[int]] = defaultdict(list)
    night: dict[str, list[int]] = defaultdict(list)  # 00:00–06:00
    for e in entries:
        sgv = e.get("sgv")
        date_ms = e.get("date")
        if not sgv or not date_ms:
            continue
        dt = datetime.fromtimestamp(date_ms / 1000, tz=TZ_AR)
        day = dt.strftime("%Y-%m-%d")
        by_day[day].append(sgv)
        if dt.hour < 6:
            night[day].append(sgv)

    daily = []
    for day in sorted(by_day):
        vals = by_day[day]
        tir = sum(1 for v in vals if BG_LOW <= v <= BG_HIGH) / len(vals) * 100
        lows = sum(1 for v in vals if v < BG_LOW)
        daily.append({
            "day": day,
            "n": len(vals),
            "avg": round(statistics.mean(vals)),
            "tir": round(tir),
            "lows": lows,
            "min": min(vals),
            "max": max(vals),
        })

    nights_with_low = [d for d, vals in night.items() if any(v < BG_LOW for v in vals)]

    # Ejercicios registrados y si hubo baja nocturna en la noche siguiente
    exercises = []
    for t in treatments:
        if t.get("eventType") == "Exercise":
            created = t.get("created_at", "")
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(TZ_AR)
            except Exception:
                continue
            day = dt.strftime("%Y-%m-%d")
            low_next_night = day in nights_with_low
            exercises.append({
                "day": day, "hora": dt.strftime("%H:%M"),
                "notes": t.get("notes", ""), "low_madrugada": low_next_night,
            })

    valid = [d for d in daily if d["n"] >= 50]  # días con datos suficientes
    avg_tir = round(statistics.mean([d["tir"] for d in valid])) if valid else None

    return {
        "days": days,
        "daily": daily,
        "avg_tir": avg_tir,
        "nights_with_low": sorted(nights_with_low),
        "exercises": exercises,
    }


def format_patterns_digest(p: dict, compact: bool = False) -> str:
    if not p["daily"]:
        return "Todavía no hay suficientes datos para detectar patrones."
    lines = []
    if p["avg_tir"] is not None:
        lines.append(f"🎯 Tiempo en rango promedio ({p['days']}d): {p['avg_tir']}%")
    recent = p["daily"][-7:]
    if not compact:
        lines.append("Por día (últimos):")
        for d in recent:
            flag = " ⚠️" if d["lows"] else ""
            lines.append(f"  {d['day']}: TIR {d['tir']}% · prom {d['avg']} · {d['lows']} bajas{flag}")
    n_low = len(p["nights_with_low"])
    if n_low:
        lines.append(f"🌙 Hipoglucemias de madrugada en {n_low} de los últimos {p['days']} días: "
                     f"{', '.join(p['nights_with_low'][-5:])}")
    ex_low = [e for e in p["exercises"] if e["low_madrugada"]]
    if ex_low:
        lines.append(f"🏒 En {len(ex_low)} de {len(p['exercises'])} días con ejercicio registrado "
                     f"hubo baja de madrugada esa noche. Patrón a comentar con el endocrinólogo.")
    return "\n".join(lines)


# ─── Predicción fisiológica (Fase 1): IOB/COB + proyección 30/60 ─
# Modelo transparente que NO calcula dosis: proyecta a dónde va la glucosa
# combinando la insulina que sigue actuando (baja) + los carbos que se están
# absorbiendo (suben) + el momento de la tendencia reciente del sensor.

def insulin_remaining_fraction(age_min: float,
                               dia: int = INSULIN_DIA_MIN,
                               peak: int = INSULIN_PEAK_MIN) -> float:
    """Fracción de un bolo que sigue activa `age_min` después (modelo exponencial oref/Loop)."""
    if age_min <= 0:
        return 1.0
    if age_min >= dia:
        return 0.0
    td, tp = float(dia), float(peak)
    tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
    a = 2 * tau / td
    s = 1 / (1 - a + (1 + a) * math.exp(-td / tau))
    t = age_min
    iob = 1 - s * (1 - a) * (((t * t) / (tau * td * (1 - a)) - t / tau - 1) * math.exp(-t / tau) + 1)
    return max(0.0, min(1.0, iob))


def carb_remaining_fraction(age_min: float, tabs: int = CARB_ABSORB_MIN) -> float:
    """Fracción de una comida que todavía no se absorbió (absorción lineal)."""
    if age_min <= 0:
        return 1.0
    if age_min >= tabs:
        return 0.0
    return 1 - age_min / tabs


def compute_iob(boluses: list, at_offset: float = 0) -> float:
    """Insulina activa (U). boluses: lista de (edad_min, unidades)."""
    return sum(u * insulin_remaining_fraction(age + at_offset) for age, u in boluses)


def compute_cob(carbs: list, at_offset: float = 0) -> float:
    """Carbos activos (g). carbs: lista de (edad_min, gramos)."""
    return sum(g * carb_remaining_fraction(age + at_offset) for age, g in carbs)


def insulin_effect(boluses: list, horizon: int) -> float:
    """Cambio de glucosa por insulina que actúa en [ahora, ahora+horizon] (negativo)."""
    acted = sum(u * (insulin_remaining_fraction(age) - insulin_remaining_fraction(age + horizon))
                for age, u in boluses)
    return -ISF * acted


def carb_effect(carbs: list, horizon: int) -> float:
    """Cambio de glucosa por carbos absorbidos en la ventana (positivo)."""
    absorbed = sum(g * (carb_remaining_fraction(age) - carb_remaining_fraction(age + horizon))
                   for age, g in carbs)
    return CSF * absorbed


def momentum_effect(slope: Optional[float], horizon: int, tau_m: float = 20.0) -> float:
    """Aporte de la tendencia reciente, decae para no dominar a 60 min."""
    if slope is None:
        return 0.0
    return slope * tau_m * (1 - math.exp(-horizon / tau_m))


def predict_physio(current: int, slope: Optional[float], boluses: list, carbs: list,
                   horizon: int) -> int:
    delta = insulin_effect(boluses, horizon) + carb_effect(carbs, horizon) + momentum_effect(slope, horizon)
    return int(max(20, min(500, round(current + delta))))


def prediction_band(horizon: int, exercise_recent: bool, carbs_active: bool) -> int:
    """Banda de incertidumbre (mg/dL), crece con el horizonte y con ejercicio/comida."""
    band = 8 + 0.30 * horizon
    if exercise_recent:
        band *= 1.6
    if carbs_active:
        band *= 1.2
    return int(round(band))


def parse_active_treatments(treatments: list, now_utc: datetime) -> tuple[list, list, bool]:
    """Extrae bolos e insulina/carbos activos y si hubo ejercicio reciente."""
    boluses, carbs = [], []
    exercise_recent = False
    for t in treatments:
        created = t.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except Exception:
            continue
        age = (now_utc - dt).total_seconds() / 60
        if age < 0:
            continue
        notes = (t.get("notes", "") or "").lower()
        ins = t.get("insulin")
        # Excluir la basal (Glargina): perfil plano, se sigue aparte.
        if ins and age < INSULIN_DIA_MIN and "basal" not in notes and "glargina" not in notes:
            try:
                boluses.append((age, float(ins)))
            except (TypeError, ValueError):
                pass
        cb = t.get("carbs")
        if cb and age < CARB_ABSORB_MIN:
            try:
                carbs.append((age, float(cb)))
            except (TypeError, ValueError):
                pass
        if t.get("eventType") == "Exercise" and age < 360:  # efecto hasta varias horas después
            exercise_recent = True
    return boluses, carbs, exercise_recent


def assess_hypo_risk(current: int, slope: Optional[float], boluses: list, carbs: list,
                     exercise_recent: bool) -> dict:
    """Nivel de riesgo de hipoglucemia (rule-based, no probabilidad calibrada)."""
    p30 = predict_physio(current, slope, boluses, carbs, 30)
    p60 = predict_physio(current, slope, boluses, carbs, 60)
    b30 = prediction_band(30, exercise_recent, bool(carbs))
    b60 = prediction_band(60, exercise_recent, bool(carbs))
    low30, low60 = p30 - b30, p60 - b60

    level = "ninguno"
    if current <= BG_URGENT_LOW or p30 <= BG_URGENT_LOW:
        level = "alto"
    elif p30 <= BG_LOW or low30 <= BG_LOW or current <= BG_LOW:
        level = "alto"
    elif p60 <= BG_LOW or low60 <= BG_LOW:
        level = "medio"
    elif exercise_recent and p60 <= BG_LOW + 20:
        level = "medio"
    return {"level": level, "p30": p30, "p60": p60, "b30": b30, "b60": b60}


async def compute_prediction() -> Optional[dict]:
    """Junta datos de Nightscout y devuelve la predicción completa (physio + ML)."""
    global ml_trained
    entry = await get_current_bg()
    if not entry:
        return None
    entries, treatments = await asyncio.gather(get_recent_entries(1), get_treatments(6))
    current = entry.get("sgv", 0)
    slope = bg_slope(entries)
    now_utc = datetime.now(timezone.utc)
    boluses, carbs, ex = parse_active_treatments(treatments, now_utc)
    risk = assess_hypo_risk(current, slope, boluses, carbs, ex)

    result = {
        "current": current, "slope": slope,
        "p30": risk["p30"], "b30": risk["b30"],
        "p60": risk["p60"], "b60": risk["b60"],
        "risk": risk["level"],
        "iob": compute_iob(boluses), "cob": compute_cob(carbs),
        "exercise_recent": ex,
        "source": "physio",
    }

    # --- Fase 2: predicción ML ---
    if mlm and ml_trained is not None:
        try:
            # Convertir entries a formato (datetime, sgv) para build_features
            points = []
            for e in sorted(entries, key=lambda x: x.get("date", 0)):
                sgv = e.get("sgv")
                date_ms = e.get("date")
                if sgv and date_ms:
                    ts = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)
                    points.append((ts, int(sgv)))

            if len(points) >= 7:
                features = mlm.build_features(points, len(points) - 1,
                                              treatments, ML_CONFIG)
                if features is not None:
                    ml_pred = mlm.predict_ml(ml_trained, features)
                    if ml_pred:
                        result["ml"] = ml_pred
                        # Combinar: promediar physio (60%) + ML (40%) si ambos están
                        if "p30" in ml_pred:
                            w_physio, w_ml = 0.6, 0.4
                            result["p30"] = round(result["p30"] * w_physio + ml_pred["p30"] * w_ml)
                            result["b30"] = ml_pred.get("band_30", result["b30"])
                            result["source"] = "hybrid"
                        if "p60" in ml_pred:
                            result["p60"] = round(result["p60"] * 0.6 + ml_pred["p60"] * 0.4)
                            result["b60"] = ml_pred.get("band_60", result["b60"])
                        # Probabilidad de hipo del ML mejora el assess de riesgo
                        hypo_prob_30 = ml_pred.get("hypo_prob_30", 0)
                        hypo_prob_60 = ml_pred.get("hypo_prob_60", 0)
                        if hypo_prob_30 >= 0.5 and result["risk"] == "ninguno":
                            result["risk"] = "alto"
                        elif hypo_prob_30 >= 0.3 and result["risk"] == "ninguno":
                            result["risk"] = "medio"
                        elif hypo_prob_60 >= 0.4 and result["risk"] == "ninguno":
                            result["risk"] = "medio"
                        result["hypo_prob_30"] = hypo_prob_30
                        result["hypo_prob_60"] = hypo_prob_60
        except Exception as e:
            logger.error(f"ML predict error (fallback a physio): {e}")

    return result


def format_prediction(pred: dict) -> str:
    """Texto compacto de la predicción para Telegram / contexto."""
    trend = f"{pred['slope'] * 60:+.0f}/h" if pred.get("slope") is not None else "s/d"
    risk_emoji = {"alto": "🔴", "medio": "🟡", "ninguno": "🟢"}.get(pred["risk"], "🟢")
    risk_txt = {"alto": "riesgo ALTO de hipo", "medio": "riesgo medio de hipo",
                "ninguno": "sin riesgo de hipo a la vista"}.get(pred["risk"])

    source = pred.get("source", "physio")
    source_label = {"physio": "fisiológico", "hybrid": "ML+fisiológico", "ml": "ML"}.get(source, source)

    lines = [
        f"🔮 Predicción ({source_label} · tendencia {trend})",
    ]

    # Bandas: si ML dio rango cuantílico, mostrar p10-p90; si no, ±banda
    ml = pred.get("ml")
    if ml and "p30_low" in ml:
        lines.append(f"  +30 min: ~{pred['p30']} mg/dL ({ml['p30_low']}-{ml['p30_high']})")
    else:
        lines.append(f"  +30 min: ~{pred['p30']} mg/dL (±{pred['b30']})")

    if ml and "p60_low" in ml:
        lines.append(f"  +60 min: ~{pred['p60']} mg/dL ({ml['p60_low']}-{ml['p60_high']})")
    else:
        lines.append(f"  +60 min: ~{pred['p60']} mg/dL (±{pred['b60']})")

    # Probabilidad de hipo del ML
    hypo_30 = pred.get("hypo_prob_30")
    if hypo_30 is not None and hypo_30 > 0.05:
        pct = round(hypo_30 * 100)
        lines.append(f"{risk_emoji} {risk_txt} (prob hipo 30min: {pct}%)")
    else:
        lines.append(f"{risk_emoji} {risk_txt}")

    lines.append(f"💉 IOB {pred['iob']:.1f}U · 🍞 COB {pred['cob']:.0f}g")

    if pred.get("exercise_recent"):
        lines.append("🏒 Ejercicio reciente: mayor incertidumbre y ojo con las bajas.")
    lines.append("ℹ️ Es una estimación, no un dato exacto.")
    return "\n".join(lines)


# ─── Cerebro: Claude (chat con herramientas + visión) ────────────
SYSTEM_PROMPT = f"""Sos VittoBot, el asistente de diabetes tipo 1 de Vittore, un adolescente de 15 años, deportista (hockey sobre patín y gimnasio). Le hablás a él y a veces a su papá (Mario).

PACIENTE: insulina rápida Apidra (bolo, dura ~3-4h) + Glargina (basal, 1 vez al día). Sensor FreeStyle Libre. Unidades mg/dL. Zona horaria Argentina (UTC-3).

⛔ REGLA DE SEGURIDAD ABSOLUTA — NUNCA la rompas:
NUNCA calcules, sugieras ni recomiendes dosis de insulina. NUNCA digas "ponete X unidades" ni des un ratio o factor de corrección. Si te piden dosis, respondé: "Eso lo define tu endocrinólogo. Yo te muestro los datos para que decidas con sus pautas." Podés informar, registrar, recordar, alertar y explicar; nunca dosificar.

RANGOS: en rango {BG_LOW}-{BG_HIGH} · bajo <{BG_LOW} · alto >{BG_HIGH} · urgente bajo ≤{BG_URGENT_LOW} · urgente alto ≥{BG_URGENT_HIGH}.

SEGUÍ EL HILO (muy importante):
- Respondé a lo ÚLTIMO que te dijeron. Si te están contestando algo que VOS preguntaste (ej. vos preguntaste "¿cuál?" y responden "la corrección"), continuá ESE tema; no cambies de tema por tu cuenta.
- Los datos de glucosa/predicción del contexto son INFORMACIÓN DE FONDO para que respondas con criterio. NO empieces cada mensaje hablando de la glucosa ni la repitas si no viene al caso. Comentala solo si te preguntan o si es directamente relevante a lo que están hablando.
- Para BORRAR o corregir un registro ya cargado, se usa el comando /borrar. NO ofrezcas "marcarlo como ignorar" ni digas que podés borrar por chat: si quieren borrar, decíles que usen /borrar.

CÓMO PENSAR (cuando el tema ES la glucosa):
- Correlacioná: relacioná glucosa con comidas, insulina activa (IOB) y ejercicio del contexto.
- Anticipá: si la tendencia va hacia una baja o suba, decilo antes de que pase.
- Usá los PATRONES del contexto (hipos nocturnas post-hockey, picos post-comida) para avisos concretos.
- Cerrá con UNA observación útil, no con frases genéricas ("todo bien, avisame").

REGISTRAR:
Cuando Vittore cuente que comió, se aplicó insulina, hizo ejercicio, midió su glucosa capilar, o quiera dejar una nota, registralo en su seguimiento.
- Si SOLO dice qué comió pero NO menciona insulina: registrá la comida y PREGUNTALE "¿cuánta Apidra te pusiste?" (en 1 línea, sin sugerir dosis).
- Si dice qué comió Y cuánta insulina: registrá ambos juntos.
- PRE-BOLO: si menciona cuántos minutos esperó entre ponerse la insulina y empezar a comer (ej. "esperé 15 min", "me la puse 20 min antes"), registralo en espera_min junto con la comida. NO le digas cuánto tiene que esperar; solo registrá lo que hizo.
- Si SOLO dice insulina sin comida: registrá la insulina.
- Si manda una foto y dice que lo comió: registrá y preguntá la insulina.
- Si manda una foto PREGUNTANDO info nutricional (sin decir que lo comió): NO registres, solo informá.
- Confirmá en 1 línea natural (ej. "Anotado 🍽 pasta ~60g · ¿cuánta Apidra te pusiste?").
- Elegí bien el tipo de evento: comida (almuerzo/cena/desayuno), colacion (snack/merienda), o correccion_baja (carbos para remontar una hipo).
- NUNCA sugieras cuánta insulina ponerse.

FORMATO TELEGRAM (importante):
- Telegram NO renderiza markdown: si escribís **negrita**, *cursiva*, # títulos o - viñetas, salen los caracteres literales (**, *, #). NO los uses. Escribí en texto plano.
- Para resaltar usá MAYÚSCULAS cortas o un emoji al inicio de la línea, no asteriscos.
- Nada de tablas. Usá renglones cortos y emojis como separadores.
- Conciso de verdad: máximo 6-8 líneas. Respondé solo lo que hace falta; no armes listas largas de recomendaciones ni repitas datos que ya ve en su app.
- Español argentino, directo, cálido pero sin ser empalagoso ni sermonear. Máx 5 emojis.
- No arranques con "¡Hola! 👋 Aquí tienes". Andá al grano con calidez.
- Sin culpa ni reto: celebrá los logros, acompañá los momentos difíciles.

Sos acompañamiento, no reemplazás al médico ni a la familia. Las estimaciones (carbs por foto, IOB, predicción) son aproximadas y las comunicás como tales."""

TOOLS = [
    {
        "name": "registrar_comida",
        "description": "Registra una comida/colación en Nightscout. Usar cuando el usuario cuenta que comió algo. Elegí el tipo_evento correcto según el contexto.",
        "input_schema": {
            "type": "object",
            "properties": {
                "descripcion": {"type": "string", "description": "Qué comió, breve."},
                "carbs_g": {"type": "number", "description": "Carbohidratos estimados en gramos, si se pueden inferir."},
                "insulina_u": {"type": "number", "description": "Unidades de insulina que se aplicó junto a la comida, si las mencionó."},
                "espera_min": {"type": "number", "description": "Pre-bolo: minutos que esperó entre ponerse la insulina y empezar a comer, si lo mencionó (ej. 'esperé 15 min', 'me la puse 20 min antes'). Solo registrar lo que dijo; no sugerir cuánto esperar."},
                "tipo_evento": {
                    "type": "string",
                    "enum": ["comida", "colacion", "correccion_baja"],
                    "description": "comida = almuerzo/cena/desayuno (Meal Bolus). colacion = snack/merienda (Snack Bolus). correccion_baja = carbos para remontar una hipoglucemia (Carb Correction). Si no es claro, usar 'comida'.",
                },
            },
            "required": ["descripcion"],
        },
    },
    {
        "name": "registrar_insulina",
        "description": "Registra una aplicación de insulina (bolo de corrección o basal) en Nightscout. NO sugiere dosis; solo registra lo que el usuario ya se aplicó.",
        "input_schema": {
            "type": "object",
            "properties": {
                "unidades": {"type": "number", "description": "Unidades aplicadas."},
                "tipo": {"type": "string", "enum": ["correccion", "basal"], "description": "Corrección (Apidra) o basal (Glargina)."},
                "nota": {"type": "string"},
            },
            "required": ["unidades"],
        },
    },
    {
        "name": "registrar_ejercicio",
        "description": "Registra actividad física (hockey, gimnasio, etc.) en Nightscout.",
        "input_schema": {
            "type": "object",
            "properties": {
                "descripcion": {"type": "string"},
                "duracion_min": {"type": "number"},
            },
            "required": ["descripcion"],
        },
    },
    {
        "name": "registrar_glucosa_capilar",
        "description": "Registra una medición de glucosa hecha con tira/glucómetro (BG Check).",
        "input_schema": {
            "type": "object",
            "properties": {"valor": {"type": "number"}},
            "required": ["valor"],
        },
    },
    {
        "name": "registrar_nota",
        "description": "Registra una nota, recordatorio o turno médico en Nightscout.",
        "input_schema": {
            "type": "object",
            "properties": {"texto": {"type": "string"}},
            "required": ["texto"],
        },
    },
    {
        "name": "recordar_dato",
        "description": "Guarda en memoria un dato estable y útil sobre Vittore (comida habitual, horario de entrenamiento, preferencia de trato, apodo).",
        "input_schema": {
            "type": "object",
            "properties": {
                "clave": {"type": "string"},
                "valor": {"type": "string"},
            },
            "required": ["clave", "valor"],
        },
    },
]


async def _run_tool(chat_id: int, name: str, args: dict) -> tuple[str, Optional[dict]]:
    """Ejecuta una herramienta. Devuelve (resultado_para_claude, accion_registrada)."""
    action = None
    try:
        if name == "registrar_comida":
            carbs = args.get("carbs_g")
            insulin = args.get("insulina_u")
            desc = args.get("descripcion", "comida")
            tipo_ev = args.get("tipo_evento", "comida")
            # Pre-bolo: minutos de espera entre insulina y comida (campo nativo preBolus de NS)
            espera = args.get("espera_min")
            try:
                espera = int(round(float(espera))) if espera is not None else None
            except (TypeError, ValueError):
                espera = None
            notes = desc
            if espera:
                notes = f"{desc} · pre-bolo {espera}min"
            # Mapear al eventType correcto de Nightscout
            if tipo_ev == "correccion_baja":
                event = "Carb Correction"
            elif tipo_ev == "colacion":
                event = "Snack Bolus"
            else:
                # comida principal (almuerzo, cena, desayuno)
                event = "Meal Bolus"
            ok = await post_treatment(event, carbs=carbs, insulin=insulin, notes=notes, preBolus=espera)
            action = {"tipo": "comida", "desc": desc, "evento_ns": event, "espera_min": espera}
            return (f"{'ok' if ok else 'error'}: {tipo_ev} registrada como {event} (carbs={carbs}, insulina={insulin}, pre-bolo={espera}min)", action)

        if name == "registrar_insulina":
            u = args.get("unidades")
            tipo = args.get("tipo", "correccion")
            event = "Temp Basal" if tipo == "basal" else "Correction Bolus"
            # Para basal usamos una nota clara; NS no modela glargina como basal temp real.
            if tipo == "basal":
                ok = await post_treatment("Note", insulin=u, notes=f"Basal Glargina {u}U")
            else:
                ok = await post_treatment(event, insulin=u, notes=args.get("nota"))
            action = {"tipo": "insulina", "u": u, "clase": tipo}
            return (f"{'ok' if ok else 'error'}: insulina {tipo} {u}U registrada", action)

        if name == "registrar_ejercicio":
            desc = args.get("descripcion", "ejercicio")
            dur = args.get("duracion_min")
            ok = await post_treatment("Exercise", duration=dur, notes=desc)
            action = {"tipo": "ejercicio", "desc": desc, "dur": dur}
            return (f"{'ok' if ok else 'error'}: ejercicio registrado ({desc}, {dur}min)", action)

        if name == "registrar_glucosa_capilar":
            val = args.get("valor")
            ok = await post_treatment("BG Check", glucose=val, glucoseType="Finger")
            action = {"tipo": "bgcheck", "valor": val}
            return (f"{'ok' if ok else 'error'}: glucosa capilar {val} registrada", action)

        if name == "registrar_nota":
            txt = args.get("texto", "")
            ok = await post_treatment("Note", notes=txt)
            action = {"tipo": "nota", "texto": txt}
            return (f"{'ok' if ok else 'error'}: nota registrada", action)

        if name == "recordar_dato":
            store.update_profile(chat_id, args.get("clave", "nota"), args.get("valor", ""))
            action = {"tipo": "memoria"}
            return ("ok: dato recordado", action)

    except Exception as e:
        logger.error(f"_run_tool {name}: {e}")
        return (f"error ejecutando {name}: {e}", None)
    return (f"herramienta desconocida: {name}", None)


async def build_full_context(chat_id: int) -> str:
    """Arma el contexto que ve Claude en cada mensaje."""
    entry, entries3, treatments = await asyncio.gather(
        get_current_bg(), get_recent_entries(3), get_treatments(6))

    parts = []
    if entry:
        parts.append("[GLUCOSA ACTUAL]\n" + format_bg_message(entry))

    # Predicción fisiológica (Fase 1): IOB/COB propios + proyección 30/60 + riesgo.
    if entry:
        slope = bg_slope(entries3)
        boluses, carbs, ex = parse_active_treatments(treatments, datetime.now(timezone.utc))
        risk = assess_hypo_risk(entry.get("sgv", 0), slope, boluses, carbs, ex)
        pred = {
            "current": entry.get("sgv", 0), "slope": slope,
            "p30": risk["p30"], "b30": risk["b30"], "p60": risk["p60"], "b60": risk["b60"],
            "risk": risk["level"], "iob": compute_iob(boluses), "cob": compute_cob(carbs),
            "exercise_recent": ex,
        }
        parts.append("[PREDICCIÓN]\n" + format_prediction(pred))

    parts.append("[ÚLTIMAS 3H]\n" + summarize_entries(entries3))

    if treatments:
        parts.append("[TRATAMIENTOS ÚLTIMAS 6H]\n" + format_treatments(treatments))

    profile = store.get_profile(chat_id)
    if profile:
        parts.append("[LO QUE SÉ DE VITTORE]\n" +
                     "\n".join(f"- {k}: {v}" for k, v in profile.items()))

    # Patrones (una vez al día es suficiente, pero lo calculamos liviano aquí).
    try:
        days_entries = await get_entries_days(14)
        patterns = analyze_patterns(days_entries, treatments, 14)
        digest = format_patterns_digest(patterns, compact=True)
        if digest:
            parts.append("[PATRONES RECIENTES]\n" + digest)
    except Exception as e:
        logger.error(f"patrones en contexto: {e}")

    return "\n\n".join(parts)


def extract_answer(text: str) -> str:
    """Descarta el bloque de razonamiento <think> de modelos como gpt-oss."""
    if not text:
        return ""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        return text.replace("<think>", "").strip()
    return text.strip()


# Telegram (por defecto, sin parse_mode) NO renderiza markdown: **negrita**, #títulos
# y - viñetas salen con los caracteres literales. Limpiamos el markdown del texto
# generado por el modelo antes de enviarlo, para no depender de que "se porte bien".
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_RE_UNDER = re.compile(r"__(.+?)__", re.S)
_RE_CODE = re.compile(r"`([^`]+)`")
_RE_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_RE_BULLET = re.compile(r"^(\s*)[-*]\s+", re.M)
_RE_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])")
_RE_BLANKS = re.compile(r"\n{3,}")


def tg_clean(text: str) -> str:
    """Convierte el markdown del modelo a texto plano apto para Telegram."""
    if not text:
        return text
    t = _RE_BOLD.sub(r"\1", text)      # **negrita** -> negrita
    t = _RE_UNDER.sub(r"\1", t)         # __x__ -> x
    t = _RE_CODE.sub(r"\1", t)          # `code` -> code
    t = _RE_HEADER.sub("", t)           # ## Título -> Título
    t = _RE_BULLET.sub(r"\1• ", t)      # "- item" / "* item" -> "• item"
    t = _RE_ITALIC.sub(r"\1", t)        # *cursiva* -> cursiva
    t = t.replace("**", "")             # cualquier ** suelto
    t = _RE_BLANKS.sub("\n\n", t)       # máx 1 línea en blanco seguida
    return t.strip()


def _openai_tools() -> list:
    """Convierte TOOLS (formato Anthropic) al formato function-calling de Groq/OpenAI."""
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    } for t in TOOLS]


def _claude_call(messages: list, tools: Optional[list] = None, system: str = SYSTEM_PROMPT,
                 max_tokens: int = 900):
    """Llamada síncrona a Claude (se corre en thread)."""
    kwargs = dict(model=ANTHROPIC_MODEL, max_tokens=max_tokens, system=system, messages=messages)
    if tools:
        kwargs["tools"] = tools
    return anthropic_client.messages.create(**kwargs)


def _groq_call(messages: list, tools: Optional[list] = None, model: Optional[str] = None,
               max_tokens: int = 900, temperature: float = 0.6):
    """Llamada síncrona a Groq (chat.completions, se corre en thread)."""
    kwargs = dict(model=model or GROQ_CHAT_MODEL, messages=messages,
                  max_tokens=max_tokens, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return groq_client.chat.completions.create(**kwargs)


# ─── Registro por extracción JSON (para modelos sin tool-calling, ej. gpt-oss) ─
_EXTRACT_SYSTEM = "Sos un extractor de datos de diabetes. Respondés SOLO JSON válido, sin texto alrededor."


def _extract_prompt(last_bot: str, user_msg: str) -> str:
    return (
        "Detectá si en el mensaje de Vittore hay algo para REGISTRAR en su "
        "seguimiento: una comida, una dosis de insulina que YA se aplicó, ejercicio, "
        "una medición de glucosa capilar, o una nota. Usá el contexto: si el bot "
        "preguntó 'cuántas unidades' y Vittore responde un número, es una insulina.\n"
        "Respondé SOLO este JSON (sin explicación):\n"
        '{"acciones": [{"tipo": "comida|insulina|ejercicio|glucosa|nota", '
        '"descripcion": "", "carbs_g": null, "insulina_u": null, "espera_min": null, "unidades": null, '
        '"clase": "correccion|basal", "tipo_comida": "comida|colacion|correccion_baja", '
        '"duracion_min": null, "valor": null, "texto": ""}]}\n'
        "Reglas:\n"
        "- NO inventes datos que no dijo.\n"
        "- espera_min (pre-bolo) solo si dijo cuántos minutos esperó entre la insulina y la comida (ej. 'esperé 15 min', 'me la puse 20 min antes de comer').\n"
        "- Insulina solo si hay UNIDADES explícitas.\n"
        "- Glucosa capilar solo si hay un VALOR numérico.\n"
        "- Si el usuario PREGUNTA por info nutricional (cuántos carbs tiene, qué "
        "proteínas tiene, cuántas calorías) pero NO dice que lo comió, NO registres comida.\n"
        "- Si dice 'no registres', 'no dije que registre', 'no anotar', devolvé acciones vacías.\n"
        'Si no hay nada concreto para registrar, devolvé {"acciones": []}.\n\n'
        f"[BOT PREGUNTÓ/DIJO]: {last_bot[:300]}\n[VITTORE DIJO]: {user_msg}"
    )


def _map_extracted_action(a: dict) -> Optional[tuple]:
    """Convierte una acción del JSON al par (nombre_herramienta, args) de _run_tool."""
    tipo = (a.get("tipo") or "").lower()
    if tipo == "comida":
        return ("registrar_comida", {
            "descripcion": a.get("descripcion") or "comida",
            "carbs_g": a.get("carbs_g"), "insulina_u": a.get("insulina_u"),
            "espera_min": a.get("espera_min"),
            "tipo_evento": a.get("tipo_comida") or "comida"})
    if tipo == "insulina":
        u = a.get("unidades") or a.get("insulina_u")
        if not u:
            return None
        return ("registrar_insulina", {
            "unidades": u, "tipo": a.get("clase") or "correccion", "nota": a.get("texto")})
    if tipo == "ejercicio":
        return ("registrar_ejercicio", {
            "descripcion": a.get("descripcion") or "ejercicio", "duracion_min": a.get("duracion_min")})
    if tipo == "glucosa":
        if not a.get("valor"):
            return None
        return ("registrar_glucosa_capilar", {"valor": a.get("valor")})
    if tipo == "nota":
        txt = a.get("texto") or a.get("descripcion")
        if not txt:
            return None
        return ("registrar_nota", {"texto": txt})
    return None


def _describe_logged(actions: list[dict]) -> str:
    partes = []
    for a in actions:
        t = a.get("tipo")
        if t == "comida":
            txt = f"🍽 {a.get('desc', 'comida')}"
            if a.get("espera_min"):
                txt += f" (pre-bolo {a['espera_min']}min)"
            partes.append(txt)
        elif t == "insulina":
            partes.append(f"💉 {a.get('u', '?')}U ({a.get('clase', 'correccion')})")
        elif t == "ejercicio":
            partes.append(f"🏃 {a.get('desc', 'ejercicio')}")
        elif t == "bgcheck":
            partes.append(f"🩸 {a.get('valor', '?')}")
        elif t == "nota":
            partes.append("📝 nota")
    return " · ".join(partes) if partes else "registro"


async def extract_and_log(chat_id: int, last_bot: str, user_message: str) -> list[dict]:
    """Extrae registros del mensaje con un llamado JSON (sin tool-calling) y los guarda."""
    if not groq_client:
        return []
    try:
        resp = await asyncio.to_thread(
            _groq_call,
            [{"role": "system", "content": _EXTRACT_SYSTEM},
             {"role": "user", "content": _extract_prompt(last_bot, user_message)}],
            None, None, 400, 0.0)
        raw = extract_answer(resp.choices[0].message.content or "")
        s, e = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[s:e + 1]) if (s >= 0 and e > s) else {}
    except Exception as ex:
        logger.error(f"extract_and_log: {type(ex).__name__}: {ex}")
        return []
    actions = []
    for a in (data.get("acciones") or [])[:4]:
        mapped = _map_extracted_action(a)
        if not mapped:
            continue
        name, args = mapped
        _, action = await _run_tool(chat_id, name, args)
        if action:
            actions.append(action)
    return actions


async def ask_llm(chat_id: int, user_message: str, ns_context: str) -> tuple[str, list[dict]]:
    """Chat con herramientas. Enruta al proveedor configurado (Groq gratis o Claude)."""
    provider = brain_provider()
    if provider == "anthropic":
        return await _ask_anthropic(chat_id, user_message, ns_context)
    if provider == "groq":
        return await _ask_groq(chat_id, user_message, ns_context)
    return ("El asistente inteligente no está configurado (falta GROQ_API_KEY o ANTHROPIC_API_KEY). "
            "Igual podés usar /glucosa, /resumen, /tratamientos y /patrones.", [])


async def _ask_anthropic(chat_id: int, user_message: str, ns_context: str) -> tuple[str, list[dict]]:
    history = store.get_history(chat_id, limit=20)
    messages = list(history)
    messages.append({"role": "user", "content": f"{ns_context}\n\n[MENSAJE DE VITTORE]\n{user_message}"})
    actions: list[dict] = []
    final_text = ""
    try:
        for _ in range(4):
            resp = await asyncio.to_thread(_claude_call, messages, TOOLS)
            text_blocks = [b.text for b in resp.content if b.type == "text"]
            if text_blocks:
                final_text = "\n".join(t for t in text_blocks if t).strip()
            if resp.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    result, action = await _run_tool(chat_id, block.name, block.input or {})
                    if action:
                        actions.append(action)
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id, "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        if not final_text:
            final_text = "Listo. ✅" if actions else "No estoy seguro de qué necesitás. ¿Me lo repetís?"
        store.add_history(chat_id, "user", user_message)
        store.add_history(chat_id, "assistant", final_text)
        return final_text, actions
    except Exception as e:
        logger.error(f"_ask_anthropic: {type(e).__name__}: {e}")
        return (f"Tuve un problema para pensar la respuesta ({type(e).__name__}). Probá de nuevo en un momento.", actions)


async def _ask_groq(chat_id: int, user_message: str, ns_context: str) -> tuple[str, list[dict]]:
    """Chat con Groq. Si el modelo rechaza las herramientas (bug conocido de
    gpt-oss + tools = 400), reintenta SIN herramientas para que el chat responda
    igual (el registro automático se pierde en ese caso, pero no se rompe el bot)."""
    history = store.get_history(chat_id, limit=20)
    base = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)
    base.append({"role": "user", "content": f"{ns_context}\n\n[MENSAJE DE VITTORE]\n{user_message}"})

    last_err = None
    for use_tools in (True, False):
        messages = [dict(m) for m in base]
        actions: list[dict] = []
        final_text = ""
        try:
            for _ in range(4):
                resp = await asyncio.to_thread(
                    _groq_call, messages, _openai_tools() if use_tools else None)
                msg = resp.choices[0].message
                content = extract_answer(msg.content or "")
                if content:
                    final_text = content
                tool_calls = getattr(msg, "tool_calls", None)
                if not tool_calls:
                    break
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    result, action = await _run_tool(chat_id, tc.function.name, args)
                    if action:
                        actions.append(action)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "name": tc.function.name, "content": result})
            # Si el modelo no registró nada por herramientas, intentar por extracción JSON
            # (necesario con gpt-oss, que tiene el tool-calling roto).
            if not actions:
                last_bot = next(
                    (m["content"] for m in reversed(history) if m.get("role") == "assistant"), "")
                logged = await extract_and_log(chat_id, last_bot, user_message)
                if logged:
                    actions = logged
                    final_text = (f"{final_text}\n\n✅ Anotado: {_describe_logged(logged)}").strip()
            if not final_text:
                final_text = "Listo. ✅" if actions else "No estoy seguro de qué necesitás. ¿Me lo repetís?"
            store.add_history(chat_id, "user", user_message)
            store.add_history(chat_id, "assistant", final_text)
            return final_text, actions
        except Exception as e:
            last_err = e
            logger.error(f"_ask_groq (tools={use_tools}): {type(e).__name__}: {e}")
            if use_tools:
                continue  # reintentar sin herramientas
    detail = (str(last_err)[:180] if last_err else "").strip()
    return (f"El modelo de Groq rechazó el pedido ({type(last_err).__name__}). "
            f"Detalle: {detail}", [])


# ─── Visión: foto de comida → carbohidratos ──────────────────────
FOOD_PROMPT = (
    "Analizá esta foto de comida para una persona con diabetes tipo 1. "
    "Estimá los carbohidratos totales. Respondé SOLO un JSON válido con esta forma:\n"
    '{"descripcion": "qué ves, breve", "carbs_g": <número>, "confianza": "alta|media|baja", '
    '"comentario": "1 frase útil, aclarando que es estimación"}\n'
    "Nunca sugieras dosis de insulina."
)
FOOD_SYSTEM = "Sos un asistente nutricional preciso para diabetes tipo 1. Respondés solo JSON."


def _parse_food_json(raw: str) -> tuple[str, Optional[float], str]:
    """Parsea el JSON del modelo de visión. Devuelve (texto, carbs, descripcion)."""
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start:end + 1]) if (start >= 0 and end > start) else {}
    desc = data.get("descripcion", "comida")
    carbs = data.get("carbs_g")
    conf = data.get("confianza", "media")
    comentario = data.get("comentario", "Es una estimación; puede variar según la porción real.")
    # Validar carbs: solo mostrar si es un número real
    carbs_float = None
    if isinstance(carbs, (int, float)) and carbs > 0:
        carbs_float = float(carbs)
        carbs_txt = f"🍞 Carbs estimados: ~{int(carbs)}g (confianza {conf})"
    else:
        carbs_txt = "🍞 Carbs: no pude estimar con certeza"
    texto = f"🍽 {desc}\n{carbs_txt}\nℹ️ {comentario}"
    return texto, carbs_float, desc


async def analyze_food_image(image_bytes: bytes, caption: str = "") -> tuple[str, Optional[float], str]:
    """Estima carbohidratos desde una foto. Devuelve (texto, carbs, descripcion)."""
    provider = brain_provider()
    if provider == "none":
        return ("No puedo analizar fotos sin un cerebro configurado (GROQ_API_KEY o ANTHROPIC_API_KEY).", None, "")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = FOOD_PROMPT + (f"\nComentario del usuario: {caption}" if caption else "")
    try:
        if provider == "anthropic":
            resp = await asyncio.to_thread(
                _claude_call,
                [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ]}],
                None, FOOD_SYSTEM, 600,
            )
            raw = "".join(b.text for b in resp.content if b.type == "text")
        else:  # groq: modelo de visión (qwen)
            resp = await asyncio.to_thread(
                _groq_call,
                [{"role": "system", "content": FOOD_SYSTEM},
                 {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                 ]}],
                None, GROQ_VISION_MODEL, 700, 0.3,
            )
            raw = extract_answer(resp.choices[0].message.content or "")
        return _parse_food_json(raw)
    except Exception as e:
        logger.error(f"analyze_food_image: {type(e).__name__}: {e}")
        return ("No pude analizar bien la foto. Probá con más luz o incluí un cubierto de referencia.", None, "")


# ─── Audio (Whisper de Groq) ─────────────────────────────────────
async def transcribe_voice(file_bytes: bytes) -> str:
    if not groq_client:
        return "[No hay transcripción de audio configurada (falta GROQ_API_KEY)]"
    try:
        def _call():
            audio_file = io.BytesIO(file_bytes)
            audio_file.name = "voice.ogg"
            return groq_client.audio.transcriptions.create(
                model="whisper-large-v3", file=audio_file, language="es",
            )
        transcript = await asyncio.to_thread(_call)
        return transcript.text
    except Exception as e:
        logger.error(f"transcribe_voice: {e}")
        return "[No pude entender el audio, intentá de nuevo]"


# ─── Handlers de Telegram ────────────────────────────────────────
def is_authorized(update: Update) -> bool:
    if not ALLOWED_USERS:
        return True
    return str(update.effective_chat.id) in ALLOWED_USERS


def is_admin(update: Update) -> bool:
    if not ADMIN_USERS:
        return True
    return str(update.effective_chat.id) in ADMIN_USERS


# Estado para el flujo de borrado por Telegram (en memoria, por chat).
_delete_index: dict[int, list[dict]] = defaultdict(list)  # última lista mostrada
_delete_pending: dict[int, dict] = {}                     # borrado a confirmar


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("No tenés autorización para usar este bot.")
        return
    await update.message.reply_text(
        "¡Hola! Soy VittoBot, tu asistente de glucosa 🩺\n\n"
        "Podés:\n"
        "• /glucosa — lectura actual\n"
        "• /prediccion — a dónde va tu glucosa (30/60 min)\n"
        "• /resumen — últimas 3 horas\n"
        "• /tratamientos — insulina y carbs (6h)\n"
        "• /registro — lo de hoy\n"
        "• /patrones — patrones de los últimos 14 días\n"
        "• /borrar — borrar un registro mal cargado (admin)\n"
        "• Escribirme, mandarme audio o una foto de comida 📸\n"
        "• Contarme qué comiste, qué insulina te pusiste o si entrenaste\n\n"
        f"Tu chat ID: {update.effective_chat.id}"
    )


async def cmd_glucosa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    entry = await get_current_bg()
    entries = await get_recent_entries(1)
    if entry:
        msg = format_bg_message(entry)
        slope = bg_slope(entries)
        pred = predict_bg(entry.get("sgv", 0), slope, PREDICT_MIN)
        if pred is not None:
            msg += f"\nEn ~{PREDICT_MIN}min podría estar cerca de {pred} mg/dL."
        await update.message.reply_text(msg)
    else:
        await update.message.reply_text("No pude obtener datos de Nightscout.")


async def cmd_prediccion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    await update.message.chat.send_action("typing")
    pred = await compute_prediction()
    if not pred:
        await update.message.reply_text("No pude obtener datos de Nightscout para predecir.")
        return
    header = f"🩸 Ahora: {pred['current']} mg/dL — {bg_indicator(pred['current'])}\n"
    await update.message.reply_text(header + format_prediction(pred))


async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    entries = await get_recent_entries(3)
    await update.message.reply_text(summarize_entries(entries))


async def cmd_tratamientos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    treatments = await get_treatments(6)
    text = "💉🍞 Tratamientos (6h):\n" + format_treatments(treatments)
    iob, cob = extract_iob_cob(await get_device_status())
    extra = []
    if iob is not None:
        extra.append(f"IOB {iob:.1f}U")
    if cob is not None:
        extra.append(f"COB {cob:.0f}g")
    if extra:
        text += "\n\n⚡ " + " · ".join(extra)
    await update.message.reply_text(text)


def _short_treatment(t: dict) -> str:
    try:
        dt = datetime.fromisoformat(t.get("created_at", "").replace("Z", "+00:00")).astimezone(TZ_AR)
        ts = dt.strftime("%d/%m %H:%M")
    except Exception:
        ts = "?"
    parts = [t.get("eventType", "?")]
    if t.get("insulin"):
        parts.append(f"💉{t['insulin']}U")
    if t.get("carbs"):
        parts.append(f"🍞{t['carbs']}g")
    if t.get("duration"):
        parts.append(f"⏱{t['duration']}min")
    notes = t.get("notes")
    if notes:
        parts.append(f"— {notes[:35]}")
    return f"{ts} {' '.join(parts)}"


async def cmd_borrar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lista tratamientos numerados; /borrar N [M ...] marca para borrar (solo admin)."""
    if not is_admin(update):
        await update.message.reply_text("Solo un administrador puede borrar registros.")
        return
    chat_id = update.effective_chat.id
    args = context.args or []
    nums = [int(a) for a in args if a.isdigit()]

    # /borrar N [M ...]  → preparar confirmación (uno o varios)
    if nums:
        items = _delete_index.get(chat_id, [])
        if not items:
            await update.message.reply_text("Primero mandá /borrar (sin número) para ver la lista actualizada.")
            return
        chosen, malos = [], []
        for n in nums:
            if 1 <= n <= len(items):
                chosen.append(items[n - 1])
            else:
                malos.append(str(n))
        if not chosen:
            await update.message.reply_text("Número(s) inválido(s). Mandá /borrar para ver la lista.")
            return
        _delete_pending[chat_id] = chosen
        lines = ["⚠️ Vas a borrar:"]
        lines += [f"  • {_short_treatment(t)}" for t in chosen]
        if malos:
            lines.append(f"(ignoro números fuera de rango: {', '.join(malos)})")
        lines.append("\nMandá /confirmar para borrar, o /cancelar para dejar.")
        await update.message.reply_text("\n".join(lines))
        return

    # /borrar (o con texto que no es número) → mostrar lista numerada
    treatments = await get_treatments(24)
    if not treatments:
        await update.message.reply_text("No hay tratamientos recientes para borrar.")
        return
    _delete_index[chat_id] = treatments
    _delete_pending.pop(chat_id, None)
    lines = ["🗑 Tratamientos (últimas 24h). Elegí por número, ej: /borrar 2  (o varios: /borrar 2 3 4)\n"]
    for i, t in enumerate(treatments, 1):
        lines.append(f"{i}. {_short_treatment(t)}")
    await update.message.reply_text("\n".join(lines))


async def cmd_confirmar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    chat_id = update.effective_chat.id
    chosen = _delete_pending.get(chat_id)
    if not chosen:
        await update.message.reply_text("No hay nada pendiente de borrar. Usá /borrar primero.")
        return
    ok, fallos = [], []
    for t in chosen:
        _id = t.get("_id")
        if _id and await delete_treatment(_id):
            ok.append(t)
        else:
            fallos.append(t)
    _delete_pending.pop(chat_id, None)
    _delete_index.pop(chat_id, None)  # los números quedan desactualizados tras borrar
    msg = ""
    if ok:
        msg += "✅ Borrado(s):\n" + "\n".join(f"  • {_short_treatment(t)}" for t in ok)
    if fallos:
        msg += ("\n❌ No pude borrar:\n" + "\n".join(f"  • {_short_treatment(t)}" for t in fallos))
    await update.message.reply_text(msg or "No se borró nada.")


async def cmd_cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    _delete_pending.pop(update.effective_chat.id, None)
    await update.message.reply_text("Cancelado. No se borró nada.")


async def cmd_registro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Registros de hoy, leídos de Nightscout (sobreviven reinicios)."""
    if not is_authorized(update):
        return
    treatments = await get_treatments(24)
    today = datetime.now(TZ_AR).strftime("%Y-%m-%d")
    todays = []
    for t in treatments:
        created = t.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(TZ_AR)
            if dt.strftime("%Y-%m-%d") == today:
                todays.append(t)
        except Exception:
            continue
    if not todays:
        await update.message.reply_text("No hay registros de hoy todavía.")
        return
    await update.message.reply_text(f"📋 Hoy ({today}):\n" + format_treatments(todays))


async def cmd_patrones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    await update.message.chat.send_action("typing")
    entries = await get_entries_days(14)
    treatments = await get_treatments(24 * 14)
    p = analyze_patterns(entries, treatments, 14)
    digest = format_patterns_digest(p, compact=False)
    # Que el cerebro lo narre para la consulta con el endocrinólogo, si está disponible.
    if brain_provider() != "none" and p["daily"]:
        try:
            narr_prompt = (
                "Estos son los patrones de glucosa de Vittore (14 días). Escribí un resumen "
                "breve y claro (máx 10 líneas, español argentino, sin tablas) para que el papá "
                "lo comente con el endocrinólogo. Destacá riesgos (hipos nocturnas, ejercicio) "
                "y lo que mejoró. NO sugieras dosis.\n\n" + digest
            )
            if brain_provider() == "anthropic":
                resp = await asyncio.to_thread(
                    _claude_call, [{"role": "user", "content": narr_prompt}], None, SYSTEM_PROMPT, 500)
                narr = "".join(b.text for b in resp.content if b.type == "text").strip()
            else:
                resp = await asyncio.to_thread(
                    _groq_call,
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": narr_prompt}],
                    None, None, 500, 0.5)
                narr = extract_answer(resp.choices[0].message.content or "")
            await update.message.reply_text(tg_clean(narr or digest))
            return
        except Exception as e:
            logger.error(f"cmd_patrones narrativa: {e}")
    await update.message.reply_text(digest)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    chat_id = update.effective_chat.id
    await update.message.chat.send_action("typing")
    ns_context = await build_full_context(chat_id)
    response, actions = await ask_llm(chat_id, update.message.text, ns_context)
    await update.message.reply_text(tg_clean(response))
    _schedule_followups(context, chat_id, actions)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    chat_id = update.effective_chat.id
    await update.message.chat.send_action("typing")
    voice = update.message.voice or update.message.audio
    file = await context.bot.get_file(voice.file_id)
    file_bytes = await file.download_as_bytearray()
    text = await transcribe_voice(bytes(file_bytes))
    if text.startswith("["):
        await update.message.reply_text(text)
        return
    await update.message.reply_text(f"🎙 Entendí: _{text}_", parse_mode="Markdown")
    ns_context = await build_full_context(chat_id)
    response, actions = await ask_llm(chat_id, text, ns_context)
    await update.message.reply_text(tg_clean(response))
    _schedule_followups(context, chat_id, actions)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    chat_id = update.effective_chat.id
    await update.message.chat.send_action("typing")
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_bytes = await file.download_as_bytearray()
    caption = update.message.caption or ""

    # Analizar la foto (estimar carbs, describir comida)
    texto, carbs, desc = await analyze_food_image(bytes(file_bytes), caption)
    texto = tg_clean(texto)

    # Detectar si el usuario QUIERE registrar o solo pregunta info nutricional.
    # Palabras que indican que COMIÓ (registrar): "comí", "almorcé", "cené", "merendé",
    # "desayuné", "estoy comiendo", "registrá", "anotá".
    # Si solo pregunta "cuántos carbs tiene" o "qué tiene esto" → NO registrar.
    caption_lower = caption.lower()
    quiere_registrar = any(w in caption_lower for w in (
        "comí", "almorcé", "cené", "merendé", "desayuné", "estoy comiendo",
        "estoy almorzando", "estoy cenando", "estoy merendando",
        "registr", "anot", "cargá", "cargalo",
    ))

    if quiere_registrar and carbs is not None:
        food_desc = desc or caption or "Comida (foto)"
        # Detectar si es colación o comida principal por el caption
        caption_lower_ev = caption_lower
        if any(w in caption_lower_ev for w in ("merienda", "snack", "colación", "colacion", "algo")):
            photo_event = "Snack Bolus"
        elif any(w in caption_lower_ev for w in ("remontar", "hipo", "baja", "azúcar rápido", "jugo")):
            photo_event = "Carb Correction"
        else:
            photo_event = "Meal Bolus"
        await post_treatment(photo_event, carbs=carbs, notes=f"{food_desc} [estimado por foto]")
        store.add_history(chat_id, "user", f"[Foto de comida] {caption}")
        store.add_history(chat_id, "assistant", texto)
        await update.message.reply_text(f"📸 {texto}\n\n✅ Registrado en Nightscout.")
    else:
        # Solo informar, NO registrar
        store.add_history(chat_id, "user", f"[Foto] {caption}")
        store.add_history(chat_id, "assistant", texto)
        msg = f"📸 {texto}"
        if not quiere_registrar and carbs is not None:
            msg += "\n\n💡 Si querés que lo registre, decime 'anotalo' o 'registralo'."
        await update.message.reply_text(msg)


def _schedule_followups(context: ContextTypes.DEFAULT_TYPE, chat_id: int, actions: list[dict]):
    """Programa recordatorios proactivos según lo que se registró."""
    if not context.job_queue:
        return
    for a in actions:
        if a.get("tipo") == "ejercicio":
            # Recordatorio de medir ~45 min después del ejercicio.
            context.job_queue.run_once(
                _post_exercise_nudge, when=45 * 60,
                data={"chat_id": chat_id, "desc": a.get("desc", "el entrenamiento")},
                name=f"nudge_ex_{chat_id}",
            )


async def _post_exercise_nudge(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    chat_id = data["chat_id"]
    entry = await get_current_bg()
    extra = f"\n{format_bg_message(entry)}" if entry else ""
    await context.bot.send_message(
        chat_id=chat_id,
        text=(f"🏒 ¿Cómo venís después de {data['desc']}? Acordate de medir.{extra}\n"
              "Ojo con las bajas en las horas siguientes y de madrugada."),
    )


# ─── Alertas inteligentes (anti-spam + predictivas) ──────────────
async def check_alerts(context: ContextTypes.DEFAULT_TYPE):
    entry = await get_current_bg()
    now = datetime.now(timezone.utc)
    state = store.get_alert_state()
    last_level = state.get("level")
    last_ts = state.get("ts")
    if isinstance(last_ts, str):
        try:
            last_ts = datetime.fromisoformat(last_ts)
        except Exception:
            last_ts = None

    # 1) Sensor caído / sin datos
    if not entry:
        if last_level != "no_data":
            _broadcast(context, "📡 No estoy recibiendo datos del sensor. Revisá el FreeStyle / LibreLinkUp.")
            store.set_alert_state({"level": "no_data", "ts": now.isoformat()})
        return

    sgv = entry.get("sgv", 0)
    date_ms = entry.get("date", 0)
    age_min = (now - datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)).total_seconds() / 60
    if age_min > SENSOR_GAP_MIN:
        if last_level != "no_data":
            _broadcast(context, f"📡 Última lectura hace {int(age_min)} min. El sensor puede estar caído.")
            store.set_alert_state({"level": "no_data", "ts": now.isoformat()})
        return

    level = classify_bg(sgv)
    entries, treatments = await asyncio.gather(get_recent_entries(1), get_treatments(6))
    slope = bg_slope(entries)
    boluses, carbs, ex = parse_active_treatments(treatments, now)
    iob = compute_iob(boluses)
    risk = assess_hypo_risk(sgv, slope, boluses, carbs, ex)

    # 2) Aviso predictivo (modelo fisiológico): en rango pero yendo hacia hipoglucemia
    predictive = level == "in_range" and risk["level"] in ("alto", "medio")
    effective = "predicted_low" if predictive else level

    # ¿Corresponde avisar?
    should = False
    if effective != last_level:
        # Cambio de estado. Avisamos si es un estado "malo", o si se RECUPERÓ a rango.
        if effective in ("urgent_low", "low", "urgent_high", "high", "predicted_low"):
            should = True
        elif effective == "in_range" and last_level in ("urgent_low", "low", "urgent_high", "high", "predicted_low"):
            _broadcast(context, f"✅ Glucosa de nuevo en rango: {sgv} mg/dL. Buen manejo.")
            store.set_alert_state({"level": "in_range", "ts": now.isoformat()})
            return
    else:
        # Mismo estado malo: re-avisar solo tras el cooldown.
        if effective in ("urgent_low", "urgent_high") and last_ts:
            if (now - last_ts).total_seconds() / 60 >= ALERT_COOLDOWN_MIN:
                should = True

    if not should:
        # Igual actualizamos el nivel base (sin re-notificar) para no perder transiciones.
        if effective == "in_range":
            store.set_alert_state({"level": "in_range", "ts": now.isoformat()})
        return

    msg = _build_alert_message(entry, effective, slope, iob, risk)
    _broadcast(context, msg)
    store.set_alert_state({"level": effective, "ts": now.isoformat()})


def _build_alert_message(entry: dict, level: str, slope: Optional[float], iob: Optional[float],
                         risk: Optional[dict] = None) -> str:
    base = format_bg_message(entry)
    iob_txt = f"\n💉 Insulina activa: {iob:.1f}U" if iob else ""
    pred_txt = ""
    if risk:
        pred_txt = f"\n🔮 Proyección: ~{risk['p30']} en 30min · ~{risk['p60']} en 60min"
    if level == "urgent_low":
        return f"🚨 HIPOGLUCEMIA URGENTE 🚨\n{base}{iob_txt}\n¡Vittore necesita azúcar rápido AHORA! Medir de nuevo en 15 min."
    if level == "predicted_low":
        return (f"⚠️ Ojo: el modelo ve una hipoglucemia en camino.{pred_txt}\n{base}{iob_txt}\n"
                "Considerá una colación/azúcar rápido y volvé a medir.")
    if level == "low":
        return f"🟡 Glucosa baja.\n{base}{iob_txt}\nConsiderá carbohidratos rápidos y medir en 15 min."
    if level == "urgent_high":
        return f"🚨 Glucosa muy alta.\n{base}{iob_txt}\nSi persiste, seguí las pautas del endocrinólogo. Ojo con cetonas."
    if level == "high":
        return f"🟠 Glucosa alta.\n{base}{iob_txt}"
    return base


def _broadcast(context: ContextTypes.DEFAULT_TYPE, text: str):
    for chat_id in CAREGIVER_CHAT_IDS:
        try:
            context.application.create_task(
                context.bot.send_message(chat_id=int(chat_id), text=text)
            )
        except Exception as e:
            logger.error(f"broadcast a {chat_id}: {e}")


# ─── Entrenamiento nocturno ML (Fase 2) ─────────────────────────
async def nightly_ml_train(context: ContextTypes.DEFAULT_TYPE):
    """Reentrena el modelo ML con toda la historia disponible. Corre de madrugada."""
    global ml_trained
    if not mlm or store.db is None:
        return

    logger.info("🧠 Iniciando reentrenamiento nocturno ML...")
    try:
        # Obtener toda la historia disponible (últimos 90 días)
        entries = await get_entries_days(90)
        treatments = await get_treatments_days(90)

        if not entries:
            logger.warning("No hay entries para entrenar. Abortando.")
            return

        logger.info(f"Datos obtenidos: {len(entries)} entries, {len(treatments)} treatments")

        # Construir dataset
        dataset = mlm.build_dataset(entries, treatments, ML_CONFIG)
        n = dataset.get("n_samples", 0)
        logger.info(f"Dataset construido: {n} muestras")

        if dataset["X"] is None or n < mlm.MIN_SAMPLES_TRAIN:
            logger.warning(f"Dataset muy chico ({n} muestras, necesito {mlm.MIN_SAMPLES_TRAIN}). "
                         "Seguimos con physio solo.")
            return

        # Entrenar
        trained = mlm.train_models(dataset)
        if trained is None:
            logger.warning("Entrenamiento no produjo modelo. Seguimos con el anterior.")
            return

        # Guardar en MongoDB
        saved = mlm.save_model_to_mongo(store.db, trained)
        if saved:
            ml_trained = trained  # actualizar en memoria
            metrics = trained.get("metrics", {})
            logger.info(f"✓ Modelo ML reentrenado: {n} muestras, "
                       f"MAE30={metrics.get('mae_30', '?')} MAE60={metrics.get('mae_60', '?')}")

            # Notificar a los cuidadores (breve, no spammear)
            mae30 = metrics.get("mae_30", "?")
            mae60 = metrics.get("mae_60", "?")
            cov30 = metrics.get("coverage_80_30", "?")
            sens = metrics.get("hypo_sensitivity_30", "?")
            _broadcast(context,
                       f"🧠 Modelo ML reentrenado (noche)\n"
                       f"  {n} datos · error ±{mae30}/{mae60} mg/dL (30/60 min)\n"
                       f"  Cobertura banda: {cov30}% · Sensibilidad hipo: {sens}%")
        else:
            logger.error("No pude guardar el modelo en MongoDB.")

    except Exception as e:
        logger.error(f"Error en reentrenamiento nocturno: {e}", exc_info=True)


# ─── Resumen diario proactivo ────────────────────────────────────
async def daily_evening_summary(context: ContextTypes.DEFAULT_TYPE):
    """Cada noche: resumen del día + datos Garmin + aviso de riesgo nocturno."""
    entries = await get_recent_entries(24)
    treatments = await get_treatments(24)
    summary = summarize_entries(entries)
    ex_today = [t for t in treatments if t.get("eventType") == "Exercise"]
    warn = ""
    if ex_today:
        warn = ("\n🌙 Hoy hubo entrenamiento: ojo con las bajas de madrugada. "
                "Puede convenir medir antes de dormir.")

    # Datos Garmin del día
    garmin_section = ""
    if garmin and garmin._authenticated:
        try:
            daily = garmin.get_daily_summary()
            parts = []
            if daily.get("sleep"):
                parts.append(garmin.format_sleep_note(daily["sleep"]))
                # Alertar si durmió poco (afecta glucosa)
                total_min = daily["sleep"].get("total_min", 0)
                if 0 < total_min < 360:  # menos de 6 horas
                    parts.append("   ⚠️ Poco sueño: puede haber resistencia a insulina mañana")
            if daily.get("steps"):
                parts.append(garmin.format_steps_note(daily["steps"]))
            vitals = garmin.format_vitals_note(
                daily.get("heart_rate"), daily.get("stress"), daily.get("body_battery"))
            if vitals:
                parts.append(vitals)
            if parts:
                garmin_section = "\n\n⌚ *Garmin hoy:*\n" + "\n".join(parts)
        except Exception as e:
            logger.error(f"Error datos Garmin para resumen: {e}")

    _broadcast(context, f"🌆 Resumen del día:\n{summary}{warn}{garmin_section}")


# ─── Garmin Bridge: polling y comando ─────────────────────────────
async def garmin_poll(context: ContextTypes.DEFAULT_TYPE):
    """Pollea Garmin Connect cada 15 min y registra actividades nuevas en Nightscout."""
    if garmin is None or not garmin.is_configured():
        return

    # Autenticar si es necesario
    if not garmin._authenticated:
        ok = await garmin.authenticate()
        if not ok:
            logger.warning("Garmin poll: no se pudo autenticar")
            return

    activities = garmin.get_recent_activities(hours=2)
    synced = 0
    for act in activities:
        treatment = garmin.format_for_nightscout(act)
        if treatment is None:
            continue
        # Postear a Nightscout
        ok = await post_treatment(
            event_type="Exercise",
            duration=treatment.get("duration"),
            notes=treatment.get("notes", ""),
            created_at=treatment.get("created_at"),
        )
        if ok:
            garmin.mark_activity_synced(act)
            synced += 1
            logger.info(f"Garmin → NS: {act.get('activityName')} ({treatment.get('duration')} min)")

    if synced > 0:
        _broadcast(context,
                   f"⌚ Garmin: {synced} actividad(es) sincronizada(s) con Nightscout automáticamente.")


async def garmin_daily_collect(context: ContextTypes.DEFAULT_TYPE):
    """Cada mañana: recolecta datos de sueño, pasos y vitales de ayer y hoy."""
    if garmin is None or not garmin._authenticated:
        return
    try:
        # Recolectar datos de ayer (sueño completo) y hoy
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        garmin.get_daily_summary(yesterday)
        today = datetime.now().strftime("%Y-%m-%d")
        daily = garmin.get_daily_summary(today)

        # Notificar resumen de sueño por la mañana
        if daily.get("sleep") and daily["sleep"].get("total_min", 0) > 0:
            sleep_note = garmin.format_sleep_note(daily["sleep"])
            total_min = daily["sleep"].get("total_min", 0)
            msg = f"☀️ Buenos días!\n{sleep_note}"
            if 0 < total_min < 360:
                msg += "\n⚠️ Pocas horas de sueño — puede haber más resistencia a insulina hoy."
            elif total_min >= 480:
                msg += "\n✅ Buen descanso."
            _broadcast(context, msg)

        logger.info(f"Garmin daily collect OK: sueño={daily.get('sleep') is not None}, "
                    f"pasos={daily.get('steps') is not None}")
    except Exception as e:
        logger.error(f"Error en garmin_daily_collect: {e}")


async def cmd_garmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra el estado del bridge Garmin + datos de hoy."""
        if not is_authorized(update):
        return
    if garmin is None:
        await update.message.reply_text("⌚ Garmin Bridge no disponible (módulo no instalado).")
        return
    if not garmin.is_configured():
        await update.message.reply_text(
            "⌚ Garmin Bridge no configurado.\n"
            "Faltan variables: GARMIN_EMAIL y GARMIN_PASSWORD en Railway.")
        return

    status = garmin.get_status()
    lines = [
        "⌚ *Estado Garmin Bridge*",
        f"  Configurado: {'✅' if status['configured'] else '❌'}",
        f"  Autenticado: {'✅' if status['authenticated'] else '❌'}",
        f"  Actividades sync: {status['activities_synced']}",
        f"  Último sync: {status['last_sync']}",
    ]

    # Si está autenticado, mostrar datos de hoy
    if garmin._authenticated:
        try:
            daily = garmin.get_daily_summary()
            lines.append("")
            lines.append("📊 *Datos de hoy:*")
            if daily.get("sleep"):
                lines.append(f"  {garmin.format_sleep_note(daily['sleep'])}")
            if daily.get("steps"):
                lines.append(f"  {garmin.format_steps_note(daily['steps'])}")
            vitals = garmin.format_vitals_note(
                daily.get("heart_rate"), daily.get("stress"), daily.get("body_battery"))
            if vitals:
                lines.append(f"  {vitals}")
        except Exception as e:
            lines.append(f"\n  Error cargando datos: {e}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ─── Main ────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("glucosa", cmd_glucosa))
    app.add_handler(CommandHandler("prediccion", cmd_prediccion))
    app.add_handler(CommandHandler("resumen", cmd_resumen))
    app.add_handler(CommandHandler("tratamientos", cmd_tratamientos))
    app.add_handler(CommandHandler("registro", cmd_registro))
    app.add_handler(CommandHandler("patrones", cmd_patrones))
    app.add_handler(CommandHandler("borrar", cmd_borrar))
    app.add_handler(CommandHandler("confirmar", cmd_confirmar))
    app.add_handler(CommandHandler("cancelar", cmd_cancelar))
    app.add_handler(CommandHandler("garmin", cmd_garmin))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Alertas cada 5 minutos (ahora con anti-spam y predicción)
    app.job_queue.run_repeating(check_alerts, interval=300, first=15)
    # Resumen diario a las 22:30 hora Argentina
    app.job_queue.run_daily(daily_evening_summary, time=dtime(22, 30, tzinfo=TZ_AR))
    # Reentrenamiento ML a las 04:00 hora Argentina (madrugada, mínimo uso)
    app.job_queue.run_daily(nightly_ml_train, time=dtime(4, 0, tzinfo=TZ_AR))
    # Garmin Bridge: polling cada 15 minutos + recolección diaria a las 08:00
    if garmin and garmin.is_configured():
        app.job_queue.run_repeating(garmin_poll, interval=900, first=30)
        app.job_queue.run_daily(garmin_daily_collect, time=dtime(8, 0, tzinfo=TZ_AR))

    ml_status = "no disponible"
    if ml_trained:
        ml_status = f"cargado (v{ml_trained.get('version', '?')}, {ml_trained.get('n_samples', '?')} muestras)"
    elif mlm:
        ml_status = "sin modelo aún (se entrena a las 04:00)"

    garmin_status = "no disponible"
    if garmin and garmin.is_configured():
        garmin_status = "configurado (polling cada 15 min)"
    elif garmin:
        garmin_status = "sin credenciales"

    persist = "MongoDB" if store.db is not None else "RAM"
    logger.info(f"VittoreD1Bot v5 iniciado ✓ | cerebro: {brain_label()} | "
                f"persistencia: {persist} | ML: {ml_status} | Garmin: {garmin_status}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
