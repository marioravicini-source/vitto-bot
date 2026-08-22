"""
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
    """Junta datos de Nightscout y devuelve la predicción completa."""
    entry = await get_current_bg()
    if not entry:
        return None
    entries, treatments = await asyncio.gather(get_recent_entries(1), get_treatments(6))
    current = entry.get("sgv", 0)
    slope = bg_slope(entries)
    boluses, carbs, ex = parse_active_treatments(treatments, datetime.now(timezone.utc))
    risk = assess_hypo_risk(current, slope, boluses, carbs, ex)
    return {
        "current": current, "slope": slope,
        "p30": risk["p30"], "b30": risk["b30"],
        "p60": risk["p60"], "b60": risk["b60"],
        "risk": risk["level"],
        "iob": compute_iob(boluses), "cob": compute_cob(carbs),
        "exercise_recent": ex,
    }


def format_prediction(pred: dict) -> str:
    """Texto compacto de la predicción para Telegram / contexto."""
    trend = f"{pred['slope'] * 60:+.0f}/h" if pred.get("slope") is not None else "s/d"
    risk_emoji = {"alto": "🔴", "medio": "🟡", "ninguno": "🟢"}.get(pred["risk"], "🟢")
    risk_txt = {"alto": "riesgo ALTO de hipo", "medio": "riesgo medio de hipo",
                "ninguno": "sin riesgo de hipo a la vista"}.get(pred["risk"])
    lines = [
        f"🔮 Predicción (tendencia {trend})",
        f"  +30 min: ~{pred['p30']} mg/dL (±{pred['b30']})",
        f"  +60 min: ~{pred['p60']} mg/dL (±{pred['b60']})",
        f"{risk_emoji} {risk_txt}",
        f"💉 IOB {pred['iob']:.1f}U · 🍞 COB {pred['cob']:.0f}g",
    ]
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

CÓMO PENSAR (esto es lo que te hace útil, no un tablero):
- Correlacioná: relacioná glucosa con comidas, insulina activa (IOB) y ejercicio del contexto. Ej: "venís bajando y todavía tenés insulina activa de la corrección de las 17:28, ojo".
- Anticipá: si la tendencia va hacia una baja o suba, decilo antes de que pase.
- Usá los PATRONES del contexto (hipos nocturnas post-hockey, picos post-comida) para dar avisos concretos y con memoria.
- Cerrá con UNA observación útil, no con frases genéricas ("todo bien, avisame").

REGISTRAR (usá las herramientas, no lo hagas a mano):
Cuando Vittore cuente que comió, se aplicó insulina, hizo ejercicio, midió su glucosa capilar, o quiera dejar una nota/turno, LLAMÁ a la herramienta correspondiente con los datos que puedas extraer (gramos de carbohidratos si los sabés, unidades de insulina, minutos de ejercicio). Si falta un dato importante y es fácil, preguntá en una línea; si no, registrá con lo que hay. Después confirmá en 1 línea: "✅ Registrado: 🍽 Pasta (~60g)".

MEMORIA: si aprendés algo estable y útil (comida habitual, horario de entrenamiento, cómo prefiere que le hablen, su apodo para vos), guardalo con recordar_dato.

FORMATO TELEGRAM:
- Nada de tablas markdown (Telegram no las renderiza). Usá texto plano y emojis como separadores.
- Conciso: máximo ~12 líneas. No repitas datos crudos que ya ve en su app.
- Español argentino, directo, cálido pero sin ser empalagoso ni sermonear. Máx 5 emojis.
- No arranques con "¡Hola! 👋 Aquí tienes". Andá al grano con calidez.
- Sin culpa ni reto: celebrá los logros, acompañá los momentos difíciles.

Sos acompañamiento, no reemplazás al médico ni a la familia. Las estimaciones (carbs por foto, IOB, predicción) son aproximadas y las comunicás como tales."""

TOOLS = [
    {
        "name": "registrar_comida",
        "description": "Registra una comida/colación en Nightscout. Usar cuando el usuario cuenta que comió algo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "descripcion": {"type": "string", "description": "Qué comió, breve."},
                "carbs_g": {"type": "number", "description": "Carbohidratos estimados en gramos, si se pueden inferir."},
                "insulina_u": {"type": "number", "description": "Unidades de insulina que se aplicó junto a la comida, si las mencionó."},
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
            event = "Meal Bolus" if insulin else "Carb Correction"
            ok = await post_treatment(event, carbs=carbs, insulin=insulin, notes=desc)
            action = {"tipo": "comida", "desc": desc}
            return (f"{'ok' if ok else 'error'}: comida registrada (carbs={carbs}, insulina={insulin})", action)

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
    history = store.get_history(chat_id, limit=20)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += list(history)
    messages.append({"role": "user", "content": f"{ns_context}\n\n[MENSAJE DE VITTORE]\n{user_message}"})
    actions: list[dict] = []
    final_text = ""
    tools = _openai_tools()
    try:
        for _ in range(4):
            resp = await asyncio.to_thread(_groq_call, messages, tools)
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
        if not final_text:
            final_text = "Listo. ✅" if actions else "No estoy seguro de qué necesitás. ¿Me lo repetís?"
        store.add_history(chat_id, "user", user_message)
        store.add_history(chat_id, "assistant", final_text)
        return final_text, actions
    except Exception as e:
        logger.error(f"_ask_groq: {type(e).__name__}: {e}")
        return (f"Tuve un problema para pensar la respuesta ({type(e).__name__}). Probá de nuevo en un momento.", actions)


# ─── Visión: foto de comida → carbohidratos ──────────────────────
FOOD_PROMPT = (
    "Analizá esta foto de comida para una persona con diabetes tipo 1. "
    "Estimá los carbohidratos totales. Respondé SOLO un JSON válido con esta forma:\n"
    '{"descripcion": "qué ves, breve", "carbs_g": <número>, "confianza": "alta|media|baja", '
    '"comentario": "1 frase útil, aclarando que es estimación"}\n'
    "Nunca sugieras dosis de insulina."
)
FOOD_SYSTEM = "Sos un asistente nutricional preciso para diabetes tipo 1. Respondés solo JSON."


def _parse_food_json(raw: str) -> tuple[str, Optional[float]]:
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start:end + 1]) if (start >= 0 and end > start) else {}
    desc = data.get("descripcion", "comida")
    carbs = data.get("carbs_g")
    conf = data.get("confianza", "media")
    comentario = data.get("comentario", "Es una estimación; puede variar según la porción real.")
    texto = (f"🍽 {desc}\n"
             f"🍞 Carbs estimados: ~{carbs}g (confianza {conf})\n"
             f"ℹ️ {comentario}")
    return texto, (float(carbs) if isinstance(carbs, (int, float)) else None)


async def analyze_food_image(image_bytes: bytes, caption: str = "") -> tuple[str, Optional[float]]:
    """Estima carbohidratos desde una foto. Enruta al proveedor configurado."""
    provider = brain_provider()
    if provider == "none":
        return ("No puedo analizar fotos sin un cerebro configurado (GROQ_API_KEY o ANTHROPIC_API_KEY).", None)
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
        return ("No pude analizar bien la foto. Probá con más luz o incluí un cubierto de referencia.", None)


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
            await update.message.reply_text(narr or digest)
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
    await update.message.reply_text(response)
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
    await update.message.reply_text(response)
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
    texto, carbs = await analyze_food_image(bytes(file_bytes), caption)
    # Registrar la comida en Nightscout con los carbs estimados
    desc = caption or "Comida (foto)"
    await post_treatment("Carb Correction", carbs=carbs, notes=f"{desc} [estimado por foto]")
    store.add_history(chat_id, "user", f"[Foto de comida] {caption}")
    store.add_history(chat_id, "assistant", texto)
    await update.message.reply_text(f"📸 {texto}\n\n✅ Registrado en Nightscout.")


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


# ─── Resumen diario proactivo ────────────────────────────────────
async def daily_evening_summary(context: ContextTypes.DEFAULT_TYPE):
    """Cada noche: resumen del día + aviso de riesgo nocturno si hubo ejercicio."""
    entries = await get_recent_entries(24)
    treatments = await get_treatments(24)
    summary = summarize_entries(entries)
    ex_today = [t for t in treatments if t.get("eventType") == "Exercise"]
    warn = ""
    if ex_today:
        warn = ("\n🌙 Hoy hubo entrenamiento: ojo con las bajas de madrugada. "
                "Puede convenir medir antes de dormir.")
    _broadcast(context, f"🌆 Resumen del día:\n{summary}{warn}")


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

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Alertas cada 5 minutos (ahora con anti-spam y predicción)
    app.job_queue.run_repeating(check_alerts, interval=300, first=15)
    # Resumen diario a las 22:30 hora Argentina
    app.job_queue.run_daily(daily_evening_summary, time=dtime(22, 30, tzinfo=TZ_AR))

    persist = "MongoDB" if store.db is not None else "RAM"
    logger.info(f"VittoreD1Bot v5 iniciado ✓ | cerebro: {brain_label()} | persistencia: {persist}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
