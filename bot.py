"""
VittoreD1Bot — Agente conversacional para diabetes tipo 1
Lee datos de glucosa de Nightscout y responde por Telegram usando Groq.
Soporta: texto, audio, fotos de comida (estimación de carbohidratos),
registro de comidas/insulina, memoria conversacional.

REGLA DE SEGURIDAD ABSOLUTA:
Este bot NUNCA calcula, sugiere ni recomienda dosis de insulina.
Solo informa, registra y alerta.
"""

import os
import io
import re
import base64
import json
import asyncio
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
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
from groq import Groq

# ─── Configuración ───────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
NIGHTSCOUT_URL = os.environ["NIGHTSCOUT_URL"]
NIGHTSCOUT_API_SECRET = os.environ.get("NIGHTSCOUT_API_SECRET", "")
ALLOWED_USERS = os.environ.get("ALLOWED_USERS", "").split(",")

# Umbrales de glucosa (mg/dL)
BG_HIGH = int(os.environ.get("BG_HIGH", "180"))
BG_LOW = int(os.environ.get("BG_LOW", "70"))
BG_URGENT_HIGH = int(os.environ.get("BG_URGENT_HIGH", "250"))
BG_URGENT_LOW = int(os.environ.get("BG_URGENT_LOW", "55"))

# Modelos Groq
CHAT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.6-27b"

# Zona horaria Argentina
TZ_AR = timezone(timedelta(hours=-3))

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("VittoBot")

# ─── Clientes ────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ─── Memoria conversacional ──────────────────────────────────────
conversation_history: dict[int, list[dict]] = defaultdict(list)
MAX_HISTORY = 20

# Registro diario: comidas, insulina, notas
daily_log: dict[int, list[dict]] = defaultdict(list)


def add_to_history(chat_id: int, role: str, content: str):
    """Agrega un mensaje al historial de conversación."""
    conversation_history[chat_id].append({"role": role, "content": content})
    if len(conversation_history[chat_id]) > MAX_HISTORY:
        conversation_history[chat_id] = conversation_history[chat_id][-MAX_HISTORY:]


def add_to_daily_log(chat_id: int, entry_type: str, detail: str):
    """Registra una entrada en el log diario."""
    now = datetime.now(TZ_AR)
    daily_log[chat_id].append({
        "tipo": entry_type,
        "detalle": detail,
        "hora": now.strftime("%H:%M"),
        "fecha": now.strftime("%Y-%m-%d"),
    })


def get_daily_log_summary(chat_id: int) -> str:
    """Resumen del log diario."""
    today = datetime.now(TZ_AR).strftime("%Y-%m-%d")
    entries = [e for e in daily_log[chat_id] if e["fecha"] == today]
    if not entries:
        return "No hay registros de hoy."

    lines = [f"Registros de hoy ({today}):"]
    for e in entries:
        emoji = {"comida": "🍽", "insulina": "💉", "nota": "📝", "ejercicio": "🏃"}.get(e["tipo"], "•")
        lines.append(f"{emoji} {e['hora']} — {e['tipo'].capitalize()}: {e['detalle']}")
    return "\n".join(lines)


SYSTEM_PROMPT = """Sos un asistente amable y claro que ayuda a una familia a entender
los datos de glucosa de su hijo Vittore (15 años, diabetes tipo 1).

Datos del paciente:
- Insulina rápida: Apidra (glulisina), DIA ~3-4 horas
- Insulina basal: Glargina (Lantus/Toujeo)
- Sensor: FreeStyle Libre (CGM)
- Unidades: mg/dL
- Zona horaria: Argentina (UTC-3)

REGLA ABSOLUTA DE SEGURIDAD:
- NUNCA calculés, sugerís ni recomendés dosis de insulina.
- NUNCA digás "podrías ponerte X unidades" ni nada similar.
- Si te preguntan por dosis, respondé: "Las dosis de insulina las debe indicar
  el endocrinólogo. Yo solo puedo mostrarte los datos."
- Podés explicar tendencias, alertar sobre valores fuera de rango, y dar info
  educativa general sobre diabetes.

Rangos de referencia:
- Normal: 70-180 mg/dL
- Bajo: <70 mg/dL (hipoglucemia)
- Alto: >180 mg/dL (hiperglucemia)
- Urgente bajo: <55 mg/dL
- Urgente alto: >250 mg/dL

CAPACIDADES DE REGISTRO:
Cuando el usuario mencione que comió algo, se puso insulina, hizo ejercicio, o quiera
anotar algo, respondé confirmando el registro. Ejemplos:
- "Almorcé pasta" → registrá como comida
- "Me puse 4 de Apidra" → registrá como insulina (NO comentes si es mucho o poco)
- "Salí a correr 30 min" → registrá como ejercicio
- "Nota: mañana turno con endocrinólogo" → registrá como nota

ESTILO:
Respondé siempre en español, de forma concisa y cálida. Usá emojis con moderación.
Cuando recibas datos de glucosa, interpretá la tendencia y dá contexto útil.
Recordá el contexto de la conversación previa para dar seguimiento proactivo.
Si hace rato que no preguntan, y tenés datos nuevos relevantes, podés mencionarlos."""

VISION_PROMPT = """Sos un asistente nutricional para una persona con diabetes tipo 1.
Analizá esta foto de comida y estimá los carbohidratos (hidratos de carbono).

Respondé en español con este formato:
1. Qué ves en la foto (describí los alimentos)
2. Estimación de carbohidratos por alimento (en gramos)
3. Total estimado de carbohidratos

IMPORTANTE:
- Aclarás que es una ESTIMACIÓN y puede variar según porciones reales.
- NUNCA sugerís dosis de insulina.
- Si no podés identificar bien la comida, pedí más detalles.
- Sé conciso y claro."""


def extract_answer(text: str) -> str:
    """Extrae la respuesta final, descartando el bloque <think>."""
    if not text:
        return ""
    # Si hay </think>, la respuesta está después
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    # Si hay <think> sin cierre (se quedó sin tokens), devolver el contenido
    if "<think>" in text:
        return text.replace("<think>", "").strip()
    return text.strip()


# ─── Funciones de Nightscout ─────────────────────────────────────
async def get_current_bg() -> Optional[dict]:
    """Obtiene la última lectura de glucosa de Nightscout."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{NIGHTSCOUT_URL}/api/v1/entries/current.json",
                headers=_ns_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 0:
                return data[0]
    except Exception as e:
        logger.error(f"Error consultando Nightscout: {e}")
    return None


async def get_recent_entries(hours: int = 3) -> list:
    """Obtiene las lecturas de las últimas N horas."""
    try:
        count = hours * 12
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{NIGHTSCOUT_URL}/api/v1/entries.json?count={count}",
                headers=_ns_headers(),
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Error consultando historial: {e}")
    return []


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


def format_bg_message(entry: dict) -> str:
    sgv = entry.get("sgv", 0)
    direction = entry.get("direction", "")
    date_ms = entry.get("date", 0)

    dt = datetime.fromtimestamp(date_ms / 1000, tz=TZ_AR)
    time_str = dt.strftime("%H:%M")

    if sgv <= BG_URGENT_LOW:
        indicator = "🔴 URGENTE BAJO"
    elif sgv <= BG_LOW:
        indicator = "🟡 BAJO"
    elif sgv >= BG_URGENT_HIGH:
        indicator = "🔴 URGENTE ALTO"
    elif sgv >= BG_HIGH:
        indicator = "🟠 ALTO"
    else:
        indicator = "🟢 En rango"

    mins_ago = int((datetime.now(TZ_AR) - dt).total_seconds() / 60)

    return (
        f"**Glucosa: {sgv} mg/dL** {indicator}\n"
        f"Tendencia: {format_direction(direction)}\n"
        f"Hora: {time_str} (hace {mins_ago} min)"
    )


def summarize_entries(entries: list) -> str:
    if not entries:
        return "No hay datos recientes."

    values = [e.get("sgv", 0) for e in entries if e.get("sgv")]
    if not values:
        return "No hay datos de glucosa disponibles."

    avg = sum(values) / len(values)
    high_count = sum(1 for v in values if v > BG_HIGH)
    low_count = sum(1 for v in values if v < BG_LOW)
    in_range = len(values) - high_count - low_count
    pct_in_range = (in_range / len(values)) * 100

    first_time = datetime.fromtimestamp(entries[-1].get("date", 0) / 1000, tz=TZ_AR)
    last_time = datetime.fromtimestamp(entries[0].get("date", 0) / 1000, tz=TZ_AR)

    return (
        f"Resumen ({first_time.strftime('%H:%M')} - {last_time.strftime('%H:%M')}):\n"
        f"- Lecturas: {len(values)}\n"
        f"- Promedio: {avg:.0f} mg/dL\n"
        f"- Mín/Máx: {min(values)}/{max(values)} mg/dL\n"
        f"- En rango: {pct_in_range:.0f}%\n"
        f"- Altos (>{BG_HIGH}): {high_count} | Bajos (<{BG_LOW}): {low_count}"
    )


# ─── Funciones adicionales de Nightscout ─────────────────────────
def _ns_headers() -> dict:
    """Headers comunes para las requests a Nightscout (SHA1 del secret)."""
    headers = {}
    if NIGHTSCOUT_API_SECRET:
        headers["api-secret"] = hashlib.sha1(
            NIGHTSCOUT_API_SECRET.encode("utf-8")
        ).hexdigest()
    return headers


async def get_treatments(hours: int = 6) -> list:
    """Obtiene tratamientos recientes (insulina, carbs, notas, etc.)."""
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        params = {
            "find[created_at][$gte]": since.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "count": 50,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{NIGHTSCOUT_URL}/api/v1/treatments.json",
                headers=_ns_headers(),
                params=params,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Error consultando treatments: {e}")
    return []


async def get_device_status() -> Optional[dict]:
    """Obtiene el último devicestatus (IOB, COB, batería, etc.)."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{NIGHTSCOUT_URL}/api/v1/devicestatus.json?count=1",
                headers=_ns_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 0:
                return data[0]
    except Exception as e:
        logger.error(f"Error consultando devicestatus: {e}")
    return None


async def get_profile() -> Optional[dict]:
    """Obtiene el perfil activo del paciente."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{NIGHTSCOUT_URL}/api/v1/profile/current",
                headers=_ns_headers(),
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Error consultando profile: {e}")
    return None


async def post_treatment(event_type: str, **kwargs) -> bool:
    """Registra un tratamiento en Nightscout."""
    try:
        now = datetime.now(timezone.utc)
        treatment = {
            "eventType": event_type,
            "created_at": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "enteredBy": "VittoBot",
        }
        treatment.update(kwargs)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{NIGHTSCOUT_URL}/api/v1/treatments",
                headers=_ns_headers(),
                json=treatment,
            )
            resp.raise_for_status()
            logger.info(f"Treatment registrado en NS: {event_type}")
            return True
    except Exception as e:
        logger.error(f"Error registrando treatment: {type(e).__name__}: {e}")
    return False


def format_treatments(treatments: list) -> str:
    """Formatea tratamientos para mostrar/enviar al LLM."""
    if not treatments:
        return "No hay tratamientos recientes."

    lines = []
    for t in treatments:
        created = t.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(TZ_AR)
            time_str = dt.strftime("%H:%M")
        except Exception:
            time_str = created[:16] if created else "?"

        event = t.get("eventType", "?")
        parts = [f"{time_str} — {event}"]

        insulin = t.get("insulin")
        if insulin:
            parts.append(f"💉 {insulin}U")
        carbs = t.get("carbs")
        if carbs:
            parts.append(f"🍞 {carbs}g carbs")
        notes = t.get("notes")
        if notes:
            parts.append(f"📝 {notes}")
        glucose = t.get("glucose")
        if glucose:
            parts.append(f"🩸 {glucose} mg/dL")
        duration = t.get("duration")
        if duration:
            parts.append(f"⏱ {duration} min")

        lines.append(" | ".join(parts))

    return "Tratamientos recientes:\n" + "\n".join(lines)


def format_iob_cob(device_status: Optional[dict]) -> str:
    """Extrae IOB y COB del devicestatus."""
    if not device_status:
        return ""

    parts = []

    # IOB puede estar en pump.iob o en openaps.iob o en loop.iob
    iob = None
    cob = None

    # OpenAPS / Loop
    for key in ["openaps", "loop"]:
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

    # Pump IOB
    pump = device_status.get("pump", {})
    if isinstance(pump, dict) and iob is None:
        pump_iob = pump.get("iob", {})
        if isinstance(pump_iob, dict):
            iob = pump_iob.get("iob", pump_iob.get("bolusiob"))
        elif isinstance(pump_iob, (int, float)):
            iob = pump_iob

    if iob is not None:
        parts.append(f"IOB: {iob:.1f}U")
    if cob is not None:
        parts.append(f"COB: {cob:.0f}g")

    return " | ".join(parts) if parts else ""


# ─── Audio (Whisper) ─────────────────────────────────────────────
async def transcribe_voice(file_bytes: bytes) -> str:
    """Transcribe audio usando Whisper de Groq."""
    if not groq_client:
        return "[Error: No hay API key de Groq configurada]"
    try:
        audio_file = io.BytesIO(file_bytes)
        audio_file.name = "voice.ogg"
        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            language="es",
        )
        return transcript.text
    except Exception as e:
        logger.error(f"Error transcribiendo audio: {e}")
        return "[No pude entender el audio, intentá de nuevo]"


# ─── Visión (fotos de comida → carbohidratos) ───────────────────
async def analyze_food_image(image_bytes: bytes, caption: str = "") -> str:
    """Analiza una foto de comida y estima carbohidratos usando visión."""
    if not groq_client:
        return "Error: No hay API key de Groq configurada."
    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        user_text = VISION_PROMPT
        if caption:
            user_text += f"\n\nEl usuario agregó este comentario: {caption}"

        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
            temperature=0.3,
        )

        result = response.choices[0].message.content or ""
        logger.info(f"Vision raw length={len(result)}, starts_with_think={'<think>' in result[:20]}")
        result = extract_answer(result)
        return result if result.strip() else "No pude analizar la imagen. Intentá con otra foto."
    except Exception as e:
        logger.error(f"Error analizando imagen: {type(e).__name__}: {e}")
        return f"Error analizando la imagen: {type(e).__name__}: {e}"


# ─── Interacción con Groq (chat) ────────────────────────────────
async def build_full_context(chat_id: int) -> str:
    """Construye el contexto completo de Nightscout para el LLM."""
    # Glucosa actual + historial
    entry = await get_current_bg()
    entries = await get_recent_entries(hours=3)

    bg_context = ""
    if entry:
        bg_context += format_bg_message(entry) + "\n\n"
    bg_context += summarize_entries(entries)

    # Treatments recientes
    treatments = await get_treatments(hours=6)
    treatments_ctx = format_treatments(treatments) if treatments else ""

    # IOB / COB
    device_status = await get_device_status()
    iob_cob = format_iob_cob(device_status)

    # Registro local del día
    log_context = get_daily_log_summary(chat_id)

    parts = [f"[GLUCOSA ACTUAL]\n{bg_context}"]
    if iob_cob:
        parts.append(f"[INSULINA/CARBS ACTIVOS]\n{iob_cob}")
    if treatments_ctx:
        parts.append(f"[TRATAMIENTOS RECIENTES (últimas 6h)]\n{treatments_ctx}")
    if log_context:
        parts.append(f"[REGISTROS LOCALES DEL DÍA]\n{log_context}")

    return "\n\n".join(parts)


async def ask_llm(chat_id: int, user_message: str, ns_context: str) -> str:
    """Envía un mensaje al LLM vía Groq con contexto completo de Nightscout."""
    if not groq_client:
        return "Error: No hay API key de Groq configurada."
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in conversation_history[chat_id]:
            messages.append(msg)

        full_context = f"{ns_context}\n\n[MENSAJE DEL USUARIO]\n{user_message}"
        messages.append({"role": "user", "content": full_context})

        response = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )

        reply = extract_answer(response.choices[0].message.content or "")

        add_to_history(chat_id, "user", user_message)
        add_to_history(chat_id, "assistant", reply)
        await detect_and_log(chat_id, user_message)

        return reply
    except Exception as e:
        logger.error(f"Error con Groq LLM: {type(e).__name__}: {e}")
        return f"Error con el asistente: {type(e).__name__}: {e}"


async def detect_and_log(chat_id: int, message: str):
    """Detecta registros en el mensaje, los guarda localmente y en Nightscout."""
    msg = message.lower()

    if any(w in msg for w in ["comí", "almorcé", "cené", "desayuné", "meriendé", "comimos", "comió", "tomé jugo", "comida"]):
        add_to_daily_log(chat_id, "comida", message)
        await post_treatment("Meal Bolus", notes=message)

    elif any(w in msg for w in ["insulina", "apidra", "glargina", "lantus", "toujeo", "me puse", "unidades", "me inyecté"]):
        add_to_daily_log(chat_id, "insulina", message)
        # Extraer unidades del mensaje si es posible
        units = None
        import re as _re
        match = _re.search(r"(\d+(?:[.,]\d+)?)\s*(?:u(?:nidades)?|de apidra|de glargina|de lantus|de toujeo)", msg)
        if match:
            units = float(match.group(1).replace(",", "."))
        kwargs = {"notes": message}
        if units:
            kwargs["insulin"] = units
        await post_treatment("Correction Bolus", **kwargs)

    elif any(w in msg for w in ["corrí", "caminé", "ejercicio", "fútbol", "natación", "bici", "gimnasio", "deporte"]):
        add_to_daily_log(chat_id, "ejercicio", message)
        await post_treatment("Exercise", notes=message)

    elif any(w in msg for w in ["nota:", "recordar:", "turno", "cita", "médico", "endocrinólogo"]):
        add_to_daily_log(chat_id, "nota", message)
        await post_treatment("Note", notes=message)


# ─── Handlers de Telegram ────────────────────────────────────────
def is_authorized(update: Update) -> bool:
    if not ALLOWED_USERS or ALLOWED_USERS == [""]:
        return True
    return str(update.effective_chat.id) in ALLOWED_USERS


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("No tenés autorización para usar este bot.")
        return

    await update.message.reply_text(
        "¡Hola! Soy el asistente de glucosa de Vittore 🩺\n\n"
        "Podés:\n"
        "• /glucosa — Ver la lectura actual\n"
        "• /resumen — Resumen de las últimas 3 horas\n"
        "• /tratamientos — Insulina y carbs de las últimas 6h\n"
        "• /registro — Ver los registros de hoy\n"
        "• Escribirme o mandarme un audio con cualquier pregunta\n"
        "• Mandarme una foto de comida para estimar carbohidratos 📸\n"
        "• Decirme qué comió, cuánta insulina se puso, si hizo ejercicio\n\n"
        f"Tu chat ID es: {update.effective_chat.id}"
    )


async def cmd_glucosa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    entry = await get_current_bg()
    if entry:
        await update.message.reply_text(format_bg_message(entry), parse_mode="Markdown")
    else:
        await update.message.reply_text("No pude obtener datos de Nightscout.")


async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    entries = await get_recent_entries(hours=3)
    summary = summarize_entries(entries)
    await update.message.reply_text(summary)


async def cmd_registro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    summary = get_daily_log_summary(update.effective_chat.id)
    await update.message.reply_text(summary)


async def cmd_tratamientos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra los tratamientos recientes de Nightscout."""
    if not is_authorized(update):
        return
    treatments = await get_treatments(hours=6)
    text = format_treatments(treatments)
    # Agregar IOB/COB si están disponibles
    device_status = await get_device_status()
    iob_cob = format_iob_cob(device_status)
    if iob_cob:
        text += f"\n\n⚡ {iob_cob}"
    await update.message.reply_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja mensajes de texto libres."""
    if not is_authorized(update):
        return

    user_text = update.message.text
    await update.message.chat.send_action("typing")

    ns_context = await build_full_context(update.effective_chat.id)
    response = await ask_llm(update.effective_chat.id, user_text, ns_context)
    await update.message.reply_text(response)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja mensajes de voz: transcribe y procesa como texto."""
    if not is_authorized(update):
        return

    if not groq_client:
        await update.message.reply_text(
            "El asistente no está habilitado. "
            "Necesito una API key de Groq (variable GROQ_API_KEY)."
        )
        return

    await update.message.chat.send_action("typing")

    voice = update.message.voice or update.message.audio
    file = await context.bot.get_file(voice.file_id)
    file_bytes = await file.download_as_bytearray()

    text = await transcribe_voice(bytes(file_bytes))
    if text.startswith("["):
        await update.message.reply_text(text)
        return

    await update.message.reply_text(f"🎙 Entendí: _{text}_", parse_mode="Markdown")

    ns_context = await build_full_context(update.effective_chat.id)
    response = await ask_llm(update.effective_chat.id, text, ns_context)
    await update.message.reply_text(response)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja fotos: analiza comida y estima carbohidratos."""
    if not is_authorized(update):
        return

    if not groq_client:
        await update.message.reply_text("El asistente no está habilitado.")
        return

    await update.message.chat.send_action("typing")

    # Tomar la foto de mayor resolución
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_bytes = await file.download_as_bytearray()

    # Caption opcional del usuario
    caption = update.message.caption or ""

    # Analizar con visión
    response = await analyze_food_image(bytes(file_bytes), caption)

    # Registrar como comida
    food_desc = caption if caption else "Foto de comida analizada"
    add_to_daily_log(update.effective_chat.id, "comida", food_desc)
    add_to_history(update.effective_chat.id, "user", f"[Envió foto de comida] {caption}")
    add_to_history(update.effective_chat.id, "assistant", response)

    await update.message.reply_text(f"📸 Análisis nutricional:\n\n{response}")


# ─── Alertas automáticas ─────────────────────────────────────────
async def check_alerts(context: ContextTypes.DEFAULT_TYPE):
    entry = await get_current_bg()
    if not entry:
        return

    sgv = entry.get("sgv", 0)
    alert_msg = None

    if sgv <= BG_URGENT_LOW:
        alert_msg = f"🚨 HIPOGLUCEMIA URGENTE 🚨\n\n{format_bg_message(entry)}\n\n¡Vittore necesita azúcar AHORA!"
    elif sgv <= BG_LOW:
        alert_msg = f"⚠️ Glucosa baja\n\n{format_bg_message(entry)}\n\nConsiderá dar carbohidratos."
    elif sgv >= BG_URGENT_HIGH:
        alert_msg = f"🚨 HIPERGLUCEMIA URGENTE 🚨\n\n{format_bg_message(entry)}\n\nConsultá con el endocrinólogo si persiste."
    elif sgv >= BG_HIGH:
        alert_msg = f"⚠️ Glucosa alta\n\n{format_bg_message(entry)}"

    if alert_msg and ALLOWED_USERS and ALLOWED_USERS != [""]:
        for chat_id in ALLOWED_USERS:
            try:
                await context.bot.send_message(
                    chat_id=int(chat_id.strip()),
                    text=alert_msg,
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.error(f"Error enviando alerta a {chat_id}: {e}")


# ─── Main ────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Comandos
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("glucosa", cmd_glucosa))
    app.add_handler(CommandHandler("resumen", cmd_resumen))
    app.add_handler(CommandHandler("registro", cmd_registro))
    app.add_handler(CommandHandler("tratamientos", cmd_tratamientos))

    # Mensajes de texto → LLM vía Groq
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Mensajes de voz → Whisper → LLM vía Groq
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Fotos → Visión (estimación de carbohidratos)
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Alertas cada 5 minutos
    app.job_queue.run_repeating(check_alerts, interval=300, first=10)

    logger.info("VittoreD1Bot v4 iniciado ✓ (carga completa NS)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
