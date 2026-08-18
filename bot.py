"""
VittoreD1Bot — Agente conversacional para diabetes tipo 1
Lee datos de glucosa de Nightscout y responde por Telegram usando Claude.

REGLA DE SEGURIDAD ABSOLUTA:
Este bot NUNCA calcula, sugiere ni recomienda dosis de insulina.
Solo informa, registra y alerta.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import anthropic

# ─── Configuración ───────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
NIGHTSCOUT_URL = os.environ["NIGHTSCOUT_URL"]  # con https://
NIGHTSCOUT_API_SECRET = os.environ.get("NIGHTSCOUT_API_SECRET", "")
ALLOWED_USERS = os.environ.get("ALLOWED_USERS", "").split(",")  # chat IDs

# Umbrales de glucosa (mg/dL)
BG_HIGH = int(os.environ.get("BG_HIGH", "180"))
BG_LOW = int(os.environ.get("BG_LOW", "70"))
BG_URGENT_HIGH = int(os.environ.get("BG_URGENT_HIGH", "250"))
BG_URGENT_LOW = int(os.environ.get("BG_URGENT_LOW", "55"))

# Zona horaria Argentina
TZ_AR = timezone(timedelta(hours=-3))

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("VittoBot")

# ─── Clientes ────────────────────────────────────────────────────
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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

Respondé siempre en español, de forma concisa y cálida. Usá emojis con moderación.
Cuando recibas datos de glucosa, interpretá la tendencia y dá contexto útil."""


# ─── Funciones de Nightscout ─────────────────────────────────────
async def get_current_bg() -> Optional[dict]:
    """Obtiene la última lectura de glucosa de Nightscout."""
    try:
        headers = {}
        if NIGHTSCOUT_API_SECRET:
            headers["api-secret"] = NIGHTSCOUT_API_SECRET
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{NIGHTSCOUT_URL}/api/v1/entries/current.json",
                headers=headers,
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
        headers = {}
        if NIGHTSCOUT_API_SECRET:
            headers["api-secret"] = NIGHTSCOUT_API_SECRET
        count = hours * 12  # ~1 lectura cada 5 min
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{NIGHTSCOUT_URL}/api/v1/entries.json?count={count}",
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Error consultando historial: {e}")
    return []


def format_direction(direction: str) -> str:
    """Convierte la dirección de tendencia a emoji."""
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
    """Formatea una lectura de glucosa para mostrar."""
    sgv = entry.get("sgv", 0)
    direction = entry.get("direction", "")
    date_ms = entry.get("date", 0)

    dt = datetime.fromtimestamp(date_ms / 1000, tz=TZ_AR)
    time_str = dt.strftime("%H:%M")

    # Indicador de rango
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
    """Genera un resumen de las últimas lecturas."""
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


# ─── Interacción con Claude ──────────────────────────────────────
async def ask_claude(user_message: str, bg_context: str) -> str:
    """Envía un mensaje a Claude con contexto de glucosa."""
    try:
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"[DATOS ACTUALES DE NIGHTSCOUT]\n{bg_context}\n\n[MENSAJE DEL USUARIO]\n{user_message}",
                }
            ],
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"Error con Claude: {e}")
        return "Perdón, tuve un problema consultando a Claude. Intentá de nuevo."


# ─── Handlers de Telegram ────────────────────────────────────────
def is_authorized(update: Update) -> bool:
    """Verifica si el usuario está autorizado."""
    if not ALLOWED_USERS or ALLOWED_USERS == [""]:
        return True  # Sin restricción si no se configuran usuarios
    return str(update.effective_chat.id) in ALLOWED_USERS


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("No tenés autorización para usar este bot.")
        return

    await update.message.reply_text(
        "¡Hola! Soy el asistente de glucosa de Vittore 🩺\n\n"
        "Podés preguntarme:\n"
        "• /glucosa - Ver la lectura actual\n"
        "• /resumen - Resumen de las últimas 3 horas\n"
        "• O escribime cualquier pregunta sobre los datos\n\n"
        f"Tu chat ID es: {update.effective_chat.id}\n"
        "(Guardalo para configurar ALLOWED_USERS)"
    )


async def cmd_glucosa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return

    entry = await get_current_bg()
    if entry:
        await update.message.reply_text(
            format_bg_message(entry), parse_mode="Markdown"
        )
    else:
        await update.message.reply_text("No pude obtener datos de Nightscout.")


async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return

    entries = await get_recent_entries(hours=3)
    summary = summarize_entries(entries)
    await update.message.reply_text(summary)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja mensajes de texto libres usando Claude."""
    if not is_authorized(update):
        return

    user_text = update.message.text

    # Obtener contexto de glucosa
    entry = await get_current_bg()
    entries = await get_recent_entries(hours=3)

    bg_context = ""
    if entry:
        bg_context += format_bg_message(entry) + "\n\n"
    bg_context += summarize_entries(entries)

    # Enviar a Claude
    await update.message.chat.send_action("typing")
    response = await ask_claude(user_text, bg_context)
    await update.message.reply_text(response)


# ─── Alertas automáticas ─────────────────────────────────────────
async def check_alerts(context: ContextTypes.DEFAULT_TYPE):
    """Revisa glucosa y envía alertas si está fuera de rango."""
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
    elif sgv >= BG_URGENT_HIGH:
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

    # Mensajes de texto libre → Claude
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Job de alertas cada 5 minutos
    app.job_queue.run_repeating(check_alerts, interval=300, first=10)

    logger.info("VittoreD1Bot iniciado ✓")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
