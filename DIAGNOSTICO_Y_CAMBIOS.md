# VittoBot — Diagnóstico y mejora de inteligencia (v5)

**Proyecto Diabete · 22 de agosto de 2026**

Este documento explica por qué el bot de Telegram "no se sentía inteligente"
ni ayudaba a mantener la glucosa en rango, y qué cambié en `bot.py` para
resolverlo. Al final están los pasos de despliegue.

> ⛔ **La regla de seguridad no se tocó.** El bot sigue sin calcular ni sugerir
> dosis de insulina. Solo informa, registra, recuerda y alerta. Las dosis las
> define el endocrinólogo.

---

## 1. Diagnóstico: por qué no ayudaba

**El cerebro no era Claude.** El documento funcional dice que el cerebro
conversacional es Claude, pero el bot en producción usaba **Groq con
`gpt-oss-20b`** para el chat y `qwen3.6-27b` para las fotos. Un modelo de 20B
razona poco: no correlaciona bien glucosa + insulina activa + ejercicio, no
anticipa, y cae en respuestas genéricas. Esta era, de lejos, la causa principal
de la falta de "inteligencia".

**El registro adivinaba por palabras clave.** Cualquier mensaje con "comí" se
registraba como `Meal Bolus` (que implica insulina) aunque no hubiera insulina;
cualquier "insulina" iba como `Correction Bolus`. Eso ensucia Nightscout y
confunde los cálculos posteriores.

**La memoria se borraba al reiniciar.** El historial y el registro del día
vivían en variables en RAM. Cada vez que Railway reiniciaba el contenedor, el
bot "olvidaba" todo.

**Las alertas eran reactivas y spameaban.** Solo miraba el umbral cada 5
minutos y reenviaba la misma alerta una y otra vez mientras el valor seguía
alto o bajo. No anticipaba una baja, no consideraba la insulina activa, y no
avisaba si el sensor se caía.

**No había análisis de patrones.** El documento describe detectar hipoglucemias
nocturnas post-hockey y picos post-comida, pero nada de eso estaba
implementado.

---

## 2. Qué cambié (v5)

**Cerebro = Claude.** Chat y estimación de carbohidratos por foto ahora usan
**`claude-sonnet-4-6`**. Groq queda únicamente para transcribir los audios
(Whisper). Si por algún motivo no hay clave de Claude, el bot sigue funcionando
con los comandos básicos y avisa que falta configurarla.

**Razonamiento real en el prompt.** El sistema instruye a Claude a correlacionar
datos, anticipar subidas/bajadas, usar los patrones y cerrar con una
observación útil en vez de frases de relleno. Ahora ve en cada mensaje: glucosa
actual + **proyección a ~20 min**, insulina/carbs activos (IOB/COB), tratamientos
de las últimas 6 h, lo que aprendió de Vittore y un resumen de patrones.

**Registro confiable con herramientas.** En vez de adivinar, Claude usa
*tools* y decide el `eventType` correcto y los datos (carbs, unidades, minutos):
`registrar_comida`, `registrar_insulina`, `registrar_ejercicio`,
`registrar_glucosa_capilar`, `registrar_nota` y `recordar_dato`. El registro del
día (`/registro`) ahora se **lee desde Nightscout**, así que sobrevive a los
reinicios.

**Análisis de patrones (14 días).** Nuevo comando `/patrones`: tiempo en rango
por día, hipoglucemias nocturnas, y correlación ejercicio → baja de madrugada.
Claude lo narra en lenguaje claro para llevar a la consulta con el endocrinólogo.

**Alertas inteligentes.**
- *Predictivas*: si viene bajando fuerte y va camino a hipoglucemia, avisa
  **antes** de cruzar el umbral.
- *Anti-spam*: solo avisa cuando cambia el estado (y re-avisa los estados
  urgentes recién después de 30 min). Avisa también cuando **se recupera** el
  rango.
- *Contexto*: incluye la insulina activa en el aviso.
- *Sensor caído*: avisa si no llegan datos hace más de 20 min.

**Proactividad.** Al registrar un entrenamiento, programa un recordatorio de
medir ~45 min después. Y cada noche a las 22:30 manda un resumen del día, con
aviso extra de riesgo nocturno si hubo ejercicio.

**Persistencia en MongoDB.** La memoria conversacional y el perfil aprendido
(comidas habituales, rutina, apodo, preferencias) se guardan en el MongoDB
Atlas que ya tenés. Si no hay `MONGODB_URI`, usa RAM como antes.

---

## 3. Despliegue

### 3.1 Variables de entorno en Railway

| Variable | Obligatoria | Descripción |
|---|---|---|
| `TELEGRAM_TOKEN` | sí | Token del bot (ya lo tenías). |
| `NIGHTSCOUT_URL` | sí | URL de Nightscout (ya lo tenías). |
| `NIGHTSCOUT_API_SECRET` | sí | Secret de Nightscout (ya lo tenías). |
| **`ANTHROPIC_API_KEY`** | **sí (nueva)** | Clave de la API de Claude. **Es lo que enciende la inteligencia.** |
| `ANTHROPIC_MODEL` | no | Por defecto `claude-sonnet-4-6`. |
| `GROQ_API_KEY` | no | Solo para transcribir audios (Whisper). |
| `MONGODB_URI` | recomendada | Para que no pierda la memoria al reiniciar. Usá la misma de tu `.env`. |
| `ALLOWED_USERS` | sí | Chat IDs autorizados, separados por coma. |
| `CAREGIVER_CHAT_IDS` | no | A quién van las alertas (por defecto, los de `ALLOWED_USERS`). |
| `BG_LOW`/`BG_HIGH`/`BG_URGENT_LOW`/`BG_URGENT_HIGH` | no | Umbrales; **confirmalos con el endocrinólogo**. |

> La clave nueva y crítica es **`ANTHROPIC_API_KEY`**. Sin ella el bot corre,
> pero en "modo básico" (comandos, sin conversación inteligente).

### 3.2 Pasos

1. En Railway, agregá `ANTHROPIC_API_KEY` (y `MONGODB_URI` si querés
   persistencia). El resto ya lo tenés.
2. Se actualizó `requirements.txt` (agrega `anthropic` y `pymongo`).
3. `git add . && git commit -m "v5: cerebro Claude, registro por tools, patrones y alertas inteligentes" && git push`. Railway redepliega solo.
4. Probá en Telegram: `/patrones`, mandale una foto de comida, contale "comí
   una pizza y me puse insulina", y dejá que corra para ver las alertas.

---

## 4. Límites y próximos pasos

- Los umbrales de alerta y todo lo relativo a insulina **deben revisarse con el
  endocrinólogo** antes de confiar en el sistema.
- Las estimaciones (carbs por foto, IOB, predicción) son aproximadas.
- La proyección de glucosa es lineal por tendencia reciente: útil para
  anticipar, no es una certeza.
- Pendiente (no incluido acá): capa de llamada telefónica real (Twilio) para
  emergencias, e integración de actividad del Garmin.
