"""
ml_model.py — Fase 2: Modelo ML personalizado para predicción de glucosa
=========================================================================
Entrena con la historia de Vittore en Nightscout y reentrena cada noche.
Predice glucosa a +30 y +60 min con banda de incertidumbre (quantile regression)
y probabilidad de hipoglucemia (clasificador).

El modelo se guarda serializado en MongoDB (colección "ml_models") para
que el bot lo cargue al arrancar.

⛔ SEGURIDAD: este módulo NUNCA calcula ni sugiere dosis. Solo predice
valores futuros de glucosa y riesgo de hipoglucemia.
"""

import io
import math
import pickle
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, Any

import numpy as np

logger = logging.getLogger("VittoBot.ML")

# ─── Constantes del modelo ──────────────────────────────────────
MODEL_VERSION = "2.0"
MIN_SAMPLES_TRAIN = 500       # ~2 días de datos cada 5 min
MIN_HYPO_EVENTS = 10          # mínimo de episodios hipo para entrenar clasificador
BG_LOW = 70                   # mg/dL, umbral hipoglucemia
BG_VALID_MIN = 20             # lecturas por debajo se descartan
BG_VALID_MAX = 500            # lecturas por encima se descartan
MAX_JUMP_PER_5MIN = 40        # mg/dL, salto máximo físicamente posible en 5 min
HORIZONS = [30, 60]           # minutos de predicción
QUANTILES = [0.10, 0.50, 0.90]  # para banda de incertidumbre

# Parámetros de curvas (mismos que bot.py, se pasan en config)
DEFAULT_CONFIG = {
    "isf": 30.0,
    "carb_ratio": 10.0,
    "insulin_dia_min": 240,
    "insulin_peak_min": 65,
    "carb_absorb_min": 180,
}


# ─── Utilidades de curvas (duplicadas de bot.py para independencia) ─
def _insulin_remaining(age_min: float, dia: int = 240, peak: int = 65) -> float:
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


def _carb_remaining(age_min: float, tabs: int = 180) -> float:
    if age_min <= 0:
        return 1.0
    if age_min >= tabs:
        return 0.0
    return 1 - age_min / tabs


# ─── Feature engineering ────────────────────────────────────────
def _compute_iob_at(boluses: list, offset: float, config: dict) -> float:
    """IOB en un momento dado (offset minutos desde 'ahora' de los bolos)."""
    dia = config.get("insulin_dia_min", 240)
    peak = config.get("insulin_peak_min", 65)
    return sum(u * _insulin_remaining(age + offset, dia, peak) for age, u in boluses)


def _compute_cob_at(carbs: list, offset: float, config: dict) -> float:
    tabs = config.get("carb_absorb_min", 180)
    return sum(g * _carb_remaining(age + offset, tabs) for age, g in carbs)


def _parse_treatments_at(treatments: list, ref_time: datetime, config: dict) -> dict:
    """Extrae bolos, carbos y ejercicio activos en ref_time."""
    boluses, carbs_list = [], []
    exercise_recent = False
    exercise_minutes_ago = None
    dia = config.get("insulin_dia_min", 240)
    tabs = config.get("carb_absorb_min", 180)

    for t in treatments:
        created = t.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except Exception:
            continue
        age = (ref_time - dt).total_seconds() / 60
        if age < 0:
            continue
        notes = (t.get("notes", "") or "").lower()
        ins = t.get("insulin")
        if ins and age < dia and "basal" not in notes and "glargina" not in notes:
            try:
                boluses.append((age, float(ins)))
            except (TypeError, ValueError):
                pass
        cb = t.get("carbs")
        if cb and age < tabs:
            try:
                carbs_list.append((age, float(cb)))
            except (TypeError, ValueError):
                pass
        if t.get("eventType") == "Exercise" and age < 720:  # 12h window
            exercise_recent = True
            if exercise_minutes_ago is None or age < exercise_minutes_ago:
                exercise_minutes_ago = age

    iob = _compute_iob_at(boluses, 0, config)
    cob = _compute_cob_at(carbs_list, 0, config)
    return {
        "iob": iob,
        "cob": cob,
        "boluses": boluses,
        "carbs": carbs_list,
        "exercise_recent": exercise_recent,
        "exercise_minutes_ago": exercise_minutes_ago,
        "total_insulin_4h": sum(u for _, u in boluses),
        "total_carbs_3h": sum(g for _, g in carbs_list),
        "time_since_last_bolus": min((a for a, _ in boluses), default=None),
        "time_since_last_carb": min((a for a, _ in carbs_list), default=None),
    }


def build_features(glucose_values: list, ref_idx: int,
                   treatments: list, config: dict) -> Optional[np.ndarray]:
    """
    Construye vector de features para un punto dado de la serie de glucosa.

    glucose_values: lista de (timestamp_utc, sgv) ordenada cronológicamente.
    ref_idx: índice del punto actual en la serie.
    treatments: lista de tratamientos de Nightscout.
    config: parámetros clínicos.

    Returns: numpy array de features o None si datos insuficientes.
    """
    if ref_idx < 6:  # necesitamos al menos 30 min de historia (6 * 5min)
        return None

    ts_now, sgv_now = glucose_values[ref_idx]
    if not (BG_VALID_MIN <= sgv_now <= BG_VALID_MAX):
        return None

    # Glucosa reciente: valores a -5, -10, -15, -20, -25, -30 min
    # Buscamos el más cercano a cada offset
    recent_sgv = []
    for offset_min in [5, 10, 15, 20, 25, 30]:
        target_ts = ts_now - timedelta(minutes=offset_min)
        best = None
        best_diff = float("inf")
        for j in range(max(0, ref_idx - 12), ref_idx):
            ts_j, sgv_j = glucose_values[j]
            diff = abs((ts_j - target_ts).total_seconds())
            if diff < best_diff and BG_VALID_MIN <= sgv_j <= BG_VALID_MAX:
                best_diff = diff
                best = sgv_j
        if best is None or best_diff > 180:  # tolerancia de 3 min
            return None  # datos insuficientes
        recent_sgv.append(best)

    # Pendiente (slope): variación por minuto en los últimos 15 min
    sgv_15 = recent_sgv[2]  # -15 min
    slope = (sgv_now - sgv_15) / 15.0

    # Aceleración: cambio en la pendiente
    sgv_30 = recent_sgv[5]  # -30 min
    slope_prev = (sgv_15 - sgv_30) / 15.0
    accel = slope - slope_prev

    # Tratamientos activos
    tx = _parse_treatments_at(treatments, ts_now, config)

    # Hora del día (features cíclicas)
    tz_ar = timezone(timedelta(hours=-3))
    local_dt = ts_now.astimezone(tz_ar)
    hour_frac = local_dt.hour + local_dt.minute / 60
    hour_sin = math.sin(2 * math.pi * hour_frac / 24)
    hour_cos = math.cos(2 * math.pi * hour_frac / 24)

    # Día de semana (0=lunes, 6=domingo)
    dow = local_dt.weekday()
    is_weekend = 1.0 if dow >= 5 else 0.0

    # Es de noche (00:00-06:00)
    is_night = 1.0 if local_dt.hour < 6 else 0.0

    features = [
        sgv_now,                          # 0: glucosa actual
        recent_sgv[0],                    # 1: glucosa -5min
        recent_sgv[1],                    # 2: glucosa -10min
        recent_sgv[2],                    # 3: glucosa -15min
        recent_sgv[3],                    # 4: glucosa -20min
        recent_sgv[4],                    # 5: glucosa -25min
        recent_sgv[5],                    # 6: glucosa -30min
        slope,                            # 7: pendiente (mg/dL/min)
        accel,                            # 8: aceleración
        tx["iob"],                        # 9: insulina activa (U)
        tx["cob"],                        # 10: carbos activos (g)
        tx["total_insulin_4h"],           # 11: insulina total en 4h
        tx["total_carbs_3h"],             # 12: carbos totales en 3h
        tx["time_since_last_bolus"] or -1,  # 13: min desde último bolo (-1 si no hay)
        tx["time_since_last_carb"] or -1,   # 14: min desde última comida (-1 si no hay)
        1.0 if tx["exercise_recent"] else 0.0,  # 15: ejercicio reciente
        tx["exercise_minutes_ago"] or -1,   # 16: min desde ejercicio (-1 si no hay)
        hour_sin,                         # 17: hora (seno)
        hour_cos,                         # 18: hora (coseno)
        is_weekend,                       # 19: fin de semana
        is_night,                         # 20: es de noche
    ]

    return np.array(features, dtype=np.float64)


FEATURE_NAMES = [
    "sgv_now", "sgv_5", "sgv_10", "sgv_15", "sgv_20", "sgv_25", "sgv_30",
    "slope", "accel", "iob", "cob", "total_insulin_4h", "total_carbs_3h",
    "time_since_last_bolus", "time_since_last_carb",
    "exercise_recent", "exercise_minutes_ago",
    "hour_sin", "hour_cos", "is_weekend", "is_night",
]


# ─── Dataset builder ────────────────────────────────────────────
def build_dataset(entries: list, treatments: list, config: dict) -> dict:
    """
    Construye matrices X, y_30, y_60, y_hypo_30, y_hypo_60 desde datos de Nightscout.

    entries: lista de entradas de Nightscout con 'sgv' y 'date' (ms).
    treatments: lista de tratamientos.
    config: parámetros clínicos.

    Returns: dict con keys 'X', 'y_30', 'y_60', 'y_hypo_30', 'y_hypo_60', 'n_samples'.
    """
    # Convertir entries a (datetime_utc, sgv), filtrar y ordenar
    points = []
    for e in entries:
        sgv = e.get("sgv")
        date_ms = e.get("date")
        if sgv is None or date_ms is None:
            continue
        if not (BG_VALID_MIN <= sgv <= BG_VALID_MAX):
            continue
        ts = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)
        points.append((ts, int(sgv)))

    points.sort(key=lambda x: x[0])

    # Filtrar saltos imposibles
    filtered = [points[0]] if points else []
    for i in range(1, len(points)):
        ts_prev, sgv_prev = filtered[-1]
        ts_cur, sgv_cur = points[i]
        dt_min = (ts_cur - ts_prev).total_seconds() / 60
        if dt_min > 0 and abs(sgv_cur - sgv_prev) / max(dt_min, 1) > MAX_JUMP_PER_5MIN / 5:
            continue  # salto imposible, descartar
        filtered.append(points[i])

    points = filtered
    if len(points) < 20:
        return {"X": None, "n_samples": 0}

    X_list, y30_list, y60_list = [], [], []

    for i in range(len(points)):
        ts_i, sgv_i = points[i]

        # Buscar glucosa real a +30 y +60 min
        target_30 = ts_i + timedelta(minutes=30)
        target_60 = ts_i + timedelta(minutes=60)

        real_30 = _find_closest_sgv(points, i, target_30)
        real_60 = _find_closest_sgv(points, i, target_60)

        if real_30 is None or real_60 is None:
            continue

        feat = build_features(points, i, treatments, config)
        if feat is None:
            continue

        X_list.append(feat)
        y30_list.append(real_30)
        y60_list.append(real_60)

    if not X_list:
        return {"X": None, "n_samples": 0}

    X = np.array(X_list)
    y_30 = np.array(y30_list, dtype=np.float64)
    y_60 = np.array(y60_list, dtype=np.float64)
    y_hypo_30 = (y_30 < BG_LOW).astype(np.int32)
    y_hypo_60 = (y_60 < BG_LOW).astype(np.int32)

    return {
        "X": X,
        "y_30": y_30,
        "y_60": y_60,
        "y_hypo_30": y_hypo_30,
        "y_hypo_60": y_hypo_60,
        "n_samples": len(X_list),
    }


def _find_closest_sgv(points: list, from_idx: int,
                      target_ts: datetime, tolerance_sec: float = 180) -> Optional[float]:
    """Busca la lectura más cercana a target_ts en la serie, tolerando 3 min."""
    best_val = None
    best_diff = float("inf")
    # Buscar hacia adelante desde from_idx
    for j in range(from_idx + 1, len(points)):
        ts_j, sgv_j = points[j]
        diff = abs((ts_j - target_ts).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_val = sgv_j
        if ts_j > target_ts + timedelta(seconds=tolerance_sec):
            break  # ya pasamos
    return best_val if best_diff <= tolerance_sec else None


# ─── Entrenamiento ──────────────────────────────────────────────
def train_models(dataset: dict) -> Optional[dict]:
    """
    Entrena modelos de regresión (quantile) y clasificación (hipo).

    Returns: dict con modelos entrenados y métricas, o None si datos insuficientes.
    """
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

    X = dataset["X"]
    n = dataset["n_samples"]

    if n < MIN_SAMPLES_TRAIN:
        logger.warning(f"Solo {n} muestras, necesito {MIN_SAMPLES_TRAIN}. No entreno.")
        return None

    # Split temporal: último 20% para validación
    split_idx = int(n * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]

    models = {}
    metrics = {}

    # --- Regresión cuantílica para cada horizonte y quantile ---
    for horizon in HORIZONS:
        y_key = f"y_{horizon}"
        y = dataset[y_key]
        y_train, y_val = y[:split_idx], y[split_idx:]

        for q in QUANTILES:
            label = f"reg_{horizon}_q{int(q*100)}"
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            models[label] = model

            if q == 0.50:  # métricas en la mediana
                mae = mean_absolute_error(y_val, pred_val)
                rmse = mean_squared_error(y_val, pred_val) ** 0.5
                metrics[f"mae_{horizon}"] = round(mae, 1)
                metrics[f"rmse_{horizon}"] = round(rmse, 1)
                logger.info(f"Regresión +{horizon}min: MAE={mae:.1f} RMSE={rmse:.1f}")

        # Cobertura de banda
        y_val_full = dataset[y_key][split_idx:]
        p10 = models[f"reg_{horizon}_q10"].predict(X_val)
        p90 = models[f"reg_{horizon}_q90"].predict(X_val)
        coverage = float(np.mean((y_val_full >= p10) & (y_val_full <= p90)))
        metrics[f"coverage_80_{horizon}"] = round(coverage * 100, 1)
        logger.info(f"Cobertura 80% +{horizon}min: {coverage*100:.1f}%")

    # --- Clasificador de hipoglucemia ---
    for horizon in HORIZONS:
        y_hypo = dataset[f"y_hypo_{horizon}"]
        n_hypo = y_hypo.sum()

        if n_hypo < MIN_HYPO_EVENTS:
            logger.warning(f"Solo {n_hypo} episodios de hipo a +{horizon}min "
                         f"(mínimo {MIN_HYPO_EVENTS}). Clasificador no entrena.")
            continue

        y_train, y_val = y_hypo[:split_idx], y_hypo[split_idx:]

        # Peso para desbalanceo (priorizar sensibilidad)
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale = n_neg / max(n_pos, 1)
        sample_weight = np.where(y_train == 1, scale, 1.0)

        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        models[f"clf_hypo_{horizon}"] = clf

        pred_proba = clf.predict_proba(X_val)
        if pred_proba.shape[1] == 2:
            proba = pred_proba[:, 1]
        else:
            proba = pred_proba[:, 0]

        # Sensibilidad al umbral 0.3 (prioriza no perderse hipos)
        pred_bin = (proba >= 0.3).astype(int)
        tp = ((pred_bin == 1) & (y_val == 1)).sum()
        fn = ((pred_bin == 0) & (y_val == 1)).sum()
        fp = ((pred_bin == 1) & (y_val == 0)).sum()
        sensitivity = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        metrics[f"hypo_sensitivity_{horizon}"] = round(sensitivity * 100, 1)
        metrics[f"hypo_precision_{horizon}"] = round(precision * 100, 1)
        logger.info(f"Clasificador hipo +{horizon}min: sens={sensitivity*100:.1f}% "
                   f"prec={precision*100:.1f}% (n_hypo={n_hypo})")

    if not models:
        return None

    return {
        "models": models,
        "metrics": metrics,
        "n_samples": n,
        "version": MODEL_VERSION,
        "feature_names": FEATURE_NAMES,
    }


# ─── Persistencia en MongoDB ────────────────────────────────────
def save_model_to_mongo(db, trained: dict) -> bool:
    """Guarda modelo entrenado en MongoDB, versionado."""
    try:
        # Serializar cada modelo con pickle
        serialized = {}
        for name, model in trained["models"].items():
            buf = io.BytesIO()
            pickle.dump(model, buf)
            serialized[name] = buf.getvalue()

        doc = {
            "version": trained["version"],
            "n_samples": trained["n_samples"],
            "metrics": trained["metrics"],
            "feature_names": trained["feature_names"],
            "models": serialized,
            "trained_at": datetime.now(timezone.utc),
        }

        # Guardar como último modelo activo y en histórico
        db.ml_models.update_one(
            {"_id": "active"},
            {"$set": doc},
            upsert=True,
        )

        # Histórico (últimos 7 para poder volver atrás)
        db.ml_model_history.insert_one({
            **doc,
            "_id": None,  # auto-generated
        })
        # Limpiar viejos
        count = db.ml_model_history.count_documents({})
        if count > 7:
            oldest = list(db.ml_model_history.find().sort("trained_at", 1).limit(count - 7))
            if oldest:
                db.ml_model_history.delete_many({"_id": {"$in": [o["_id"] for o in oldest]}})

        logger.info(f"Modelo ML guardado en MongoDB (v{trained['version']}, "
                   f"{trained['n_samples']} muestras)")
        return True
    except Exception as e:
        logger.error(f"Error guardando modelo ML: {e}")
        return False


def load_model_from_mongo(db) -> Optional[dict]:
    """Carga el modelo activo desde MongoDB."""
    try:
        doc = db.ml_models.find_one({"_id": "active"})
        if not doc or "models" not in doc:
            logger.info("No hay modelo ML en MongoDB todavía.")
            return None

        models = {}
        for name, data in doc["models"].items():
            buf = io.BytesIO(data)
            models[name] = pickle.load(buf)

        result = {
            "models": models,
            "metrics": doc.get("metrics", {}),
            "n_samples": doc.get("n_samples", 0),
            "version": doc.get("version", "?"),
            "feature_names": doc.get("feature_names", FEATURE_NAMES),
            "trained_at": doc.get("trained_at"),
        }
        logger.info(f"Modelo ML cargado (v{result['version']}, "
                   f"{result['n_samples']} muestras, "
                   f"entrenado: {result['trained_at']})")
        return result
    except Exception as e:
        logger.error(f"Error cargando modelo ML: {e}")
        return None


# ─── Inferencia ─────────────────────────────────────────────────
def predict_ml(trained: dict, features: np.ndarray) -> Optional[dict]:
    """
    Predice glucosa a +30/+60 min con bandas y probabilidad de hipo.

    Returns: dict con p30, p60, bandas, probabilidad de hipo, o None si falla.
    """
    if trained is None or features is None:
        return None

    models = trained["models"]
    result = {}

    for horizon in HORIZONS:
        h = str(horizon)

        # Regresión cuantílica
        q50_key = f"reg_{horizon}_q50"
        q10_key = f"reg_{horizon}_q10"
        q90_key = f"reg_{horizon}_q90"

        if q50_key in models:
            x = features.reshape(1, -1)
            p50 = float(models[q50_key].predict(x)[0])
            p10 = float(models[q10_key].predict(x)[0]) if q10_key in models else p50 - 20
            p90 = float(models[q90_key].predict(x)[0]) if q90_key in models else p50 + 20

            # Clamp a rango válido
            p50 = max(BG_VALID_MIN, min(BG_VALID_MAX, p50))
            p10 = max(BG_VALID_MIN, min(p50, p10))
            p90 = max(p50, min(BG_VALID_MAX, p90))

            result[f"p{horizon}"] = round(p50)
            result[f"p{horizon}_low"] = round(p10)
            result[f"p{horizon}_high"] = round(p90)
            result[f"band_{horizon}"] = round((p90 - p10) / 2)

        # Clasificador de hipo
        clf_key = f"clf_hypo_{horizon}"
        if clf_key in models:
            x = features.reshape(1, -1)
            proba = models[clf_key].predict_proba(x)
            if proba.shape[1] == 2:
                hypo_prob = float(proba[0, 1])
            else:
                hypo_prob = float(proba[0, 0])
            result[f"hypo_prob_{horizon}"] = round(hypo_prob, 3)

    if not result:
        return None

    result["source"] = "ml"
    result["version"] = trained.get("version", "?")
    result["n_samples"] = trained.get("n_samples", 0)
    return result


# ─── Tracking de errores de predicción ──────────────────────────
def log_prediction_error(db, prediction: dict, actual_30: Optional[float],
                         actual_60: Optional[float]):
    """Registra error de predicción para calibración continua."""
    try:
        doc = {
            "ts": datetime.now(timezone.utc),
            "predicted_30": prediction.get("p30"),
            "predicted_60": prediction.get("p60"),
            "actual_30": actual_30,
            "actual_60": actual_60,
        }
        if actual_30 is not None and prediction.get("p30"):
            doc["error_30"] = actual_30 - prediction["p30"]
        if actual_60 is not None and prediction.get("p60"):
            doc["error_60"] = actual_60 - prediction["p60"]

        db.ml_prediction_log.insert_one(doc)

        # Mantener solo los últimos 30 días de logs
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        db.ml_prediction_log.delete_many({"ts": {"$lt": cutoff}})
    except Exception as e:
        logger.error(f"Error registrando predicción: {e}")


def get_model_stats(db) -> Optional[dict]:
    """Estadísticas del modelo para el comando /prediccion."""
    try:
        doc = db.ml_models.find_one({"_id": "active"})
        if not doc:
            return None
        return {
            "version": doc.get("version"),
            "n_samples": doc.get("n_samples"),
            "metrics": doc.get("metrics", {}),
            "trained_at": doc.get("trained_at"),
        }
    except Exception:
        return None
