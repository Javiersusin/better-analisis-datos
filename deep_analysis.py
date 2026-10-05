#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ventanas horarias de referencia:
    sleep_window   = 00:00-05:00
    wake_window    = 09:00-22:00
    pre_sleep_window = 22:00-00:00


Nota de zona horaria:
    Todos los timestamps del proyecto son UTC (sufijo 'Z').

Qué es este módulo:
    Biblioteca pura de análisis: NO lee ni escribe ficheros por su cuenta.
    Expone build_deep_analysis(), que devuelve el dict del bloque
    `deep_analysis`. La llaman dos sitios:
      - process_health_json.build_deep_analysis_block (flujo principal,
        con hr_samples_override para no re-leer el JSON) y
      - process_deep_analysis.py (wrapper sobre JSON ya procesados).

Entradas esperadas:
    health_processed : dict del *_health_processed.json (usa heart_rate_raw).
    phone_processed  : dict del *_phone_usage_processed.json (daily_blocks).
    hr_samples_override: lista [{"time": datetime, "value": int}] para tests
                       o para cuando las muestras ya están en memoria.

Salida:
    Dict con las claves time_windows, data_quality, mobile_sleep_window,
    pre_sleep_mobile, circadian_heart_rate, heart_rate_delta_events,
    heart_rate_mobile_events, compulsive_events_heart_rate y before_sleep.
    Si no hay HR ni móvil, devuelve el mismo esqueleto con null/False:
    no se inventan métricas.

Límite importante:
    Aquí solo se mide RELACIÓN TEMPORAL entre uso del móvil y HR. No hay
    inferencia causal ni diagnóstico de ningún tipo.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import mean, median, pstdev, stdev
from typing import Any, Iterable

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Ventanas temporales de referencia (minutos desde medianoche, en UTC)
# ─────────────────────────────────────────────────────────────────────
SLEEP_WINDOW = (0, 300)        # 00:00 - 05:00
WAKE_WINDOW = (540, 1320)      # 09:00 - 22:00
PRE_SLEEP_WINDOW = (1320, 1440)  # 22:00 - 00:00

TIME_WINDOWS = {
    "sleep_window": "00:00-05:00",
    "wake_window": "09:00-22:00",
    "pre_sleep_window": "22:00-00:00",
}

# Configuración de sincronización móvil ↔ HR.
SYNC_HR_WINDOW_SECONDS = 300.0  # ±5 minutos para buscar HR antes/después de un evento
SHORT_SESSION_THRESHOLD_SECONDS = 5.0  # sesión "compulsiva corta" (< 5 s)
RAPID_REOPENING_WINDOW_SECONDS = 10.0  # reapertura rápida (misma app <= 10 s)
P10 = 10
P90 = 90
MAX_DELTA_SAMPLE_EVENTS = 1000  # tope de eventos de delta guardados en el JSON

DESCRIPTIVE_FIELDS = (
    "window_start",
    "window_end",
    "filename",
    "participant_id",
)
# Nota: DESCRIPTIVE_FIELDS hoy no se usa en este módulo (está pensado para
# copiar esos metadatos descriptivos al bloque). Igual que `mean`, `pstdev` y
# `stdev` del import de statistics: solo `median` y las funciones de numpy se
# usan realmente. Se dejan sin tocar por si se retoma.


# ─────────────────────────────────────────────────────────────────────
# Utilidades de tiempo
# ─────────────────────────────────────────────────────────────────────
def parse_iso_datetime(value: str) -> datetime:
    """Convierte una fecha ISO ('Z') en un datetime UTC con tzinfo."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def minute_of_day(value: datetime) -> float:
    """Devuelve los minutos desde medianoche (0-1440) de un instante dado.

    Argumentos:
        value: datetime (preferentemente con tz UTC; si es naive se asume UTC).
    Hijo del cálculo de ventanas: permite comprobar si un instante cae dentro
    de la ventana de sueño, vigilia o pre-sueño de forma directa.
    """
    utc = value.astimezone(timezone.utc)
    return utc.hour * 60.0 + utc.minute + utc.second / 60.0


def overlap_minutes(start: datetime, end: datetime, win: tuple[int, int]) -> float:
    """Minutos de solapamiento entre un intervalo [start,end] y una ventana.

    Argumentos:
        start/end: límites del intervalo (p. ej. una sesión de móvil).
        win: ventana en minutos del día (start_min, end_min).
    Devuelve:
        minutos solapados (float >= 0). Si el intervalo cruza medianoche no se
        necesita soporte aquí porque las sesiones se dividen por día antes.
    """
    if end <= start:
        return 0.0
    win_start, win_end = win
    inter_start = max(minute_of_day(start), float(win_start))
    inter_end = min(minute_of_day(end), float(win_end))
    return max(0.0, inter_end - inter_start)


def in_window(value: datetime, win: tuple[int, int]) -> bool:
    """Indica si un instante cae dentro de una ventana de minutos del día."""
    win_start, win_end = win
    m = minute_of_day(value)
    return win_start <= m < win_end


def is_weekend_date(value: datetime) -> bool:
    """True si el día (UTC) es sábado o domingo.

    Basado en weekday() de Python (lunes=0 ... domingo=6). La separación
    weekday/weekend es necesaria para estudiar si el patrón circadiano cambia
    entre días laborables y fines de semana.
    """
    return value.astimezone(timezone.utc).weekday() >= 5


def bucket_day(value: datetime) -> str:
    """Fecha UTC (YYYY-MM-DD) de un instante."""
    return value.astimezone(timezone.utc).date().isoformat()


# ─────────────────────────────────────────────────────────────────────
# Carga de datos desde los JSON procesados
# ─────────────────────────────────────────────────────────────────────
def load_hr_samples(health_processed: dict[str, Any]) -> list[dict[str, Any]]:
    """Extrae las muestras de HR en bruto del JSON de salud procesado.

    Argumentos:
        health_processed: dict del archivo *_health_processed.json.
    Devuelve:
        Lista de muestras [{"time": datetime, "value": int}] ordenadas por
        tiempo. Si no existe `heart_rate_raw`, devuelve [].
    """
    raw = health_processed.get("heart_rate_raw") or []
    samples: list[dict[str, Any]] = []
    for entry in raw:
        try:
            t = parse_iso_datetime(entry["time"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            v = int(entry["value"])
        except (KeyError, TypeError, ValueError):
            continue
        samples.append({"time": t, "value": v})
    return sorted(samples, key=lambda s: s["time"])


def load_sessions(phone_processed: dict[str, Any]) -> list[dict[str, Any]]:
    """Reúne todas las sesiones de los daily_blocks del uso del móvil.

    Argumentos:
        phone_processed: dict del archivo *_phone_usage_processed.json.
    Devuelve:
        Lista de sesiones [{"category","start","end","duration_seconds",
        "unlock_to_open_seconds","date"}]. Se conservan inicio y fin de cada
        sesión y su categoría .
    """
    sessions: list[dict[str, Any]] = []
    blocks = phone_processed.get("daily_blocks") or []
    if not isinstance(blocks, list):
        return sessions
    for block in blocks:
        date = block.get("date")
        for s in block.get("sessions") or []:
            try:
                start = parse_iso_datetime(s["start_time"])
                end = parse_iso_datetime(s["end_time"])
            except (KeyError, TypeError, ValueError):
                continue
            sessions.append(
                {
                    "category": s.get("category"),
                    "start": start,
                    "end": end,
                    "duration_seconds": float(s.get("duration_seconds", 0.0) or 0.0),
                    "unlock_to_open_seconds": s.get("unlock_to_open_seconds"),
                    "date": date,
                }
            )
    return sorted(sessions, key=lambda s: s["start"])


def load_unlock_events(phone_processed: dict[str, Any]) -> list[datetime]:
    """Extrae los eventos de desbloqueo con su timestamp original."""
    unlocked: list[datetime] = []
    blocks = phone_processed.get("daily_blocks") or []
    if not isinstance(blocks, list):
        return unlocked
    for block in blocks:
        for stamp in block.get("unlock_events") or []:
            try:
                unlocked.append(parse_iso_datetime(stamp))
            except (TypeError, ValueError):
                continue
    return sorted(unlocked)


# ─────────────────────────────────────────────────────────────────────
# Uso del móvil durante la ventana de sueño (00:00-05:00)
# ─────────────────────────────────────────────────────────────────────
def mobile_usage_during_sleep_window(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Métrica de uso del móvil dentro de la ventana de sueño.

    Qué recibe:
        sessions: sesiones de móvil (cada una con start/end).
    Qué devuelve:
        Dict con mobile_usage_during_sleep_window_minutes,
        mobile_sessions_during_sleep_window, mobile_usage_percentage_during_sleep_window
        y un desglose por fecha.
    Si faltan datos:
        Se devuelve 0.0 / 0 / 0.0 (no hay sesiones no se inventa nada).
    """
    total_usage = sum(s["duration_seconds"] for s in sessions) / 60.0
    window_minutes = 0.0
    count = 0
    per_date: dict[str, dict[str, Any]] = defaultdict(lambda: {"minutes": 0.0, "sessions": 0})

    for s in sessions:
        ov = overlap_minutes(s["start"], s["end"], SLEEP_WINDOW)
        if ov > 0:
            window_minutes += ov
            count += 1
            date = bucket_day(s["start"])
            per_date[date]["minutes"] += ov
            per_date[date]["sessions"] += 1

    percentage = (window_minutes / total_usage * 100.0) if total_usage > 0 else 0.0

    return {
        "mobile_usage_during_sleep_window_minutes": round(window_minutes, 2),
        "mobile_sessions_during_sleep_window": count,
        "mobile_usage_percentage_during_sleep_window": round(percentage, 2),
        "total_mobile_usage_minutes": round(total_usage, 2),
        "per_date": [
            {"date": d, **per_date[d]} for d in sorted(per_date)
        ],
    }


# ─────────────────────────────────────────────────────────────────────
# Uso del móvil antes de dormir (22:00-00:00)
# ─────────────────────────────────────────────────────────────────────
def mobile_usage_before_sleep(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Uso del móvil en la ventana 22:00-00:00, total y por categoría.

    Qué recibe:
        sessions: sesiones de móvil con categoria y start/end.
    Qué devuelve:
        Dict con total_minutes, mobile_usage_before_sleep_minutes y by_category.
    Hipótesis:
        pre_sleep_window = 22:00-00:00 (periodo inmediatamente previo a la
        ventana de sueño).
    """
    total = 0.0
    by_category: dict[str, float] = defaultdict(float)
    count = 0
    for s in sessions:
        ov = overlap_minutes(s["start"], s["end"], PRE_SLEEP_WINDOW)
        if ov > 0:
            total += ov
            count += 1
            cat = s.get("category") or "other"
            by_category[cat] += ov

    return {
        "pre_sleep_window": TIME_WINDOWS["pre_sleep_window"],
        "total_minutes": round(total, 2),
        "mobile_usage_before_sleep_minutes": round(total, 2),
        "sessions_count": count,
        "by_category": {
            cat: round(minutes, 2) for cat, minutes in sorted(by_category.items())
        },
    }


def _consecutive_pairs_by_category(sessions: list[dict[str, Any]]):
    """Itera pares consecutivos que comparten categoría."""
    for prev, curr in zip(sessions, sessions[1:]):
        if prev.get("category") == curr.get("category"):
            yield prev, curr


def compulsive_behavior_before_sleep(
    sessions: list[dict[str, Any]],
    unlock_events: list[datetime],
) -> dict[str, Any]:
    """Métricas compulsivas dentro de 22:00-00:00.


    Qué recibe:
        sessions de móvil y unlock_events (timestamps).
    Qué devuelve:
        Dict con rapid_reopenings_before_sleep, unlock_count_before_sleep,
        app_switches_before_sleep y short_sessions_before_sleep.

    """
    pre = [s for s in sessions if overlap_minutes(s["start"], s["end"], PRE_SLEEP_WINDOW) > 0]
    rapid = sum(
        1
        for prev, curr in _consecutive_pairs_by_category(pre)
        if 0 <= (curr["start"] - prev["end"]).total_seconds() <= RAPID_REOPENING_WINDOW_SECONDS
    )
    unlocks = sum(1 for u in unlock_events if in_window(u, PRE_SLEEP_WINDOW))
    switches = sum(
        1
        for prev, curr in zip(pre, pre[1:])
        if prev.get("category") != curr.get("category")
        and overlap_minutes(prev["start"], prev["end"], PRE_SLEEP_WINDOW) > 0
        and overlap_minutes(curr["start"], curr["end"], PRE_SLEEP_WINDOW) > 0
    )
    short = sum(
        1
        for s in pre
        if s["duration_seconds"] < SHORT_SESSION_THRESHOLD_SECONDS
    )

    return {
        "rapid_reopenings_before_sleep": rapid,
        "unlock_count_before_sleep": unlocks,
        "app_switches_before_sleep": switches,
        "short_sessions_before_sleep": short,
    }


# ─────────────────────────────────────────────────────────────────────
# Filtrado P10-P90 para el análisis circadiano (no toca los originales)
# ─────────────────────────────────────────────────────────────────────
def filter_p10_p90(values: list[float]) -> tuple[list[float], float, float]:
    """Elimina el 10 % inferior y el 10 % superior de una serie.

    Argumentos:
        values: serie numérica 
    Devuelve:
        (filtered, lower_cutoff, upper_cutoff). Si no hay suficientes datos
        (len < 2) devuelve (values, None, None) para no inventar percentiles.
    Metodología:
        lower_cutoff = percentil P10, upper_cutoff = percentil P90. Solo se
        usa como señal de trabajo para el análisis circadiano; los originales
        no se modifican ni se borran del JSON.

    """
    if len(values) < 2:
        return list(values), None, None
    lo = float(np.percentile(values, P10))
    hi = float(np.percentile(values, P90))
    filtered = [v for v in values if lo <= v <= hi]
    return filtered, lo, hi


# ─────────────────────────────────────────────────────────────────────
# Estadística circadiana de la frecuencia cardíaca
# ─────────────────────────────────────────────────────────────────────
def _hr_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Estadísticas básicas de una lista de muestras de HR."""
    if not samples:
        return {"count": 0}
    values = [s["value"] for s in samples]
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "range": int(max(values) - min(values)),
        "mean": round(float(np.mean(values)), 2),
        "std": round(float(np.std(values)), 2),
    }


def hour_of_daily_extremes(samples: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Hora (0-23) en la que se dan el mínimo y máximo de HR del periodo.

    Argumentos:
        samples: muestras de HR ordenadas por tiempo.
    Devuelve:
        (hour_of_min, hour_of_max) en 0-23, o None si no hay muestras.
    Metodología:
        Localiza temporalmente la muestra con menor/mayor HR y devuelve su
        hora UTC. Si hay empates se usa la mediana de horas.

    """
    if not samples:
        return None, None
    min_val = min(s["value"] for s in samples)
    max_val = max(s["value"] for s in samples)
    hrs_min = [s["time"].astimezone(timezone.utc).hour for s in samples if s["value"] == min_val]
    hrs_max = [s["time"].astimezone(timezone.utc).hour for s in samples if s["value"] == max_val]
    h_min = int(round(median(hrs_min))) % 24
    h_max = int(round(median(hrs_max))) % 24
    return h_min, h_max


def circadian_heart_rate_analysis(hr_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Análisis circadiano completo de la HR con separación weekday/weekend.

    Qué hace:
        Para all_days, weekday y weekend calcula: estadísticas de la señal en
        bruto y de la señal filtrada P10-P90, recuentos, cortes y la hora del
        mínimo/máximo diario.
    Qué devuelve:
        Dict con key por grupo ('all_days','weekday','weekend').
    """
    def analyze_group(groups: list[dict[str, Any]]) -> dict[str, Any]:
        if not groups:
            return {"has_enough_data": False, "count": 0}
        raw = [s["value"] for s in groups]
        filtered, lo, hi = filter_p10_p90(raw)

        f_samples = [s for s in groups if (lo is None or lo <= s["value"] <= hi)]

        h_min, h_max = hour_of_daily_extremes(f_samples)

        # Perfil horario (media por hora de día del grupo).
        by_hour: dict[int, list[int]] = defaultdict(list)
        for s in groups:
            by_hour[s["time"].astimezone(timezone.utc).hour].append(s["value"])

        hourly_profile = [
            {
                "hour": h,
                "count": len(vals),
                "mean_raw": round(float(np.mean(vals)), 2),
            }
            for h, vals in sorted(by_hour.items())
        ]

        raw_stats = _hr_stats(groups)
        filtered_stats = _hr_stats(f_samples)

        # Métricas diarias por fecha (devuelve además lista, para no perder
        # la resolución diaria).
        per_day: dict[str, list[int]] = defaultdict(list)
        for s in groups:
            per_day[bucket_day(s["time"])].append(s["value"])

        per_day_stats = []
        for d, vals in sorted(per_day.items()):
            fv, flo, fhi = filter_p10_p90(vals)
            day_minutes_samples = [s for s in groups if bucket_day(s["time"]) == d]
            dm, dx = hour_of_daily_extremes(
                [s for s in day_minutes_samples if (flo is None or flo <= s["value"] <= fhi)]
            )
            per_day_stats.append(
                {
                    "date": d,
                    "count": len(vals),
                    "min": int(min(vals)),
                    "max": int(max(vals)),
                    "mean": round(float(np.mean(vals)), 2),
                    "hour_of_min": dm,
                    "hour_of_max": dx,
                }
            )

        return {
            "has_enough_data": True,
            "days": sorted(per_day),
            "computed": sorted(set(per_day)),
            "count_days": len(per_day),
            "raw": raw_stats,
            "filtered": {**filtered_stats, "lower_cutoff": lo, "upper_cutoff": hi},
            "hour_of_daily_min_hr": h_min,
            "hour_of_daily_max_hr": h_max,
            "hourly_profile": hourly_profile,
            "per_day": per_day_stats,
        }

    all_days = list(hr_samples)
    weekday = [s for s in all_days if not is_weekend_date(s["time"])]
    weekend = [s for s in all_days if is_weekend_date(s["time"])]

    return {
        "all_days": analyze_group(all_days),
        "weekday": analyze_group(weekday),
        "weekend": analyze_group(weekend),
    }


# ─────────────────────────────────────────────────────────────────────
# Cambios instantáneos de HR entre mediciones consecutivas
# ─────────────────────────────────────────────────────────────────────
def heart_rate_delta_events(hr_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Cambios de HR entre dos mediciones consecutivas (resolución de evento).

    Qué hace:
        Para cada par consecutivo (t1,t2) calcula delta_hr = HR(t2)-HR(t1) y
        delta_time = t2-t1, con los datos originales (sin agregación en
        bloques de 5 min).
    Qué devuelve:
        Resumen estadístico y un muestreo de eventos para su posterior
        análisis. No se asume que un cambio de 1-2 bpm implique respuesta
        fisiológica real; simplemente se detecta el evento.
    """
    if len(hr_samples) < 2:
        return {"count": 0}

    deltas = []
    deltas_secs = []
    for prev, curr in zip(hr_samples, hr_samples[1:]):
        dt = (curr["time"] - prev["time"]).total_seconds()
        d = (curr["value"] - prev["value"], dt)
        deltas.append(d)
        deltas_secs.append(dt)

    values = [d for d, _ in deltas]
    positive = sum(1 for v in values if v > 0)
    negative = sum(1 for v in values if v < 0)
    zero = sum(1 for v in values if v == 0)

    # Construir sample_events: los últimos N deltas con sus muestras originales.
    # deltas[i] = transición de hr_samples[i] → hr_samples[i+1].
    n_events = min(len(deltas), MAX_DELTA_SAMPLE_EVENTS)
    start_idx = len(deltas) - n_events  # índice del primer delta a incluir
    sample_events = []
    for i in range(start_idx, len(deltas)):
        prev = hr_samples[i]
        curr = hr_samples[i + 1]
        d_val, dt = deltas[i]
        sample_events.append(
            {
                "time1": prev["time"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "time2": curr["time"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "hr1": prev["value"],
                "hr2": curr["value"],
                "delta_hr": d_val,
                "delta_time_seconds": round(dt, 1),
            }
        )

    return {
        "count": len(deltas),
        "mean_delta_hr": round(float(np.mean(values)), 3) if values else None,
        "std_delta_hr": round(float(np.std(values)), 3) if values else None,
        "min_delta_hr": int(min(values)) if values else None,
        "max_delta_hr": int(max(values)) if values else None,
        "mean_delta_time_seconds": round(float(np.mean(deltas_secs)), 1) if deltas_secs else None,
        "positive_changes": positive,
        "negative_changes": negative,
        "zero_changes": zero,
        "sample_events": sample_events,
    }


# ─────────────────────────────────────────────────────────────────────
# Sincronización temporal móvil ↔ HR + análisis antes/durante/después
# ─────────────────────────────────────────────────────────────────────
def _nearest_hr(
    hr_samples: list[dict[str, Any]],
    moment: datetime,
    window_seconds: float = SYNC_HR_WINDOW_SECONDS,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """HR más próxima antes y después de un instante dentro de una ventana.

    Argumentos:
        hr_samples: muestras ordenadas por tiempo.
        moment: instante de referencia (evento del móvil).
        window_seconds: ventana de máxima distancia temporal.
    Devuelve:
        (before, after): muestra previa y posterior (o None).

    """
    times = [s["time"] for s in hr_samples]
    idx = bisect_left(times, moment)
    before = hr_samples[idx - 1] if idx > 0 else None
    after = hr_samples[idx] if idx < len(hr_samples) else None

    if before is not None and (moment - before["time"]).total_seconds() > window_seconds:
        before = None
    if after is not None and (after["time"] - moment).total_seconds() > window_seconds:
        after = None
    return before, after


def analyze_session_hr(
    session: dict[str, Any],
    hr_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Análisis HR antes/durante/después de una sesión de app.

    Qué devuelve:
        Dict con hr_before, hr_during_mean, hr_after y los deltas
        (delta_before_start, delta_end_after, delta_before_after), además de
        indicadores de calidad (has_enough...).
    Metodología:
        HR_before = medición inmediatamente anterior al inicio de la sesión.
        HR_during = media de las mediciones dentro de [start,end].
        HR_after = medición inmediatamente posterior al final de la sesión.
        Los deltas describen la relación temporal SIN implicar causalidad.
    """
    before, _ = _nearest_hr(hr_samples, session["start"])
    _, after = _nearest_hr(hr_samples, session["end"])

    contains = []
    for s in hr_samples:
        if session["start"] <= s["time"] <= session["end"]:
            contains.append(s)

    during_mean = round(float(np.mean([s["value"] for s in contains])), 2) if contains else None

    hr_before = before["value"] if before else None
    hr_after = after["value"] if after else None

    # Identifica la primera y última medición dentro de la sesión para los deltas.
    hr_first = contains[0]["value"] if contains else hr_before
    hr_last = contains[-1]["value"] if contains else hr_after

    delta_before_start = (hr_first - hr_before) if (hr_before is not None and hr_first is not None) else None
    delta_end_after = (hr_after - hr_last) if (hr_after is not None and hr_last is not None) else None
    delta_before_after = (hr_after - hr_before) if (hr_before is not None and hr_after is not None) else None

    has_enough = bool(
        before is not None
        and after is not None
        and (before["time"] != after["time"])
    )

    return {
        "category": session.get("category"),
        "start_time": session["start"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "end_time": session["end"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "duration_seconds": round(session.get("duration_seconds", 0.0), 1),
        "hr_before": hr_before,
        "hr_during_mean": during_mean,
        "hr_during_count": len(contains),
        "hr_after": hr_after,
        "delta_before_start": delta_before_start,
        "delta_end_after": delta_end_after,
        "delta_before_after": delta_before_after,
        "has_enough_hr_before_after": has_enough,
    }


def heart_rate_mobile_events_analysis(
    sessions: list[dict[str, Any]],
    hr_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sincroniza cada sesión con su HR y agrega por categoría.

    Qué hace:
        Para cada sesión del móvil calcula antes/durante/después y agrega
        por categoría (atención especial a social).
    Qué devuelve:
        Dict con sessions_analysis (lista) y by_category (resumen).
    """
    analyzed = [analyze_session_hr(s, hr_samples) for s in sessions]

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in analyzed:
        if a["category"] is not None:
            by_category[a["category"]].append(a)

    cat_summary = {}
    for cat, items in sorted(by_category.items()):
        deltas = [i["delta_before_after"] for i in items if i["delta_before_after"] is not None]
        dur = [i["hr_during_mean"] for i in items if i["hr_during_mean"] is not None]
        cat_summary[cat] = {
            "sessions_count": len(items),
            "mean_delta_before_after": round(float(np.mean(deltas)), 2) if deltas else None,
            "mean_hr_during": round(float(np.mean(dur)), 2) if dur else None,
            "sessions_with_increase": sum(1 for d in deltas if d > 0),
            "sessions_with_decrease": sum(1 for d in deltas if d < 0),
        }

    return {
        "sync_window_seconds": SYNC_HR_WINDOW_SECONDS,
        "sessions_analysis": analyzed,
        "by_category": cat_summary,
    }


# ─────────────────────────────────────────────────────────────────────
# Eventos compulsivos y su relación temporal con la HR
# ─────────────────────────────────────────────────────────────────────
def _compile_compulsive_events(
    sessions: list[dict[str, Any]],
    unlock_events: list[datetime],
) -> list[dict[str, Any]]:
    """Reúne eventos compulsivos con su timestamp.

    Tipos:
        unlock (desbloqueo), rapid_reopening (reapertura rápida),
        short_session (sesión < 5 s), app_switch (cambio de categoría).
    """
    events: list[dict[str, Any]] = []

    for u in unlock_events:
        events.append({"type": "unlock", "time": u, "category": None})

    for prev, curr in _consecutive_pairs_by_category(sessions):
        gap = (curr["start"] - prev["end"]).total_seconds()
        if 0 <= gap <= RAPID_REOPENING_WINDOW_SECONDS:
            events.append({"type": "rapid_reopening", "time": curr["start"], "category": curr.get("category")})

    for s in sessions:
        if s["duration_seconds"] < SHORT_SESSION_THRESHOLD_SECONDS:
            events.append({"type": "short_session", "time": s["start"], "category": s.get("category")})

    for prev, curr in zip(sessions, sessions[1:]):
        if prev.get("category") != curr.get("category"):
            events.append({"type": "app_switch", "time": curr["start"], "category": curr.get("category")})

    return sorted(events, key=lambda e: e["time"])


def compulsive_events_heart_rate(
    sessions: list[dict[str, Any]],
    unlock_events: list[datetime],
    hr_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """HR inmediatamente antes/después de cada evento compulsivo.

    Qué hace:
        Para cada evento compulsivo (unlock, rapid_reopening, short_session,
        app_switch) busca la HR más próxima antes y después (dentro de la
        ventana de sync) y calcula el delta.
    Qué devuelve:
        Dict agrupado por tipo con métricas hr_before_compulsive_event,
        hr_after_compulsive_event y delta_hr_compulsive_event cuando hay datos.
    """
    events = _compile_compulsive_events(sessions, unlock_events)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        before, after = _nearest_hr(hr_samples, ev["time"])
        record = {
            "time": ev["time"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "category": ev.get("category"),
            "hr_before": before["value"] if before else None,
            "hr_after": after["value"] if after else None,
        }
        if record["hr_before"] is not None and record["hr_after"] is not None:
            record["delta_hr"] = record["hr_after"] - record["hr_before"]
        else:
            record["delta_hr"] = None
        by_type[ev["type"]].append(record)

    result: dict[str, Any] = {}
    for etype, items in sorted(by_type.items()):
        deltas = [i["delta_hr"] for i in items if i["delta_hr"] is not None]
        before = [i["hr_before"] for i in items if i["hr_before"] is not None]
        after = [i["hr_after"] for i in items if i["hr_after"] is not None]
        result[etype] = {
            "events_count": len(items),
            "hr_before_compulsive_event": round(float(np.mean(before)), 2) if before else None,
            "hr_after_compulsive_event": round(float(np.mean(after)), 2) if after else None,
            "delta_hr_compulsive_event": round(float(np.mean(deltas)), 2) if deltas else None,
            "events": items,
        }

    return result


# ─────────────────────────────────────────────────────────────────────
# Análisis específico ANTES DE DORMIR (22:00-00:00)
# ─────────────────────────────────────────────────────────────────────
def before_sleep_analysis(
    sessions: list[dict[str, Any]],
    unlock_events: list[datetime],
    hr_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Análisis conjunto móvil + HR restringido a 22:00-00:00.

    Considera la relación entre el uso del móvil antes de dormir y la HR:
        A) Tiempo total de uso (mobile_usage_before_sleep_minutes).
        B) Desglose por categoría (social, messaging, ...).
        C) Comportamiento compulsivo.
    Además calcula la estadística de HR dentro de la ventana (heart_rate_before_sleep).
    """
    mobile = mobile_usage_before_sleep(sessions)
    compulsive = compulsive_behavior_before_sleep(sessions, unlock_events)

    hr_in_window = [s for s in hr_samples if in_window(s["time"], PRE_SLEEP_WINDOW)]
    hr_stats = _hr_stats(hr_in_window)

    return {
        "pre_sleep_window": TIME_WINDOWS["pre_sleep_window"],
        "mobile": mobile,
        "compulsive_behavior": compulsive,
        "heart_rate_before_sleep": hr_stats,
    }


# ─────────────────────────────────────────────────────────────────────
# Bloque completo deep_analysis
# ─────────────────────────────────────────────────────────────────────
def build_deep_analysis(
    health_processed: dict[str, Any] | None = None,
    phone_processed: dict[str, Any] | None = None,
    hr_samples_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construye el bloque completo de deep_analysis para un participante.

    Qué hace:
        Combina los datos de salud (HR en bruto) y de uso del móvil (sesiones,
        eventos de desbloqueo) y calcula todos los análisis definidos en el
        proyecto: ventanas temporales, uso móvil en sueño, pre-sueño,
        circadiano HWeekday/weekend, deltas de HR, sincronización móvil ↔ HR
        y eventos compulsivos.
    Qué recibe:
        health_processed (dict del *_health_processed.json),
        phone_processed (dict del *_phone_usage_processed.json), o
        hr_samples_override (lista) para usar muestras directas (tests).
    Qué devuelve:
        Dict con la estructura del bloque deep_analysis, o {} si no hay datos
        de HR ni de móvil.
    """
    # Carga de datos.
    if hr_samples_override is not None:
        hr_samples = hr_samples_override
    else:
        hr_samples = load_hr_samples(health_processed or {})

    if phone_processed:
        sessions = load_sessions(phone_processed)
        unlock_events = load_unlock_events(phone_processed)
    else:
        sessions = []
        unlock_events = []

    has_hr = bool(hr_samples)
    has_mobile = bool(sessions)

    # Cuando no hay nada, devolvemos un bloque mínimo honesto (todas las
    # claves canónicas presentes pero con valores nulos/False).
    if not has_hr and not has_mobile:
        return {
            "time_windows": dict(TIME_WINDOWS),
            "data_quality": {
                "has_heart_rate_data": False,
                "has_mobile_data": False,
            },
            "mobile_sleep_window": None,
            "pre_sleep_mobile": None,
            "circadian_heart_rate": None,
            "heart_rate_delta_events": None,
            "heart_rate_mobile_events": None,
            "compulsive_events_heart_rate": None,
            "before_sleep": None,
        }

    day_dates = sorted({bucket_day(s["time"]) for s in hr_samples}) if has_hr else []

    return {
        "time_windows": dict(TIME_WINDOWS),
        "unclassified_note": (
            "Las horas 05:00-09:00 y 22:00-00:00 no se clasifican automáticamente "
            "como sueño ni como vigilia en estos análisis."
        ),
        "data_quality": {
            "has_heart_rate_data": has_hr,
            "has_mobile_data": has_mobile,
            "hr_raw_count": len(hr_samples),
            "mobile_sessions_count": len(sessions),
            "unlock_events_count": len(unlock_events),
            "days_with_hr": day_dates,
        },
        "mobile_sleep_window": mobile_usage_during_sleep_window(sessions) if has_mobile else None,
        "pre_sleep_mobile": (
            {
                "mobile": mobile_usage_before_sleep(sessions),
                "compulsive_behavior": compulsive_behavior_before_sleep(sessions, unlock_events),
            }
            if has_mobile
            else None
        ),
        "circadian_heart_rate": (
            circadian_heart_rate_analysis(hr_samples) if has_hr else None
        ),
        "heart_rate_delta_events": (
            heart_rate_delta_events(hr_samples) if has_hr else None
        ),
        "heart_rate_mobile_events": (
            heart_rate_mobile_events_analysis(sessions, hr_samples)
            if has_mobile and has_hr
            else None
        ),
        "compulsive_events_heart_rate": (
            compulsive_events_heart_rate(sessions, unlock_events, hr_samples)
            if has_mobile and has_hr
            else None
        ),
        "before_sleep": (
            before_sleep_analysis(sessions, unlock_events, hr_samples)
            if has_mobile and has_hr
            else None
        ),
    }