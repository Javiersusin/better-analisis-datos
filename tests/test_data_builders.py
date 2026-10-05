#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Datos de prueba REALISTAS para el deep_analysis.

Estos generadores construyen JSON procesados con PATRONES CONOCIDOS para
verificar mediante assertions que el análisis los detecta correctamente.

Casos cubiertos:
    Caso 1: uso de una app `social` con aumento de HR temporalmente asociado.
    Caso 2: uso de una app `music` SIN aumento relevante de HR.
    Caso 3: comportamiento compulsivo (unlock -> social -> close repetido).
    Caso 4: uso de redes sociales antes de dormir (22:00-00:00).
    Caso 5: uso del móvil durante la ventana de sueño (00:00-05:00).
    Caso 6: curva circadiana realista con valores extremos artificiales.

Los datos son sintéticos pero estructurados como los JSON reales del
proyecto para poder ejecutar el pipeline real de deep_analysis sobre ellos.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any

UT = timezone.utc


def z(dt: datetime) -> str:
    """Formatea un datetime a ISO con sufijo Z (como los JSON del proyecto)."""
    return dt.astimezone(UT).isoformat().replace("+00:00", "Z")


def iso(y: int, m: int, d: int, hh: int, mm: int = 0, ss: int = 0) -> str:
    """Construye un timestamp ISO/UTC con fecha-hora dados."""
    return z(datetime(y, m, d, hh, mm, ss, tzinfo=UT))


def make_hr_sample(y, m, d, hh, mm, ss, value) -> dict[str, Any]:
    """Una muestra de HR en bruto en el mismo formato que heart_rate_raw."""
    return {"time": iso(y, m, d, hh, mm, ss), "value": int(value)}


def make_session(
    y, m, d,
    start_hh, start_mm,
    duration_seconds: int,
    category: str,
    unlock_to_open: float | None = None,
) -> dict[str, Any]:
    """Una sesión de app en el formato de sessions del JSON procesado."""
    start = datetime(y, m, d, start_hh, start_mm, 0, tzinfo=UT)
    end = start + timedelta(seconds=duration_seconds)
    return {
        "category": category,
        "start_time": z(start),
        "end_time": z(end),
        "duration_seconds": round(duration_seconds, 1),
        "unlock_to_open_seconds": unlock_to_open,
    }


def make_phone_processed(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Envuelve bloques diarios en la estructura del JSON de uso del móvil."""
    return {"daily_blocks": blocks}


def make_health_processed(
    hr_raw: list[dict[str, Any]],
    daily_summary=None,
) -> dict[str, Any]:
    """Envuelve HR en bruto en la estructura del JSON de salud procesado."""
    return {
        "filename": "TEST_health_processed.json",
        "file_type": "health",
        "window_start": hr_raw[0]["time"] if hr_raw else iso(2026, 5, 9, 0, 0),
        "window_end": hr_raw[-1]["time"] if hr_raw else iso(2026, 5, 16, 23, 59),
        "heart_rate_raw": hr_raw,
        "daily_summary": daily_summary if daily_summary is not None else [],
    }


# ─────────────────────────────────────────────────────────────────────
# Caso 1: app social con aumento de HR temporalmente asociado.
# Antes: 60 bpm · Durante: 62-64 bpm · Después: 68 bpm.
# ─────────────────────────────────────────────────────────────────────
def build_caso1_social():
    hr = [
        # Noche/madrugada estable (para no interferir) - 05-13 (miércoles)
        make_hr_sample(2026, 5, 13, 0, 30, 0, 50),
        make_hr_sample(2026, 5, 13, 1, 0, 0, 50),
        # Mañana/mediodía actividad moderada
        make_hr_sample(2026, 5, 13, 11, 0, 0, 62),
        make_hr_sample(2026, 5, 13, 12, 0, 0, 63),
        # FC justo ANTES de abrir la app social (22:14 -> 60 bpm)
        make_hr_sample(2026, 5, 13, 22, 14, 0, 60),
        # DURANTE el uso de la app social (22:15-22:40) -> 62-64 bpm
        make_hr_sample(2026, 5, 13, 22, 18, 0, 62),
        make_hr_sample(2026, 5, 13, 22, 25, 0, 63),
        make_hr_sample(2026, 5, 13, 22, 33, 0, 64),
        # DESPUÉS de cerrar la app social (22:45 -> 68 bpm)
        make_hr_sample(2026, 5, 13, 22, 45, 0, 68),
    ]
    session = make_session(2026, 5, 13, 22, 15, duration_seconds=1500, category="social")
    block = {
        "date": "2026-05-13",
        "metrics": {},
        "compulsive_metrics": {},
        "sessions": [session],
        "unlock_events": [],
    }
    return make_health_processed(hr), make_phone_processed([block])


# ─────────────────────────────────────────────────────────────────────
# Caso 2: app music SIN incremento de HR (60 -> 60-61 -> 60).
# ─────────────────────────────────────────────────────────────────────
def build_caso2_music():
    hr = [
        make_hr_sample(2026, 5, 14, 17, 59, 0, 60),
        make_hr_sample(2026, 5, 14, 18, 10, 0, 60),
        make_hr_sample(2026, 5, 14, 18, 20, 0, 61),
        # Muestra "después" dentro de ±5 min tras el cierre (18:30)
        make_hr_sample(2026, 5, 14, 18, 33, 0, 60),
        make_hr_sample(2026, 5, 14, 19, 0, 0, 60),
    ]
    session = make_session(2026, 5, 14, 18, 0, duration_seconds=1800, category="music")
    block = {
        "date": "2026-05-14",
        "metrics": {},
        "compulsive_metrics": {},
        "sessions": [session],
        "unlock_events": [],
    }
    return make_health_processed(hr), make_phone_processed([block])


# ─────────────────────────────────────────────────────────────────────
# Caso 3: comportamiento compulsivo unlock -> social -> close repetido.
# EXPECted: se detecta el patrón compulsivo y las reaperturas rápidas.
# ─────────────────────────────────────────────────────────────────────
def build_caso3_compulsivo():
    # Patrón compulsivo: unlock -> social(4s) -> close, repetido cada 6 s en
    # 23:50-23:51 del miércoles 13/05.
    # - La sesión dura 4 s (< 5 s) => short_session.
    # - El hueco entre cierre y siguiente apertura es de 2 s (<= 10 s) =>
    #   rapid_reopening.
    sessions = []
    unlock_events = []
    base = datetime(2026, 5, 13, 23, 50, 0, tzinfo=UT)
    for i in range(4):
        unlock_events.append(z(base + timedelta(seconds=i * 6)))
        s_open = base + timedelta(seconds=i * 6 + 1)
        s_close = base + timedelta(seconds=i * 6 + 5)
        sessions.append(
            {
                "category": "social",
                "start_time": z(s_open),
                "end_time": z(s_close),
                "duration_seconds": 4.0,
                "unlock_to_open_seconds": 1.0,
            }
        )
    hr = [
        make_hr_sample(2026, 5, 13, 23, 49, 0, 58),
        make_hr_sample(2026, 5, 13, 23, 52, 0, 62),
        make_hr_sample(2026, 5, 13, 23, 55, 0, 66),
    ]
    block = {
        "date": "2026-05-13",
        "metrics": {},
        "compulsive_metrics": {},
        "sessions": sessions,
        "unlock_events": unlock_events,
    }
    return make_health_processed(hr), make_phone_processed([block])


# ─────────────────────────────────────────────────────────────────────
# Caso 4: uso de redes sociales antes de dormir (22:00-00:00).
# EXPECted: social_minutes_before_sleep se calcula correctamente.
# ─────────────────────────────────────────────────────────────────────
def build_caso4_pre_sleep_social():
    # Tres sesiones sociales de 10 min cada una entre 22:30 y 23:30 del 13/05
    sessions = []
    for offset_min in (30, 55, 80):  # 22:30, 22:55, 23:20
        base = datetime(2026, 5, 13, 22, 0, 0, tzinfo=UT) + timedelta(minutes=offset_min)
        sessions.append(
            {
                "category": "social",
                "start_time": z(base),
                "end_time": z(base + timedelta(seconds=600)),
                "duration_seconds": 600.0,
                "unlock_to_open_seconds": None,
            }
        )
    # Una sesión de música de 5 min (prevista para no confundir) 23:40
    sessions.append(make_session(2026, 5, 13, 23, 40, duration_seconds=300, category="music"))

    hr = [make_hr_sample(2026, 5, 13, 23, 45, 0, 64)]
    block = {
        "date": "2026-05-13",
        "metrics": {},
        "compulsive_metrics": {},
        "sessions": sessions,
        "unlock_events": [],
    }
    return make_health_processed(hr), make_phone_processed([block])


# ─────────────────────────────────────────────────────────────────────
# Caso 5: uso del móvil DURANTE la ventana de sueño (00:00-05:00).
# EXPECted: mobile_usage_during_sleep_window_minutes contabiliza sesiones.
# ─────────────────────────────────────────────────────────────────────
def build_caso5_sleep_window():
    # Sesión 1: 00:30-00:40 (10 min) del 14/05
    # Sesión 2: 02:10-02:25 (15 min) del 14/05
    sessions = [
        make_session(2026, 5, 14, 0, 30, duration_seconds=600, category="social"),
        make_session(2026, 5, 14, 2, 10, duration_seconds=900, category="other"),
    ]
    hr = [
        make_hr_sample(2026, 5, 14, 0, 30, 0, 55),
        make_hr_sample(2026, 5, 14, 0, 40, 0, 56),
        make_hr_sample(2026, 5, 14, 2, 25, 0, 54),
    ]
    block = {
        "date": "2026-05-14",
        "metrics": {},
        "compulsive_metrics": {},
        "sessions": sessions,
        "unlock_events": [],
    }
    return make_health_processed(hr), make_phone_processed([block])


# ─────────────────────────────────────────────────────────────────────
# Caso 6: curva circadiana realista con valores extremos artificiales.
# Valores más bajos de noche, más altos de día, + extremos (overy/under para
# comprobar el filtro P10-P90).
# ─────────────────────────────────────────────────────────────────────
def build_caso6_circadiano():
    hr = []
    # Curva suave basada en seno: nadir ~ 22:00? para simplificar usamos
    # plantilla explícita por hora con variación natural.
    base_profile = {
        0: 47, 1: 47, 2: 45, 3: 45, 4: 45, 5: 47, 6: 52, 7: 56,
        8: 60, 9: 62, 10: 64, 11: 64, 12: 65, 13: 65, 14: 65, 15: 66,
        16: 67, 17: 67, 18: 66, 19: 64, 20: 62, 21: 58, 22: 52, 23: 49,
    }
    # Dado que es un análisis por tiempo de día, generamos un día entero
    # (13/05 es miércoles) hora a hora (2 muestras por hora para variación).
    for hour, base in base_profile.items():
        hr.append(make_hr_sample(2026, 5, 13, hour, 20, 0, base))
        hr.append(make_hr_sample(2026, 5, 13, hour, 45, 0, base + 1))
    # Extremos artificiales: uno muy bajo a las 03:30 (35 bpm) y uno muy alto
    # a las 12:10 (130 bpm). Deben quedar fuera del filtro P10-P90.
    hr.append(make_hr_sample(2026, 5, 13, 3, 30, 0, 35))
    hr.append(make_hr_sample(2026, 5, 13, 12, 10, 0, 130))

    block = {
        "date": "2026-05-13",
        "metrics": {},
        "compulsive_metrics": {},
        "sessions": [],
        "unlock_events": [],
    }
    return make_health_processed(hr), make_phone_processed([block])


# ─────────────────────────────────────────────────────────────────────
# Conjunto completo de pruebas (documentación de EXPECTEDs).
# ─────────────────────────────────────────────────────────────────────
def build_all_cases() -> dict[str, tuple[dict, dict]]:
    return {
        "caso1_social": build_caso1_social(),
        "caso2_music": build_caso2_music(),
        "caso3_compulsivo": build_caso3_compulsivo(),
        "caso4_pre_sleep_social": build_caso4_pre_sleep_social(),
        "caso5_sleep_window": build_caso5_sleep_window(),
        "caso6_circadiano": build_caso6_circadiano(),
    }