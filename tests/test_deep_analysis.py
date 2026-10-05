#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests del bloque deep_analysis con datos sintéticos REALISTAS.

Cada test incluye comentarios EXPECTED indicando el resultado esperado, y
comprueba de verdad el resultado mediante assertions (no solo imprimir).

Cómo ejecutar (desde la raíz del proyecto):
    python -m pytest tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permitimos importar deep_analysis (raíz del proyecto) desde tests/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import deep_analysis as da
import pytest
from tests import test_data_builders as tdb


# ─────────────────────────────────────────────────────────────────────
# Caso 1: red social + aumento de HR temporalmente asociado.
# ─────────────────────────────────────────────────────────────────────
def test_caso1_social_detecta_incremento_hrtemporal():
    """
    EXPECTED:
    - La sesión social de 22:15-22:40 del 13/05 se encuentra en el análisis.
    - Se detecta un aumento de HR temporalmente cercano a la sesión social:
        hr_before   = 60
        hr_during   ≈ 63 (media de 62,63,64)
        hr_after    = 68
        delta_before_after = +8
    """
    health, phone = tdb.build_caso1_social()
    deep = da.build_deep_analysis(health, phone)

    # Los análisis se agrupan en heart_rate_mobile_events.sessions_analysis.
    sessions_hr = deep["heart_rate_mobile_events"]["sessions_analysis"]
    assert len(sessions_hr) == 1

    s = sessions_hr[0]
    assert s["category"] == "social"
    assert s["hr_before"] == 60, "EXPECTED: hr_before = 60"
    assert s["hr_after"] == 68, "EXPECTED: hr_after = 68"
    assert s["hr_during_mean"] == pytest.approx(63.0, abs=0.01), "EXPECTED: media durante ≈ 63"
    assert s["delta_before_after"] == 8, "EXPECTED: delta_before_after = +8"
    assert s["delta_before_after"] > 0

    # Resumen por categoría: social presente.
    by_cat = deep["heart_rate_mobile_events"]["by_category"]
    assert "social" in by_cat
    assert by_cat["social"]["sessions_with_increase"] >= 1


# ─────────────────────────────────────────────────────────────────────
# Caso 2: música SIN incremento relevante.
# ─────────────────────────────────────────────────────────────────────
def test_caso2_music_no_es_detectado_como_gran_cambio():
    """
    EXPECTED:
    - La sesión de music NO se marca como un gran cambio: delta_before_after
      pequeño (0-1) y sin incremento relevante.
    """
    health, phone = tdb.build_caso2_music()
    deep = da.build_deep_analysis(health, phone)

    sessions_hr = deep["heart_rate_mobile_events"]["sessions_analysis"]
    assert len(sessions_hr) == 1
    s = sessions_hr[0]
    assert s["category"] == "music"
    # hr_before=60 (17:59), hr_after=60 (19:00) o 60 (18:40 según ventana).
    assert s["delta_before_after"] is not None
    assert abs(s["delta_before_after"]) <= 1, (
        f"EXPECTED: cambio pequeño <=1, se obtuvo {s['delta_before_after']}"
    )


# ─────────────────────────────────────────────────────────────────────
# Caso 3: comportamiento compulsivo (unlock->social->close repetido).
# ─────────────────────────────────────────────────────────────────────
def test_caso3_compulsivo_detecta_patron_y_relacion_hr():
    """
    EXPECTED:
    - Se detectan eventos compulsivos (unlock, rapid_reopening, short_session,
      app_switch) en la ventana 22:00-00:00.
    - Hay al menos 3 reaperturas rápidas (misma categoría social separadas
      <= 10 s).
    - Existe relación temporal: delta_hr_compulsive_event > 0.
    """
    health, phone = tdb.build_caso3_compulsivo()

    # A) Uso móvil antes de dormir: rápido y compulsivo.
    sessions = da.load_sessions(phone)
    unlocks = da.load_unlock_events(phone)
    compulsive_pre = da.compulsive_behavior_before_sleep(sessions, unlocks)
    assert compulsive_pre["unlock_count_before_sleep"] >= 3, "EXPECTED: >= 3 desbloqueos en pre-sueño"
    assert compulsive_pre["rapid_reopenings_before_sleep"] >= 3, "EXPECTED: >= 3 reaperturas rápidas"
    assert compulsive_pre["short_sessions_before_sleep"] >= 3, "EXPECTED: >= 3 sesiones cortas"

    # B) Relación con HR.
    deep = da.build_deep_analysis(health, phone)
    ce = deep["compulsive_events_heart_rate"]
    assert "rapid_reopening" in ce or "short_session" in ce, "EXPECTED: eventos compulsivos detectados"

    total_deltas = []
    for etype, data in ce.items():
        for ev in data["events"]:
            if ev["delta_hr"] is not None:
                total_deltas.append(ev["delta_hr"])
    assert len(total_deltas) >= 1, "EXPECTED: al menos un delta de HR asociado a evento compulsivo"
    # Los datos del caso 3 suben la HR con el patrón (58->62->66), así que
    # algunos deltas deben ser > 0.
    assert any(d > 0 for d in total_deltas), "EXPECTED: algún delta_hr positivo"


# ─────────────────────────────────────────────────────────────────────
# Caso 4: uso de redes sociales antes de dormir.
# ─────────────────────────────────────────────────────────────────────
def test_caso4_uso_social_before_sleep_calculado():
    """
    EXPECTED:
    - 30 minutos de móvil antes de dormir (3 sesiones de 10 min: 22:30, 22:55,
      23:20) + 5 min de música (23:40) = 35 min totales.
    - 30 minutos de categoría social.
    - 5 minutos de categoría music.
    """
    health, phone = tdb.build_caso4_pre_sleep_social()

    sessions = da.load_sessions(phone)
    by_cat = da.mobile_usage_before_sleep(sessions)

    assert by_cat["mobile_usage_before_sleep_minutes"] == pytest.approx(35.0, abs=0.01), (
        "EXPECTED: 35 min de uso total antes de dormir"
    )
    assert by_cat["by_category"]["social"] == pytest.approx(30.0, abs=0.01), (
        "EXPECTED: 30 min de social antes de dormir"
    )
    assert by_cat["by_category"]["music"] == pytest.approx(5.0, abs=0.01), (
        "EXPECTED: 5 min de music antes de dormir"
    )

    # El bloque deep_analysis expone social_minutes_before_sleep indirectamente
    # vía pre_sleep_mobile.mobile.by_category.
    deep = da.build_deep_analysis(health, phone)
    assert deep["pre_sleep_mobile"]["mobile"]["by_category"]["social"] == pytest.approx(30.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────
# Caso 5: uso durante la ventana de sueño.
# ─────────────────────────────────────────────────────────────────────
def test_caso5_uso_durante_ventana_sueno():
    """
    EXPECTED:
    - 10 min (00:30-00:40) + 15 min (02:10-02:25) = 25 min en la ventana de
      sueño 00:00-05:00.
    - 2 sesiones en la ventana de sueño.
    """
    health, phone = tdb.build_caso5_sleep_window()

    sessions = da.load_sessions(phone)
    metric = da.mobile_usage_during_sleep_window(sessions)

    assert metric["mobile_usage_during_sleep_window_minutes"] == pytest.approx(25.0, abs=0.01), (
        "EXPECTED: 25 min en ventana de sueño"
    )
    assert metric["mobile_sessions_during_sleep_window"] == 2, (
        "EXPECTED: 2 sesiones en ventana de sueño"
    )

    # La sesión 01 (00:30-00:40) tiene 10 min => 40 % de la ventana dentro de
    # sueño; la otra 15 min, etc. Comprobación de coherencia del porcentaje.
    total = metric["total_mobile_usage_minutes"]
    assert total == pytest.approx(25.0 + 0.0 + 0.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────
# Caso 6: circadianidad + filtro P10-P90.
# ─────────────────────────────────────────────────────────────────────
def test_caso6_p10_p90_elimina_extremos_y_conserva_forma():
    """
    EXPECTED:
    - El filtro P10-P90 elimina los valores extremos (35 y 130 bpm).
    - La sociedad no se destruye: la media filtrada queda dentro de un rango
      razonable (~51-66 bpm por la curva de base).
    - Se guardan raw_heart_rate_count y filtered_heart_rate_count.
    """
    health, phone = tdb.build_caso6_circadiano()

    hr = da.load_hr_samples(health)
    values = [s["value"] for s in hr]

    filtered, lo, hi = da.filter_p10_p90(values)
    # EXPECTED: los extremos quedan fuera del rango P10-P90.
    assert 35 not in filtered, "EXPECTED: el extremo bajo (35) se elimina del análisis P10-P90"
    assert 130 not in filtered, "EXPECTED: el extremo alto (130) se elimina del análisis P10-P90"
    assert all(lo <= v <= hi for v in filtered)
    assert lo is not None and hi is not None

    # La forma general se conserva: media de la señal filtrada debe ser ~53-56.
    mean_filtered = sum(filtered) / len(filtered)
    assert 45 <= mean_filtered <= 70, f"EXPECTED: forma conservada, media {mean_filtered:.1f}"

    deep = da.build_deep_analysis(health, phone)
    circ = deep["circadian_heart_rate"]["all_days"]
    assert circ["raw"]["count"] == 50, "EXPECTED: raw_heart_rate_count"
    assert circ["filtered"]["count"] < circ["raw"]["count"], (
        "EXPECTED: el filtro P10-P90 descarta valores de las colas"
    )
    assert circ["filtered"]["count"] == len(filtered)
    assert circ["filtered"]["lower_cutoff"] == pytest.approx(lo)
    assert circ["filtered"]["upper_cutoff"] == pytest.approx(hi)


# ─────────────────────────────────────────────────────────────────────
# Separación weekday / weekend.
# ─────────────────────────────────────────────────────────────────────
def test_weekday_weekend_separacion():
    """
    EXPECTED:
    - Un día entre semana (lunes-viernes) solo aparece en weekday.
    - Un día de fin de semana (sábado-domingo) solo aparece en weekend.
    """
    # Creamos HR con 13/05 (miércoles) y 16/05 (sábado).
    hr = [
        tdb.make_hr_sample(2026, 5, 13, 12, 0, 0, 60),
        tdb.make_hr_sample(2026, 5, 16, 12, 0, 0, 65),
    ]
    health = tdb.make_health_processed(hr)
    deep = da.build_deep_analysis(health, None)

    circ = deep["circadian_heart_rate"]
    assert circ["all_days"]["count_days"] == 2
    assert "2026-05-13" in circ["weekday"]["computed"]
    assert "2026-05-13" not in circ["weekend"]["computed"]
    assert "2026-05-16" in circ["weekend"]["computed"]
    assert "2026-05-16" not in circ["weekday"]["computed"]


# ─────────────────────────────────────────────────────────────────────
# Hora del mínimo/máximo diario.
# ─────────────────────────────────────────────────────────────────────
def test_hour_of_daily_min_max():
    """
    EXPECTED:
    - El máximo diario se da por la tarde y el mínimo por la noche/madrugada
      en la curva del caso 6.
    """
    health, phone = tdb.build_caso6_circadiano()
    circ = da.circadian_heart_rate_analysis(da.load_hr_samples(health))
    all_days = circ["all_days"]

    # El máximo ocurre a las 16-17 h; el mínimo a las 3-4 h.
    assert all_days["hour_of_daily_max_hr"] in (16, 17), (
        f"EXPECTED: hora del máximo ~16-17, se obtuvo {all_days['hour_of_daily_max_hr']}"
    )
    assert all_days["hour_of_daily_min_hr"] in (3, 4, 2), (
        f"EXPECTED: hora del mínimo ~3-4, se obtuvo {all_days['hour_of_daily_min_hr']}"
    )


# ─────────────────────────────────────────────────────────────────────
# Deltas de HR entre mediciones consecutivas.
# ─────────────────────────────────────────────────────────────────────
def test_hr_delta_events():
    """
    EXPECTED:
    - Con las muestras del caso 2 (5 muestras) hay 4 pares consecutivos.
    - Algunos deltas son 0/±1 y se registran correctamente.
    """
    health, phone = tdb.build_caso2_music()
    hr = da.load_hr_samples(health)
    deltas = da.heart_rate_delta_events(hr)

    assert deltas["count"] == 4, "EXPECTED: 4 deltas entre 5 mediciones consecutivas"
    assert len(deltas["sample_events"]) == 4
    # Las dos primeras muestras: 60 -> 60 (delta 0) y 60 -> 61 (delta +1)
    assert deltas["sample_events"][0]["delta_hr"] == 0
    assert any(e["delta_hr"] in (1, -1) for e in deltas["sample_events"])


# ─────────────────────────────────────────────────────────────────────
# Datos ausentes: no se inventa nada.
# ─────────────────────────────────────────────────────────────────────
def test_sin_datos_hr_ni_movil():
    """
    EXPECTED:
    - Si no hay datos de HR ni de móvil, el bloque de deep_analysis devuelve
      has_heart_rate_data=False y has_mobile_data=False, sin inventar métricas.
    """
    health = tdb.make_health_processed([])
    deep = da.build_deep_analysis(health, None)
    assert deep["data_quality"]["has_heart_rate_data"] is False
    assert deep["data_quality"]["has_mobile_data"] is False
    assert deep["mobile_sleep_window"] is None


# ─────────────────────────────────────────────────────────────────────
# La sesión social del caso 1 también debe aparecer en el bloque
# antes de dormir (22:00-00:00).
# ─────────────────────────────────────────────────────────────────────
def test_caso1_social_en_before_sleep():
    """
    EXPECTED:
    - La sesión social del caso 1 ocurre a las 22:15-22:40 (dentro de
      22:00-00:00), por lo que el análisis before_sleep debe reflejarla.
    - HR antes de dormir (dentro de 22:00-00:00) tiene datos.
    """
    health, phone = tdb.build_caso1_social()
    deep = da.build_deep_analysis(health, phone)

    bs = deep["before_sleep"]
    assert bs is not None
    # Uso antes de dormir: 25 min (1500 s) solo sociedad.
    assert bs["mobile"]["mobile_usage_before_sleep_minutes"] == pytest.approx(25.0, abs=0.01)
    assert bs["mobile"]["by_category"]["social"] == pytest.approx(25.0, abs=0.01)
    # HR dentro de la ventana: muestras a 22:18/22:25/22:33/22:45 -> count > 0.
    assert bs["heart_rate_before_sleep"]["count"] >= 4