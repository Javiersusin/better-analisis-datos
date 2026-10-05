#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paint_deep_analysis.py - Gráficas del análisis profundo (por día).

Genera gráficas por cada día del daily_summary en graficas_deep/:

  1. Uso del móvil por hora + resumen nocturno
  2. Panel Día vs Noche
  3. Efecto temporal ΔHR por categoría + event study
  4. Uso nocturno + HR como línea
  5. Ritmo circadiano (todas las juntas)

Qué lee:
    - health: heart_rate_raw (HR medición a medición), daily_summary (la fecha
      de cada día y su copia resumida de deep_analysis) y deep_analysis global
      (heart_rate_mobile_events -> sesiones con HR antes/después).
    - phone: daily_blocks (sesiones, métricas por hora y categorías).

Qué escribe: PNG en graficas_deep/ con el patrón deep_<n>_<tipo>_<fecha>.png.
Qué NO hace: no modifica ningún JSON. Backend Agg (sin pantalla).

Filtros que aplican SOLO al dibujo (los datos del JSON no se tocan):
    - se excluye la categoría "other" (apps del sistema sin clasificar)
    - se ignoran sesiones de menos de MIN_SESSION_SECONDS
    - el gráfico 5 exige al menos MIN_CIRCADIAN_SAMPLES muestras por hora
    - solo se pintan las categorías presentes en los datos (las que no están
      en CAT_COLORS salen con el gris de `other`)

Uso:
  python paint_deep_analysis.py <health_processed.json> <phone_usage_processed.json>
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─── Colores (académicos, sobrios) ─────────────────────────────────
BAR_BLUE = "#4A90D9"
HR_RED = "#D94A4A"
SLEEP_BG = "#E8F0FE"
WAKE_BG = "#E8F8E8"
PRE_SLEEP_BG = "#FFF3E0"

CAT_COLORS = {
    "social": "#E74C3C",
    "messaging": "#3498DB",
    "video_streaming": "#9B59B6",
    "productivity": "#27AE60",
    "music": "#F39C12",
    "navigation": "#1ABC9C",
    "other": "#95A5A6",
}

TEXT_DARK = "#2C3E50"
TEXT_GRAY = "#7F8C8D"
TEXT_LIGHT = "#BDC3C7"

# Filtros
MIN_SESSION_SECONDS = 5      # Sesiones mínimas para gráficos 3 y 4
MIN_CIRCADIAN_SAMPLES = 30   # Muestras mínimas por hora para gráfico 5
CATEGORIES_TO_SHOW = {"social", "messaging", "video_streaming", "productivity", "music", "navigation", "shopping", "education"}
# Nota: CATEGORIES_TO_SHOW hoy no se usa en el código; el filtrado real de
# categorías se hace excluyendo "other" (gráficos 3 y 4). Se deja por si se
# decide pasar a una lista blanca explícita.


# ─── Estilo base ────────────────────────────────────────────────────
def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": TEXT_DARK,
        "xtick.color": TEXT_DARK,
        "ytick.color": TEXT_DARK,
        "text.color": TEXT_DARK,
        "grid.color": "#EEEEEE",
        "grid.alpha": 0.8,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
    })


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  -> {name}")


def _ts(t: str) -> datetime:
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Extracción de datos por día ───────────────────────────────────
def extract_hr_by_hour_by_day(health: dict) -> dict[str, dict[int, list[float]]]:
    """HR por hora, separado por día. {fecha: {hora: [valores]}}"""
    hr_raw = health.get("heart_rate_raw", [])
    result: dict[str, dict[int, list[float]]] = {}
    for sample in hr_raw:
        t = _ts(sample["time"])
        day = t.strftime("%Y-%m-%d")
        h = t.hour
        result.setdefault(day, {}).setdefault(h, []).append(float(sample["value"]))
    return result


def extract_sessions_by_hour_by_day(phone: dict) -> dict[str, dict[int, float]]:
    """Minutos de uso por hora, separado por día."""
    result: dict[str, dict[int, float]] = {}
    for block in phone.get("daily_blocks", []):
        day = block.get("date", "")
        for entry in block.get("metrics", {}).get("usage_by_hour", []):
            t = _ts(entry["hour"])
            h = t.hour
            result.setdefault(day, {})
            result[day][h] = result[day].get(h, 0.0) + entry.get("minutes", 0.0)
    return result


def extract_sessions_by_hour_day_cat(phone: dict) -> dict[str, dict[int, dict[str, float]]]:
    """Minutos por hora y categoría, separado por día."""
    result: dict[str, dict[int, dict[str, float]]] = {}
    for block in phone.get("daily_blocks", []):
        day = block.get("date", "")
        for s in block.get("sessions", []):
            try:
                start = _ts(s["start_time"])
            except (KeyError, ValueError):
                continue
            h = start.hour
            cat = s.get("category") or "other"
            dur_min = s.get("duration_seconds", 0) / 60.0
            result.setdefault(day, {})
            result[day].setdefault(h, {})
            result[day][h][cat] = result[day][h].get(cat, 0.0) + dur_min
    return result


def extract_hr_sessions_by_day(health: dict) -> dict[str, list[dict]]:
    """Sesiones con HR (de heart_rate_mobile_events), separadas por día."""
    deep = health.get("deep_analysis", {})
    mobile_events = deep.get("heart_rate_mobile_events", {})
    all_sessions = mobile_events.get("sessions_analysis", [])
    result: dict[str, list[dict]] = {}
    for s in all_sessions:
        day = s.get("start_time", "")[:10]
        result.setdefault(day, []).append(s)
    return result


def extract_all_sessions_by_day(phone: dict) -> dict[str, list[dict]]:
    """Todas las sesiones de daily_blocks, separadas por día."""
    result: dict[str, list[dict]] = {}
    for block in phone.get("daily_blocks", []):
        day = block.get("date", "")
        for s in block.get("sessions", []):
            result.setdefault(day, []).append(s)
    return result


# ─── Gráfica 1: Uso del móvil por hora + resumen nocturno ──────────
def graf1_mobile_usage(
    date: str,
    sessions_by_hour: dict[int, float],
    sessions_by_hour_cat: dict[int, dict[str, float]],
    deep: dict,
    out_dir: str,
):
    """Barras de uso del móvil por hora con fondo sueño/vigilia."""
    pre_sleep = deep.get("pre_sleep_mobile", {}).get("mobile", {})
    mobile_sleep = deep.get("mobile_sleep_window", {})

    night_cats = pre_sleep.get("by_category", {})
    night_total_min = pre_sleep.get("total_minutes", 0)
    night_sessions = pre_sleep.get("sessions_count", 0)
    sleep_min = mobile_sleep.get("mobile_usage_during_sleep_window_minutes", 0)
    sleep_sessions = mobile_sleep.get("mobile_sessions_during_sleep_window", 0)

    fig, (ax_main, ax_night) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [7, 2.5]},
    )

    # Panel izquierdo: barras de uso por hora
    hours = list(range(24))
    minutes = [sessions_by_hour.get(h, 0.0) for h in hours]

    ax_main.axvspan(0, 5, color=SLEEP_BG, zorder=0)
    ax_main.axvspan(9, 22, color=WAKE_BG, zorder=0)
    ax_main.axvspan(22, 24, color=PRE_SLEEP_BG, zorder=0)
    ax_main.axvspan(5, 9, color="#F8F8F8", zorder=0)

    for h in hours:
        if h < 5 or h >= 22:
            cats = sessions_by_hour_cat.get(h, {})
            bottom = 0
            for cat, dur in cats.items():
                color = CAT_COLORS.get(cat, CAT_COLORS["other"])
                ax_main.bar(h, dur, bottom=bottom, color=color, alpha=0.85,
                            width=0.7, zorder=2, edgecolor="white", linewidth=0.3)
                bottom += dur
        else:
            ax_main.bar(h, minutes[h], color=BAR_BLUE, alpha=0.85,
                        width=0.7, zorder=2, edgecolor="white", linewidth=0.3)

    for h in hours:
        if (h < 5 or h >= 22) and minutes[h] > 0:
            ax_main.text(h, minutes[h] + 0.5, f"{minutes[h]:.0f}",
                         ha="center", va="bottom", fontsize=7, color=TEXT_GRAY)

    ax_main.set_xlabel("Hora del día (UTC)")
    ax_main.set_ylabel("Minutos de uso")
    ax_main.set_xticks(hours)
    ax_main.set_xticklabels([f"{h:02d}" for h in hours], fontsize=7)
    ax_main.set_xlim(-0.5, 23.5)
    ax_main.set_ylim(0, max(minutes) * 1.15 if minutes else 30)
    ax_main.grid(True, axis="y", linewidth=0.3)
    ax_main.set_title(f"Uso del móvil por hora — {date}", fontsize=12,
                       fontweight="bold", color=TEXT_DARK, pad=10)

    # Leyenda: zonas de fondo + categorías presentes
    legend_items = [
        mpatches.Patch(color=SLEEP_BG, label="Sueño (00–05)"),
        mpatches.Patch(color=WAKE_BG, label="Vigilia (09–22)"),
        mpatches.Patch(color=PRE_SLEEP_BG, label="Pre-sueño (22–00)"),
    ]
    # Añadir solo categorías que aparecen en los datos de noche
    cats_present = set()
    for h in hours:
        if h < 5 or h >= 22:
            cats_present.update(sessions_by_hour_cat.get(h, {}).keys())
    for cat in sorted(cats_present):
        color = CAT_COLORS.get(cat, CAT_COLORS["other"])
        legend_items.append(mpatches.Patch(color=color, alpha=0.85, label=cat.capitalize()))
    # Siempre mostrar día como referencia
    legend_items.append(mpatches.Patch(color=BAR_BLUE, alpha=0.85, label="Día (uso normal)"))

    ax_main.legend(handles=legend_items, loc="upper left", fontsize=7,
                   framealpha=0.9, edgecolor="#CCCCCC", ncol=2)

    # Panel derecho: resumen nocturno
    ax_night.axis("off")
    ax_night.set_xlim(0, 10)
    ax_night.set_ylim(0, 10)

    ax_night.text(5, 9.5, "Uso nocturno", ha="center", va="top",
                  fontsize=13, fontweight="bold", color=TEXT_DARK)
    ax_night.axhline(y=9.2, xmin=0.1, xmax=0.9, color=TEXT_LIGHT, linewidth=0.8)

    ax_night.text(5, 8.8, "Pre-sueño (22–00)", ha="center", va="top",
                  fontsize=10, fontweight="bold", color="#E67E22")
    y = 8.2
    ax_night.text(5, y, f"{night_total_min:.0f} min · {night_sessions} sesiones",
                  ha="center", va="top", fontsize=9, color=TEXT_DARK)
    y -= 0.6

    if night_cats:
        for cat, dur in sorted(night_cats.items(), key=lambda x: -x[1]):
            color = CAT_COLORS.get(cat, CAT_COLORS["other"])
            ax_night.barh(y, dur / night_total_min * 8 if night_total_min else 0,
                          height=0.4, left=1, color=color, alpha=0.8, zorder=2)
            ax_night.text(0.8, y, f"{cat.capitalize()}", ha="right", va="center",
                          fontsize=8, color=TEXT_DARK)
            ax_night.text(1 + dur / night_total_min * 8 + 0.2 if night_total_min else 1.2,
                          y, f"{dur:.0f} min", ha="left", va="center",
                          fontsize=8, color=TEXT_GRAY)
            y -= 0.6

    y -= 0.2
    ax_night.axhline(y=y, xmin=0.1, xmax=0.9, color=TEXT_LIGHT, linewidth=0.5)
    y -= 0.3

    ax_night.text(5, y, "Ventana de sueño (00–05)", ha="center", va="top",
                  fontsize=10, fontweight="bold", color="#2980B9")
    y -= 0.6
    ax_night.text(5, y, f"{sleep_min:.0f} min · {sleep_sessions} sesiones",
                  ha="center", va="top", fontsize=9, color=TEXT_DARK)
    y -= 0.6
    pct = mobile_sleep.get("mobile_usage_percentage_during_sleep_window", 0)
    ax_night.text(5, y, f"{pct:.1f}% del uso total",
                  ha="center", va="top", fontsize=8, color=TEXT_GRAY)

    fig.tight_layout()
    _save(fig, out_dir, f"deep_01_mobile_usage_{date}.png")


# ─── Gráfica 2: Panel Día vs Noche ─────────────────────────────────
def graf2_day_vs_night(
    date: str,
    hr_by_hour: dict[int, list[float]],
    sessions_by_hour: dict[int, float],
    phone_sessions: list[dict],
    deep: dict,
    out_dir: str,
):
    """Panel visual de resumen: DÍA vs NOCHE."""
    day_hours = list(range(9, 22))
    night_sleep_hours = list(range(0, 5))
    night_pre_hours = list(range(22, 24))

    day_hr = []
    night_sleep_hr = []
    night_pre_hr = []
    for h, vals in hr_by_hour.items():
        if h in day_hours:
            day_hr.extend(vals)
        if h in night_sleep_hours:
            night_sleep_hr.extend(vals)
        if h in night_pre_hours:
            night_pre_hr.extend(vals)

    day_min = sum(sessions_by_hour.get(h, 0) for h in day_hours)
    night_sleep_min = sum(sessions_by_hour.get(h, 0) for h in night_sleep_hours)
    night_pre_min = sum(sessions_by_hour.get(h, 0) for h in night_pre_hours)

    day_sessions = 0
    night_sleep_sessions = 0
    night_pre_sessions = 0
    for s in phone_sessions:
        try:
            h = _ts(s["start_time"]).hour
        except (KeyError, ValueError):
            continue
        if h in day_hours:
            day_sessions += 1
        elif h in night_sleep_hours:
            night_sleep_sessions += 1
        elif h in night_pre_hours:
            night_pre_sessions += 1

    mobile_sleep = deep.get("mobile_sleep_window", {})
    pre_sleep = deep.get("pre_sleep_mobile", {}).get("mobile", {})

    fig, (ax_day, ax_night) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Resumen Día vs Noche — {date}", fontsize=14,
                 fontweight="bold", color=TEXT_DARK, y=1.02)

    # Panel DÍA
    ax_day.set_facecolor(WAKE_BG)
    ax_day.set_xlim(0, 10)
    ax_day.set_ylim(0, 10)
    ax_day.axis("off")

    ax_day.text(5, 9.3, "DÍA", ha="center", va="top",
                fontsize=16, fontweight="bold", color="#27AE60")
    ax_day.text(5, 8.5, "(09:00 – 22:00)", ha="center", va="top",
                fontsize=9, color=TEXT_GRAY)

    y = 7.5
    metrics_day = [
        ("Pulso medio", f"{np.mean(day_hr):.1f} bpm" if day_hr else "—"),
        ("Min. uso móvil", f"{day_min:.0f} min"),
        ("Sesiones", f"{day_sessions}"),
    ]
    for label, value in metrics_day:
        ax_day.text(2, y, label, ha="left", va="center", fontsize=10, color=TEXT_GRAY)
        ax_day.text(8, y, value, ha="right", va="center", fontsize=11,
                    fontweight="bold", color=TEXT_DARK)
        y -= 1.2

    # Panel NOCHE
    ax_night.set_facecolor(SLEEP_BG)
    ax_night.set_xlim(0, 10)
    ax_night.set_ylim(0, 10)
    ax_night.axis("off")

    ax_night.text(5, 9.3, "NOCHE", ha="center", va="top",
                  fontsize=16, fontweight="bold", color="#2980B9")
    ax_night.text(5, 8.5, "(dividido en dos ventanas)", ha="center", va="top",
                  fontsize=9, color=TEXT_GRAY)

    y = 7.5
    ax_night.text(1, y, "Ventana de sueño (00–05)", ha="left", va="center",
                  fontsize=10, fontweight="bold", color="#2980B9")
    y -= 0.7
    for label, value in [
        ("Pulso medio", f"{np.mean(night_sleep_hr):.1f} bpm" if night_sleep_hr else "—"),
        ("Min. uso", f"{mobile_sleep.get('mobile_usage_during_sleep_window_minutes', 0):.0f} min"),
        ("Sesiones", f"{mobile_sleep.get('mobile_sessions_during_sleep_window', 0)}"),
    ]:
        ax_night.text(2, y, label, ha="left", va="center", fontsize=9, color=TEXT_GRAY)
        ax_night.text(8, y, value, ha="right", va="center", fontsize=10,
                      fontweight="bold", color=TEXT_DARK)
        y -= 0.6

    y -= 0.4
    ax_night.axhline(y=y, xmin=0.1, xmax=0.9, color=TEXT_LIGHT, linewidth=0.5)
    y -= 0.4

    ax_night.text(1, y, "Pre-sueño (22–00)", ha="left", va="center",
                  fontsize=10, fontweight="bold", color="#E67E22")
    y -= 0.7
    for label, value in [
        ("Pulso medio", f"{np.mean(night_pre_hr):.1f} bpm" if night_pre_hr else "—"),
        ("Min. uso", f"{pre_sleep.get('total_minutes', 0):.0f} min"),
        ("Sesiones", f"{pre_sleep.get('sessions_count', 0)}"),
    ]:
        ax_night.text(2, y, label, ha="left", va="center", fontsize=9, color=TEXT_GRAY)
        ax_night.text(8, y, value, ha="right", va="center", fontsize=10,
                      fontweight="bold", color=TEXT_DARK)
        y -= 0.6

    fig.tight_layout()
    _save(fig, out_dir, f"deep_02_day_vs_night_{date}.png")


# ─── Gráfica 3: Efecto temporal ΔHR por categoría ─────────────────
def graf3_temporal_effect(
    date: str,
    hr_sessions: list[dict],
    out_dir: str,
):
    """ΔHR por categoría + event study."""
    cat_data: dict[str, dict] = {}
    for s in hr_sessions:
        cat = s.get("category")
        if not cat:
            continue
        # Excluir 'other' - son apps del sistema no clasificadas
        if cat == "other":
            continue
        # Filtrar sesiones muy cortas (< 5 seg) - no representan uso real
        if s.get("duration_seconds", 0) < MIN_SESSION_SECONDS:
            continue
        hr_before = s.get("hr_before")
        hr_after = s.get("hr_after")
        hr_during = s.get("hr_during_mean")
        delta = s.get("delta_before_after")

        if hr_before is None and hr_after is None:
            continue

        cat_data.setdefault(cat, {
            "before": [], "after": [], "during": [],
            "delta": [], "durations": [],
        })
        if hr_before is not None:
            cat_data[cat]["before"].append(float(hr_before))
        if hr_after is not None:
            cat_data[cat]["after"].append(float(hr_after))
        if hr_during is not None:
            cat_data[cat]["during"].append(float(hr_during))
        if delta is not None:
            cat_data[cat]["delta"].append(float(delta))
        cat_data[cat]["durations"].append(s.get("duration_seconds", 0) / 60.0)

    if not cat_data:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Efecto temporal del pulso — {date}", fontsize=13,
                     fontweight="bold", color=TEXT_DARK, y=0.98)
        for ax in axes:
            ax.axis("off")
            ax.text(0.5, 0.5, "Sin datos de HR para este día",
                    ha="center", va="center", fontsize=12, color=TEXT_GRAY,
                    transform=ax.transAxes)
        _save(fig, out_dir, f"deep_03_temporal_effect_{date}.png")
        return

    cats_with_data = {c: d for c, d in cat_data.items() if len(d["delta"]) >= 2}
    cats_few = {c: d for c, d in cat_data.items() if len(d["delta"]) < 2}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6),
                                       gridspec_kw={"width_ratios": [1, 1.3]})
    fig.suptitle(f"Efecto temporal del pulso — {date}",
                 fontsize=13, fontweight="bold", color=TEXT_DARK, y=0.98)

    # Panel A: ΔHR por categoría
    all_cats = sorted(cat_data.keys(), key=lambda c: -len(cat_data[c]["delta"]))
    y_positions = np.arange(len(all_cats))

    for i, cat in enumerate(all_cats):
        d = cat_data[cat]
        deltas = d["delta"]
        n = len(deltas)
        color = CAT_COLORS.get(cat, CAT_COLORS["other"])

        if n >= 2:
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas) if n > 1 else 0

            jitter = np.random.normal(0, 0.12, n)
            ax_a.scatter(deltas, [i + j for j in jitter],
                        color=color, alpha=0.5, s=30, zorder=3, edgecolors="white", linewidth=0.3)

            ax_a.errorbar(mean_delta, i, xerr=std_delta, fmt="o",
                         color=color, markersize=8, capsize=4, capthick=1.5,
                         linewidth=1.5, zorder=4)

            arrow = "↑" if mean_delta > 0 else "↓" if mean_delta < 0 else "→"
            ax_a.text(mean_delta, i + 0.35, f"{arrow} {mean_delta:+.1f}",
                     ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
        else:
            if deltas:
                ax_a.scatter(deltas, [i], color=color, alpha=0.5, s=30, zorder=3)
                ax_a.text(deltas[0], i + 0.2, f"{deltas[0]:+.1f}",
                         ha="center", va="bottom", fontsize=8, color=color)

        ax_a.text(1.02, i, f"n={n}", ha="left", va="center", fontsize=9, color=TEXT_GRAY,
                 transform=ax_a.get_yaxis_transform())

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([c.capitalize() for c in all_cats], fontsize=10)
    ax_a.set_xlabel("ΔHR (bpm): HR después − HR antes", fontsize=10)
    ax_a.axvline(x=0, color=TEXT_LIGHT, linestyle="--", linewidth=1, zorder=1)
    ax_a.grid(True, axis="x", linewidth=0.3)
    ax_a.set_title("A. Cambio de pulso por categoría", fontsize=11,
                   fontweight="bold", color=TEXT_DARK, pad=10)

    if cats_few:
        note = "Categorías con n<2: " + ", ".join(
            f"{c.capitalize()} (n={len(d['delta'])})" for c, d in cats_few.items()
        )
        ax_a.text(0.5, -0.12, note, ha="center", va="top",
                 fontsize=8, color=TEXT_LIGHT, style="italic",
                 transform=ax_a.transAxes)

    # Panel B: Event study
    ax_b.set_title("B. Evolución del pulso alrededor de la sesión",
                   fontsize=11, fontweight="bold", color=TEXT_DARK, pad=10)

    time_labels = ["-5 min", "Inicio", "Durante", "Final", "+5 min"]
    time_x = [0, 1, 2, 3, 4]

    plotted_any = False
    for cat in sorted(cats_with_data.keys()):
        d = cats_with_data[cat]
        color = CAT_COLORS.get(cat, CAT_COLORS["other"])
        n = len(d["before"])

        if n < 3:
            continue

        hr_before = np.mean(d["before"]) if d["before"] else None
        hr_during = np.mean(d["during"]) if d["during"] else None
        hr_after = np.mean(d["after"]) if d["after"] else None

        if hr_before is None or hr_after is None:
            continue

        profile = [hr_before, hr_before, hr_during or hr_before, hr_after, hr_after]

        ax_b.plot(time_x, profile, color=color, linewidth=2, marker="o",
                 markersize=6, label=f"{cat.capitalize()} (n={n})", zorder=3)

        stds = []
        for vals in [d["before"], d["before"], d["during"], d["after"], d["after"]]:
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        profile_upper = [p + s for p, s in zip(profile, stds)]
        profile_lower = [p - s for p, s in zip(profile, stds)]
        ax_b.fill_between(time_x, profile_lower, profile_upper,
                         color=color, alpha=0.1, zorder=1)

        plotted_any = True

    if not plotted_any:
        ax_b.text(0.5, 0.5, "Necesario ≥3 sesiones por categoría\npara mostrar evolución temporal",
                 ha="center", va="center", fontsize=10, color=TEXT_GRAY,
                 transform=ax_b.transAxes)

    ax_b.set_xticks(time_x)
    ax_b.set_xticklabels(time_labels, fontsize=9)
    ax_b.set_ylabel("HR (bpm)", fontsize=10)
    ax_b.grid(True, axis="y", linewidth=0.3)
    if plotted_any:
        ax_b.legend(fontsize=8, framealpha=0.9, edgecolor="#CCCCCC", loc="best")

    ax_b.axvline(x=1, color=TEXT_LIGHT, linestyle=":", linewidth=1, zorder=1)
    ax_b.axvline(x=3, color=TEXT_LIGHT, linestyle=":", linewidth=1, zorder=1)

    fig.subplots_adjust(bottom=0.15, top=0.88, left=0.08, right=0.95)
    _save(fig, out_dir, f"deep_03_temporal_effect_{date}.png")


# ─── Gráfica 4: Uso nocturno + pulso ──────────────────────────────
def graf4_night_hr_sessions(
    date: str,
    hr_sessions: list[dict],
    hr_raw_day: list[dict],
    out_dir: str,
):
    """HR nocturno (22–05) + sesiones como bloques coloreados."""
    night_sessions = []
    for s in hr_sessions:
        try:
            t = _ts(s["start_time"])
            h = t.hour
            if h >= 22 or h < 5:
                night_sessions.append(s)
        except (KeyError, ValueError):
            continue

    night_hr_points = []
    for sample in hr_raw_day:
        t = _ts(sample["time"])
        h = t.hour
        m = t.minute
        if h >= 22:
            minute_offset = (h - 22) * 60 + m
        elif h < 5:
            minute_offset = (h + 2) * 60 + m
        else:
            continue
        night_hr_points.append((minute_offset, float(sample["value"])))

    night_hr_points.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Uso nocturno y frecuencia cardíaca — {date}",
                 fontsize=13, fontweight="bold", color=TEXT_DARK, y=0.98)

    ax.axvspan(0, 120, color=PRE_SLEEP_BG, zorder=0, label="Pre-sueño (22–00)")
    ax.axvspan(120, 420, color=SLEEP_BG, zorder=0, label="Sueño (00–05)")

    if night_hr_points:
        mins = [p[0] for p in night_hr_points]
        hrs = [p[1] for p in night_hr_points]
        ax.plot(mins, hrs, color=HR_RED, linewidth=1.5, alpha=0.8, zorder=4, label="HR")
        ax.fill_between(mins, hrs, alpha=0.05, color=HR_RED, zorder=1)

    for s in night_sessions:
        # Excluir 'other' - son apps del sistema no clasificadas
        if s.get("category") == "other":
            continue
        # Filtrar sesiones muy cortas
        if s.get("duration_seconds", 0) < MIN_SESSION_SECONDS:
            continue
        try:
            t_start = _ts(s["start_time"])
            t_end = _ts(s["end_time"])
        except (KeyError, ValueError):
            continue

        h_start, m_start = t_start.hour, t_start.minute
        h_end, m_end = t_end.hour, t_end.minute

        min_start = (h_start - 22) * 60 + m_start if h_start >= 22 else (h_start + 2) * 60 + m_start
        min_end = (h_end - 22) * 60 + m_end if h_end >= 22 else (h_end + 2) * 60 + m_end

        cat = s.get("category", "other")
        color = CAT_COLORS.get(cat, CAT_COLORS["other"])
        duration = min_end - min_start

        ax.barh(0, duration, left=min_start, height=8, color=color,
                alpha=0.7, zorder=2, edgecolor="white", linewidth=0.5)

        if duration > 10:
            ax.text(min_start + duration / 2, 4, cat[:4].capitalize(),
                   ha="center", va="center", fontsize=7, color="white",
                   fontweight="bold", zorder=3)

        hr_before = s.get("hr_before")
        hr_after = s.get("hr_after")
        if hr_before is not None:
            ax.plot(min_start, hr_before, "v", color=color, markersize=6, zorder=5, alpha=0.8)
        if hr_after is not None:
            ax.plot(min_end, hr_after, "^", color=color, markersize=6, zorder=5, alpha=0.8)

    tick_mins = list(range(0, 421, 60))
    tick_labels = []
    for m in tick_mins:
        h = (m // 60 + 22) % 24
        tick_labels.append(f"{h:02d}:00")

    ax.set_xticks(tick_mins)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlim(0, 420)

    if night_hr_points:
        all_hr = [p[1] for p in night_hr_points]
        ax.set_ylim(min(all_hr) - 5, max(all_hr) + 10)
    ax.set_ylabel("HR (bpm)", fontsize=10)
    ax.set_xlabel("Hora del día (UTC)", fontsize=10)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)

    legend_items = [
        plt.Line2D([0], [0], color=HR_RED, linewidth=1.5, label="HR"),
        mpatches.Patch(color=PRE_SLEEP_BG, label="Pre-sueño (22–00)"),
        mpatches.Patch(color=SLEEP_BG, label="Sueño (00–05)"),
    ]
    # Solo mostrar categorías clasificadas (excluir 'other')
    cats_present = set(s.get("category") for s in night_sessions
                       if s.get("category") and s.get("category") != "other")
    for cat in sorted(cats_present):
        color = CAT_COLORS.get(cat, CAT_COLORS["other"])
        legend_items.append(mpatches.Patch(color=color, alpha=0.7, label=cat.capitalize()))

    ax.legend(handles=legend_items, loc="upper right", fontsize=7,
             framealpha=0.9, edgecolor="#CCCCCC", ncol=2)

    ax.text(0.5, -0.15, "Triángulos ▼ = HR antes de sesión · ▲ = HR después de sesión",
           ha="center", va="top", fontsize=8, color=TEXT_GRAY,
           transform=ax.transAxes)

    fig.tight_layout()
    _save(fig, out_dir, f"deep_04_night_hr_sessions_{date}.png")


# ─── Gráfica 5: Ritmo circadiano por día ──────────────────────────
def graf5_circadian_rhythm(
    date: str,
    hr_by_hour: dict[int, list[float]],
    out_dir: str,
):
    """Perfil circadiano de HR para un día específico.

    Solo muestra horas con ≥30 muestras para garantizar calidad.
    """
    if not hr_by_hour:
        print("  [5] %s: Sin datos HR." % date)
        return

    # Filtrar horas con suficientes datos
    hours_with_data = []
    for h in range(24):
        vals = hr_by_hour.get(h, [])
        if len(vals) >= MIN_CIRCADIAN_SAMPLES:
            hours_with_data.append((h, np.mean(vals), len(vals)))

    if len(hours_with_data) < 2:
        print("  [5] %s: Insuficientes horas con datos >=%d muestras." % (date, MIN_CIRCADIAN_SAMPLES))
        return

    hours = [h for h, _, _ in hours_with_data]
    hr_means = [m for _, m, _ in hours_with_data]
    counts = [c for _, _, c in hours_with_data]

    # Calcular medias día/noche
    day_vals = [m for h, m, _ in hours_with_data if 9 <= h < 22]
    night_vals = [m for h, m, _ in hours_with_data if h < 5]

    fc_day_mean = np.mean(day_vals) if day_vals else None
    fc_night_mean = np.mean(night_vals) if night_vals else None

    min_val = min(hr_means)
    max_val = max(hr_means)
    min_hour = hours[hr_means.index(min_val)]
    max_hour = hours[hr_means.index(max_val)]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Ritmo circadiano de frecuencia cardíaca — {date}",
                 fontsize=13, fontweight="bold", color=TEXT_DARK, y=0.98)

    # Franjas de fondo
    ax.axvspan(0, 5, color=SLEEP_BG, zorder=0)
    ax.axvspan(9, 22, color=WAKE_BG, zorder=0)
    ax.axvspan(22, 24, color=PRE_SLEEP_BG, zorder=0)
    ax.axvspan(5, 9, color="#F8F8F8", zorder=0)

    # Línea principal
    ax.plot(hours, hr_means, color=HR_RED, linewidth=2.5, zorder=4,
            marker="o", markersize=6, label="HR media horaria")

    # Añadir conteo de muestras debajo de cada punto
    for h, m, c in zip(hours, hr_means, counts):
        ax.text(h, m - 2, f"n={c}", ha="center", va="top", fontsize=7, color=TEXT_GRAY)

    # Líneas de referencia día/noche
    if fc_day_mean is not None:
        ax.axhline(y=fc_day_mean, color="#27AE60", linestyle="--",
                   linewidth=1.2, alpha=0.8, zorder=3)
        ax.text(max(hours) + 0.3, fc_day_mean + 0.3, f"Día: {fc_day_mean:.1f}",
                fontsize=8, color="#27AE60", fontweight="bold")
    if fc_night_mean is not None:
        ax.axhline(y=fc_night_mean, color="#2980B9", linestyle="--",
                   linewidth=1.2, alpha=0.8, zorder=3)
        ax.text(max(hours) + 0.3, fc_night_mean - 0.8, f"Noche: {fc_night_mean:.1f}",
                fontsize=8, color="#2980B9", fontweight="bold")

    # Marcadores de mínimo y máximo
    ax.annotate(
        f"Mín: {min_val:.0f} bpm\n({min_hour}:00)",
        xy=(min_hour, min_val),
        xytext=(min_hour + 1.5, min_val - 3),
        arrowprops=dict(arrowstyle="->", color="#2980B9", lw=1.5),
        fontsize=9, color="#2980B9", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#2980B9", alpha=0.9),
    )
    ax.annotate(
        f"Máx: {max_val:.0f} bpm\n({max_hour}:00)",
        xy=(max_hour, max_val),
        xytext=(max_hour + 1.5, max_val + 3),
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5),
        fontsize=9, color="#E74C3C", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#E74C3C", alpha=0.9),
    )

    ax.set_xlabel("Hora del día (UTC)")
    ax.set_ylabel("BPM")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    ax.grid(True, axis="both", linewidth=0.3, alpha=0.5)

    # Leyenda
    handles = [
        plt.Line2D([0], [0], color=HR_RED, linewidth=2, marker="o",
                   markersize=6, label="HR media horaria"),
        mpatches.Patch(color=SLEEP_BG, label="Noche (00–05)"),
        mpatches.Patch(color=WAKE_BG, label="Día (09–22)"),
    ]
    if fc_day_mean is not None:
        handles.append(plt.Line2D([0], [0], color="#27AE60", linestyle="--",
                                  linewidth=1, label=f"FC día media: {fc_day_mean:.1f}"))
    if fc_night_mean is not None:
        handles.append(plt.Line2D([0], [0], color="#2980B9", linestyle="--",
                                  linewidth=1, label=f"FC noche media: {fc_night_mean:.1f}"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9,
              edgecolor="#CCCCCC")

    fig.tight_layout()
    _save(fig, out_dir, f"deep_05_circadian_rhythm_{date}.png")


# ─── Main ──────────────────────────────────────────────────────────
def main() -> int:
    if len(sys.argv) < 3:
        print("Uso: python paint_deep_analysis.py <health_processed.json> <phone_usage_processed.json>")
        return 1

    health_path = sys.argv[1]
    phone_path = sys.argv[2]

    if not os.path.exists(health_path):
        print(f"[ERROR] No existe: {health_path}")
        return 1
    if not os.path.exists(phone_path):
        print(f"[ERROR] No existe: {phone_path}")
        return 1

    print("Cargando datos...")
    health = load_json(health_path)
    phone = load_json(phone_path)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficas_deep")
    print(f"Generando gráficas en {out_dir}/")

    _style()

    # Extraer datos
    hr_by_hour_by_day = extract_hr_by_hour_by_day(health)
    sessions_by_hour_by_day = extract_sessions_by_hour_by_day(phone)
    sessions_by_hour_day_cat = extract_sessions_by_hour_day_cat(phone)
    hr_sessions_by_day = extract_hr_sessions_by_day(health)
    all_sessions_by_day = extract_all_sessions_by_day(phone)
    deep = health.get("deep_analysis", {})

    # Días del daily_summary
    dates = [d.get("date") for d in health.get("daily_summary", []) if d.get("date")]
    if not dates:
        # Si no hay daily_summary, usar los días encontrados en los datos
        dates = sorted(set(list(hr_by_hour_by_day.keys()) + list(sessions_by_hour_by_day.keys())))

    print(f"\nDías encontrados: {dates}")
    total_graf = 0

    for date in dates:
        print(f"\n--- {date} ---")
        hr_by_hour = hr_by_hour_by_day.get(date, {})
        sessions_by_hour = sessions_by_hour_by_day.get(date, {})
        sessions_by_hour_cat = sessions_by_hour_day_cat.get(date, {})
        hr_sessions = hr_sessions_by_day.get(date, [])
        phone_sessions = all_sessions_by_day.get(date, [])

        # HR raw de este día
        hr_raw_day = [s for s in health.get("heart_rate_raw", [])
                      if s.get("time", "")[:10] == date]

        # Deep analysis de este día (de daily_summary)
        day_summary = next((d for d in health.get("daily_summary", []) if d.get("date") == date), {})
        day_deep = day_summary.get("deep_analysis", {})

        graf1_mobile_usage(date, sessions_by_hour, sessions_by_hour_cat, day_deep, out_dir)
        graf2_day_vs_night(date, hr_by_hour, sessions_by_hour, phone_sessions, day_deep, out_dir)
        graf3_temporal_effect(date, hr_sessions, out_dir)
        graf4_night_hr_sessions(date, hr_sessions, hr_raw_day, out_dir)
        graf5_circadian_rhythm(date, hr_by_hour, out_dir)
        total_graf += 5

    print(f"\n[OK] {total_graf} gráficas generadas en {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
