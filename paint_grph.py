#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paint_grph.py - Gráficas de resumen de SALUD -> carpeta graficas/.

Qué lee:
    Un `*_health_processed.json`: heart_rate_resume (pulso en bloques de
    5 min), general_resume (pasos por día), blood_oxygen_resume (SpO2) y
    daily_summary (actividad, calorías, sueño, workouts, HRV, FC en reposo).

Qué escribe (PNG en graficas/):
    6 globales:
        01_pulso_completo      - una fila por día con la curva de pulso
        02_pulso_comparativo   - media diaria de BPM
        03_pasos_semanales     - pasos por día con objetivo 8000
        04_calorias_semanales  - kcal por día
        05_entrenamientos      - minutos/sesiones de workout (SOLO si hay)
        06_resumen_semanal     - panel multiparamétrico (pasos, FC, reposo,
                                 HRV, calorías; los dos últimos solo si existen)
    1 por día:
        07_diario_<fecha>_<d>.png - detalle del día (FC, stats, pasos+calorías,
                                 SpO2/HRV y tarta de sueño)
    Por eso en el repo puede faltar 05: no sale nada si daily_summary no
    tiene sección `workouts`.

Qué NO hace:
    No modifica ningún JSON: solo lee y dibuja. Backend Agg (sin pantalla).

Uso:
    python paint_grph.py [ruta/al/health_processed.json]
    Sin argumento usa DATA_FILE de más abajo (una ruta por defecto hardcodeada).
"""

import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Archivo por defecto: se puede cambiar pasando la ruta como primer argumento.
DATA_FILE = os.path.join(SCRIPT_DIR, "260516_1728_health_processed.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "graficas")

if len(sys.argv) > 1:
    DATA_FILE = sys.argv[1]

C = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "grid": "#21262d",
    "text": "#c9d1d9",
    "dim": "#8b949e",
    "title": "#f0f6fc",
    "hr": "#ff6b6b",
    "hr_avg": "#ffa657",
    "steps": "#58a6ff",
    "steps_dim": "#1f6feb",
    "spo2": "#7ee787",
    "cal": "#f0883e",
    "cal_dim": "#bd561d",
    "workout": "#d2a8ff",
    "sleep_deep": "#388bfd",
    "sleep_light": "#79c0ff",
    "sleep_rem": "#d2a8ff",
    "sleep_awake": "#f85149",
    "accent": "#58a6ff",
    "z_rest": "#238636",
    "z_fat": "#d29922",
    "z_cardio": "#da3633",
    "z_peak": "#f85149",
}

WD = {0: "Lun", 1: "Mar", 2: "Mie", 3: "Jue", 4: "Vie", 5: "Sab", 6: "Dom"}
# Tramos de BPM solo para pintar el fondo (descanso/grasa/cardio). Es una
# referencia visual genérica, NO una medición ni un juicio clínico.
HR_ZONES = [(40, 80, C["z_rest"]), (80, 110, C["z_fat"]), (110, 140, C["z_cardio"])]


def load():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def style():
    plt.rcParams.update({
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["panel"],
        "axes.edgecolor": C["grid"],
        "axes.labelcolor": C["text"],
        "xtick.color": C["text"],
        "ytick.color": C["text"],
        "text.color": C["text"],
        "grid.color": C["grid"],
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
    })


def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=120, facecolor=fig.get_facecolor(), pad_inches=0.2)
    plt.close(fig)
    print(f"  -> {name}")


def day_label(d):
    dt = datetime.strptime(d, "%Y-%m-%d")
    return f"{WD[dt.weekday()]} {dt.day}/{dt.month}"


def ts(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))



def get_summary(data, date):
    return next((s for s in data.get("daily_summary", []) if s["date"] == date), None)


def apply_hr_bg(ax):
    for lo, hi, clr in HR_ZONES:
        ax.axhspan(lo, hi, alpha=0.07, color=clr, zorder=0)


def hr_formatter(ax, times, interval=None):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if interval is None:
        span = (times[-1] - times[0]).total_seconds()
        interval = max(1, int(span / 7200))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)


# ─── GRÁFICAS GLOBALES ──────────────────────────────────────────────

def graf_pulso_completo(data):
    print("\n  [1/7] Pulso completo por dia...")
    dp = data.get("heart_rate_resume", {}).get("daily_points", [])
    if not dp:
        print("    Sin datos de pulso.")
        return
    n = len(dp)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Frecuencia Cardiaca - Vista Completa por Dia",
                 color=C["title"], fontsize=14, fontweight="bold", y=0.998)

    for i, d in enumerate(dp):
        ax = axes[i]
        pts = d.get("points", [])
        t = [ts(p["time"]) for p in pts]
        b = [p.get("bpm", 0) for p in pts]
        lbl = day_label(d.get("date", ""))
        avg, mn, mx = np.mean(b), min(b), max(b)

        ax.fill_between(t, b, alpha=0.12, color=C["hr"], zorder=1)
        ax.plot(t, b, color=C["hr"], linewidth=0.5, alpha=0.4, zorder=2)
        ax.axhline(avg, color=C["hr_avg"], linewidth=1, ls="--", alpha=0.7)
        ax.text(t[-1], avg + 1.5, f" {avg:.0f}", fontsize=7, color=C["hr_avg"], va="bottom", ha="right")

        n_pts = len(b)
        ax.set_title(f"{lbl} ({d['date']})  Min:{mn:.0f}  Max:{mx:.0f}  Avg:{avg:.0f}  n={n_pts}",
                     loc="left", fontsize=9, color=C["title"], pad=4)
        ax.set_ylabel("BPM", fontsize=8)
        hr_formatter(ax, t)
        ax.grid(True, axis="y", linewidth=0.3)
        ax.set_xlim(t[0], t[-1])
        apply_hr_bg(ax)

    fig.text(0.5, 0.002, "Zonas: Verde=Descanso(<80) Amarillo=Quemar grasa(80-110) Rojo=Cardio(110-140)",
             ha="center", fontsize=7, color=C["dim"])
    fig.tight_layout(rect=[0, 0.015, 1, 0.985])
    save(fig, "01_pulso_completo.png")


def graf_pulso_comparativo(data):
    print("  [2/7] Pulso comparativo semanal...")
    da = data.get("heart_rate_resume", {}).get("daily_average_bpm", [])
    if not da:
        print("    Sin datos de FC promedio.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Frecuencia Cardiaca Promedio - Comparativa Semanal",
                 color=C["title"], fontsize=13, fontweight="bold")

    labels = [day_label(d["date"]) for d in da]
    bpms = [d.get("bpm", 0) for d in da]
    samples = [d.get("samples", 0) for d in da]
    clrs = [C["z_rest"] if b < 80 else C["z_fat"] if b < 100 else C["z_cardio"] for b in bpms]

    bars = ax.bar(labels, bpms, color=clrs, edgecolor="white", linewidth=0.3, alpha=0.85, width=0.55)
    for bar, b, s in zip(bars, bpms, samples):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{b:.1f}", ha="center", fontsize=10, fontweight="bold", color=C["title"])
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"n={s}", ha="center", va="center", fontsize=7, color="white", alpha=0.8)

    avg_l = np.mean(bpms)
    ax.axhline(avg_l, color=C["hr_avg"], linewidth=1.5, ls="--", alpha=0.8)
    ax.text(len(labels)-0.5, avg_l+1.5, f"Media: {avg_l:.1f}", fontsize=8, color=C["hr_avg"], ha="right")
    ax.set_ylabel("BPM Promedio")
    ax.grid(True, axis="y", linewidth=0.3)
    ax.set_ylim(0, max(bpms) + 20)
    fig.tight_layout()
    save(fig, "02_pulso_comparativo.png")


def graf_pasos_semanales(data):
    print("  [3/7] Pasos semanales...")
    sd = data.get("general_resume", {}).get("steps_per_day", [])
    if not sd:
        print("    Sin datos de pasos.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Pasos por Dia - Vista Semanal", color=C["title"], fontsize=13, fontweight="bold")

    labels = [day_label(d["date"]) for d in sd]
    steps = [d.get("steps", 0) for d in sd]
    avg = np.mean(steps)
    clrs = [C["steps"] if s >= avg else "#6e7681" for s in steps]

    bars = ax.bar(labels, steps, color=clrs, edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
    for bar, s in zip(bars, steps):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+150,
                f"{s:,}", ha="center", fontsize=9, fontweight="bold", color=C["title"])

    ax.axhline(avg, color=C["hr_avg"], linewidth=1.5, ls="--", alpha=0.7)
    ax.text(len(labels)-0.5, avg+200, f"Promedio: {avg:,.0f}", fontsize=8, color=C["hr_avg"], ha="right")
    ax.axhline(8000, color=C["z_rest"], linewidth=1, ls=":", alpha=0.5)
    ax.text(len(labels)-0.5, 8200, "Objetivo: 8000", fontsize=7, color=C["z_rest"], ha="right", alpha=0.7)
    ax.set_ylabel("Pasos")
    ax.grid(True, axis="y", linewidth=0.3)
    ax.set_ylim(0, max(steps)*1.25)
    fig.tight_layout()
    save(fig, "03_pasos_semanales.png")


def graf_calorias_semanales(data):
    print("  [4/7] Calorias semanales...")
    days_with_cal = [ds for ds in data.get("daily_summary", [])
                     if "activity" in ds and "calories" in ds.get("activity", {})]
    if not days_with_cal:
        print("    Sin datos de calorias.")
        return

    all_dates = [d["date"] for d in data["general_resume"]["steps_per_day"]]
    cal_map = {ds["date"]: ds["activity"]["calories"] for ds in days_with_cal}
    labels = [day_label(d) for d in all_dates]
    cals = [cal_map.get(d, 0) for d in all_dates]
    clrs = [C["cal"] if c > 0 else "#6e7681" for c in cals]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Calorias Quemadas por Dia", color=C["title"], fontsize=13, fontweight="bold")

    bars = ax.bar(labels, cals, color=clrs, edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
    for bar, c in zip(bars, cals):
        if c > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    f"{c:.0f}", ha="center", fontsize=10, fontweight="bold", color=C["title"])

    avg_c = np.mean([c for c in cals if c > 0]) if any(c > 0 for c in cals) else 0
    if avg_c > 0:
        ax.axhline(avg_c, color=C["hr_avg"], linewidth=1.5, ls="--", alpha=0.7)
        ax.text(len(labels)-0.5, avg_c+5, f"Media: {avg_c:.0f} kcal", fontsize=8, color=C["hr_avg"], ha="right")

    ax.set_ylabel("Calorias (kcal)")
    ax.grid(True, axis="y", linewidth=0.3)
    ax.set_ylim(0, max(cals)*1.25 if max(cals) > 0 else 100)
    fig.tight_layout()
    save(fig, "04_calorias_semanales.png")


def graf_entrenamientos(data):
    print("  [5/7] Entrenamientos...")
    ws = [ds for ds in data.get("daily_summary", []) if "workouts" in ds]
    if not ws:
        print("    Sin datos de entrenamiento.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Resumen de Entrenamientos", color=C["title"], fontsize=13, fontweight="bold")

    dates = [day_label(s["date"]) for s in ws]
    mins = [s.get("workouts", {}).get("total_workout_minutes", 0) for s in ws]
    cnts = [s.get("workouts", {}).get("workout_count", 0) for s in ws]

    # Minutos
    bars1 = axes[0].bar(dates, mins, color=C["cal"], edgecolor="white", linewidth=0.3, alpha=0.85)
    for bar, m in zip(bars1, mins):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"{m:.1f}m", ha="center", fontsize=9, fontweight="bold", color=C["title"])
    axes[0].set_title("Minutos Entrenados", loc="left", color=C["title"])
    axes[0].set_ylabel("Minutos")
    axes[0].grid(True, axis="y", linewidth=0.3)

    # Sesiones
    bars2 = axes[1].bar(dates, cnts, color=C["accent"], edgecolor="white", linewidth=0.3, alpha=0.85)
    for bar, c in zip(bars2, cnts):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                     str(c), ha="center", fontsize=10, fontweight="bold", color=C["title"])
    axes[1].set_title("Sesiones", loc="left", color=C["title"])
    axes[1].set_ylabel("Sesiones")
    axes[1].set_ylim(0, max(cnts)+1)
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].grid(True, axis="y", linewidth=0.3)

    # Calorias por workout (si disponible)
    cal_days = [ds for ds in data.get("daily_summary", [])
                if "activity" in ds and "calories" in ds.get("activity", {})]
    if cal_days:
        cd_dates = [day_label(ds["date"]) for ds in cal_days]
        cd_cals = [ds["activity"]["calories"] for ds in cal_days]
        bars3 = axes[2].bar(cd_dates, cd_cals, color="#f0883e", edgecolor="white", linewidth=0.3, alpha=0.85)
        for bar, c in zip(bars3, cd_cals):
            axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                         f"{c:.0f}", ha="center", fontsize=9, fontweight="bold", color=C["title"])
        axes[2].set_title("Calorias Totales", loc="left", color=C["title"])
        axes[2].set_ylabel("kcal")
        axes[2].grid(True, axis="y", linewidth=0.3)
    else:
        axes[2].text(0.5, 0.5, "Sin calorias", transform=axes[2].transAxes, ha="center", color=C["dim"])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "05_entrenamientos.png")


def graf_resumen_semanal(data):
    print("  [6/7] Resumen semanal...")
    sd = data.get("general_resume", {}).get("steps_per_day", [])
    da = data.get("heart_rate_resume", {}).get("daily_average_bpm", [])
    if not sd:
        print("    Sin datos de pasos.")
        return
    cal_map = {}
    hrv_map = {}
    resting_map = {}
    for ds in data.get("daily_summary", []):
        act = ds.get("activity", {})
        if "calories" in act:
            cal_map[ds["date"]] = act["calories"]
        hr = ds.get("heart_rate", {})
        if "resting_bpm" in hr:
            resting_map[ds["date"]] = hr["resting_bpm"]
        hrv = ds.get("heart_rate_variability", {})
        if "avg_ms" in hrv:
            hrv_map[ds["date"]] = hrv["avg_ms"]

    has_hrv = bool(hrv_map)
    has_resting = bool(resting_map)

    n_rows = 3 + (1 if has_resting else 0) + (1 if has_hrv else 0)
    row_idx = 0
    fig_h = 9 + (2.5 if has_resting else 0) + (2.5 if has_hrv else 0)

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, fig_h), sharex=True)
    fig.suptitle("Resumen Semanal - Multi-Metrica", color=C["title"], fontsize=14, fontweight="bold")

    x = np.arange(len(labels := [day_label(d["date"]) for d in sd]))
    steps = [d.get("steps", 0) for d in sd]
    bpms = [d.get("bpm", 0) for d in da] if da else [0] * len(sd)
    cals = [cal_map.get(d["date"], 0) for d in sd]

    # Pasos
    clrs_s = [C["steps"] if s >= np.mean(steps) else "#6e7681" for s in steps]
    axes[0].bar(x, steps, color=clrs_s, edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
    for j, s in enumerate(steps):
        axes[0].text(j, s + 150, f"{s:,}", ha="center", fontsize=8, fontweight="bold", color=C["title"])
    axes[0].axhline(np.mean(steps), color=C["hr_avg"], linewidth=1, ls="--", alpha=0.6)
    axes[0].set_ylabel("Pasos")
    axes[0].set_title("Pasos", loc="left", color=C["steps"], fontsize=10)
    axes[0].grid(True, axis="y", linewidth=0.3)
    axes[0].set_ylim(0, max(steps)*1.3)
    row_idx += 1

    # FC promedio
    if bpms:
        clrs_h = [C["z_rest"] if b < 80 else C["z_fat"] if b < 100 else C["z_cardio"] for b in bpms]
        axes[row_idx].bar(x[:len(bpms)], bpms, color=clrs_h, edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
        for j, b in enumerate(bpms):
            axes[row_idx].text(j, b + 1.5, f"{b:.1f}", ha="center", fontsize=9, fontweight="bold", color=C["title"])
        axes[row_idx].axhline(np.mean(bpms), color=C["hr_avg"], linewidth=1, ls="--", alpha=0.6)
        axes[row_idx].set_ylabel("BPM")
        axes[row_idx].set_title("Frecuencia Cardiaca Promedio", loc="left", color=C["hr"], fontsize=10)
        axes[row_idx].grid(True, axis="y", linewidth=0.3)
        axes[row_idx].set_ylim(0, max(bpms)+20)
    else:
        axes[row_idx].set_visible(False)
    row_idx += 1

    # FC en reposo (si disponible)
    if has_resting:
        resting_vals = [resting_map.get(d["date"], 0) for d in sd]
        has_data_r = [v > 0 for v in resting_vals]
        if any(has_data_r):
            clrs_r = [C["spo2"] if v > 0 else "#6e7681" for v in resting_vals]
            bars_r = axes[row_idx].bar(x, resting_vals, color=clrs_r, edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
            for j, v in enumerate(resting_vals):
                if v > 0:
                    axes[row_idx].text(j, v + 0.5, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold", color=C["title"])
            valid_r = [v for v in resting_vals if v > 0]
            if valid_r:
                axes[row_idx].axhline(np.mean(valid_r), color=C["hr_avg"], linewidth=1, ls="--", alpha=0.6)
            axes[row_idx].set_ylabel("BPM")
            axes[row_idx].set_title("FC en Reposo", loc="left", color=C["spo2"], fontsize=10)
            axes[row_idx].grid(True, axis="y", linewidth=0.3)
            axes[row_idx].set_ylim(0, max(valid_r)+15 if valid_r else 100)
        else:
            axes[row_idx].set_visible(False)
        row_idx += 1

    # HRV (si disponible)
    if has_hrv:
        hrv_vals = [hrv_map.get(d["date"], 0) for d in sd]
        has_data_h = [v > 0 for v in hrv_vals]
        if any(has_data_h):
            clrs_hrv = [C["workout"] if v > 0 else "#6e7681" for v in hrv_vals]
            bars_hrv = axes[row_idx].bar(x, hrv_vals, color=clrs_hrv, edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
            for j, v in enumerate(hrv_vals):
                if v > 0:
                    axes[row_idx].text(j, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold", color=C["title"])
            valid_h = [v for v in hrv_vals if v > 0]
            if valid_h:
                axes[row_idx].axhline(np.mean(valid_h), color=C["hr_avg"], linewidth=1, ls="--", alpha=0.6)
            axes[row_idx].set_ylabel("ms")
            axes[row_idx].set_title("HRV (Variabilidad FC)", loc="left", color=C["workout"], fontsize=10)
            axes[row_idx].grid(True, axis="y", linewidth=0.3)
            axes[row_idx].set_ylim(0, max(valid_h)*1.3 if valid_h else 100)
        else:
            axes[row_idx].set_visible(False)
        row_idx += 1

    # Calorias
    clrs_c = [C["cal"] if c > 0 else "#6e7681" for c in cals]
    axes[row_idx].bar(x, cals, color=clrs_c, edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
    for j, c in enumerate(cals):
        if c > 0:
            axes[row_idx].text(j, c + 3, f"{c:.0f}", ha="center", fontsize=9, fontweight="bold", color=C["title"])
    if any(c > 0 for c in cals):
        valid = [c for c in cals if c > 0]
        axes[row_idx].axhline(np.mean(valid), color=C["hr_avg"], linewidth=1, ls="--", alpha=0.6)
    axes[row_idx].set_ylabel("kcal")
    axes[row_idx].set_title("Calorias Quemadas", loc="left", color=C["cal"], fontsize=10)
    axes[row_idx].grid(True, axis="y", linewidth=0.3)
    axes[row_idx].set_ylim(0, max(cals)*1.3 if max(cals) > 0 else 100)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "06_resumen_semanal.png")


# ─── GRÁFICA DIARIA ─────────────────────────────────────────────────

def graf_diaria(data, idx):
    dp = data["heart_rate_resume"]["daily_points"][idx]
    date = dp["date"]
    pts = dp["points"]
    lbl = day_label(date)
    summ = get_summary(data, date)

    print(f"  [7/7] Detalle diario: {lbl}...")
    t = [ts(p["time"]) for p in pts]
    b = [p["bpm"] for p in pts]
    avg = np.mean(b)

    has_cal = summ and "activity" in summ and "hourly_calories" in summ.get("activity", {})
    has_workout = summ and "workouts" in summ
    has_spo2 = any(s["date"] == date for s in data.get("blood_oxygen_resume", {}).get("daily_summary", []))
    has_sleep = summ and "sleep" in summ
    hr_data = summ.get("heart_rate", {}) if summ else {}
    has_resting = "resting_bpm" in hr_data
    hrv_data = summ.get("heart_rate_variability", {}) if summ else {}
    has_hrv = "avg_ms" in hrv_data

    n_rows = 4 if has_spo2 or has_sleep else 3
    fig = plt.figure(figsize=(14, 12 if n_rows > 3 else 9.5))
    fig.suptitle(f"Detalle Diario: {lbl} ({date})",
                 color=C["title"], fontsize=13, fontweight="bold", y=0.985)

    gs = gridspec.GridSpec(n_rows, 2, hspace=0.45, wspace=0.30,
                           left=0.07, right=0.97, top=0.93, bottom=0.05)

    # ── 1. FC (ocupa 2 columnas) ──
    ax_hr = fig.add_subplot(gs[0, :])
    ax_hr.fill_between(t, b, alpha=0.12, color=C["hr"], zorder=1)
    ax_hr.plot(t, b, color=C["hr"], linewidth=0.5, alpha=0.4, zorder=2)
    ax_hr.axhline(avg, color=C["hr_avg"], linewidth=1, ls="--", alpha=0.8)
    ax_hr.text(t[-1], avg+2, f"Avg: {avg:.1f}", fontsize=7, color=C["hr_avg"], ha="right", va="bottom")
    if has_resting:
        ax_hr.axhline(hr_data["resting_bpm"], color=C["spo2"], linewidth=1, ls=":", alpha=0.7)
        ax_hr.text(t[0], hr_data["resting_bpm"]+1.5, f"Reposo: {hr_data['resting_bpm']:.0f}",
                   fontsize=7, color=C["spo2"], va="bottom")
    ax_hr.set_title("Frecuencia Cardiaca", loc="left", color=C["title"])
    ax_hr.set_ylabel("BPM")
    hr_formatter(ax_hr, t)
    ax_hr.grid(True, alpha=0.3)
    ax_hr.set_xlim(t[0], t[-1])
    apply_hr_bg(ax_hr)

    # ── 2. Estadisticas FC + info extra ──
    ax_st = fig.add_subplot(gs[1, 0])
    mn_b, mx_b = min(b), max(b)
    p25, p50, p75 = np.percentile(b, [25, 50, 75])
    metrics = ["Min", "P25", "Mediana", "P75", "Max"]
    vals = [mn_b, p25, p50, p75, mx_b]
    clrs = [C["z_rest"], C["z_fat"], C["hr_avg"], C["z_fat"], C["z_cardio"]]

    if has_resting:
        metrics.insert(0, "Reposo")
        vals.insert(0, hr_data["resting_bpm"])
        clrs.insert(0, C["spo2"])

    y = np.arange(len(metrics))
    ax_st.barh(y, vals, color=clrs, edgecolor="white", linewidth=0.3, alpha=0.8, height=0.5)
    for j, (m, v) in enumerate(zip(metrics, vals)):
        ax_st.text(v+1, j, f"{v:.1f}", va="center", fontsize=9, fontweight="bold", color=C["title"])
    ax_st.set_yticks(y)
    ax_st.set_yticklabels(metrics, fontsize=8)
    n_pts = len(b)
    title_extra = ""
    if has_hrv:
        title_extra = f"  HRV:{hrv_data['avg_ms']:.1f}ms"
    ax_st.set_title(f"Estadisticas FC (n={n_pts}){title_extra}", loc="left", color=C["title"])
    ax_st.set_xlabel("BPM")
    ax_st.grid(True, axis="x", alpha=0.3)
    ax_st.set_xlim(0, mx_b+15)

    # ── 3. Pasos + Calorias (apilados si hay cal) ──
    ax_p = fig.add_subplot(gs[1, 1])
    steps_val = summ.get("activity", {}).get("steps", 0) if summ else 0
    cal_val = summ.get("activity", {}).get("calories", 0) if summ else 0
    avg_steps = np.mean([d["steps"] for d in data["general_resume"]["steps_per_day"]])

    x_pos = [0]
    x_labels = [lbl]
    bar_w = 0.4

    ax_p.bar(x_pos, [steps_val], color=C["steps"] if steps_val >= avg_steps else "#6e7681",
             edgecolor="white", linewidth=0.3, alpha=0.85, width=bar_w)
    ax_p.text(0, steps_val + 150, f"{steps_val:,}", fontsize=11, fontweight="bold",
              color=C["title"], ha="center")

    ax_p.axhline(avg_steps, color=C["accent"], linewidth=1, ls="--", alpha=0.5)
    ax_p.text(0.3, avg_steps + 100, f"Promedio: {avg_steps:,.0f}", fontsize=7, color=C["accent"], alpha=0.7)

    if cal_val > 0:
        ax_p2 = ax_p.twinx()
        ax_p2.bar([0.45], [cal_val], color=C["cal"], edgecolor="white", linewidth=0.3, alpha=0.85, width=bar_w)
        ax_p2.text(0.45, cal_val + 5, f"{cal_val:.0f} kcal", fontsize=9, fontweight="bold",
                   color=C["cal"], ha="center")
        ax_p2.set_ylabel("Calorias", color=C["cal"], fontsize=8)
        ax_p2.tick_params(axis="y", labelcolor=C["cal"], labelsize=7)
        ax_p2.set_ylim(0, cal_val * 1.4)
        ax_p2.set_xlim(-0.5, 1)

    ax_p.set_title("Pasos" + (" + Calorias" if cal_val > 0 else ""), loc="left", color=C["title"])
    ax_p.set_ylabel("Pasos")
    ax_p.set_ylim(0, max(steps_val, avg_steps) * 1.3)
    ax_p.grid(True, axis="y", alpha=0.3)
    ax_p.set_xticks([])
    ax_p.set_xlim(-0.5, 1)

    # ── 4. SpO2 / HRV ──
    ax_sp = fig.add_subplot(gs[2, 0])
    spo2_data = data.get("blood_oxygen_resume", {})
    sp = next((s for s in spo2_data.get("daily_summary", []) if s.get("date") == date), None)
    if sp:
        ax_sp.bar([lbl], [sp["avg"]], color=C["spo2"], edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
        ax_sp.axhline(95, color=C["hr_avg"], linewidth=1, ls="--", alpha=0.5)
        ax_sp.text(0, sp["avg"]+0.5, f"{sp['avg']:.0f}%", fontsize=12, fontweight="bold",
                   color=C["title"], ha="center")
        ax_sp.set_ylim(min(90, sp["avg"]-5), 100)
        detail = f"Min:{sp['min']:.0f} Max:{sp['max']:.0f} n={sp['measurements']}"
        ax_sp.text(0.5, 0.05, detail, transform=ax_sp.transAxes, fontsize=7, color=C["dim"], ha="center")
        ax_sp.set_title("SpO2 Sanguineo", loc="left", color=C["title"])
    elif has_hrv:
        ax_sp.bar([lbl], [hrv_data["avg_ms"]], color=C["workout"], edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
        ax_sp.text(0, hrv_data["avg_ms"]+0.5, f"{hrv_data['avg_ms']:.1f}ms", fontsize=12, fontweight="bold",
                   color=C["title"], ha="center")
        ax_sp.set_ylim(0, hrv_data["avg_ms"] * 1.4)
        detail = f"n={hrv_data.get('samples', '?')} muestras"
        ax_sp.text(0.5, 0.05, detail, transform=ax_sp.transAxes, fontsize=7, color=C["dim"], ha="center")
        ax_sp.set_title("HRV (Variabilidad FC)", loc="left", color=C["workout"])
    else:
        ax_sp.text(0.5, 0.5, "Sin datos SpO2/HRV", transform=ax_sp.transAxes, ha="center", fontsize=11, color=C["dim"])
        ax_sp.set_ylim(90, 100)
    ax_sp.set_ylabel("%" if sp else "ms")
    ax_sp.grid(True, axis="y", alpha=0.3)
    ax_sp.set_xticks([])

    # ── 5. Sueno ──
    ax_sl = fig.add_subplot(gs[2, 1])
    if has_sleep:
        sl = summ["sleep"]
        total = sl.get("total_sleep_minutes", 0)
        deep = sl.get("deep_minutes", 0)
        light = sl.get("light_minutes", 0)
        rem = sl.get("rem_minutes", 0)
        awake = sl.get("awake_minutes", 0)
        eff = sl.get("sleep_efficiency", 0)
        sh = sl.get("sleep_start", "?")
        sm_ = sl.get("sleep_end", "?")

        parts, pcol, plbl = [], [], []
        if deep > 0:
            parts.append(deep); pcol.append(C["sleep_deep"]); plbl.append(f"Profundo ({deep}m)")
        if light > 0:
            parts.append(light); pcol.append(C["sleep_light"]); plbl.append(f"Ligero ({light}m)")
        if rem > 0:
            parts.append(rem); pcol.append(C["sleep_rem"]); plbl.append(f"REM ({rem}m)")
        if awake > 0:
            parts.append(awake); pcol.append(C["sleep_awake"]); plbl.append(f"Despierto ({awake}m)")

        if sum(parts) > 0:
            ax_sl.pie(parts, labels=plbl, colors=pcol, startangle=90,
                      textprops={"fontsize": 7, "color": C["text"]},
                      wedgeprops={"edgecolor": C["panel"], "linewidth": 1})
            h, m = total // 60, total % 60
            ax_sl.set_title(f"Sueno ({h}h {m}m, Eff:{eff}%)", loc="left", color=C["title"])
        else:
            ax_sl.text(0.5, 0.6, f"Horario: {sh} - {sm_}", transform=ax_sl.transAxes,
                       fontsize=10, color=C["text"], ha="center")
            ax_sl.text(0.5, 0.4, f"Total: {total} min (Eff:{eff}%)", transform=ax_sl.transAxes,
                       fontsize=9, color=C["dim"], ha="center")
            ax_sl.text(0.5, 0.2, "Desglose no disponible", transform=ax_sl.transAxes,
                       fontsize=8, color=C["dim"], ha="center", style="italic")
            ax_sl.set_title("Sueno", loc="left", color=C["title"])
    else:
        ax_sl.text(0.5, 0.5, "Sin datos de sueno", transform=ax_sl.transAxes,
                   ha="center", fontsize=11, color=C["dim"])
        ax_sl.set_title("Sueno", loc="left", color=C["title"])

    # ── Info workouts en pie de grafica ──
    footer_parts = []
    if has_workout:
        wk = summ["workouts"]
        footer_parts.append(f"Entrenamiento: {wk.get('workout_count', 0)} sesiones, {wk.get('total_workout_minutes', 0):.1f} min")
    if has_resting:
        footer_parts.append(f"FC Reposo: {hr_data['resting_bpm']:.0f} bpm")
    if has_hrv:
        footer_parts.append(f"HRV: {hrv_data['avg_ms']:.1f} ms (n={hrv_data.get('samples', '?')})")
    if footer_parts:
        fig.text(0.5, 0.005, "  |  ".join(footer_parts),
                 ha="center", fontsize=8, color=C["workout"], alpha=0.8)

    tag = date.replace("-", "")
    dp_ = lbl.split()[1].replace("/", "")
    save(fig, f"07_diario_{tag}_{dp_}.png")


# ─── MAIN ───────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  GENERADOR DE GRAFICAS - DATOS DE SALUD")
    print("=" * 55)
    print(f"  Archivo: {os.path.basename(DATA_FILE)}")

    data = load()
    style()

    period = f"{data.get('window_start', '?')[:10]} a {data.get('window_end', '?')[:10]}"
    n_days = len(data.get("heart_rate_resume", {}).get("daily_points", []))
    print(f"\nPeriodo: {period}  |  Dias: {n_days}")

    # Las 6 globales. Ojo: graf_entrenamientos no escribe nada si no hay
    # workouts en daily_summary (imprime "Sin datos de entrenamiento"), así
    # que `total = 6 + n_days` de abajo es el máximo, no siempre el real.
    graf_pulso_completo(data)
    graf_pulso_comparativo(data)
    graf_pasos_semanales(data)
    graf_calorias_semanales(data)
    graf_entrenamientos(data)
    graf_resumen_semanal(data)

    for i in range(n_days):
        graf_diaria(data, i)

    total = 6 + n_days
    print(f"\n{'=' * 55}")
    print(f"  {total} graficas generadas en: {OUTPUT_DIR}")
    print(f"  (6 globales + {n_days} diarias)")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
