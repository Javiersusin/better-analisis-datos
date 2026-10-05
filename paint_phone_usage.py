#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paint_phone_usage.py - Gráficas de uso del móvil -> carpeta graficas_uso_movil/.

Qué lee:
    Un `*_phone_usage_processed.json` (raíz `daily_blocks`). Si el JSON no
    tiene `daily_blocks`, se trata el propio objeto como un único bloque.

Qué escribe (4 PNG por día, en graficas_uso_movil/):
    01_<n>_resumen_general       - pantalla, desbloqueos, cambios de app, reaperturas
    02_<n>_distribucion_categorias - tarta de tiempo + barras de aperturas
    03_<n>_timeline_horario      - minutos por hora con media y máximo
    04_<n>_patrones_compulsivos  - sesiones por duración + métricas de refuerzo
    (<n> es el índice del día, empezando en 1)

Qué NO hace:
    No modifica ningún JSON. Backend Agg (sin pantalla).

Funciones definidas pero NO llamadas desde main():
    - merge_daily_blocks() y blank_metrics(): para agregar varios días en un
      solo bloque resumen; de momento no se usa (y si se usa, sus claves
      `short_sessions` son "<30s"/"<60s" mientras que en el JSON por día son
      "<5s"/"5-10s": habría que unificarlas).
    - graf_duracion_sesiones(): histograma de duraciones. OJO: escribe con el
      mismo prefijo "04_" que graf_patrones_compulsivos, así que habría que
      renumerar (05_) antes de llamarla.

Uso:
    python paint_phone_usage.py [ruta/al/phone_usage_processed.json]
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
DATA_FILE = os.path.join(SCRIPT_DIR, "usomvl", "260623_1438_phone_usage_processed.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "graficas_uso_movil")

if len(sys.argv) > 1:
    DATA_FILE = sys.argv[1]

C = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "grid": "#21262d",
    "text": "#c9d1d9",
    "dim": "#8b949e",
    "title": "#f0f6fc",
    "social": "#f97583",
    "messaging": "#79c0ff",
    "productivity": "#56d364",
    "video_streaming": "#d2a8ff",
    "music": "#ffa657",
    "system": "#8b949e",
    "other": "#6e7681",
    "navigation": "#7ee787",
    "shopping": "#f0883e",
    "education": "#58a6ff",
    "games": "#da3633",
    "accent": "#58a6ff",
    "short": "#f85149",
    "medium": "#d29922",
    "long": "#238636",
    "unlock": "#7ee787",
    "rapid": "#f85149",
    "avg_line": "#ffa657",
}

CATEGORY_COLORS = {
    "social": C["social"],
    "messaging": C["messaging"],
    "productivity": C["productivity"],
    "video_streaming": C["video_streaming"],
    "music": C["music"],
    "system": C["system"],
    "other": C["other"],
    "navigation": C["navigation"],
    "shopping": C["shopping"],
    "education": C["education"],
    "games": C["games"],
}

WD = {0: "Lun", 1: "Mar", 2: "Mie", 3: "Jue", 4: "Vie", 5: "Sab", 6: "Dom"}


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
        "font.weight": "bold",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
    })


def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=120, facecolor=fig.get_facecolor(), pad_inches=0.2)
    plt.close(fig)
    print(f"    -> {name}")


def ts(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def day_label(d):
    if not d:
        return "?"
    dt = datetime.strptime(d, "%Y-%m-%d")
    return f"{WD[dt.weekday()]} {dt.day}/{dt.month}"


def fmt_min(m):
    if m >= 60:
        h = int(m // 60)
        mn = int(m % 60)
        return f"{h}h {mn}m"
    return f"{m:.1f}m"


def fmt_duration(seconds):
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    elif seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    return f"{seconds:.1f}s"


def get_cat_color(cat):
    return CATEGORY_COLORS.get(cat, C["other"])


def blank_metrics():
    """Estructura vacía de `metrics` con todas las claves presentes.

    Se usa dentro de merge_daily_blocks (hoy no llamado desde main) para que
    el bloque agregado tenga la misma forma que un bloque diario normal.
    """
    return {
        "screen_time_minutes": 0,
        "unlock_count": 0,
        "app_switches": 0,
        "short_sessions": {},
        "categories": {},
        "usage_by_hour": [],
        "events_by_category": {},
        "average_session_seconds": 0,
    }


def merge_daily_blocks(blocks):
    """Agrega varios daily_blocks en un único bloque resumen.

    Suma pantalla, desbloqueos, cambios de app, reaperturas y uso por hora y
    por categoría a lo largo de todos los días, y concatena las sesiones.

    Estado: NO se llama desde main() (ver cabecera del módulo). Si se activa,
    unificar las claves de short_sessions: aquí se escriben "<30s"/"<60s"
    mientras que los datos de origen vienen como "<5s"/"5-10s".
    """
    if not blocks:
        return None

    merged = {
        "daily_blocks": blocks,
        "date": blocks[0].get("date", "?"),
        "date_end": blocks[-1].get("date", blocks[0].get("date", "?")),
        "date_range": {
            "start": blocks[0].get("date", "?"),
            "end": blocks[-1].get("date", blocks[0].get("date", "?")),
            "days": len(blocks),
        },
        "window_end": blocks[-1].get("window_end", "?"),
        "metrics": blank_metrics(),
        "compulsive_metrics": {
            "rapid_reopenings": 0,
            "average_session_seconds": 0,
            "unlock_to_social_average_seconds": None,
        "short_session_pressure": {
            "<5s": 0,
            "5-10s": 0,
        },
            "app_switches": 0,
        },
        "sessions": [],
    }

    category_totals = {}
    category_opens = {}
    category_events = {}
    category_duration_sum = {}
    hourly_usage = {}
    total_session_seconds = 0.0
    unlock_count = 0
    app_switches = 0
    average_session_weight = 0.0
    session_count = 0
    rapid_reopenings = 0
    social_unlock_values = []
    short_under_30 = 0
    short_under_60 = 0

    for block in blocks:
        metrics = block.get("metrics", {})
        compulsive = block.get("compulsive_metrics", {})
        sessions = block.get("sessions", [])

        total_session_seconds += metrics.get("screen_time_minutes", 0) * 60.0
        unlock_count += metrics.get("unlock_count", 0)
        app_switches += metrics.get("app_switches", 0)
        rapid_reopenings += compulsive.get("rapid_reopenings", 0)
        short_under_30 += compulsive.get("short_session_pressure", {}).get("<5s", 0)
        short_under_60 += compulsive.get("short_session_pressure", {}).get("5-10s", 0)

        for hour_entry in metrics.get("usage_by_hour", []):
            hour = hour_entry.get("hour")
            minutes = hour_entry.get("minutes", 0)
            hourly_usage[hour] = hourly_usage.get(hour, 0) + minutes

        for category, values in metrics.get("categories", {}).items():
            category_totals[category] = category_totals.get(category, 0) + values.get("minutes", 0)
            category_opens[category] = category_opens.get(category, 0) + values.get("opens", 0)
            category_events[category] = category_events.get(category, 0) + values.get("events", 0)
            category_duration_sum[category] = category_duration_sum.get(category, 0) + values.get("average_session_seconds", 0) * values.get("opens", 0)

        for session in sessions:
            session_count += 1
            average_session_weight += session.get("duration_seconds", 0)
            if session.get("category") == "social" and session.get("unlock_to_open_seconds") is not None:
                social_unlock_values.append(session.get("unlock_to_open_seconds"))

    merged["metrics"] = {
        "screen_time_minutes": round(total_session_seconds / 60.0, 2),
        "unlock_count": unlock_count,
        "app_switches": app_switches,
        "short_sessions": {
            "<30s": short_under_30,
            "<60s": short_under_60,
        },
        "categories": {
            category: {
                "minutes": round(category_totals[category], 2),
                "opens": category_opens.get(category, 0),
                "average_session_seconds": round(category_duration_sum.get(category, 0) / category_opens.get(category, 1), 1),
                "events": category_events.get(category, 0),
                "time_percentage": round((category_totals[category] / total_session_seconds) * 100, 2) if total_session_seconds else 0,
            }
            for category in sorted(category_totals)
        },
        "usage_by_hour": [
            {"hour": hour, "minutes": round(minutes, 2)}
            for hour, minutes in sorted(hourly_usage.items())
        ],
        "events_by_category": {category: category_events[category] for category in sorted(category_events)},
        "average_session_seconds": round(average_session_weight / session_count, 1) if session_count else 0,
    }

    merged["compulsive_metrics"] = {
        "rapid_reopenings": rapid_reopenings,
        "average_session_seconds": round(average_session_weight / session_count, 1) if session_count else 0,
        "unlock_to_social_average_seconds": round(sum(social_unlock_values) / len(social_unlock_values), 1) if social_unlock_values else None,
        "short_session_pressure": {
            "<5s": short_under_30,
            "5-10s": short_under_60,
        },
        "app_switches": app_switches,
    }

    merged["sessions"] = [session for block in blocks for session in block.get("sessions", [])]
    return merged


# ─── GRÁFICAS GLOBALES ──────────────────────────────────────────────

def graf_resumen_general(data, day_idx, total_days):
    prefix = f"01_{day_idx}_"
    date = data.get("date", "?")
    print(f"  [{day_idx}/{total_days}] Resumen general: {date}...")
    metrics = data.get("metrics", {})
    cm = data.get("compulsive_metrics", {})

    screen = metrics.get("screen_time_minutes", 0)
    unlocks = metrics.get("unlock_count", 0)
    switches = metrics.get("app_switches", 0)
    avg_session = metrics.get("average_session_seconds", 0)
    rapid = cm.get("rapid_reopenings", 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Resumen General - {day_label(date)} ({date})", color=C["title"], fontsize=13, fontweight="bold")

    labels = ["Tiempo\nPantalla", "Desbloqueos", "Cambio\nApps", "Apertura\nRapida"]
    values = [screen, unlocks, switches, rapid]
    clrs = [C["accent"], C["unlock"], C["medium"], C["rapid"]]

    bars = ax.bar(labels, values, color=clrs, edgecolor="white", linewidth=0.3, alpha=0.85, width=0.5)
    for bar, v, lbl in zip(bars, values, labels):
        txt = fmt_min(v) if "Tiempo" in lbl else f"{v:,}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                txt, ha="center", fontsize=11, fontweight="bold", color=C["title"])

    ax.set_title("Resumen General", loc="left", color=C["title"], fontsize=10, fontweight="bold")
    ax.set_ylabel("Valor", fontweight="bold")
    ax.grid(True, axis="y", linewidth=0.3)
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 10)
    fig.tight_layout()
    save(fig, f"{prefix}resumen_general.png")


def graf_distribucion_categorias(data, day_idx, total_days):
    prefix = f"02_{day_idx}_"
    date = data.get("date", "?")
    print(f"  [{day_idx}/{total_days}] Distribucion por categorias: {date}...")
    cats = data.get("metrics", {}).get("categories", {})
    if not cats:
        print("    Sin datos de categorias.")
        return

    sorted_cats = sorted(cats.items(), key=lambda x: x[1]["minutes"], reverse=True)
    labels = [c[0] for c in sorted_cats]
    minutes = [c[1]["minutes"] for c in sorted_cats]
    clrs = [get_cat_color(c) for c in labels]
    total_min = sum(minutes)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Distribucion de Uso por Categoria - {day_label(date)} ({date})", color=C["title"], fontsize=13, fontweight="bold")

    if sum(minutes) > 0:
        wedges, _ = axes[0].pie(
            minutes, labels=None, colors=clrs, startangle=90,
            wedgeprops={"edgecolor": C["panel"], "linewidth": 1.5},
        )
        legend_labels = [f"{l}  {fmt_min(m)}  ({m/total_min*100:.1f}%)" for l, m in zip(labels, minutes)]
        axes[0].legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(-0.15, 0.5),
                       fontsize=9, frameon=False, labelspacing=1.2, prop={"weight": "bold"})
    else:
        axes[0].text(0.5, 0.5, "Sin datos", transform=axes[0].transAxes,
                     ha="center", va="center", fontsize=14, color=C["dim"], fontweight="bold")
    axes[0].set_title("Tiempo Total por Categoria", loc="left", color=C["title"], fontsize=10, fontweight="bold")

    opens = [cats[l]["opens"] for l in labels]
    bars = axes[1].barh(labels[::-1], opens[::-1], color=clrs[::-1], edgecolor="white", linewidth=0.3, alpha=0.85)
    for bar, o in zip(bars, opens[::-1]):
        axes[1].text(bar.get_width() + max(opens)*0.02, bar.get_y() + bar.get_height()/2,
                     str(o), va="center", fontsize=9, fontweight="bold", color=C["title"])
    axes[1].set_title("Numero de Aperturas", loc="left", color=C["title"], fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Aperturas", fontweight="bold")
    axes[1].grid(True, axis="x", linewidth=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, f"{prefix}distribucion_categorias.png")


def graf_timeline_horario(data, day_idx, total_days):
    prefix = f"03_{day_idx}_"
    date = data.get("date", "?")
    print(f"  [{day_idx}/{total_days}] Timeline horario: {date}...")
    usage = data.get("metrics", {}).get("usage_by_hour", [])

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Uso del Movil por Hora - {day_label(date)} ({date})", color=C["title"], fontsize=13, fontweight="bold")

    if not usage:
        ax.text(0.5, 0.5, "Sin datos horarios", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=C["dim"], fontweight="bold")
        ax.set_title("Uso por Hora", loc="left", color=C["title"], fontsize=10, fontweight="bold")
        fig.tight_layout()
        save(fig, f"{prefix}timeline_horario.png")
        return

    usage_map = {u["hour"]: u["minutes"] for u in usage}
    times_raw = [ts(u["hour"]) for u in usage]

    from datetime import timedelta
    min_t = min(times_raw).replace(minute=0, second=0, microsecond=0)
    max_t = max(times_raw).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    all_times = []
    all_mins = []
    cursor = min_t
    while cursor < max_t:
        hour_key = cursor.strftime("%Y-%m-%dT%H:00:00Z")
        all_times.append(cursor)
        all_mins.append(usage_map.get(hour_key, 0))
        cursor += timedelta(hours=1)

    # Segunda creación de la figura más abajo: la de arriba solo sirve para el
    # camino "sin datos" (se dibuja y se guarda ahí mismo). Aquí se recrea
    # para poder añadir el total de pantalla del día al título.
    fig, ax = plt.subplots(figsize=(14, 5))
    screen_min = data.get("metrics", {}).get("screen_time_minutes", 0)
    h = int(screen_min // 60)
    m = int(screen_min % 60)
    uso_str = f"{h}h {m:02d}min"
    fig.suptitle(f"Uso del Movil por Hora - {day_label(date)} ({date})  |  Uso del dia: {uso_str}", color=C["title"], fontsize=13, fontweight="bold")

    ax.fill_between(all_times, all_mins, alpha=0.3, color=C["accent"], zorder=1, step="mid")
    ax.plot(all_times, all_mins, color=C["accent"], linewidth=1.5, alpha=0.8, zorder=2, drawstyle="steps-mid")
    ax.bar(all_times, all_mins, width=0.03, color=C["accent"], alpha=0.5, zorder=3, align="center")

    avg = np.mean(all_mins)
    ax.axhline(avg, color=C["avg_line"], linewidth=1.5, ls="--", alpha=0.8)
    ax.text(all_times[-1], avg + max(all_mins)*0.03, f"Promedio: {fmt_min(avg)}",
            fontsize=8, color=C["avg_line"], ha="right", va="bottom", fontweight="bold")

    max_idx = np.argmax(all_mins)
    ax.annotate(f"Max: {fmt_min(all_mins[max_idx])}",
                xy=(all_times[max_idx], all_mins[max_idx]),
                xytext=(10, 15), textcoords="offset points",
                fontsize=8, color=C["title"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["avg_line"]))

    ax.set_title("Uso por Hora", loc="left", color=C["title"], fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Minutos de Uso", fontweight="bold")
    ax.grid(True, axis="both", linewidth=0.3)
    fig.tight_layout()
    save(fig, f"{prefix}timeline_horario.png")


def graf_duracion_sesiones(data, day_idx, total_days):
    """Histograma de duraciones de sesión + reparto por categoría.

    Estado: NO se llama desde main() (ver cabecera del módulo). Si se activa,
    renombrar su prefijo "04_" a "05_" para no pisar a
    graf_patrones_compulsivos, que usa el mismo.
    """
    prefix = f"04_{day_idx}_"
    date = data.get("date", "?")
    print(f"  [{day_idx}/{total_days}] Distribucion de duracion de sesiones: {date}...")
    sessions = data.get("sessions", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Analisis de Duracion de Sesiones - {day_label(date)} ({date})", color=C["title"], fontsize=13, fontweight="bold")

    if not sessions:
        for ax in axes:
            ax.text(0.5, 0.5, "Sin sesiones", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color=C["dim"], fontweight="bold")
        axes[0].set_title("Histograma de Duracion", loc="left", color=C["title"], fontsize=10, fontweight="bold")
        axes[1].set_title("Duracion por Categoria", loc="left", color=C["title"], fontsize=10, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save(fig, f"{prefix}duracion_sesiones.png")
        return

    durations = [s["duration_seconds"] for s in sessions]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Analisis de Duracion de Sesiones - {day_label(date)} ({date})", color=C["title"], fontsize=13, fontweight="bold")

    bins = [0, 1, 5, 10, 30, 60, 120, 300, 600, max(durations)+1]
    counts = []
    bin_labels = []
    for i in range(len(bins)-1):
        c = sum(1 for d in durations if bins[i] <= d < bins[i+1])
        counts.append(c)
        if bins[i] < 1:
            bin_labels.append("<1s")
        elif bins[i+1]-1 < 60:
            bin_labels.append(f"{int(bins[i])}-{int(bins[i+1])}s")
        else:
            bin_labels.append(f"{int(bins[i]//60)}-{int(bins[i+1]//60)}m")

    colors = [C["short"] if b < 30 else C["medium"] if b < 60 else C["long"] for b in bins[:-1]]
    bars = axes[0].bar(bin_labels, counts, color=colors, edgecolor="white", linewidth=0.3, alpha=0.85)
    for bar, c in zip(bars, counts):
        if c > 0:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(counts)*0.02,
                         str(c), ha="center", fontsize=8, fontweight="bold", color=C["title"])
    axes[0].set_title("Histograma de Duracion", loc="left", color=C["title"], fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Duracion", fontweight="bold")
    axes[0].set_ylabel("Numero de Sesiones", fontweight="bold")
    axes[0].grid(True, axis="y", linewidth=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    cat_durations = {}
    for s in sessions:
        cat = s["category"]
        if cat not in cat_durations:
            cat_durations[cat] = []
        cat_durations[cat].append(s["duration_seconds"])

    sorted_cats = sorted(cat_durations.keys(), key=lambda c: len(cat_durations[c]), reverse=True)
    bin_edges = [0, 1, 5, 10, 30, 60, 120, 300, 600]
    bin_lbls_r = ["<1s", "1-5s", "5-10s", "10-30s", "30s-1m", "1-2m", "2-5m", "5-10m"]

    cat_hist = {}
    for cat in sorted_cats:
        hist_vals = []
        for i in range(len(bin_edges)-1):
            c = sum(1 for d in cat_durations[cat] if bin_edges[i] <= d < bin_edges[i+1])
            hist_vals.append(c)
        cat_hist[cat] = hist_vals

    x = np.arange(len(bin_lbls_r))
    bar_width = 0.8 / len(sorted_cats)

    for idx, cat in enumerate(sorted_cats):
        offset = (idx - len(sorted_cats)/2 + 0.5) * bar_width
        bars_r = axes[1].bar(x + offset, cat_hist[cat], bar_width,
                             color=get_cat_color(cat), edgecolor="white", linewidth=0.2, alpha=0.85, label=cat)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_lbls_r, fontsize=7, rotation=45, ha="right")
    axes[1].set_title("Duracion por Categoria", loc="left", color=C["title"], fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Duracion", fontweight="bold")
    axes[1].set_ylabel("Sesiones", fontweight="bold")
    axes[1].grid(True, axis="y", linewidth=0.3)
    axes[1].legend(fontsize=7, frameon=False, ncol=2, loc="upper right", prop={"weight": "bold"})

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, f"{prefix}duracion_sesiones.png")


def graf_patrones_compulsivos(data, day_idx, total_days):
    prefix = f"04_{day_idx}_"
    date = data.get("date", "?")
    print(f"  [{day_idx}/{total_days}] Patrones compulsivos: {date}...")
    cm = data.get("compulsive_metrics", {})
    sessions = data.get("sessions", [])

    social_avg = cm.get("unlock_to_social_average_seconds")
    rapid = cm.get("rapid_reopenings", 0)
    avg_session = cm.get("average_session_seconds", 0)
    short = cm.get("short_session_pressure", {})
    under_5 = short.get("<5s", 0)
    between_5_10 = short.get("5-10s", 0)
    over_10 = len(sessions) - under_5 - between_5_10
    total = len(sessions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.suptitle(f"Patrones Compulsivos - {day_label(date)} ({date})", color=C["title"], fontsize=13, fontweight="bold")

    vals = [under_5, between_5_10, over_10]
    labels_s = ["<5s", "5-10s", ">10s"]
    clrs_s = [C["short"], C["medium"], C["long"]]

    if sum(vals) > 0:
        wedges, _ = axes[0].pie(
            vals, labels=None, colors=clrs_s, startangle=90,
            wedgeprops={"edgecolor": C["panel"], "linewidth": 1.5},
        )
        legend_labels = [f"{l}: {v} ({v/total*100:.0f}%)" if total > 0 else f"{l}: {v}"
                         for l, v in zip(labels_s, vals)]
        axes[0].legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(-0.15, 0.5),
                       fontsize=9, frameon=False, labelspacing=1.2, prop={"weight": "bold"})
    else:
        axes[0].text(0.5, 0.5, "Sin sesiones", transform=axes[0].transAxes,
                     ha="center", fontsize=11, color=C["dim"], fontweight="bold")
    axes[0].set_title(f"Sesiones por Duracion (n={total})", loc="left", color=C["title"], fontsize=10, fontweight="bold")

    axes[1].axis("off")
    metrics_text = [
        ("Reaperturas rapidas (<10s)", f"{rapid}", C["rapid"]),
        ("Sesion promedio", f"{avg_session:.0f}s  ({avg_session/60:.1f} min)", C["accent"]),
        ("Desbloqueo -> Social", f"{social_avg:.1f}s" if social_avg is not None else "N/A", C["social"]),
    ]
    y = 0.85
    for label, value, color in metrics_text:
        axes[1].text(0.05, y, label + ":", transform=axes[1].transAxes,
                     fontsize=12, color=C["dim"], fontweight="bold", va="top")
        axes[1].text(0.95, y, value, transform=axes[1].transAxes,
                     fontsize=16, color=color, fontweight="bold", va="top", ha="right")
        y -= 0.25

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, f"{prefix}patrones_compulsivos.png")


# ─── MAIN ───────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  GENERADOR DE GRAFICAS - USO DEL MOVIL")
    print("=" * 55)
    print(f"  Archivo: {os.path.basename(DATA_FILE)}")

    raw = load()
    style()

    blocks = raw.get("daily_blocks", None)
    if blocks is None:
        blocks = [raw]

    total = len(blocks)
    total_graphs = 0
    print(f"\n  {total} dia(s) encontrado(s)")

    for i, block in enumerate(blocks):
        day_idx = i + 1
        date = block.get("date", "?")
        screen = block.get("metrics", {}).get("screen_time_minutes", 0)
        n_sessions = len(block.get("sessions", []))
        print(f"\n  --- Dia {day_idx}/{total}: {day_label(date)} ({date})  "
              f"Pantalla: {fmt_min(screen)}  Sesiones: {n_sessions} ---")

        graf_resumen_general(block, day_idx, total)
        graf_distribucion_categorias(block, day_idx, total)
        graf_timeline_horario(block, day_idx, total)
        graf_patrones_compulsivos(block, day_idx, total)
        total_graphs += 4

    print(f"\n{'=' * 55}")
    print(f"  {total_graphs} graficas generadas en: {OUTPUT_DIR}")
    print(f"  ({total} dias x 4 graficas)")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
