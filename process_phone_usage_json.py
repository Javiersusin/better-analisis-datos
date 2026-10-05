#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_phone_usage_json.py - Procesado y anonimización del uso del móvil.

Qué hace:
    Toma el export de Android UsageStats y escribe al lado un
    `*_phone_usage_processed.json` anonimizado y agregado por día UTC.

Entrada:
    `usomvl/YYMMDD_HHMM_phone_usage.json` (crudo). De `records` solo se leen
    `records.events`; `appUsageStats`, `eventStats` y `configurationStats` se
    ignoran.

Salida:
    `usomvl/YYMMDD_HHMM_phone_usage_processed.json` con UNA sola raíz:
        { "daily_blocks": [ {cabecera, date, window_start, window_end,
                              metrics, compulsive_metrics,
                              unlock_events, sessions}, ... ] }

Qué se queda y qué se tira:
    - Se conservan 5 tipos de evento: ACTIVITY_RESUMED (inicio),
      ACTIVITY_PAUSED / ACTIVITY_STOPPED (fin) y SCREEN_INTERACTIVE /
      KEYGUARD_HIDDEN (desbloqueo). El resto de tipos de Android se descarta.
    - NO se vuelcan al resultado: packageName/className (solo sirven para
      asignar categoría), deviceManufacturer, deviceModel, androidSdk,
      collectorName/Version, standbyBucket, notificationChannelId, config...
    - Un paquete desconocido cae en la categoría `other`.

Uso:
    python process_phone_usage_json.py usomvl/                  # carpeta entera
    python process_phone_usage_json.py usomvl/x_phone_usage.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


FILENAME_RE = re.compile(
    r"^(?P<date>\d{6})_(?P<time>\d{4})_(?P<file_type>.+)$",
    re.IGNORECASE,
)

HEADER_KEYS = [
    "participant_id",
    "filename",
    "file_type",
    "timestamp_from_filename",
    "window_start",
    "window_end",
    "parse_status",
]

# Vocabulario de eventos de Android UsageStats que usamos:
# - SESSION_START_EVENT_TYPES: el SO trae la app a primer plano -> abre sesión.
# - SESSION_END_EVENT_TYPES: la sesión se pausa o se detiene -> la cerramos.
# - UNLOCK_EVENT_TYPES: desbloqueo de pantalla. SCREEN_INTERACTIVE llega al
#   encender la pantalla y KEYGUARD_HIDDEN al quitar el candado; los dos sirven
#   para medir "desbloqueo -> apertura de app", aunque `unlock_count` solo
#   cuenta KEYGUARD_HIDDEN (para no contar dos veces el mismo gesto).
SESSION_END_EVENT_TYPES = {"ACTIVITY_PAUSED", "ACTIVITY_STOPPED"}
SESSION_START_EVENT_TYPES = {"ACTIVITY_RESUMED"}
UNLOCK_EVENT_TYPES = {"SCREEN_INTERACTIVE", "KEYGUARD_HIDDEN"}

# Reglas de clasificación: packageName (o className como fallback) -> categoría.
# Se comparan por `startswith`, así que se listan prefijos/paquetes raíz.
# Es una lista CERRADA: cualquier app que no esté aquí cae en DEFAULT_CATEGORY.
# Si aparece una app nueva, aquí se añade (no hay que tocar más código).
PHONE_CATEGORY_RULES: dict[str, tuple[str, ...]] = {
    "social": (
        "com.facebook.katana",
        "com.instagram.android",
        "com.twitter.android",
        "com.xcorp.android",
        "com.reddit.frontpage",
        "org.telegram.messenger",
        "com.snapchat.android",
        "com.tiktok.android",
    ),
    "messaging": (
        "com.whatsapp",
        "com.google.android.apps.messaging",
        "com.samsung.android.messaging",
        "com.android.mms",
        "com.viber.voip",
        "jp.naver.line.android",
        "com.discord",
        "com.microsoft.team",
        "com.skype.raider",
    ),
    "productivity": (
        "com.better.better",
        "com.microsoft.office",
        "com.microsoft.teams",
        "com.microsoft.word",
        "com.microsoft.excel",
        "com.microsoft.powerpoint",
        "com.google.android.apps.docs",
        "com.google.android.apps.docs.editors.docs",
        "com.google.android.apps.docs.editors.sheets",
        "com.google.android.apps.docs.editors.slides",
        "com.notion",
        "com.trello",
        "com.asana.android",
        "com.todoist",
        "com.euroforum.euroforumevents",
        "com.google.android.calendar",
    ),
    "games": (
        "com.supercell.",
        "com.king.",
        "com.roblox.client",
        "com.mojang.minecraftpe",
        "com.tencent.ig",
        "com.epicgames.fortnite",
    ),
    "video_streaming": (
        "com.google.android.youtube",
        "com.netflix.mediaclient",
        "com.disney.disneyplus",
        "com.hbo.hbonow",
        "com.amazon.avod.thirdpartyclient",
        "com.primevideo",
        "rtve.tablet.android",
    ),
    "music": (
        "com.spotify.music",
        "com.youtube.music",
        "com.apple.android.music",
        "com.google.android.apps.youtube.music",
        "com.soundcloud.android",
        "com.deezer.android.app",
    ),
    "education": (
        "org.khanacademy.android",
        "com.duolingo",
        "com.coursera.android",
        "edx.edxonline.org",
        "com.udemy.android",
        "com.brilliant.android",
    ),
    "navigation": (
        "com.google.android.apps.maps",
        "com.google.android.googlequicksearchbox",
        "com.opera.browser",
        "com.waze",
        "com.here.app.maps",
        "com.michelin.android",
        "com.lyft.android.zaragozaapp",
        "eu.livesport.FlashScore_com",
    ),
    "shopping": (
        "com.amazon.mshop.android.shopping",
        "com.amazon.mShop.android.shopping",
        "com.ebay.mobile",
        "com.alibaba.aliexpresshd",
        "com.mercadolibre",
        "com.zalando.mobile",
    ),
}

DEFAULT_CATEGORY = "other"
# Tramos de duración para `metrics.short_sessions`: se generan las claves
# "<5s" y "5-10s" (las sesiones de >=10 s no se contabilizan en ese bloque).
SHORT_SESSION_THRESHOLDS = (5, 10)
UNLOCK_TO_APP_WINDOW_SECONDS = 15.0
RAPID_REOPENING_WINDOW_SECONDS = 10.0  # Reabrir tras 10 segundos.
#   ↑ OJO: estos dos últimos (y el 5 s de SHORT_SESSION) están replicados en
#     deep_analysis.py. Si se cambian aquí, hay que cambiarlos ahí también.


@dataclass(slots=True)
class NormalizedEvent:
    timestamp: datetime  # Momento exacto del evento (UTC).
    event_type: str  # Tipo de evento usado en la logica.
    category: str  # Categoria anonima asignada a la app.
    package_name: str | None  # Paquete Android original (si existe).
    class_name: str | None  # Clase Android original (fallback/clasificacion).
    event_type_name: str | None  # Nombre bruto del evento en el JSON.



@dataclass(slots=True)
class Session:
    category: str  # Categoria de la app durante la sesion.
    start_time: datetime  # Inicio de la sesion.
    end_time: datetime  # Fin de la sesion.
    duration_seconds: float  # Duracion total en segundos.
    unlock_to_open_seconds: float | None = None  # Segundos desde desbloqueo a apertura.


def parse_iso_datetime(value: str) -> datetime:
    """Convierte una fecha ISO del JSON en un datetime de Python."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def millis_to_datetime(value: int | float | str | None) -> datetime | None:
    """Convierte milisegundos Unix a datetime UTC."""
    if value is None:
        return None

    try:
        numeric_value = float(value)
        if numeric_value <= 0:
            return None
        return datetime.fromtimestamp(numeric_value / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def bucket_hour(value: datetime) -> str:
    """Agrupa un instante por hora en formato UTC."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def bucket_day(value: datetime) -> str:
    """Agrupa un instante por día en formato YYYY-MM-DD."""
    return value.astimezone(timezone.utc).date().isoformat()


def isoformat_utc(value: datetime) -> str:
    """Formatea un datetime en ISO-8601 con sufijo Z."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Escribe JSON con sangria estable y salto final."""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.write("\n")


def first_existing_value(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def parse_filename_metadata(path: Path) -> dict[str, Any]:
    """Saca del nombre del archivo la fecha, la hora y el tipo."""
    match = FILENAME_RE.match(path.stem)
    if not match:
        return {
            "filename": path.name,
            "file_type": None,
            "timestamp_from_filename": None,
        }

    raw_date = match.group("date")
    raw_time = match.group("time")
    parsed_datetime = datetime.strptime(raw_date + raw_time, "%y%m%d%H%M")

    return {
        "filename": path.name,
        "file_type": match.group("file_type"),
        "timestamp_from_filename": parsed_datetime.strftime("%Y-%m-%dT%H:%M:00"),
    }


def build_metadata(path: Path, original: dict[str, Any], participant_id: str | None) -> dict[str, Any]:
    """Construye la cabecera base del archivo de salida."""
    filename_metadata = parse_filename_metadata(path)
    window_start = first_existing_value(original, ["window_start", "windowStart"])
    window_end = first_existing_value(original, ["window_end", "windowEnd"])

    if window_start is None:
        window_start_dt = millis_to_datetime(original.get("queryStartMillis"))
        window_start = window_start_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if window_start_dt else None

    if window_end is None:
        window_end_dt = millis_to_datetime(original.get("queryEndMillis"))
        window_end = window_end_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if window_end_dt else None

    return {
        "participant_id": participant_id or first_existing_value(original, ["participant_id", "participantId"]),
        "filename": first_existing_value(original, ["filename", "fileName"]) or filename_metadata["filename"],
        "file_type": first_existing_value(original, ["file_type", "fileType"]) or filename_metadata["file_type"],
        "timestamp_from_filename": first_existing_value(
            original,
            ["timestamp_from_filename", "timestampFromFilename"],
        )
        or filename_metadata["timestamp_from_filename"],
        "window_start": window_start,
        "window_end": window_end,
        "parse_status": first_existing_value(original, ["parse_status", "parseStatus"]) or "parsed",
    }


def load_json(path: Path) -> Any:
    """Carga un JSON desde disco."""
    return read_json(path)


def extract_events(payload: Any) -> list[dict[str, Any]]:
    """Obtiene la lista de eventos del lote de uso del telefono.

    Nota: el crudo trae records como objeto con varias claves
    (events, appUsageStats, eventStats, configurationStats) y aquí solo nos
    quedamos con `events`; las otras se ignoran a propósito. También se
    acepta la variante en la que `records` es ya una lista plana de eventos.
    """
    if not isinstance(payload, dict):
        return []

    records = payload.get("records")
    if isinstance(records, dict):
        events = records.get("events")
        if isinstance(events, list):
            return [event for event in events if isinstance(event, dict)]

    if isinstance(records, list):
        return [event for event in records if isinstance(event, dict)]

    return []



def classify_package(package_name: str | None, class_name: str | None = None) -> str:
    """Convierte un packageName en una categoria estable y ampliable."""
    candidate = (package_name or class_name or "").strip().lower()
    if not candidate: #sino, le ponemos other, ya iré clasificando poco a poco según salagan  → hoy la lista ya vive en PHONE_CATEGORY_RULES; lo que no esté ahí sigue cayendo en `other`
        return DEFAULT_CATEGORY

    for category, prefixes in PHONE_CATEGORY_RULES.items():
        for prefix in prefixes:
            if candidate.startswith(prefix.lower()):
                return category

    return DEFAULT_CATEGORY


def normalize_event(raw_event: dict[str, Any]) -> NormalizedEvent | None:
    """Normaliza un evento crudo para trabajar con el resto del pipeline."""
    timestamp = millis_to_datetime(raw_event.get("timestamp"))
    event_type_name = raw_event.get("eventTypeName")

    if timestamp is None or not event_type_name:
        return None

    package_name = raw_event.get("packageName")
    class_name = raw_event.get("className")

    return NormalizedEvent(
        timestamp=timestamp,
        event_type=str(event_type_name),
        category=classify_package(package_name, class_name),
        package_name=package_name if isinstance(package_name, str) else None,
        class_name=class_name if isinstance(class_name, str) else None,
        event_type_name=str(event_type_name),
    )



def build_sessions(events: list[NormalizedEvent], window_end: datetime | None = None) -> list[Session]:
    """Construye sesiones a partir de eventos de actividad y desbloqueo."""
    if not events:
        return []

    sorted_events = sorted(events, key=lambda item: item.timestamp)
    sessions: list[Session] = []
    current_session: dict[str, Any] | None = None
    last_unlock_time: datetime | None = None

    def close_current_session(end_time: datetime) -> None:
        nonlocal current_session
        if current_session is None:
            return

        start_time = current_session["start_time"]
        if end_time < start_time:
            end_time = start_time

        duration_seconds = max((end_time - start_time).total_seconds(), 0.0)
        sessions.append(
            Session(
                category=current_session["category"],
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_seconds,
                unlock_to_open_seconds=current_session["unlock_to_open_seconds"],
            )
        )
        current_session = None

    for event in sorted_events:
        if event.event_type in UNLOCK_EVENT_TYPES:
            last_unlock_time = event.timestamp
            continue

        if event.event_type not in SESSION_START_EVENT_TYPES and event.event_type not in SESSION_END_EVENT_TYPES:
            continue

        if event.event_type in SESSION_START_EVENT_TYPES:
            if current_session is not None:
                same_category = current_session["category"] == event.category
                same_package = current_session["package_name"] == event.package_name
                if same_category and same_package:
                    continue

                close_current_session(event.timestamp)

            unlock_delta = None
            if last_unlock_time is not None:
                delta_seconds = (event.timestamp - last_unlock_time).total_seconds()
                if 0 <= delta_seconds <= UNLOCK_TO_APP_WINDOW_SECONDS:
                    unlock_delta = delta_seconds

            current_session = {
                "category": event.category,
                "package_name": event.package_name,
                "start_time": event.timestamp,
                "unlock_to_open_seconds": unlock_delta,
            }
            continue

        if current_session is None:
            continue

        if current_session["package_name"] != event.package_name:
            continue

        close_current_session(event.timestamp)

    if current_session is not None:
        fallback_end = window_end or sorted_events[-1].timestamp
        close_current_session(fallback_end)

    return sessions


def add_duration_by_hour(hourly_totals: dict[str, float], start_dt: datetime, end_dt: datetime, value_seconds: float) -> None:
    """Reparte una duracion entre las horas UTC que cruza la sesion."""
    start_utc = start_dt.astimezone(timezone.utc)
    end_utc = end_dt.astimezone(timezone.utc)
    duration_seconds = (end_utc - start_utc).total_seconds()

    if duration_seconds <= 0:
        hourly_totals[bucket_hour(start_utc)] += float(value_seconds)
        return

    cursor = start_utc
    while cursor < end_utc:
        hour_start = cursor.replace(minute=0, second=0, microsecond=0)
        next_hour = hour_start + timedelta(hours=1)
        segment_end = min(end_utc, next_hour)
        segment_seconds = (segment_end - cursor).total_seconds()
        if segment_seconds > 0:
            hour_key = bucket_hour(cursor)
            hourly_totals[hour_key] += float(value_seconds) * (segment_seconds / duration_seconds)
        cursor = segment_end


def split_session_by_day(session: Session) -> list[Session]:
    """Divide una sesion en segmentos de un solo dia UTC si cruza medianoche."""
    start_utc = session.start_time.astimezone(timezone.utc)
    end_utc = session.end_time.astimezone(timezone.utc)

    if end_utc <= start_utc:
        return [session]

    segments: list[Session] = []
    cursor = start_utc
    keep_unlock_delta = True

    while cursor < end_utc:
        next_day = cursor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        segment_end = min(end_utc, next_day)
        segment_seconds = max((segment_end - cursor).total_seconds(), 0.0)

        if segment_seconds > 0:
            segments.append(
                Session(
                    category=session.category,
                    start_time=cursor,
                    end_time=segment_end,
                    duration_seconds=segment_seconds,
                    unlock_to_open_seconds=session.unlock_to_open_seconds if keep_unlock_delta else None,
                )
            )

        cursor = segment_end
        keep_unlock_delta = False

    return segments or [session]


def serialize_sessions(sessions: list[Session]) -> list[dict[str, Any]]:
    """Convierte sesiones a la estructura JSON de salida."""
    return [
        {
            "category": session.category,
            "start_time": isoformat_utc(session.start_time),
            "end_time": isoformat_utc(session.end_time),
            "duration_seconds": round(session.duration_seconds, 1),
            "unlock_to_open_seconds": round(session.unlock_to_open_seconds, 1) if session.unlock_to_open_seconds is not None else None,
        }
        for session in sessions
    ]


def compute_block_window(
    sessions: list[Session],
    events: list[NormalizedEvent],
    day: str,
) -> tuple[str, str]:
    """Calcula la ventana temporal de un bloque diario."""
    day_start = datetime.fromisoformat(f"{day}T00:00:00+00:00")
    day_end = day_start + timedelta(days=1)

    timestamps = [event.timestamp for event in events]
    for session in sessions:
        timestamps.append(session.start_time)
        timestamps.append(session.end_time)

    if timestamps:
        return isoformat_utc(min(timestamps)), isoformat_utc(max(timestamps))

    return isoformat_utc(day_start), isoformat_utc(day_end)


def build_daily_block(
    metadata: dict[str, Any],
    day: str,
    sessions: list[Session],
    events: list[NormalizedEvent],
) -> dict[str, Any]:
    """Construye un bloque diario con la misma estructura analitica que el bloque global."""
    metrics = compute_metrics(sessions, events)
    compulsive_metrics = detect_compulsive_patterns(sessions, events)
    window_start, window_end = compute_block_window(sessions, events, day)

    # Conservamos los eventos de desbloqueo con su timestamp original (estructura aditiva).
    # Los deep_analysis los necesitan para sincronizar frecuencia cardíaca con uso del móvil.
    unlock_events = [
        isoformat_utc(event.timestamp)
        for event in events
        if event.event_type in UNLOCK_EVENT_TYPES
    ]

    return {
        **{key: metadata.get(key) for key in HEADER_KEYS},
        "date": day,
        "window_start": window_start,
        "window_end": window_end,
        "metrics": metrics,
        "compulsive_metrics": compulsive_metrics,
        "unlock_events": unlock_events,
        "sessions": serialize_sessions(sessions),
    }


def build_daily_blocks(
    metadata: dict[str, Any],
    sessions: list[Session],
    events: list[NormalizedEvent],
) -> list[dict[str, Any]]:
    """Agrupa sesiones y eventos por dia UTC."""
    sessions_by_day: dict[str, list[Session]] = defaultdict(list)
    events_by_day: dict[str, list[NormalizedEvent]] = defaultdict(list)

    for session in sessions:
        for segment in split_session_by_day(session):
            sessions_by_day[bucket_day(segment.start_time)].append(segment)

    for event in events:
        events_by_day[bucket_day(event.timestamp)].append(event)

    all_days = sorted(set(sessions_by_day) | set(events_by_day))
    return [
        build_daily_block(
            metadata,
            day,
            sorted(sessions_by_day.get(day, []), key=lambda item: item.start_time),
            sorted(events_by_day.get(day, []), key=lambda item: item.timestamp),
        )
        for day in all_days
    ]


def compute_metrics(
    sessions: list[Session],
    events: list[NormalizedEvent],
) -> dict[str, Any]:
    """Calcula el primer bloque de metricas anonimizadas del uso del telefono."""
    total_usage_seconds = sum(session.duration_seconds for session in sessions)

    unlock_count = sum(1 for event in events if event.event_type == "KEYGUARD_HIDDEN")
    app_switches = max(len(sessions) - 1, 0)

    category_totals_seconds: dict[str, float] = defaultdict(float)
    category_opens: dict[str, int] = defaultdict(int)
    category_event_counts: dict[str, int] = defaultdict(int)
    hourly_usage_seconds: dict[str, float] = defaultdict(float)

    for session in sessions:
        category_totals_seconds[session.category] += session.duration_seconds
        category_opens[session.category] += 1

    for session in sessions:
        add_duration_by_hour(hourly_usage_seconds, session.start_time, session.end_time, session.duration_seconds)

    for event in events:
        category_event_counts[event.category] += 1

    short_session_counts = {}
    prev = 0
    for threshold in SHORT_SESSION_THRESHOLDS:
        label = f"{prev}-{threshold}s" if prev > 0 else f"<{threshold}s"
        short_session_counts[label] = sum(1 for session in sessions if prev <= session.duration_seconds < threshold)
        prev = threshold

    categories: dict[str, dict[str, Any]] = {}
    for category in sorted(category_totals_seconds):
        minutes = round(category_totals_seconds[category] / 60.0, 2)
        opens = category_opens.get(category, 0)
        category_sessions = [session.duration_seconds for session in sessions if session.category == category]
        average_session_seconds = round(mean(category_sessions), 1) if category_sessions else 0.0
        events_count = category_event_counts.get(category, 0)
        percentage = round((category_totals_seconds[category] / total_usage_seconds) * 100, 2) if total_usage_seconds else 0.0

        categories[category] = {
            "minutes": minutes,
            "opens": opens,
            "average_session_seconds": average_session_seconds,
            "events": events_count,
            "time_percentage": percentage,
        }

    average_session_seconds = round(mean([session.duration_seconds for session in sessions]), 1) if sessions else 0.0

    usage_by_hour = [
        {
            "hour": hour_key,
            "minutes": round(hourly_usage_seconds[hour_key] / 60.0, 2),
        }
        for hour_key in sorted(hourly_usage_seconds)
    ]

    return {
        "screen_time_minutes": round(total_usage_seconds / 60.0, 2),
        "unlock_count": unlock_count,
        "app_switches": app_switches,
        "short_sessions": short_session_counts,
        "categories": categories,
        "usage_by_hour": usage_by_hour,
        "events_by_category": {
            category: category_event_counts[category]
            for category in sorted(category_event_counts)
        },
        "average_session_seconds": average_session_seconds,
    }


def detect_compulsive_patterns(sessions: list[Session], events: list[NormalizedEvent]) -> dict[str, Any]:
    """Deja preparado el bloque de deteccion de patrones compulsivos."""
    rapid_reopenings = 0
    previous_session: Session | None = None

    for session in sessions:
        if previous_session is not None:
            gap_seconds = (session.start_time - previous_session.end_time).total_seconds()
            if previous_session.category == session.category and 0 <= gap_seconds <= RAPID_REOPENING_WINDOW_SECONDS:
                rapid_reopenings += 1
        previous_session = session

    social_unlock_deltas = [
        session.unlock_to_open_seconds
        for session in sessions
        if session.category == "social" and session.unlock_to_open_seconds is not None
    ]

    return {
        "rapid_reopenings": rapid_reopenings,
        "average_session_seconds": round(mean([session.duration_seconds for session in sessions]), 1) if sessions else 0.0,
        "unlock_to_social_average_seconds": round(mean(social_unlock_deltas), 1) if social_unlock_deltas else None,
        "short_session_pressure": {
            **(
                {f"<{SHORT_SESSION_THRESHOLDS[0]}s": sum(1 for s in sessions if s.duration_seconds < SHORT_SESSION_THRESHOLDS[0])}
                if SHORT_SESSION_THRESHOLDS
                else {}
            ),
            **{
                f"{SHORT_SESSION_THRESHOLDS[i]}-{SHORT_SESSION_THRESHOLDS[i+1]}s": sum(
                    1 for s in sessions if SHORT_SESSION_THRESHOLDS[i] <= s.duration_seconds < SHORT_SESSION_THRESHOLDS[i + 1]
                )
                for i in range(len(SHORT_SESSION_THRESHOLDS) - 1)
            },
        },
        "app_switches": max(len(sessions) - 1, 0),
    }


def anonymize_data(
    path: Path,
    payload: dict[str, Any],
    participant_id: str | None,
    sessions: list[Session],
    events: list[NormalizedEvent],
) -> dict[str, Any]:
    """Construye el JSON de salida sin campos identificativos del dispositivo.

    Qué hace:
        Devuelve SOLO `{"daily_blocks": [...]}`. Al no copiar el resto del
        payload original se caen de golpe deviceManufacturer, deviceModel,
        androidSdk, collectorName/Version, queryStartMillis/EndMillis,
        collectedAt, hasGap y todo lo que acompaña a `records`.
    Qué NO se serializa dentro de los bloques:
        ni packageName ni className de las sesiones (solo `category`), ni el
        resto de eventos que no sean desbloqueos. Quedan tiempos, categorías y
        métricas anónimas.
    """
    metadata = build_metadata(path, payload, participant_id)
    daily_blocks = build_daily_blocks(metadata, sessions, events)
    return {"daily_blocks": daily_blocks}


def export_json(path: Path, payload: Any) -> None:
    """Guarda el resultado anonimizado en disco."""
    write_json(path, payload)


def output_path_for(source_path: Path) -> Path:
    """Genera el nombre del archivo de salida en la misma carpeta."""
    return source_path.with_name(f"{source_path.stem}_processed{source_path.suffix}")


def iter_source_files(root: Path) -> list[Path]:
    """Devuelve un unico archivo o todos los JSON de una carpeta."""
    if root.is_file():
        return [root]

    return sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() == ".json"
        and not candidate.name.lower().endswith("_processed.json")
    )


def process_file(source_path: Path, participant_id: str | None) -> Path:
    """Procesa un unico archivo JSON de uso del telefono."""
    payload = load_json(source_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {source_path.name}, pero se obtuvo {type(payload).__name__}.")

    raw_events = extract_events(payload)
    events = [event for raw_event in raw_events if (event := normalize_event(raw_event)) is not None]

    window_end = millis_to_datetime(payload.get("queryEndMillis"))
    sessions = build_sessions(events, window_end=window_end)
    anonymized = anonymize_data(source_path, payload, participant_id, sessions, events)

    target_path = output_path_for(source_path)
    export_json(target_path, anonymized)
    return target_path


def process_path(root: Path, participant_id: str | None) -> list[Path]:
    """Procesa un archivo o una carpeta completa y devuelve las rutas creadas."""
    processed_files: list[Path] = []
    for source_path in iter_source_files(root):
        processed_files.append(process_file(source_path, participant_id))
    return processed_files


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Anonimiza archivos JSON de uso del telefono Android.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Ruta al archivo JSON o a la carpeta que contiene varios JSON.",
    )
    parser.add_argument(
        "--participant-id",
        default=None,
        help="Sobrescribe participant_id cuando no se pueda inferir del nombre de la carpeta.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"No existe la ruta indicada: {source}")

    processed_files = process_path(source, args.participant_id)

    for processed_path in processed_files:
        print(f"Archivo procesado: {processed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())