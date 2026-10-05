#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_health_json.py - Procesado de los JSON crudos de salud.

Qué hace:
    Toma un export de salud (Android/Samsung Health o iOS/HealthKit) y escribe
    al lado un `*_health_processed.json` enriquecido, SIN tocar el original.

Entrada:
    `health/YYMMDD_HHMM_health.json` (o una carpeta entera con varios).

Salida:
    `health/YYMMDD_HHMM_health_processed.json`, con esta estructura:
      - cabecera (HEADER_KEYS) al principio
      - general_resume      -> pasos por día (fuentes saneadas)
      - heart_rate_resume   -> media de pulso por bloques de 5 min
      - heart_rate_raw      -> pulso medición a medición (para deep_analysis)
      - blood_oxygen_resume -> SpO2 por día
      - daily_summary       -> una fila por día UTC
      - deep_analysis       -> solo si se pasa --phone-processed
      - el resto del JSON original, sin `records` (ver build_output_payload)

Qué NO hace:
    No entrena modelos ni diagnostica nada: es un paso de ETL + resúmenes.

Uso:
    python process_health_json.py health/260516_1728_health.json
    python process_health_json.py health/                       # carpeta entera
    python process_health_json.py health/x.json --phone-processed usomvl/y_processed.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from pathlib import Path
from statistics import median
from typing import Any, Iterable


FILENAME_RE = re.compile(
    r"^(?P<date>\d{6})_(?P<time>\d{4})_(?P<file_type>.+)$",
    re.IGNORECASE,
)

# Expresiones y constantes globales usadas en la lógica:
# - FILENAME_RE: extrae fecha, hora y tipo desde el nombre del archivo (ej. 260516_1728_health)
#uso redes sociales, patrones compulsibos abrir cerrar, desbloqueos constante, sepearacion por tipo app, fliltrado apps privadas
#wliminar id telefono
#   ↑ Nota: esa lista es del bloque de USO DEL MÓVIL, que vive en
#     process_phone_usage_json.py, no aquí. Todo lo de ahí arriba ya está
#     resuelto en ese módulo: sociales -> PHONE_CATEGORY_RULES, patrones
#     compulsivos -> detect_compulsive_patterns, desbloqueos -> unlock_count,
#     separación por tipo de app -> classify_package y "apps privadas" ->
#     se dejan en la categoría `other` (nunca se vuelca el packageName).
#     El "eliminar id teléfono" es la anonimización de `anonymize_data`.
GAP_THRESHOLD_SECONDS = 1.0
PRIMARY_STEPS_SOURCE = "com.sec.android.app.shealth"
SECONDARY_STEPS_SOURCE = "android"
HEART_RATE_BUCKET_MINUTES = 5
HEART_RATE_MIN_BPM = 35
HEART_RATE_MAX_BPM = 270
HRV_TYPE = "HEART_RATE_VARIABILITY_SDNN" #iosexlc
#   ↑ "iosexlc" = tipo de registro propio de iOS/exclusivo: solo aparece en los
#     exports de Apple Health, por eso HRV y FC en reposo se leen dentro del
#     `if CURRENT_PLATFORM == "ios"` de build_daily_summary.
BLOOD_OXYGEN_TYPE = "BLOOD_OXYGEN"
TOTAL_CALORIES_TYPE = "TOTAL_CALORIES_BURNED"
DISTANCE_DELTA_TYPE = "DISTANCE_DELTA"
WORKOUT_TYPE = "WORKOUT"
BODY_TEMPERATURE_TYPE = "BODY_TEMPERATURE"
STRESS_TYPE = "STRESS"
SLEEP_SESSION_TYPE = "SLEEP_SESSION"
SLEEP_PHASE_TYPES = {
    "SLEEP_ASLEEP",
    "SLEEP_AWAKE_IN_BED",
    "SLEEP_AWAKE",
    "SLEEP_DEEP",
    "SLEEP_LIGHT",
    "SLEEP_REM",
}
ALLOWED_HEALTH_SOURCES = {PRIMARY_STEPS_SOURCE, SECONDARY_STEPS_SOURCE}
HEADER_KEYS = [
    "participant_id",
    "filename",
    "file_type",
    "timestamp_from_filename",
    "window_start",
    "window_end",
    "parse_status",
]
WINDOW_KEYS_TO_REMOVE = {"windowStart", "windowEnd", "window_start", "window_end"}
CURRENT_PLATFORM: str | None = None #detecto arriba ya si es ios
#   ↑ Se asigna al principio de build_output_payload a partir de
#     payload["platform"] ("android" / "ios"). Es una variable de módulo
#     (no un parámetro) porque la usan funciones muy internas como
#     collect_heart_rate_samples, collect_step_intervals y build_daily_summary.

# Notas sobre las constantes:
# - GAP_THRESHOLD_SECONDS: umbral mínimo (segundos) para considerar que hay un "gap" entre intervalos
# - PRIMARY/SECONDARY_STEPS_SOURCE: prioridades para elegir entre fuentes de pasos cuando hay solapamientos
# - HEART_RATE_*: parámetros para agrupar y filtrar muestras de pulso, val limite dado para desviacion logica
# - HEADER_KEYS: orden y nombres de las claves que añadimos como cabecera enriquecida
# - ALLOWED_HEALTH_SOURCES: mismo par de fuentes que para pasos; en Android se
#   filtra por ellas y en iOS se acepta cualquier fuente (los exports de
#   Apple Health traen sourceName de reloj, teléfono, etc.).
# - HEART_RATE_BUCKET_MINUTES: solo se usa para heart_rate_resume (el bucket de
#   5 min). heart_rate_raw NO se agrupa: mantiene la medición original.


def parse_iso_datetime(value: str) -> datetime:
    """Convierte una fecha ISO del JSON en un datetime de Python."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def bucket_hour(value: datetime) -> str:
    """Agrupa un instante por hora en formato UTC."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def bucket_day(value: datetime) -> str:
    """Agrupa un instante por día en formato YYYY-MM-DD."""
    return value.astimezone(timezone.utc).date().isoformat()


def intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Dice si dos intervalos se pisan en el tiempo."""
    return left["start"] < right["end"] and right["start"] < left["end"]


def bucket_heart_rate_time(value: datetime) -> str:
    """Agrupa un instante en bloques de 5 minutos para el pulso."""
    utc_value = value.astimezone(timezone.utc)
    bucket_minute = (utc_value.minute // HEART_RATE_BUCKET_MINUTES) * HEART_RATE_BUCKET_MINUTES
    bucket_start = utc_value.replace(minute=bucket_minute, second=0, microsecond=0)
    return bucket_start.strftime("%Y-%m-%dT%H:%M:00Z")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Escribe JSON con una línea en blanco entre la cabecera y el bloque original."""
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if isinstance(payload, dict):
        lines = text.splitlines()
        separator_line = f'  "{HEADER_KEYS[-1]}":'

        for index, line in enumerate(lines):
            if line.startswith(separator_line):
                lines.insert(index + 1, "")
                break

        text = "\n".join(lines)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.write("\n")


def first_existing_value(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def infer_participant_id(path: Path, override: str | None) -> Any:
    """Obtiene el identificador del participante.

    Si el usuario lo pasa por parámetro, ese valor manda.
    Si no, solo se usa el nombre de la carpeta cuando parece un hash.

    Estado actual (decisión consciente): NO se infiere nada más. Sin
    `--participant-id` se devuelve None y la cabecera queda en null, para no
    colar en la salida el nombre de la carpeta (que puede ser un hash o un
    nombre personal). `path` queda ahí por si más adelante se decide usarlo.
    """
    if override:
        return override


    return None


def parse_filename_metadata(path: Path) -> dict[str, Any]:
    """Saca del nombre del archivo la fecha, la hora y el tipo.

    Ejemplo:
    260516_1728_health.json -> 2026-05-16 17:28, tipo health.
    """
    match = FILENAME_RE.match(path.stem)
    if not match:
        # Si el nombre no sigue el patrón, devolvemos solo lo seguro.
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


def build_metadata(path: Path, original: dict[str, Any]) -> dict[str, Any]:
    """Construye solo la cabecera que queremos añadir o completar."""
    filename_metadata = parse_filename_metadata(path)

    metadata = {
        # Si ya existe la clave en el JSON original, se respeta su valor.
        "participant_id": first_existing_value(original, ["participant_id", "participantId"]),
        "filename": first_existing_value(original, ["filename", "fileName"]) or filename_metadata["filename"],
        "file_type": first_existing_value(original, ["file_type", "fileType"]) or filename_metadata["file_type"],
        "timestamp_from_filename": first_existing_value(
            original,
            ["timestamp_from_filename", "timestampFromFilename"],
        )
        or filename_metadata["timestamp_from_filename"],
        "window_start": first_existing_value(original, ["window_start", "windowStart"]),
        "window_end": first_existing_value(original, ["window_end", "windowEnd"]),
        "parse_status": first_existing_value(original, ["parse_status", "parseStatus"]) or "parsed",
    }
    return metadata


def build_output_payload(
    path: Path,
    payload: Any,
    participant_id: str | None, # Lo dejo vacío de momemto  → sin --participant-id llega None y la cabecera sale en null (ver infer_participant_id)
    phone_processed: dict[str, Any] | None = None,
) -> Any:
    """Añade la cabecera nueva al JSON sin borrar el resto del contenido.

    Si se pasa `phone_processed`,
    además se calcula en el mismo paso el bloque `deep_analysis`, usando las
    muestras de HR en bruto ya presentes en memoria (los records originales),
    sin necesidad de re-leer el JSON procesado.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path.name}, pero se obtuvo {type(payload).__name__}.")

    global CURRENT_PLATFORM
    CURRENT_PLATFORM = str(payload.get("platform") or "").lower() or None

    metadata = build_metadata(path, payload)
    metadata["participant_id"] = metadata.get("participant_id") or infer_participant_id(path, participant_id)

    enriched: dict[str, Any] = {}

    # Cabecera nueva, en el orden que nos interesa.
    for key in HEADER_KEYS:
        enriched[key] = metadata.get(key)

    # Primero va schemaVersion dentro de la parte original.
    if "schemaVersion" in payload:
        enriched["schemaVersion"] = payload["schemaVersion"]

    general_resume = build_general_resume(payload) #resumen pasos inicial
    if general_resume is not None:
        enriched["general_resume"] = general_resume

    heart_rate_resume = build_heart_rate_resume(payload) #resumen pulso después pasos(bloques 5min)
    if heart_rate_resume is not None:
        enriched["heart_rate_resume"] = heart_rate_resume

    # Datos en bruto de frecuencia cardíaca, medición a medición, con su timestamp original.
    # ADTIVO: no sustituye heart_rate_resume (bloques de 5 min) sino que lo complementa
    # para poder hacer análisis de alta resolución temporal (deep_analysis).
    heart_rate_raw = build_heart_rate_raw(payload)
    if heart_rate_raw:
        enriched["heart_rate_raw"] = heart_rate_raw

    blood_oxygen_resume = build_blood_oxygen_resume(payload)
    if blood_oxygen_resume is not None:
        enriched["blood_oxygen_resume"] = blood_oxygen_resume

    # Luego copiamos el resto del JSON original, evitando las ventanas duplicadas.
    for key, value in payload.items():
        if key == "schemaVersion":
            continue
        if key in WINDOW_KEYS_TO_REMOVE:
            continue
        enriched[key] = value

    # Eliminamos listas grandes que no queremos en la salida final.
    enriched.pop("requestedDataTypes", None)
    enriched.pop("exportedDataTypes", None)

    # Reordenamos `records` para que todos los registros `STEPS` queden juntos al inicio,
    # preservando el orden relativo dentro de cada grupo.
    records = enriched.get("records")
    if isinstance(records, list):
        steps_records = [r for r in records if isinstance(r, dict) and r.get("type") == "STEPS"]
        other_records = [r for r in records if not (isinstance(r, dict) and r.get("type") == "STEPS")]
        enriched["records"] = steps_records + other_records

    daily_summary = build_daily_summary(enriched)
    if daily_summary:
        enriched["daily_summary"] = daily_summary

    # Análisis fisiológico y circadiano profundo (ADITIVO).
    if not enriched.get("deep_analysis"):
        deep = build_deep_analysis_block(payload, phone_processed)
        if deep:
            enriched["deep_analysis"] = deep

    # Añadimos deep_analysis a cada día del daily_summary
    # para que sea fácil ver qué pasó cada día
    attach_deep_analysis_to_daily_summary(
        enriched.get("daily_summary", []),
        enriched.get("deep_analysis"),
    )

    # Por último quitamos `records`: todo lo útil ya está resumido arriba
    # (general_resume, heart_rate_resume, heart_rate_raw, blood_oxygen_resume,
    # daily_summary). Al salir sin records, `add_steps_summary_to_records` y
    # `prune_summary_records` (que se llaman después en process_file/process.py)
    # no encuentran records y hoy no modifican nada. Si algún día se quiere
    # conservar `dataset_summary`/los intervals en la salida, este pop hay que
    # moverlo o eliminarlo.
    enriched.pop("records", None)

    return enriched


def build_deep_analysis_block(
    raw_payload: dict[str, Any],
    phone_processed: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Construye el bloque `deep_analysis` reutilizando las muestras de HR en bruto.

    Qué hace:
        Durante el procesado ya tenemos los records de HR brutos en memoria
        (collect_heart_rate_samples + filtro del módulo actual). Convierte esas
        muestras al formato que espera deep_analysis y llama a build_deep_analysis
        usando hr_samples_override, para no re-leer el JSON procesado.
    Qué recibe:
        raw_payload: JSON original de salud con los records sin procesar.
        phone_processed: dict del *_phone_usage_processed.json (opcional).
    Qué devuelve:
        El bloque deep_analysis (dict), o None si no hay HR ni móvil.
    """
    from deep_analysis import build_deep_analysis

    samples = collect_heart_rate_samples(raw_payload)
    accepted, _ = filter_heart_rate_samples(samples)
    hr_samples = [{"time": s["time"], "value": int(s["value"])} for s in accepted]

    if not hr_samples and not phone_processed:
        return None

    return build_deep_analysis(
        health_processed=None,
        phone_processed=phone_processed or {},
        hr_samples_override=hr_samples,
    )


def attach_deep_analysis_to_daily_summary(
    daily_summary: list[dict[str, Any]],
    deep_analysis: dict[str, Any] | None,
) -> None:
    """Añade un bloque deep_analysis a cada día del daily_summary.

    Qué hace:
        Recorre cada día del daily_summary y le añade un sub-bloco
        deep_analysis con los datos que corresponden a ese día concreto.
        Esto facilita ver qué pasó cada día sin tener que buscar en el
        bloque deep_analysis global.

    Qué añade a cada día:
        - hr_stats: estadísticas de HR de ese día (min, max, mean, hour_of_min/max)
        - mobile_sleep_window: uso del móvil en ventana de sueño de ese día
        - hr_delta_events: eventos de cambio de HR de ese día
        - pre_sleep_compulsive: comportamiento compulsivo antes de dormir

    No añade datos globales que son iguales para todos los días:
        - circadian_profile (perfil horario aggregate)
        - mobile_pre_sleep (pre-sleep global)
        - hr_mobile_events (análisis global de sesiones)
        - compulsive_events_heart_rate (eventos globales)

    No retorna nada. Modifica daily_summary in-place.
    """
    if not deep_analysis or not daily_summary:
        return

    # Indexamos datos por día para acceso rápido
    circadian = deep_analysis.get("circadian_heart_rate") or {}
    all_days_circadian = circadian.get("all_days") or {}
    per_day_stats = all_days_circadian.get("per_day") or []

    mobile_sleep = deep_analysis.get("mobile_sleep_window") or {}
    mobile_sleep_per_date = mobile_sleep.get("per_date") or []

    pre_sleep = deep_analysis.get("pre_sleep_mobile") or {}
    pre_sleep_compulsive = pre_sleep.get("compulsive_behavior") or {}

    hr_delta = deep_analysis.get("heart_rate_delta_events") or {}
    hr_delta_samples = hr_delta.get("sample_events") or []

    # Para cada día del daily_summary, buscamos los datos que le corresponden
    for day_entry in daily_summary:
        date = day_entry.get("date")
        if not date:
            continue

        day_deep: dict[str, Any] = {}

        # 1. Stats de HR del día (de per_day_stats)
        hr_day = next((d for d in per_day_stats if d.get("date") == date), None)
        if hr_day:
            day_deep["hr_stats"] = {
                "min": hr_day.get("min"),
                "max": hr_day.get("max"),
                "mean": hr_day.get("mean"),
                "samples": hr_day.get("count"),
                "hour_of_min": hr_day.get("hour_of_min"),
                "hour_of_max": hr_day.get("hour_of_max"),
            }

        # 2. Uso del móvil en ventana de sueño de ese día
        sleep_day = next((d for d in mobile_sleep_per_date if d.get("date") == date), None)
        if sleep_day:
            day_deep["mobile_sleep_window"] = {
                "minutes": sleep_day.get("minutes"),
                "sessions": sleep_day.get("sessions"),
            }

        # 3. Eventos de delta HR de ese día (filtramos por timestamp)
        # Un delta tiene time1 y time2. Si time1 es del día anterior pero
        # time2 es de este día, el delta pertenece al día de time2.
        if hr_delta_samples:
            day_events = [
                e for e in hr_delta_samples
                if e.get("time2", "")[:10] == date
            ]
            if day_events:
                deltas = [e.get("delta_hr", 0) for e in day_events]
                day_deep["hr_delta_events"] = {
                    "count": len(day_events),
                    "mean_delta": round(sum(deltas) / len(deltas), 3) if deltas else 0,
                    "max_increase": max(deltas) if deltas else 0,
                    "max_decrease": min(deltas) if deltas else 0,
                }

        # 4. Comportamiento compulsivo antes de dormir (es global, no por día)
        # Solo lo añadimos al último día como referencia
        # (o al primero si hay pre-sleep data)
        #   ↑ Sin implementar: `compulsive_behavior` es global, así que de
        #     momento no se copia a ningún día concreto. Si se decide copiarlo,
        #     habría que elegir aquí el día destino.

        # Solo añadimos deep_analysis si hay datos
        if day_deep:
            day_entry["deep_analysis"] = day_deep


def prune_summary_records(
    payload: dict[str, Any],
    clean_up_steps: bool = True,
    filter_heart_rate: bool = True,
) -> dict[str, Any]:
    """Elimina detalle crudo ya resumido, preservando metadatos y resúmenes."""
    records = payload.get("records")
    if not isinstance(records, list):
        return payload

    record_types_to_remove = set()
    if clean_up_steps:
        record_types_to_remove.add("STEPS")
    if filter_heart_rate:
        record_types_to_remove.add("HEART_RATE")

    if not record_types_to_remove:
        return payload

    updated_records: list[Any] = []
    for record in records:
        if not isinstance(record, dict):
            updated_records.append(record)
            continue

        record_type = record.get("type")
        if record_type == "HEART_RATE" and filter_heart_rate:
            # HEART_RATE ya queda resumido en heart_rate_resume.
            continue

        if record_type == "STEPS" and clean_up_steps and record.get("representation") == "interval_series":
            # Conservamos dataset_summary y metadatos, pero quitamos muestras crudas.
            cleaned_record = {
                key: value
                for key, value in record.items()
                if key not in {"intervals", "dataset_summary"}
            }
            updated_records.append(cleaned_record)
            continue

        updated_records.append(record)

    updated_payload = dict(payload)
    updated_payload["records"] = updated_records
    return updated_payload


def collect_heart_rate_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Recoge muestras de pulso de series regulares y sparse."""
    records = payload.get("records")
    if not isinstance(records, list):
        return []

    # `collected`: lista de muestras normalizadas con campos comunes:
    # {"representation","sourceName","startTime","time","value"}
    collected: list[dict[str, Any]] = []
    # Fuentes que aceptamos para construir el resumen , me ha parecido mas válida la primaria(definida arriba)
    allowed_sources = {PRIMARY_STEPS_SOURCE, SECONDARY_STEPS_SOURCE}

    for record in records:
        if not isinstance(record, dict):
            continue

        if record.get("type") != "HEART_RATE":
            continue

        source_name = record.get("sourceName")

        is_allowed = (
            CURRENT_PLATFORM == "ios"
            or source_name in ALLOWED_HEALTH_SOURCES
        )

        if not is_allowed:
            continue

        if record.get("representation") == "regular_series":
            start_time = record.get("startTime")
            values = record.get("values")
            sample_period_millis = int(record.get("samplePeriodMillis", 60000) or 60000)
            if not start_time or not isinstance(values, list):
                continue

            start_datetime = parse_iso_datetime(start_time)
            step = timedelta(milliseconds=sample_period_millis)

            for index, value in enumerate(values):
                collected.append(
                    {
                        "representation": "regular_series",
                        "sourceName": source_name,
                        # `startTime`: ISO string en UTC, `time`: datetime para ordenación/comparaciones
                        "startTime": (start_datetime + (step * index)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "time": start_datetime + (step * index),
                        "value": int(value),
                    }
                )

        elif record.get("representation") == "sparse_series":
            samples = record.get("samples")
            if not isinstance(samples, list):
                continue

            for sample in samples:
                if not isinstance(sample, dict):
                    continue

                time_value = sample.get("time")
                if not time_value:
                    continue

                collected.append(
                    {
                        "representation": "sparse_series",
                        "sourceName": source_name,
                        "startTime": time_value,
                        # Convertimos `time` a datetime para posterior filtrado y agrupado
                        "time": parse_iso_datetime(time_value),
                        "value": int(sample.get("value", 0) or 0),
                    }
                )

        elif record.get("representation") == "interval_series":
            intervals = record.get("intervals")
            if not isinstance(intervals, list):
                continue

            for interval in intervals:
                if not isinstance(interval, dict):
                    continue

                start_time = interval.get("startTime")
                if not start_time:
                    continue

                collected.append(
                    {
                        "representation": "interval_series",
                        "sourceName": source_name,
                        "startTime": start_time,
                        "time": parse_iso_datetime(start_time),
                        "value": int(interval.get("value", 0) or 0),
                    }
                )

    return collected


def filter_heart_rate_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Elimina muestras claramente anómalas usando el contexto de las anteriores aceptadas.

    Qué hace hoy (implementación real):
        Ordena por tiempo y descarta solo las `sparse_series` fuera del rango
        lógico 35-270 bpm; las `regular_series` e `interval_series` se aceptan
        tal cual, porque vienen de un sensor con muesteo fijo.
    Qué devuelve:
        (aceptadas, rechazadas). Las rechazadas llevan un `reason` y acaban en
        `heart_rate_resume.rejected_samples` (auditable, no se descartan en
        silencio).
    Pendiente:
        El "contexto de las anteriores aceptadas" (detectar saltos bruscos en
        relación con la muestra previa) todavía no está implementado.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for sample in sorted(samples, key=lambda item: item["time"]):
        bpm = int(sample["value"])
        if sample.get("representation") != "sparse_series":
            accepted.append(sample)
            continue

        if bpm < HEART_RATE_MIN_BPM or bpm > HEART_RATE_MAX_BPM:
            rejected.append({**sample, "reason": "out_of_range"})
            continue

        accepted.append(sample)

    return accepted, rejected


def build_heart_rate_resume(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Construye un resumen diario del pulso en bloques de 5 minutos."""
    samples = collect_heart_rate_samples(payload)
    if not samples:
        return None

    accepted_samples, rejected_samples = filter_heart_rate_samples(samples)
    if not accepted_samples:
        return None

    daily_values = group_heart_rate_samples_by_day(accepted_samples)
    daily_buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for sample in accepted_samples:
        day_key = bucket_day(sample["time"])
        bucket_key = bucket_heart_rate_time(sample["time"])
        daily_buckets[day_key][bucket_key].append(sample["value"])

    daily_average_bpm: list[dict[str, Any]] = []
    daily_points: list[dict[str, Any]] = []

    for day_key in sorted(daily_buckets):
        bucket_items = []

        for bucket_key in sorted(daily_buckets[day_key]):
            values = daily_buckets[day_key][bucket_key]
            bucket_average = round(sum(values) / len(values), 1)
            bucket_items.append({"time": bucket_key, "bpm": bucket_average})

        day_values = daily_values.get(day_key, [])
        daily_average = round(sum(day_values) / len(day_values), 1) if day_values else None

        daily_average_bpm.append(
            {
                "date": day_key,
                "bpm": daily_average,
                "samples": len(daily_values.get(day_key, [])),
            }
        )
        daily_points.append(
            {
                "date": day_key,
                "points": bucket_items,
            }
        )

    return {
        "type": "HEART_RATE",
        "bucket_minutes": HEART_RATE_BUCKET_MINUTES,
        "daily_average_bpm": daily_average_bpm,
        "daily_points": daily_points,
        "rejected_samples": [
            {
                "representation": sample["representation"],
                "startTime": sample["startTime"],
                "value": sample["value"],
                "reason": sample["reason"],
            }
            for sample in rejected_samples
        ],
    }


def build_heart_rate_raw(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Construye la lista de muestras de pulso en bruto (medición a medición).

    Qué hace:
        Conserva cada medición de frecuencia cardíaca con su timestamp original,
        sin agrupar en bloques de 5 minutos.
    Qué recibe:
        El payload original (dict) con los records a procesar.
    Qué devuelve:
        Lista de dicts {"time": ISO, "value": int} ordenados por tiempo,
        o [] si no hay muestras válidas.
    Hipótesis/metodología:
        El análisis fisiológico profundo requiere resolución temporal de evento.
        Se mantienen las muestras que ya pasan el filtro estándar del proyecto
        (mismas reglas que heart_rate_resume: rango 35-270 bpm y fuentes permitidas).
    Si faltan datos:
        Devuelve [] (lista vacía).
    Limitaciones:
        No contiene las muestras rechazadas por el filtro de rango lógico.
    """
    samples = collect_heart_rate_samples(payload)
    accepted, _ = filter_heart_rate_samples(samples)
    serialized: list[dict[str, Any]] = []
    for s in sorted(accepted, key=lambda item: item["time"]):
        serialized.append(
            {
                "time": s["startTime"],
                "value": int(s["value"]),
            }
        )
    return serialized


def group_heart_rate_samples_by_day(samples: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Agrupa muestras aceptadas de pulso por día UTC."""
    heart_rate_by_day: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        heart_rate_by_day[bucket_day(sample["time"])].append(int(sample["value"]))
    return heart_rate_by_day


def collect_blood_oxygen_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Recoge muestras BLOOD_OXYGEN en formato de punto individual."""
    records = payload.get("records")
    if not isinstance(records, list):
        return []

    collected: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        if record.get("type") != BLOOD_OXYGEN_TYPE:
            continue

        if record.get("representation") != "point":
            continue

        start_time = record.get("startTime")
        value = record.get("value")
        if not start_time or not isinstance(value, dict):
            continue

        numeric_value = value.get("numericValue")
        if numeric_value is None:
            continue

        sample_time = parse_iso_datetime(start_time)
        collected.append(
            {
                "time": sample_time,
                "date": bucket_day(sample_time),
                "value": float(numeric_value),
            }
        )

    return collected


def build_blood_oxygen_resume(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Construye un resumen diario de BLOOD_OXYGEN."""
    samples = collect_blood_oxygen_samples(payload)
    if not samples:
        return None

    daily_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"min": None, "max": None, "sum": 0.0, "measurements": 0}
    )

    for sample in samples:
        day_key = sample["date"]
        value = float(sample["value"])
        stats = daily_stats[day_key]

        stats["min"] = value if stats["min"] is None else min(stats["min"], value)
        stats["max"] = value if stats["max"] is None else max(stats["max"], value)
        stats["sum"] = float(stats["sum"]) + value
        stats["measurements"] = int(stats["measurements"]) + 1

    daily_summary: list[dict[str, Any]] = []
    for day_key in sorted(daily_stats):
        stats = daily_stats[day_key]
        measurements = int(stats["measurements"])
        average = round(float(stats["sum"]) / measurements, 1) if measurements else None

        daily_summary.append(
            {
                "date": day_key,
                "min": round(float(stats["min"]), 1) if stats["min"] is not None else None,
                "max": round(float(stats["max"]), 1) if stats["max"] is not None else None,
                "avg": average,
                "measurements": measurements,
            }
        )

    return {
        "type": BLOOD_OXYGEN_TYPE,
        "daily_summary": daily_summary,
    }

	
def extract_record_day(record: dict[str, Any]) -> str | None:
    """Devuelve el día UTC asociado a un registro."""
    for key in ("startTime", "time", "endTime"):
        raw_value = record.get(key)
        if raw_value:
            return bucket_day(parse_iso_datetime(raw_value))
    return None
def extract_record_range(record: dict[str, Any]) -> tuple[datetime, datetime] | None:
    """Devuelve el rango temporal de un registro si existe."""
    start_time = record.get("startTime") or record.get("time")
    end_time = record.get("endTime") or record.get("time") or start_time
    if not start_time or not end_time:
        return None
    return parse_iso_datetime(start_time), parse_iso_datetime(end_time)
def extract_numeric_record_value(record: dict[str, Any]) -> float | None:
    """Normaliza el valor numérico de un registro."""
    value = record.get("value")
    if isinstance(value, dict):
        for key in ("numericValue", "value"):
            candidate = value.get(key)
            if candidate is not None:
                return float(candidate)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_workout_energy_burned(record: dict[str, Any]) -> float | None:
    """Saca las calorías de un WORKOUT si vienen dentro del objeto `value`."""
    value = record.get("value")
    if not isinstance(value, dict):
        return None

    candidate = value.get("totalEnergyBurned")
    if isinstance(candidate, (int, float)):
        return float(candidate)
    return None


def add_interval_value_by_hour(
    hourly_totals: dict[str, float],
    start_dt: datetime,
    end_dt: datetime,
    value: float,
) -> None:
    """Reparte un valor entre las horas UTC que cruza el intervalo."""
    start_utc = start_dt.astimezone(timezone.utc)
    end_utc = end_dt.astimezone(timezone.utc)
    duration_seconds = (end_utc - start_utc).total_seconds()

    if duration_seconds <= 0:
        hourly_totals[bucket_hour(start_utc)] += float(value)
        return

    cursor = start_utc
    while cursor < end_utc:
        hour_start = cursor.replace(minute=0, second=0, microsecond=0)
        next_hour = hour_start + timedelta(hours=1)
        segment_end = min(end_utc, next_hour)
        segment_seconds = (segment_end - cursor).total_seconds()
        if segment_seconds > 0:
            hour_key = bucket_hour(cursor)
            hourly_totals[hour_key] += float(value) * (segment_seconds / duration_seconds)
        cursor = segment_end


def add_interval_value_by_day(
    daily_totals: dict[str, float],
    start_dt: datetime,
    end_dt: datetime,
    value: float,
) -> None:
    """Reparte un valor entre los días UTC que cruza el intervalo."""
    start_utc = start_dt.astimezone(timezone.utc)
    end_utc = end_dt.astimezone(timezone.utc)
    duration_seconds = (end_utc - start_utc).total_seconds()

    if duration_seconds <= 0:
        daily_totals[bucket_day(start_utc)] += float(value)
        return

    cursor = start_utc
    while cursor < end_utc:
        day_start = cursor.replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = day_start + timedelta(days=1)
        segment_end = min(end_utc, next_day)
        segment_seconds = (segment_end - cursor).total_seconds()
        if segment_seconds > 0:
            day_key = bucket_day(cursor)
            daily_totals[day_key] += float(value) * (segment_seconds / duration_seconds)
        cursor = segment_end


def build_daily_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Construye un resumen por día con las métricas principales."""
    records = payload.get("records")
    if not isinstance(records, list):
        return []
    daily: dict[str, dict[str, Any]] = defaultdict(dict)
    def day_bucket(day: str) -> dict[str, Any]:
        if day not in daily:
            daily[day] = {}
        return daily[day]
    def ensure_section(day: str, section: str, initial: dict[str, Any]) -> dict[str, Any]:
        bucket = day_bucket(day)
        if section not in bucket:
            bucket[section] = initial
        return bucket[section]
    step_intervals = collect_step_intervals(payload)
    if step_intervals:
        selected_step_intervals = select_non_overlapping_step_intervals(step_intervals)
        step_totals: dict[str, int] = defaultdict(int)
        for interval in selected_step_intervals:
            step_totals[bucket_day(interval["start"])] += int(interval["value"] or 0)
        for day, total_steps in step_totals.items():
            ensure_section(day, "activity", {"steps": 0})["steps"] += total_steps
    heart_rate_samples = filter_heart_rate_samples(collect_heart_rate_samples(payload))[0]
    if heart_rate_samples:
        heart_rate_by_day = group_heart_rate_samples_by_day(heart_rate_samples)
        for day, values in heart_rate_by_day.items():
            ensure_section(day, "heart_rate", {"avg_bpm": None, "samples": 0})
            daily[day]["heart_rate"] = {
                "avg_bpm": round(sum(values) / len(values), 1),
                "samples": len(values),
            }
	
    if CURRENT_PLATFORM == "ios":
        hrv_by_day: dict[str, list[float]] = defaultdict(list)
        resting_by_day: dict[str, list[float]] = defaultdict(list)

        for record in records:
            if not isinstance(record, dict):
                continue
            if record.get("representation") != "point":
                continue

            day = extract_record_day(record)
            value = extract_numeric_record_value(record)
            if day is None or value is None:
                continue

            if record.get("type") == HRV_TYPE:
                hrv_by_day[day].append(float(value))
            elif record.get("type") == "RESTING_HEART_RATE":
                resting_by_day[day].append(float(value))

        for day, values in hrv_by_day.items():
            daily[day]["heart_rate_variability"] = {
                "avg_ms": round(sum(values) / len(values), 1),
                "samples": len(values),
            }

        for day, values in resting_by_day.items():
            ensure_section(day, "heart_rate", {"avg_bpm": None, "samples": 0})
            daily[day]["heart_rate"]["resting_bpm"] = round(sum(values) / len(values), 1)

    blood_oxygen_samples = collect_blood_oxygen_samples(payload)
    if blood_oxygen_samples:
        blood_oxygen_by_day: dict[str, list[float]] = defaultdict(list)
        for sample in blood_oxygen_samples:
            blood_oxygen_by_day[sample["date"]].append(float(sample["value"]))
        for day, values in blood_oxygen_by_day.items():
            daily[day]["blood_oxygen"] = {
                "min": round(min(values), 1),
                "max": round(max(values), 1),
                "avg": round(sum(values) / len(values), 1),
                "measurements": len(values),
            }

    calorie_totals_by_day: dict[str, float] = defaultdict(float)
    calorie_totals_by_hour: dict[str, float] = defaultdict(float)

    for record in records:
        if not isinstance(record, dict):
            continue
        record_type = record.get("type")

        if CURRENT_PLATFORM == "ios":
            tipos_calorias = {
                "ACTIVE_ENERGY_BURNED",
                "BASAL_ENERGY_BURNED",
            }
        else:
            tipos_calorias = {
                "TOTAL_CALORIES_BURNED",
            }

        if record_type in tipos_calorias:
            intervals = record.get("intervals")
            if not isinstance(intervals, list):
                continue

            for interval in intervals:
                start_time = interval.get("startTime")
                end_time = interval.get("endTime")
                value = interval.get("value")

                if start_time is None or end_time is None or value is None:
                    continue

                start_dt = parse_iso_datetime(start_time)
                end_dt = parse_iso_datetime(end_time)

                add_interval_value_by_day(
                    calorie_totals_by_day,
                    start_dt,
                    end_dt,
                    float(value),
                )
                add_interval_value_by_hour(
                    calorie_totals_by_hour,
                    start_dt,
                    end_dt,
                    float(value),
                )

            continue
        # Distancia recorrida (Android: DISTANCE_DELTA / iOS: DISTANCE_WALKING_RUNNING).
        # AVISO: `day` y `value` se calculan aquí pero NO se acumulan en ningún
        # sitio (no hay `distance_totals_by_day` análogo a `calorie_totals_by_day`),
        # así que `activity.distance_km` nunca llega a la salida. El `continue`
        # final evita además que el registro caiga en los demás tipos. Para
        # activar la distancia hay que añadir el acumulador y escribirlo en el
        # bucle final de días.
        if CURRENT_PLATFORM == "ios":
            distance_types = {"DISTANCE_WALKING_RUNNING"}
        else:
            distance_types = {DISTANCE_DELTA_TYPE}
        if record_type in distance_types:
            if record.get("representation") == "point":
                day = extract_record_day(record)
                value = extract_numeric_record_value(record)
                if day is None or value is None:
                    continue
            elif record.get("representation") == "interval_series":
                intervals = record.get("intervals", [])
                for interval in intervals:
                    start_dt = parse_iso_datetime(interval["startTime"])
                    day = bucket_day(start_dt)
                    value = float(interval["value"])
            continue
        if record_type == WORKOUT_TYPE:
            record_range = extract_record_range(record)
            value = extract_workout_energy_burned(record)
            if record_range is None or value is None:
                continue
            start_dt, end_dt = record_range
            minutes = max((end_dt - start_dt).total_seconds() / 60.0, 0.0)
            add_interval_value_by_day(calorie_totals_by_day, start_dt, end_dt, value)
            add_interval_value_by_hour(calorie_totals_by_hour, start_dt, end_dt, value)
            day = bucket_day(start_dt)
            workouts = ensure_section(day, "workouts", {"total_workout_minutes": 0, "workout_count": 0})
            workouts["total_workout_minutes"] = round(float(workouts.get("total_workout_minutes", 0)) + minutes, 1)
            workouts["workout_count"] = int(workouts.get("workout_count", 0)) + 1
            continue
        if record_type == BODY_TEMPERATURE_TYPE:
            day = extract_record_day(record)
            value = extract_numeric_record_value(record)
            if day is None or value is None:
                continue
            bucket = ensure_section(day, "body_temperature", {"_values": []})
            bucket.setdefault("_values", []).append(value)
            continue
        if record_type == STRESS_TYPE:
            day = extract_record_day(record)
            value = extract_numeric_record_value(record)
            if day is None or value is None:
                continue
            bucket = ensure_section(day, "stress", {"_values": []})
            bucket.setdefault("_values", []).append(value)
            continue
        if record_type == SLEEP_SESSION_TYPE or record_type in SLEEP_PHASE_TYPES:
            day = extract_record_day(record)
            value = extract_numeric_record_value(record)
            record_range = extract_record_range(record)
            if day is None or value is None or record_range is None:
                continue
            start_dt, end_dt = record_range
            sleep = ensure_section(
                day,
                "sleep",
                {
                    "sleep_start": None,
                    "sleep_end": None,
                    "total_sleep_minutes": 0,
                    "sleep_efficiency": None,
                    "deep_minutes": 0,
                    "light_minutes": 0,
                    "rem_minutes": 0,
                    "awake_minutes": 0,
                    "_span_start": None,
                    "_span_end": None,
                },
            )
            sleep["_span_start"] = start_dt if sleep["_span_start"] is None else min(sleep["_span_start"], start_dt)
            sleep["_span_end"] = end_dt if sleep["_span_end"] is None else max(sleep["_span_end"], end_dt)
            sleep["sleep_start"] = sleep["_span_start"].astimezone(timezone.utc).strftime("%H:%M")
            sleep["sleep_end"] = sleep["_span_end"].astimezone(timezone.utc).strftime("%H:%M")
            if record_type == SLEEP_SESSION_TYPE:
                sleep["total_sleep_minutes"] = int(float(sleep.get("total_sleep_minutes", 0)) + value)
            elif record_type == "SLEEP_DEEP":
                sleep["deep_minutes"] = int(float(sleep.get("deep_minutes", 0)) + value)
            elif record_type == "SLEEP_LIGHT":
                sleep["light_minutes"] = int(float(sleep.get("light_minutes", 0)) + value)
            elif record_type == "SLEEP_REM":
                sleep["rem_minutes"] = int(float(sleep.get("rem_minutes", 0)) + value)
            elif record_type == "SLEEP_AWAKE":
                sleep["awake_minutes"] = int(float(sleep.get("awake_minutes", 0)) + value)
            continue
    for day, sections in daily.items():
        if "body_temperature" in sections:
            values = sections["body_temperature"].pop("_values", [])
            sections["body_temperature"] = {"temp_avg": round(sum(values) / len(values), 1)} if values else {}
        if "stress" in sections:
            values = sections["stress"].pop("_values", [])
            sections["stress"] = {
                "stress_avg": round(sum(values) / len(values), 1) if values else None,
                "stress_max": round(max(values), 1) if values else None,
            }
        if "sleep" in sections:
            sleep = sections["sleep"]
            span_start = sleep.pop("_span_start", None)
            span_end = sleep.pop("_span_end", None)
            if span_start and span_end:
                # eficiencia = ( min dormido / total tiempo en cama ) * 100
                session_minutes = max((span_end - span_start).total_seconds() / 60.0, 1.0)
                sleep["sleep_efficiency"] = round(float(sleep.get("total_sleep_minutes", 0)) / session_minutes * 100)
            else:
                sleep["sleep_efficiency"] = None
        if "activity" in sections:
            activity = ensure_section(
                day,
                "activity",
                {
                    "steps": 0,
                    "distance_km": 0.0,
                    "calories": 0.0,
                },
            )

            if "distance_km" in activity:
                activity["distance_km"] = round(float(activity.get("distance_km", 0.0)), 2)

            if day in calorie_totals_by_day:
                activity["calories"] = round(float(calorie_totals_by_day[day]), 1)
        hourly_totals_for_day = {
            hour_key: calories
            for hour_key, calories in calorie_totals_by_hour.items()
            if bucket_day(parse_iso_datetime(hour_key)) == day
        }
        if hourly_totals_for_day:
            activity["hourly_calories"] = [
                {"hour": hour, "calories": round(hourly_totals_for_day[hour], 1)}
                for hour in sorted(hourly_totals_for_day)
            ]

    return [
        {"date": day, **sections}
        for day, sections in sorted(daily.items())
    ]

def collect_step_intervals(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Recoge los intervalos STEPS de las dos fuentes que nos interesan."""
    records = payload.get("records")
    if not isinstance(records, list):
        return []

    # `collected` contendrá objetos con fecha parseada (`start`/`end`) y metadatos
    collected: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        if record.get("representation") != "interval_series" or record.get("type") != "STEPS":
            continue

        source_name = record.get("sourceName")

        if CURRENT_PLATFORM != "ios":
            if source_name not in ALLOWED_HEALTH_SOURCES:
                continue

        intervals = record.get("intervals")
        if not isinstance(intervals, list):
            continue

        priority = 0 if source_name == PRIMARY_STEPS_SOURCE else 1

        for interval in intervals:
            if not isinstance(interval, dict):
                continue

            start_time = interval.get("startTime")
            end_time = interval.get("endTime")
            if not start_time or not end_time:
                continue

            collected.append(
                {
                    "sourceName": source_name,
                    "priority": priority,
                    # `start`/`end`: datetime usados para ordenar y detectar gaps/solapamientos
                    "start": parse_iso_datetime(start_time),
                    "end": parse_iso_datetime(end_time),
                    # `startTime`/`endTime`: cadenas originales para salida sin modificar
                    "startTime": start_time,
                    "endTime": end_time,
                    "value": int(interval.get("value", 0) or 0),
                }
            )

    return collected


def select_non_overlapping_step_intervals(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Se queda con los intervalos que no se pisan, priorizando la fuente principal."""
    selected: list[dict[str, Any]] = []

    for interval in sorted(intervals, key=lambda item: (item["priority"], item["start"])):
        if any(intervals_overlap(interval, chosen) for chosen in selected):
            continue
        selected.append(interval)

    return selected


def build_general_resume(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Construye el resumen general de pasos que va antes de records."""
    intervals = collect_step_intervals(payload)
    if not intervals:
        return None

    selected_intervals = select_non_overlapping_step_intervals(intervals)
    daily_totals: dict[str, int] = defaultdict(int)

    for interval in selected_intervals:
        day_key = bucket_day(interval["start"])
        daily_totals[day_key] += interval["value"]

    return {
        "type": "STEPS",
        "steps_per_day": [
            {"date": day, "steps": daily_totals[day]}
            for day in sorted(daily_totals)
        ],
    }


def build_steps_summary(intervals: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume los intervalos de pasos con gaps, totales por día y totales por hora."""
    if not intervals:
        return {
            "has_gap": False,
            "average_gaps_per_day": 0,
            "gaps": [],
            "daily_steps": [],
            "hourly_steps": [],
        }

    # Normalizamos los intervalos en una estructura con `start`/`end` como datetimes
    # para poder ordenarlos y calcular gaps de forma fiable.
    normalized_intervals: list[dict[str, Any]] = []
    for interval in intervals:
        start_time = interval.get("startTime")
        end_time = interval.get("endTime")
        if not start_time or not end_time:
            continue

        normalized_intervals.append(
            {
                "start": parse_iso_datetime(start_time),
                "end": parse_iso_datetime(end_time),
                "startTime": start_time,
                "endTime": end_time,
                "value": interval.get("value", 0),
            }
        )

    normalized_intervals.sort(key=lambda item: item["start"])

    gaps: list[dict[str, Any]] = []
    daily_totals: dict[str, int] = defaultdict(int)
    hourly_totals: dict[str, int] = defaultdict(int)
    daily_gaps: dict[str, list[dict[str, Any]]] = defaultdict(list)

    previous_interval = None
    for interval in normalized_intervals:
        steps_value = int(interval["value"] or 0)

        day_key = bucket_day(interval["start"])
        hour_key = bucket_hour(interval["start"])
        daily_totals[day_key] += steps_value
        hourly_totals[hour_key] += steps_value

        if previous_interval is not None:
            gap_seconds = (interval["start"] - previous_interval["end"]).total_seconds()
            if gap_seconds >= GAP_THRESHOLD_SECONDS:
                gap_day = bucket_day(previous_interval["end"])
                gap_entry = {
                    "from": previous_interval["endTime"],
                    "to": interval["startTime"],
                    "gap_seconds": gap_seconds,
                }
                # Guardamos cada gap detectado para incluir en el resumen y en la estadística diaria
                gaps.append(gap_entry)
                daily_gaps[gap_day].append(gap_entry)

        previous_interval = interval

    day_keys = sorted(daily_totals)
    average_gaps_per_day = round(len(gaps) / len(day_keys), 2) if day_keys else 0

    return {
        "has_gap": bool(gaps),
        "average_gaps_per_day": average_gaps_per_day,
        "gaps": gaps,
        "daily_steps": [
            {
                "date": day,
                "steps": daily_totals[day],
                "gaps_today": len(daily_gaps.get(day, [])),
            }
            for day in day_keys
        ],
        "hourly_steps": [
            {"hour": hour, "steps": hourly_totals[hour]}
            for hour in sorted(hourly_totals)
        ],
    }


def add_steps_summary_to_records(payload: dict[str, Any]) -> dict[str, Any]:
    """Añade un resumen antes de los intervalos en los registros STEPS."""
    records = payload.get("records")
    if not isinstance(records, list):
        return payload

    updated_records: list[Any] = []
    for record in records:
        if not isinstance(record, dict):
            updated_records.append(record)
            continue

        is_steps_record = record.get("representation") == "interval_series" and record.get("type") == "STEPS"
        intervals = record.get("intervals")

        if not is_steps_record or not isinstance(intervals, list):
            updated_records.append(record)
            continue

        steps_summary = build_steps_summary(intervals)
        updated_record: dict[str, Any] = {}

        for key, value in record.items():
            if key == "intervals":
                # Insertamos `dataset_summary` con la estadística calculada
                updated_record["dataset_summary"] = steps_summary
                # Conservamos el campo `intervals` original para trazabilidad
                updated_record["intervals"] = value
                continue

            updated_record[key] = value

        if "intervals" not in updated_record:
            updated_record["dataset_summary"] = steps_summary

        updated_records.append(updated_record)

    updated_payload = dict(payload)
    updated_payload["records"] = updated_records
    return updated_payload


def output_path_for(source_path: Path) -> Path:
    """Genera el nombre del archivo de salida en la misma carpeta."""
    return source_path.with_name(f"{source_path.stem}_processed{source_path.suffix}")


def iter_source_files(root: Path) -> list[Path]:
    """Devuelve un único archivo o todos los JSON de una carpeta."""
    if root.is_file():
        return [root]

    return sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() == ".json"
        and not candidate.name.lower().endswith("_processed.json")
    )


def process_file(
    source_path: Path,
    participant_id: str | None,
    phone_processed: dict[str, Any] | None = None,
) -> Path:
    """Procesa un solo archivo JSON y guarda la versión resultante."""
    payload = read_json(source_path)

    enriched = build_output_payload(
        source_path,
        payload,
        participant_id,
        phone_processed=phone_processed,
    )

    enriched = add_steps_summary_to_records(enriched)
    enriched = prune_summary_records(enriched)

    target_path = output_path_for(source_path)
    write_json(target_path, enriched)
    return target_path


def process_path(
    root: Path,
    participant_id: str | None,
    phone_processed: dict[str, Any] | None = None,
) -> list[Path]:
    """Procesa un archivo o una carpeta completa y devuelve las rutas creadas."""

    processed_files: list[Path] = []
    for source_path in iter_source_files(root):
        processed_files.append(
            process_file(
                source_path,
                participant_id,
                phone_processed=phone_processed,
            )
        )
    return processed_files


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Enriquece archivos JSON de salud con una cabecera inferible.",
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
    parser.add_argument(
        "--phone-processed",
        default=None,
        help="Ruta al JSON de uso del móvil ya procesado (*_phone_usage_processed.json). "
        "Si se indica, el mismo paso calcula el bloque deep_analysis con los datos de sesiones.",
    )
    return parser.parse_args()

#   python process_health_json.py health/260516_1728_health.json

#   procesar todos los JSON
#   python process_health_json.py health/

#   indicar un identificador user
#   python process_health_json.py health/260516_1728_health.json --participant-id tu_id

def main() -> int:
    args = parse_args()
    source = Path(args.path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"No existe la ruta indicada: {source}")

    phone_processed = None
    if args.phone_processed:
        phone_path = Path(args.phone_processed).expanduser().resolve()
        if not phone_path.exists():
            raise FileNotFoundError(f"No existe el JSON de móvil indicado: {phone_path}")
        phone_processed = read_json(phone_path)

    processed_files = process_path(
        source,
        args.participant_id,
        phone_processed=phone_processed,
    )

    for processed_path in processed_files:
        print(f"Archivo procesado: {processed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
