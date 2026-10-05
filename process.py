#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process.py - Script unificado de procesamiento.
python process.py health/260905_1723_health.json usomvl/260905_2130_phone_usage.json
Este script hace TODO el pipeline en un solo comando:
  1. Procesa uso del móvil (raw → procesado)
  2. Procesa salud (raw → procesado) + deep_analysis integrado

Así que de un JSON crudo de cada tipo, obtienes todo procesado.

Uso:
  python process.py <health.json> <phone_usage.json>
  python process.py <health.json>  (solo salud, sin deep_analysis de móvil)

No inventa código. Reutiliza las funciones de:
  - process_health_json.py
  - process_phone_usage_json.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importamos las funciones de los otros scripts.
# Python busca en el directorio actual, así que funciona si ejecutamos
# desde la carpeta del proyecto.

# Funciones para procesar salud
from process_health_json import (
    read_json,
    write_json,
    build_output_payload,
    add_steps_summary_to_records,
    prune_summary_records,
    output_path_for as health_output_path,
)

# Funciones para procesar uso del móvil
from process_phone_usage_json import (
    process_file as process_phone_file,
)


def main() -> int:
    """
    Función principal. Recibe los argumentos de la línea de comandos.

    Retorna 0 si todo va bien, 1 si hay error.
    """

    # ─── PASO 0: Leer argumentos ───────────────────────────────────
    # Esperamos: python process.py <health.json> [phone_usage.json]
    if len(sys.argv) < 2:
        print("Uso: python process.py <health.json> [phone_usage.json]")
        print("")
        print("  health.json       → JSON crudo de salud (obligatorio)")
        print("  phone_usage.json  → JSON crudo de uso del móvil (opcional)")
        print("")
        print("Ejemplo:")
        print("  python process.py health/260905_1723_health.json usomvl/260905_2130_phone_usage.json")
        return 1

    # ─── PASO 1: Validar que los archivos existen ──────────────────
    health_path = Path(sys.argv[1]).expanduser().resolve()
    if not health_path.exists():
        print(f"[ERROR] No existe el archivo de salud: {health_path}")
        return 1

    phone_path = None
    if len(sys.argv) >= 3:
        phone_path = Path(sys.argv[2]).expanduser().resolve()
        if not phone_path.exists():
            print(f"[ERROR] No existe el archivo de uso del móvil: {phone_path}")
            return 1

    # ─── PASO 2: Procesar uso del móvil (si se proporcionó) ────────
    # Esto genera el *_phone_usage_processed.json
    phone_processed_data = None
    if phone_path:
        print(f"[1/2] Procesando uso del móvil: {phone_path.name} ...")

        # process_phone_file() hace todo el procesado del móvil
        # y devuelve la ruta del archivo procesado
        processed_phone_path = process_phone_file(phone_path, participant_id=None)
        print(f"      -> Guardado: {processed_phone_path.name}")

        # Cargamos el JSON procesado para pasarlo a salud
        # (necesitamos los daily_blocks con sesiones y desbloqueos)
        phone_processed_data = read_json(processed_phone_path)

    # ─── PASO 3: Procesar salud + deep_analysis ────────────────────
    # Esto genera el *_health_processed.json con todo incluido
    print(f"[2/2] Procesando salud: {health_path.name} ...")

    # Leemos el JSON crudo de salud
    health_payload = read_json(health_path)

    # build_output_payload() hace todo el procesado:
    #   - Extrae HR, pasos, calorías, etc.
    #   - Genera heart_rate_resume, daily_summary, etc.
    #   - Si le pasamos phone_processed, genera deep_analysis también
    enriched = build_output_payload(
        health_path,
        health_payload,
        participant_id=None,
        phone_processed=phone_processed_data,
    )

    # Añade resumen de pasos a records
    # Nota: build_output_payload ya ha borrado `records`, así que hoy estas dos
    # llamadas no modifican nada. Se mantienen por si se decide conservar
    # `records`/`dataset_summary` en la salida (ver README §1.4).
    enriched = add_steps_summary_to_records(enriched)

    # Elimina registros crudos ya resumidos (para ahorrar espacio)
    enriched = prune_summary_records(enriched)

    # Calcula la ruta de salida y guardamos
    target_path = health_output_path(health_path)
    write_json(target_path, enriched)
    print(f"      -> Guardado: {target_path.name}")

    # ─── PASO 4: Resumen ───────────────────────────────────────────
    print("")
    print("=== Procesamiento completado ===")
    print(f"  Salud:     {target_path.name}")

    if phone_path:
        print(f"  Móvil:     {processed_phone_path.name}")

    # Mostramos info básica del deep_analysis si se generó
    deep = enriched.get("deep_analysis")
    if deep:
        dq = deep.get("data_quality", {})
        print(f"  Deep analysis: HR={dq.get('hr_raw_count', 0)} muestras, "
              f"móvil={dq.get('mobile_sessions_count', 0)} sesiones")
    else:
        print("  Deep analysis: no generado (sin datos HR ni móvil)")

    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
