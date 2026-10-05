#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_deep_analysis: añade el bloque `deep_analysis` a un *_processed.json.

Desde la integración en process_health_json.py, este script ya NO es el paso
principal: ahora el bloque `deep_analysis` se calcula en el MISMO paso de
procesado de salud (python process_health_json.py <health> --phone-processed
<phone>), con acceso directo a los records bruto en memoria.

Este script queda como WRAPPER para el caso de que ya existan los JSON
procesados y solo se quiera (re)calcular/insertar el bloque y generar las
gráficas, sin reprocesar el archivo completo.

Requisito:
    El JSON de salud debe tener `heart_rate_raw`. Sin esa clave, las muestras
    de HR salen vacías y el bloque solo contendrá la parte de móvil (con las
    claves de HR a None). Los JSON procesados con versiones antiguas del
    pipeline pueden no tenerla.

Uso:
    python process_deep_analysis.py <health_processed.json> <phone_usage_processed.json>
    python process_deep_analysis.py <health_processed.json> <phone_usage_processed.json> --no-graphs

Principio:
    No sustituye ni rompe el procesamiento existente; solo añade el bloque.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import deep_analysis as da

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GRAPHS_DIR = os.path.join(SCRIPT_DIR, "graficas_deep")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    """Escribe el JSON con sangría estable y salto final, sin tocar lo demás."""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.write("\n")


def insert_deep_analysis(payload: dict, deep_analysis_block: dict) -> dict:
    """Inserta el bloque deep_analysis justo después de daily_summary.

    - Si `daily_summary` existe, se recoloca para que aparezca tras él,
      manteniendo el resto de claves en el orden original.
    - Si no existe (JSON sin daily_summary), se añade al final.
    - Nunca se borra ni se modifica el contenido previo.
    """
    if "daily_summary" not in payload:
        ordered = dict(payload)
        ordered["deep_analysis"] = deep_analysis_block
        return ordered

    ordered: dict = {}
    for key, value in payload.items():
        ordered[key] = value
        if key == "daily_summary":
            ordered["deep_analysis"] = deep_analysis_block
    return ordered


def generate_graphs(deep: dict, hr_samples: list, sessions: list, out_dir: Path) -> None:
    """Genera las 7 gráficas del deep_analysis (import perezoso para no
    depender de matplotlib si solo se quiere el JSON)."""
    from paint_deep_analysis import render_all
    render_all(deep, hr_samples, sessions, out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Añade (o recalcula) el bloque deep_analysis a un *_health_processed.json "
            "sin reprocesar el archivo completo."
        )
    )
    parser.add_argument("health_processed", help="Ruta al JSON de salud procesado")
    parser.add_argument("phone_processed", nargs="?", default=None,
                        help="Ruta al JSON de uso del móvil procesado (opcional)")
    parser.add_argument("--graphs", action="store_true", default=True,
                        help="Genera las gráficas (por defecto activas)")
    parser.add_argument("--no-graphs", action="store_false", dest="graphs")
    parser.add_argument("--graphs-dir", default=DEFAULT_GRAPHS_DIR)
    args = parser.parse_args()

    health_path = Path(args.health_processed).expanduser().resolve()
    if not health_path.exists():
        raise FileNotFoundError(f"No existe: {health_path}")

    health = read_json(health_path)

    phone = None
    if args.phone_processed:
        phone_path = Path(args.phone_processed).expanduser().resolve()
        if not phone_path.exists():
            raise FileNotFoundError(f"No existe: {phone_path}")
        phone = read_json(phone_path)

    print(f"[deep_analysis] Lectura de HR en bruto desde {health_path.name} ...")
    hr_samples = da.load_hr_samples(health)
    sessions = da.load_sessions(phone) if phone else []
    print(f"  HR muestras: {len(hr_samples)} | Sesiones: {len(sessions)}")

    deep = da.build_deep_analysis(health, phone)
    combined = insert_deep_analysis(health, deep)

    write_json(health_path, combined)
    print(f"[deep_analysis] Bloque insertado en {health_path.name}")

    if args.graphs:
        print("[deep_analysis] Generando gráficas ...")
        generate_graphs(deep, hr_samples, sessions, Path(args.graphs_dir))
        print(f"[deep_analysis] Gráficas en {args.graphs_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())