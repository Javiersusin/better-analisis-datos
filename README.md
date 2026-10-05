# Análisis de datos de salud y uso del móvil

Proyecto de tratamiento y análisis exploratorio de datos para un TFG: datos de
salud (frecuencia cardíaca, pasos, calorías, SpO2, sueño), uso del móvil
(sesiones de apps, desbloqueos) y auto-reporte de la app (diario, emoción,
hábito, test diario).

> **Aviso metodológico:** aquí **no se entrena ningún modelo de Machine Learning**
> y **no se diagnostica depresión ni ninguna condición**, ni se afirma causalidad
> entre el uso del móvil y la frecuencia cardíaca. Todo el análisis es
> exploratorio y descriptivo. Las ventanas horarias son una *hipótesis de
> trabajo*, no una medición del sueño real del participante.

---

## 1. Situación del proyecto

**Fase actual: ingeniería de datos (ETL) + análisis circadiano descriptivo.**

| Estado | Qué |
|--------|-----|
| ✅ Hecho | Procesado de JSON crudos de salud → `*_health_processed.json` |
| ✅ Hecho | Procesado/anonimización de JSON crudos de uso del móvil → `*_phone_usage_processed.json` |
| ✅ Hecho | Bloque `deep_analysis` (ventanas circadianas, HR, móvil, eventos compulsivos) integrado en el procesado de salud |
| ✅ Hecho | Tres generadores de gráficas (salud, uso del móvil, análisis profundo) |
| ✅ Hecho | Suite de tests con datos sintéticos (11 tests, en verde) |
| ⬜ Pendiente | Consolidación diaria a tabla (CSV/dataset) juntando salud + móvil + auto-reporte (`appComent/`) |
| ⬜ Pendiente | Modelado / inferencia (fuera del alcance de este repositorio por ahora) |

### Cosas que conviene saber antes de tocar nada

1. **Los `*_processed.json` incluidos no son todos de la misma versión del
   pipeline.** Algunos no tienen `heart_rate_raw` ni `deep_analysis`, y los de
   móvil más antiguos llevan `metrics.app_usage_stats` y no llevan
   `unlock_events` (claves que el código actual ya no emite / ya emite).
   Para tenerlos coherentes hay que regenerarlos (ver §4).
2. **Las carpetas `graficas/`, `graficas_deep/` y `graficas_uso_movil/` son
   salidas regenerables**, no fuentes. Se pueden borrar y volver a crear.
3. **`records` no aparece en el JSON de salud procesado**: `build_output_payload`
   lo elimina al final (se resumen antes en `general_resume`, `heart_rate_resume`,
   `blood_oxygen_resume` y `daily_summary`). Consecuencia: `add_steps_summary_to_records`
   y `prune_summary_records` hoy no encuentran `records` y son no-op en el flujo
   normal (se mantienen por si se reactiva el volcado de records).
4. **`paint_phone_usage.merge_daily_blocks` y `paint_phone_usage.graf_duracion_sesiones`
   están definidas pero no se llaman desde `main()`** (la gráfica 04 usa el mismo
   prefijo que patrones compulsivos, así que habría que renumerar si se activa).
5. Hay imports sin uso (p. ej. `median` en `process_health_json.py`). No afectan
   al resultado; se dejan para no tocar código funcional.
6. **`distance_km` no llega a la salida.** En `build_daily_summary` los records
   de distancia (`DISTANCE_DELTA` / `DISTANCE_WALKING_RUNNING`) se localizan y
   se les calcula `day`/`value`, pero no se suman a ningún acumulador (los de
   calorías sí lo tienen). Por eso `activity` solo lleva `steps`, `calories` y
   `hourly_calories`. Para activarlo hay que añadir el acumulador análogo.

---

## 2. Requisitos

- Python 3.10+ (probado con 3.14)
- `numpy` (análisis circadiano y gráficas)
- `matplotlib` (gráficas, backend `Agg`: no necesita pantalla)
- `pytest` (tests)

```bash
pip install numpy matplotlib pytest
```

No hay más dependencias: todo el pipeline es Python estándar + numpy/matplotlib.

---

## 3. Estructura del repositorio

```
.
├── process.py                     # Orquestador: móvil + salud en un solo comando
├── process_phone_usage_json.py    # Crudo móvil → *_phone_usage_processed.json (anonimiza)
├── process_health_json.py         # Crudo salud → *_health_processed.json (enriquece)
├── deep_analysis.py               # Módulo de análisis circadiano/fisiológico
├── process_deep_analysis.py       # Wrapper: (re)inserta deep_analysis en un JSON ya procesado
│
├── paint_grph.py                  # Gráficas de salud        → graficas/
├── paint_phone_usage.py           # Gráficas de uso móvil    → graficas_uso_movil/
├── paint_deep_analysis.py         # Gráficas circadianas     → graficas_deep/
│
├── health/                        # JSON crudos de salud (Samsung Health / Health Connect / iOS)
│   └── *_health_processed.json    #   y sus versiones procesadas
├── usomvl/                        # JSON crudos de uso del móvil (Android UsageStats)
│   └── *_phone_usage_processed.json
├── appComent/                     # Auto-reporte de la app (AÚN SIN PROCESAR en este repo)
│   └── YYMMDD_HHMM_{diary,emotion,habit,activity,test}.json
│
├── tests/
│   ├── test_data_builders.py      # Generadores de datos sintéticos (6 casos)
│   └── test_deep_analysis.py      # 11 assertions sobre deep_analysis
│
├── graficas/                      # Salidas (regenerables)
├── graficas_deep/
└── graficas_uso_movil/
```

### Formato de los archivos de datos

| Carpeta | Nombre | Ejemplo |
|---------|--------|---------|
| `health/` | `YYMMDD_HHMM_health.json` | `260516_1728_health.json` |
| `usomvl/` | `YYMMDD_HHMM_phone_usage.json` | `260516_1721_phone_usage.json` |
| `appComent/` | `YYMMDD_HHMM_<tipo>.json` | `260521_1106_emotion.json` |

La fecha del nombre (`260516` = 2026-05-16) es la que se usa como
`timestamp_from_filename` en la cabecera. El contenido se agrupa siempre por
**día UTC**, no por la fecha del nombre.

---

## 4. Cómo se ejecuta

Todo se ejecuta **desde la raíz de este repositorio** (los scripts importan
entre sí por nombre de módulo).

### 4.1 Pipeline completo (recomendado)

```bash
python process.py health/260905_1723_health.json usomvl/260905_2130_phone_usage.json
```

Hace las dos cosas en orden:

1. Procesa el móvil crudo → `usomvl/260905_2130_phone_usage_processed.json`
2. Procesa la salud cruda **ya con** `--phone-processed` implícito →
   `health/260905_1723_health_processed.json` (incluye `deep_analysis`)

Sin segundo argumento solo procesa salud (habrá `deep_analysis` de HR si hay
muestras, pero sin bloque de móvil):

```bash
python process.py health/260905_1723_health.json
```

### 4.2 Paso a paso (equivalente)

```bash
# 1) Solo móvil: crudo → procesado (anonimizado)
python process_phone_usage_json.py usomvl/                 # carpeta entera
python process_phone_usage_json.py usomvl/260516_1721_phone_usage.json

# 2) Salud: crudo → procesado, con deep_analysis usando el móvil ya procesado
python process_health_json.py health/260516_1728_health.json \
    --phone-processed usomvl/260516_1727_phone_usage_processed.json

# Otros flags
python process_health_json.py health/                      # todos los de la carpeta
python process_health_json.py health/x_health.json --participant-id p01
```

### 4.3 Recalcular solo `deep_analysis` sobre un JSON ya procesado

Útil si ya tienes los `*_processed.json` y solo quieres (re)insertar el bloque
y generar sus gráficas, sin reprocesar nada:

```bash
python process_deep_analysis.py health/260516_1728_health_processed.json \
    usomvl/260516_1727_phone_usage_processed.json

python process_deep_analysis.py ... --no-graphs          # solo el JSON
python process_deep_analysis.py ... --graphs-dir salida  # otra carpeta
```

El bloque se coloca **justo después de `daily_summary`** y no borra nada de lo
que ya había.

### 4.4 Gráficas

```bash
# Salud: 6 globales + 1 por día → graficas/
python paint_grph.py health/260516_1728_health_processed.json

# Uso del móvil: 4 por día → graficas_uso_movil/
python paint_phone_usage.py usomvl/260623_1438_phone_usage_processed.json

# Análisis profundo: 5 por día → graficas_deep/
python paint_deep_analysis.py health/260516_1728_health_processed.json \
    usomvl/260516_1727_phone_usage_processed.json
```

Los dos primeros aceptan la ruta por argumento; si no se pasa, usan un archivo
por defecto hardcodeado arriba del script (`DATA_FILE`).

### 4.5 Tests

```bash
python -m pytest tests -v
```

---

## 5. Qué procesa y qué limpia cada script

### `process_phone_usage_json.py` — crudo móvil → procesado

**Entrada:** JSON de Android UsageStats (`records.events`).
**Salida:** `*_phone_usage_processed.json` con una única raíz: `daily_blocks`.

Limpieza / transformación:

- Solo lee `records.events`. **Ignora** `appUsageStats`, `eventStats` y
  `configurationStats` (llegan en el crudo pero no se usan).
- De ~50 tipos de evento de Android se conservan solo 5:
  `ACTIVITY_RESUMED` (inicio de sesión de app), `ACTIVITY_PAUSED` /
  `ACTIVITY_STOPPED` (fin) y `SCREEN_INTERACTIVE` / `KEYGUARD_HIDDEN`
  (desbloqueo). El resto se descarta.
- Descarta eventos con `timestamp` ausente o `<= 0`, y sin `eventTypeName`.
- **Anonimización:** se eliminan del crudo `deviceManufacturer`, `deviceModel`,
  `androidSdk`, `collectorName/Version`, `standbyBucket`,
  `notificationChannelId`, `configuration`… La salida no contiene ninguno.
- **`packageName` / `className` nunca se vuelcan**: se convierten en una
  categoría (`social`, `messaging`, `music`, …) y en la sesión solo se guarda
  `category`. Un paquete desconocido cae en `other`.
- Construye sesiones: se cierran al cambiar de paquete/categoría, o con
  `queryEndMillis` si el evento quedó abierto al final de la ventana.
- `unlock_to_open_seconds`: si entre el último desbloqueo y abrir la app hay
  `0–15 s`, se guarda como métrica de "impulsividad".
- `unlock_count` solo cuenta `KEYGUARD_HIDDEN`. Los `SCREEN_INTERACTIVE` se
  conservan para la lógica de sesión y para `unlock_events`, pero no se cuentan
  como desbloqueo (evita duplicar el mismo gesto).
- Sesiones que cruzan **medianoche UTC** se parten en segmentos de un día.
- El tiempo de cada sesión se reparte proporcionalmente entre las horas que
  cruza (`usage_by_hour`).
- Calcula `metrics` (pantalla, desbloqueos, cambios de app, sesiones cortas,
  % por categoría, aperturas) y `compulsive_metrics` (reaperturas rápidas
  ≤ 10 s, presión de sesiones cortas, desbloqueo→social).
- Añade la cabecera (`participant_id`, `filename`, `file_type`,
  `timestamp_from_filename`, `window_start`, `window_end`, `parse_status`)
  **dentro de cada bloque diario**.

### `process_health_json.py` — crudo salud → procesado

**Entrada:** JSON de salud (Android/Samsung Health o iOS/HealthKit).
**Salida:** `*_health_processed.json`.

Limpieza / transformación:

- Añade la cabecera `HEADER_KEYS` al principio del documento.
- **Elimina** `windowStart` / `windowEnd` (duplicados de
  `window_start` / `window_end`), `requestedDataTypes` y `exportedDataTypes`.
- **Pasos:** solo se aceptan las fuentes `com.sec.android.app.shealth` y
  `android` (en iOS se acepta cualquier fuente). Si hay intervalos solapados
  entre fuentes, gana la fuente principal (`select_non_overlapping_step_intervals`).
- **Frecuencia cardíaca:** fuentes permitidas (en iOS, cualquiera) y filtro de
  rango lógico **35–270 bpm** aplicado a las `sparse_series` (las
  `regular_series` e `interval_series` se aceptan tal cual). Lo rechazado
  queda listado en `heart_rate_resume.rejected_samples` para poder auditar el
  filtro: no se pierde información silenciosamente.
- La HR sale en **dos formas a propósito**:
  - `heart_rate_resume`: media por buckets de 5 minutos (para gráficas de resumen).
  - `heart_rate_raw`: medición a medición con su timestamp original
    (resolución de evento, imprescindible para `deep_analysis`).
- **SpO2:** solo puntos con `numericValue`; resumen diario min/max/media.
- **Calorías:** se reparten proporcionalmente entre los días y horas que cruza
  cada intervalo (no se asignan al día de inicio). En iOS se usan
  `ACTIVE_ENERGY_BURNED` + `BASAL_ENERGY_BURNED`, en Android
  `TOTAL_CALORIES_BURNED`; los `WORKOUT` también suman y alimentan la sección
  `workouts`.
- **Distancia:** se localizan `DISTANCE_DELTA` (Android) y
  `DISTANCE_WALKING_RUNNING` (iOS) pero **hoy no se acumulan** (ver §1.7).
- **Sueño:** agrupa `SLEEP_SESSION` y fases (`SLEEP_DEEP/LIGHT/REM/AWAKE…`);
  `sleep_efficiency = minutos dormidos / minutos en cama × 100`.
- HRV (`HEART_RATE_VARIABILITY_SDNN`) y FC en reposo solo en iOS.
- **Reordena `records`** con los `STEPS` al principio y después **`records`
  se elimina de la salida** (ver §1.4).
- Genera `daily_summary` (uno por día UTC) y, opcionalmente, `deep_analysis`.
- Copia `deep_analysis` resumido dentro de cada día de `daily_summary`
  (`hr_stats`, `mobile_sleep_window`, `hr_delta_events`) para leerlo sin
  recorrer el bloque global.

### `process.py` — orquestador

No tiene lógica propia: valida rutas, llama a `process_phone_usage_json.process_file`
y luego a `process_health_json.build_output_payload` + `add_steps_summary_to_records`
+ `prune_summary_records`, y guarda. Sirve para no repetir órdenes.

### `deep_analysis.py` — módulo de análisis

No escribe nada por sí solo; es la biblioteca que llaman
`process_health_json.py` y `process_deep_analysis.py`. Ver §6.

### `process_deep_analysis.py` — wrapper

Lee un `*_health_processed.json` ya existente, recálcula el bloque
`deep_analysis` (desde `heart_rate_raw` y del móvil procesado), lo inserta tras
`daily_summary` reordenando claves sin borrar nada, reescribe el mismo archivo
y opcionalmente genera las gráficas.

### Scripts `paint_*.py`

Solo leen JSON y escriben PNG. No modifican datos.

---

## 6. Análisis circadiano (`deep_analysis`)

### Ventanas horarias de referencia (UTC)

| Ventana | Rango | Uso |
|---------|-------|-----|
| `sleep_window` | 00:00 – 05:00 | Detección de uso nocturno |
| `wake_window` | 09:00 – 22:00 | Referencia de vigilia |
| `pre_sleep_window` | 22:00 – 00:00 | Uso antes de dormir |

**05:00–09:00 y 22:00–00:00 no se clasifican automáticamente** como sueño ni
como vigilia. El JSON lo deja explícito en `unclassified_note`. Esto es
deliberado: no se puede afirmar que alguien estuviera durmiendo a las 03:00
solo porque sea de noche.

### Qué calcula

1. **`data_quality`** — recuentos y días con datos (qué hay y qué no).
2. **`mobile_sleep_window`** — minutos/sesiones de móvil dentro de 00:00–05:00,
   con desglose `per_date`.
3. **`pre_sleep_mobile`** — uso en 22:00–00:00 por categoría + métricas
   compulsivas (reaperturas rápidas, desbloqueos, cambios de app, sesiones < 5 s).
4. **`circadian_heart_rate`** — perfil horario de HR separando
   `all_days` / `weekday` / `weekend`, con estadísticas en bruto y **filtradas
   P10–P90**, hora del mínimo y máximo diario, y métrica `per_day`.
5. **`heart_rate_delta_events`** — Δ entre mediciones consecutivas (resolución
   de evento, sin agregar). Tope de `MAX_DELTA_SAMPLE_EVENTS = 1000` eventos
   guardados en el JSON para no inflarlo.
6. **`heart_rate_mobile_events`** — por cada sesión de móvil: HR antes /
   durante / después (búsqueda ±5 min, `SYNC_HR_WINDOW_SECONDS`) y Δ, agregado
   por categoría.
7. **`compulsive_events_heart_rate`** — lo mismo para eventos puntuales:
   `unlock`, `rapid_reopening`, `short_session`, `app_switch`.
8. **`before_sleep`** — combinación móvil + HR restringida a 22:00–00:00.

### Estructura del bloque

```json
{
  "time_windows": { ... },
  "unclassified_note": "...",
  "data_quality": {
    "has_heart_rate_data": true,
    "has_mobile_data": true,
    "hr_raw_count": 9120,
    "mobile_sessions_count": 684,
    "unlock_events_count": 129,
    "days_with_hr": ["2026-05-09", "..."]
  },
  "mobile_sleep_window": { ... },
  "pre_sleep_mobile": { "mobile": { ... }, "compulsive_behavior": { ... } },
  "circadian_heart_rate": { "all_days": { ... }, "weekday": { ... }, "weekend": { ... } },
  "heart_rate_delta_events": { ... },
  "heart_rate_mobile_events": { ... },
  "compulsive_events_heart_rate": { ... },
  "before_sleep": { ... }
}
```

Si no hay HR ni móvil, se devuelve el mismo esqueleto con claves canónicas a
`null`/`false` (`data_quality.has_heart_rate_data = false`): **no se inventan
métricas**.

### Gráficas de `paint_deep_analysis.py` (5 por día)

| # | Archivo | Contenido |
|---|---------|-----------|
| 1 | `deep_01_mobile_usage_<fecha>.png` | Uso por hora + panel nocturno |
| 2 | `deep_02_day_vs_night_<fecha>.png` | Día (09–22) vs noche (00–05 / 22–00) |
| 3 | `deep_03_temporal_effect_<fecha>.png` | ΔHR por categoría + event study |
| 4 | `deep_04_night_hr_sessions_<fecha>.png` | HR nocturna + sesiones superpuestas |
| 5 | `deep_05_circadian_rhythm_<fecha>.png` | Perfil horario de HR del día |

Filtros aplicados solo en el dibujo (los datos del JSON no se tocan): se excluye
la categoría `other`, se ignoran sesiones < 5 s y el gráfico 5 exige ≥ 30
muestras por hora.

---

## 7. Decisiones de tratamiento del código y los datos

1. **Todo en UTC.** Todos los timestamps del proyecto son UTC (sufijo `Z`) y
   todos los cortes por día/hora se hacen en UTC. Esto evita mezclar husos
   distintos entre fuentes, pero significa que "día" = día UTC.
2. **Anonimización por diseño en el móvil.** El `*_phone_usage_processed.json`
   solo contiene categorías, tiempos y métricas: ni paquetes de app, ni modelo
   de dispositivo, ni SDK, ni identificador de recolector.
3. **`participant_id` no se infiere.** `infer_participant_id` solo devuelve
   valor si se pasa `--participant-id`; no se lee el nombre de la carpeta
   (evita colar un hash o un nombre personal en la salida).
4. **Aditivo por defecto.** Se añaden bloques (`general_resume`,
   `heart_rate_resume`, `heart_rate_raw`, `blood_oxygen_resume`,
   `daily_summary`, `deep_analysis`) sin reescribir el resto. Lo que se
   elimina es explícito y comentado (`windowStart/windowEnd`,
   `requestedDataTypes`, `exportedDataTypes`, `records`).
5. **No se relee el JSON.** El bloque `deep_analysis` se calcula en el mismo
   paso del procesado de salud, reutilizando las muestras de HR ya en memoria
   (`hr_samples_override`), en vez de volver a abrir el fichero.
6. **El filtro P10–P90 es solo de análisis.** Recorta colas para el perfil
   circadiano, pero los originales siguen intactos en `heart_rate_raw` y se
   publican los cortes (`lower_cutoff` / `upper_cutoff`).
7. **Los huecos se declaran, no se rellenan.** 05–09 y 22–00 no se clasifican;
   cuando faltan datos, las claves salen en `null`/`false` y los recuentos en
   `data_quality`.
8. **`other` se dibuja pero se excluye de los gráficos analíticos.** Es el
   cajón de apps del sistema; incluirlas en el ΔHR daría ruido, así que
   `paint_deep_analysis` la filtra (y filtra sesiones < 5 s).
9. **Coherencia de umbrales entre módulos.** `RAPID_REOPENING_WINDOW_SECONDS = 10`
   y `SHORT_SESSION_THRESHOLD_SECONDS = 5` están replicados en
   `process_phone_usage_json.py` y `deep_analysis.py`: si se cambia uno, hay
   que cambiarlo en ambos.
10. **Compatibilidad con JSON antiguos.** Si la clave ya existe se respeta su
    valor (`first_existing_value`), y `insert_deep_analysis` reordena sin borrar.
11. **`process_deep_analysis.py` se mantiene como wrapper** en lugar de
    eliminarlo: permite reprocesar sin repetir todo el pipeline.
12. **Backends sin pantalla.** Los scripts de gráficas fuerzan
    `matplotlib.use("Agg")` para que funcionen en servidores/CI.
13. **No se rompe lo existente.** Los scripts de procesado siguen siendo
    usables por separado; `process.py` es una capa opcional encima.

---

## 8. Tests

```bash
python -m pytest tests -v
```

`tests/test_data_builders.py` construye 6 escenarios sintéticos con la misma
estructura que los JSON reales; `tests/test_deep_analysis.py` comprueba con
assertions (no con prints) que el análisis los detecta:

| Test | Qué verifica |
|------|--------------|
| `test_caso1_social_detecta_incremento_hrtemporal` | Sesión social con HR antes=60, durante≈63, después=68 |
| `test_caso2_music_no_es_detectado_como_gran_cambio` | Δ ≤ 1 sin cambio relevante |
| `test_caso3_compulsivo_detecta_patron_y_relacion_hr` | ≥3 reaperturas rápidas y algún ΔHR > 0 |
| `test_caso4_uso_social_before_sleep_calculado` | 35 min pre-sueño (30 social + 5 music) |
| `test_caso5_uso_durante_ventana_sueno` | 25 min y 2 sesiones en 00:00–05:00 |
| `test_caso6_p10_p90_elimina_extremos_y_conserva_forma` | 35 y 130 bpm fuera, media conservada |
| `test_weekday_weekend_separacion` | Miércoles en `weekday`, sábado en `weekend` |
| `test_hour_of_daily_min_max` | Máximo ~16–17 h, mínimo ~3–4 h |
| `test_hr_delta_events` | 4 deltas entre 5 mediciones |
| `test_sin_datos_hr_ni_movil` | Sin datos → flags en `False`, nada inventado |
| `test_caso1_social_en_before_sleep` | La sesión de las 22:15 entra en `before_sleep` |

**Estado: 11/11 en verde** (`python -m pytest tests -q` → `11 passed`).

---

## 9. Pendientes / cómo continuar

1. **Dataset diario.** Escribir de nuevo `build_dataset.py` (+ su test) que
   lea `health/*_processed.json`, `usomvl/*_processed.json` y `appComent/*.json`
   y produzca una fila por fecha con: HR (media/min/max/nocturno), pasos,
   minutos de pantalla por ventana, desbloqueos, sesiones cortas, emoción
   dominante, valencia, palabras del diario, flags de hábito y respuesta al
   test diario. Los nombres de funciones esperadas están en
   `.pytest_cache/v/cache/nodeids`.
2. **Integrar `appComent/`.** Hoy está sin procesar. Hay 5 tipos:
   `emotion` (emotion/intensity/freeText), `diary` (title/body o plantilla
   `templateResponses`), `habit`, `activity` y `test` (`start_of_day` /
   `end_of_day`). Los ficheros están en UTF-8 válido; si la consola muestra
   acentos rotos es la página de códigos de la terminal, no el archivo.
3. **Regenerar los `*_processed.json`** con el código actual para que todos
   tengan la misma estructura (ver §1.2).
4. **Decidir el destino de `records`** (ver §1.4): si se quiere conservar
   `dataset_summary` en la salida, hay que mover el `pop("records")`.
5. **Modelado**, cuando haya dataset: fuera del alcance de este repositorio
   hasta ahora.

---

## 10. Limitaciones

- Las ventanas horarias son fijas y en UTC: **no son una medición del sueño**.
- ΔHR alrededor de una sesión describe **relación temporal, no causalidad**.
  Un aumento de 1–2 bpm no implica respuesta fisiológica real.
- El auto-reporte (`appComent/`) es de un solo participante y con entradas de
  prueba; no sirve para generalizar.
- El filtro de fuentes de pasos/HR está pensado para los export concretos de
  este proyecto; con otra exportación puede que haya que ampliar
  `ALLOWED_HEALTH_SOURCES` o `PHONE_CATEGORY_RULES`.
- `PHONE_CATEGORY_RULES` es una lista cerrada de prefijos: cualquier app nueva
  cae en `other` hasta que se añada.
