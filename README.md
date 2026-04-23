# Soccernet Tracking Analysis

Trabajo de Fin de Grado — Ingeniería del Software  
Análisis automático de partidos de fútbol mediante técnicas de visión artificial sobre el dataset SoccerNet.

---

## Descripción

Este proyecto explora el uso de algoritmos de tracking multiobbjeto (MOT) aplicados a secuencias de vídeo de partidos de fútbol profesional. Utilizando el dataset **SoccerNet-Tracking**, se detectan y siguen jugadores y el balón frame a frame, con el objetivo de analizar el rendimiento de distintos algoritmos y generar visualizaciones útiles para el análisis deportivo.

---

## Dataset

Se utiliza el dataset [SoccerNet Tracking](https://www.soccer-net.org/tasks/tracking), que proporciona:

- Secuencias de videos en formato MOT (frames individuales)
- Metadatos de cada secuencia (`seqinfo.ini`, `gameinfo.ini`)

> Los datos **no están incluidos** en este repositorio por su tamaño. Consulta `src/descarga_soccernet_tracking.py` para descargarlos.
---

## Funcionalidades implementadas

- Reproducción de secuencias de vídeo a partir de frames individuales
- Visualización de bounding boxes con identificadores únicos por jugador
- Asignación de color único por jugador para facilitar el seguimiento visual
- Detección y diferenciación del balón mediante filtrado por área y proporción geométrica

---

## Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales:

- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics (`ultralytics`) — para experimentos con YOLOv8
- SoccerNet (`SoccerNet`) — para la descarga del dataset

---

## Autor

**cgaleam**  
Grado en Ingeniería del Software  
Trabajo de Fin de Grado — 2025/2026
