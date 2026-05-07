# 🔍 Detección de Comportamiento Agresivo — App Extendida

Extensión de la aplicación Flask original **sin modificar ningún archivo del sistema principal**.

---

## 📁 Estructura completa del proyecto

```
agresion_app/                          ← SISTEMA PRINCIPAL (NO MODIFICAR)
└── agresion_app/
    ├── app.py                         ← Flask app original ✅ INTACTO
    ├── infer_video.py                 ← Pipeline de inferencia ✅ INTACTO
    ├── lstm_tsm.py                    ← Arquitecturas de modelos ✅ INTACTO
    ├── bilstm_f3_lr1e-04_bs32_T40_best.pt   ← Checkpoint PyTorch
    ├── yolov8l-pose.pt                ← Modelo YOLOv8
    ├── model.onnx                     ← [OPCIONAL] Modelo exportado a ONNX
    ├── kp_videos/                     ← Videos generados con keypoints (auto-creado)
    ├── uploads/                       ← Temporal (se auto-limpia)
    ├── requirements.txt               ← Dependencias originales
    └── templates/
        └── index.html                 ← ⚠️ Reemplazar con la versión de agresion_app_v2

agresion_app_v2/                       ← EXTENSIÓN (nuevos archivos)
├── app_extended.py                    ← Punto de entrada: python app_extended.py
├── requirements_extended.txt          ← Dependencias adicionales (solo onnxruntime)
├── README.md                          ← Este archivo
│
├── blueprints/                        ← Módulos Flask por funcionalidad
│   ├── __init__.py
│   ├── history_bp.py                  ← /history — Historial de predicciones
│   ├── stats_bp.py                    ← /stats — Reportes y estadísticas
│   ├── alerts_bp.py                   ← /alerts — Centro de alertas
│   ├── help_bp.py                     ← /help — Centro de ayuda
│   └── onnx_bp.py                     ← /onnx — Inferencia con modelo ONNX
│
├── database/                          ← Capa de datos SQLite
│   ├── __init__.py
│   ├── db.py                          ← CRUD, init_db, save_prediction, get_statistics...
│   └── agresion_data.db               ← Base de datos (se crea automáticamente)
│
└── templates/                         ← Plantillas HTML
    ├── base.html                      ← Layout con navbar global (heredado por todos)
    ├── index.html                     ← Reemplaza al index.html original (con navbar)
    ├── history.html                   ← Historial con tabla, filtros y modal de video KP
    ├── stats.html                     ← Estadísticas con Chart.js
    ├── alerts.html                    ← Centro de alertas
    ├── help.html                      ← Centro de ayuda + FAQ
    └── onnx.html                      ← Página de inferencia ONNX
```

---

## 🚀 Cómo iniciar

### 1. Instalar dependencias adicionales (opcional, solo para ONNX)
```bash
pip install onnxruntime
```

### 2. Copiar plantillas al sistema principal
```bash
cp agresion_app_v2/templates/*.html agresion_app/agresion_app/templates/
```
> ⚠️ El `index.html` extendido reemplaza al original, pero mantiene la misma lógica visual.
> La única diferencia es que ahora hereda de `base.html` (que agrega el navbar).

### 3. Lanzar la app extendida
```bash
# Desde el directorio raíz del proyecto
python agresion_app_v2/app_extended.py

# O con variable de entorno si la carpeta principal está en otra ruta:
MAIN_APP_DIR=/ruta/a/agresion_app/agresion_app python agresion_app_v2/app_extended.py
```

### 4. Abrir en el navegador
```
http://localhost:5000
```

---

## ⚡ Exportar modelo a ONNX

Para usar la página `/onnx`, necesitas exportar tu modelo BiLSTM:

```python
import torch
from lstm_tsm import BiLSTMClassifier

# Cargar modelo entrenado
model = BiLSTMClassifier(input_size=170)
ckpt  = torch.load("bilstm_f3_lr1e-04_bs32_T40_best.pt", map_location="cpu")
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()

# Exportar
dummy = torch.randn(1, 40, 2, 17, 5)   # [batch, T, personas, keypoints, canales]
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}},
    opset_version=14,
)
print("✅ model.onnx generado")
```

Coloca `model.onnx` en la misma carpeta que `app.py`.

---

## 🧩 Funcionalidades agregadas

| Ruta | Funcionalidad |
|------|--------------|
| `/` | Análisis principal (igual que antes + navbar + video KP inline) |
| `/history` | Historial con filtros, paginación, modal de video KP, borrado |
| `/stats` | KPIs, gráficas (dona, línea temporal, diagnósticos) con Chart.js |
| `/alerts` | Centro de alertas con niveles CRITICAL / WARNING / INFO |
| `/help` | Centro de ayuda, guía de inicio rápido, FAQ, referencia técnica |
| `/onnx` | Página de inferencia idéntica usando modelo .onnx en vez de .pt |

### Base de datos (SQLite — `agresion_data.db`)
- **`predictions`**: guarda cada predicción (timestamp, clase, confianza, metadatos, ruta al video KP)
- **`alerts`**: alertas automáticas vinculadas a predicciones (críticas y de baja confianza)

---

## 📝 Notas de diseño

- **El `app.py` original NO se modifica**. `app_extended.py` lo importa y agrega funcionalidades mediante Blueprints.
- La única excepción es la vista `/predict`, que se envuelve (monkey-patch) para interceptar el resultado e guardarlo en DB + generar el video con keypoints — sin cambiar la lógica interna de la función `run_inference`.
- El `index.html` sí se reemplaza, pero el código JS de análisis es idéntico al original; solo se agrega la sección de video KP y el navbar heredado de `base.html`.
- Todas las páginas comparten el mismo sistema de colores CSS (variables `--bg`, `--accent`, `--danger`, etc.) del diseño original.
