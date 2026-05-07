# Detección de Comportamiento Agresivo — Flask App

## Estructura
```
agresion_app/
├── app.py                          ← Servidor Flask
├── infer_video.py                  ← COPIA AQUÍ tu script de inferencia
├── lstm_tsm.py                     ← COPIA AQUÍ tu archivo de modelos
├── bilstm_f3_lr1e-04_bs32_T40_best.pt  ← COPIA AQUÍ tu checkpoint
├── requirements.txt
├── templates/
│   └── index.html
└── uploads/                        ← Videos temporales (se auto-limpian)
```

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
```bash
python app.py
# Abrir: http://localhost:5000
```

## Variables de entorno opcionales
```bash
CHECKPOINT_PATH=otro_modelo.pt MODEL_ARCH=tsm TARGET_T=30 python app.py
```

## Notas
- Copia `infer_video.py`, `lstm_tsm.py` y el checkpoint `.pt` en esta carpeta antes de arrancar.
- El video se elimina del servidor automáticamente tras cada análisis.
- Máximo 200 MB por video (configurable en app.py → MAX_CONTENT_MB).
