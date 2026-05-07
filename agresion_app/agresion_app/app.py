"""
app.py
══════════════════════════════════════════════════════════════════════════════
Flask — Detección de comportamiento agresivo en video
Usa ONNX Runtime para inferencia (en lugar de PyTorch clf_model).
Uso: python app.py  →  http://localhost:5000
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import uuid
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
#  Importar funciones de inferencia  (infer_video.py debe estar en la misma carpeta)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer_video import (
    extract_frames,
    extract_keypoints,
    add_velocity_channels,
    resize_temporal,
    normalize_pose,
    LABELS,
    DEFAULT_T,
    DEFAULT_N_FRAMES,
    CONF_THRESHOLD,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Configuración
# ─────────────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER   = os.path.join(os.path.dirname(__file__), "uploads")
ONNX_MODEL_PATH = os.environ.get(
    "ONNX_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "bilstm_fp32.onnx"),
)
MODEL_ARCH      = "onnx"
TARGET_T        = int(os.environ.get("TARGET_T", DEFAULT_T))
N_FRAMES        = int(os.environ.get("N_FRAMES", DEFAULT_N_FRAMES))
ALLOWED_EXT     = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_MB  = 200

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# ─────────────────────────────────────────────────────────────────────────────
#  Dispositivo (usado para YOLO; ONNX corre en CPU/GPU según providers)
# ─────────────────────────────────────────────────────────────────────────────
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Init] Dispositivo: {device}")

init_errors = []

# ── Sesión ONNX Runtime ───────────────────────────────────────────────────────
onnx_session = None
try:
    import onnxruntime as ort

    abs_onnx = os.path.abspath(ONNX_MODEL_PATH)
    if not os.path.exists(abs_onnx):
        msg = (f"Modelo ONNX no encontrado: {abs_onnx}\n"
               f"       Archivos .onnx en la carpeta: "
               f"{[f for f in os.listdir(os.path.dirname(abs_onnx) or '.') if f.endswith('.onnx')]}")
        print(f"[Init] ❌ {msg}")
        init_errors.append(msg)
    else:
        # Usar GPU si está disponible, con fallback a CPU
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        onnx_session = ort.InferenceSession(abs_onnx, providers=providers)
        print(f"[Init] ✅ Modelo ONNX cargado: {abs_onnx}")
        print(f"[Init]    Providers activos: {onnx_session.get_providers()}")
        print(f"[Init]    Input  : {[i.name for i in onnx_session.get_inputs()]}")
        print(f"[Init]    Output : {[o.name for o in onnx_session.get_outputs()]}")

except Exception as e:
    onnx_session = None
    msg = f"Error cargando modelo ONNX: {e}\n{traceback.format_exc()}"
    print(f"[Init] ❌ {msg}")
    init_errors.append(msg)

# ── YOLOv8l-pose ──────────────────────────────────────────────────────────────
yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8l-pose.pt")
    yolo_model.to(device)
    print("[Init] ✅ YOLOv8l-pose listo")
except Exception as e:
    yolo_model = None
    msg = f"Error cargando YOLOv8: {e}\n{traceback.format_exc()}"
    print(f"[Init] ❌ {msg}")
    init_errors.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def run_inference(video_path: str) -> dict:
    """
    Ejecuta el pipeline completo con ONNX Runtime y retorna un dict con el resultado.
    """
    if onnx_session is None or yolo_model is None:
        detail = "\n".join(init_errors) if init_errors else "Revisa la terminal para más detalles."
        raise RuntimeError(f"Modelos no disponibles. Errores de inicio:\n{detail}")

    # 1. Extraer frames
    frames, total_orig, fps_orig = extract_frames(video_path, N_FRAMES)
    if len(frames) == 0:
        raise ValueError("No se pudieron extraer frames del video.")

    # 2. Detectar keypoints
    seq_raw = extract_keypoints(frames, yolo_model)        # [T, 2, 17, 3]

    # 3. Canales de velocidad y aceleración
    seq_5ch = add_velocity_channels(seq_raw)               # [T, 2, 17, 5]

    # 4. Ajustar ventana temporal
    seq_T = resize_temporal(seq_5ch, TARGET_T)             # [TARGET_T, 2, 17, 5]

    # 5. Forward pass ONNX
    x_np = seq_T[np.newaxis].astype(np.float32)           # [1, T, 2, 17, 5]
    input_name  = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    logits_np   = onnx_session.run([output_name], {input_name: x_np})[0]  # [1, 2]

    logits_t = torch.from_numpy(logits_np)
    probs    = F.softmax(logits_t, dim=1).squeeze(0).numpy()

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    # Diagnóstico de confianza
    conf_pct = confidence * 100
    if conf_pct >= 90:
        diagnosis = ("muy_segura", "Predicción muy segura — el modelo discrimina con claridad.")
    elif conf_pct >= 70:
        diagnosis = ("confiable", "Predicción confiable.")
    elif conf_pct >= 55:
        diagnosis = ("incierta",
                     "Predicción incierta (zona ambigua). Verifica que el video muestre 2 personas claramente.")
    else:
        diagnosis = ("muy_incierta",
                     "Predicción muy incierta. El video puede no contener 2 personas visibles, "
                     "tener baja resolución o duración muy corta.")

    return {
        "class_id":          pred_class,
        "class_label":       LABELS[pred_class],
        "confidence":        round(confidence, 4),
        "probs": {
            LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
        "logits":            [round(float(v), 4) for v in logits_np.squeeze(0)],
        "n_frames":          len(frames),
        "total_frames":      total_orig,
        "fps":               round(fps_orig, 2),
        "diagnosis_level":   diagnosis[0],
        "diagnosis_message": diagnosis[1],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Rutas
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_ok = onnx_session is not None
    yolo_ok  = yolo_model is not None
    return render_template(
        "index.html",
        model_ok=model_ok,
        yolo_ok=yolo_ok,
        device=str(device),
        model_arch="ONNX",
        target_t=TARGET_T,
        n_frames=N_FRAMES,
        checkpoint=os.path.basename(ONNX_MODEL_PATH),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Recibe el video via multipart/form-data y devuelve JSON con la predicción.
    """
    if "video" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo."}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"error": "El nombre del archivo está vacío."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Formato no soportado. Usa: {', '.join(ALLOWED_EXT)}"
        }), 400

    ext       = file.filename.rsplit(".", 1)[1].lower()
    filename  = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        result = run_inference(save_path)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": onnx_session is not None,
        "model_type":   "onnx",
        "yolo_loaded":  yolo_model is not None,
        "device":       str(device),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)