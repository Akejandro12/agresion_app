"""
blueprints/onnx_bp_16.py
══════════════════════════════════════════════════════════════════════════════
Blueprint — Inferencia con ONNX (GPU NVIDIA + fallback a CPU)
══════════════════════════════════════════════════════════════════════════════
"""

import os
import uuid
import numpy as np
from flask import Blueprint, render_template, request, jsonify, current_app

onnx_bp = Blueprint("onnx", __name__, url_prefix="/onnx")

ALLOWED_EXT = {"mp4", "avi", "mov", "mkv", "webm"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ─────────────────────────────────────────────────────────────────────────────
# 🚀 CARGA ONNX (GPU NVIDIA + fallback CPU)
# ─────────────────────────────────────────────────────────────────────────────
def load_onnx_session():
    if hasattr(load_onnx_session, "_session"):
        return load_onnx_session._session

    try:
        import onnxruntime as ort
        import onnx

        model_path = current_app.config.get("ONNX_MODEL_PATH", "model.onnx")

        if not os.path.exists(model_path):
            return None, f"Modelo ONNX no encontrado: {model_path}"

        print(f"[ONNX] 📦 Cargando modelo: {model_path}")

        # Validar modelo
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        so = ort.SessionOptions()
        so.log_severity_level = 0

        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider"
        ]

        sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=providers
        )

        print("[ONNX] ✅ Sesión cargada")
        print("[ONNX] 🧠 Providers activos:", sess.get_providers())

        # Mostrar tipo de entrada del modelo
        input_info = sess.get_inputs()[0]
        print("[ONNX] 🧾 Input name :", input_info.name)
        print("[ONNX] 🧾 Input type :", input_info.type)

        load_onnx_session._session = (sess, None)
        return sess, None

    except Exception as e:
        print("[ONNX ERROR ❌]", str(e))
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 INFERENCIA
# ─────────────────────────────────────────────────────────────────────────────
def run_onnx_inference(video_path: str, yolo_model) -> dict:
    import sys

    main_dir = current_app.config.get("MAIN_APP_DIR", ".")
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    from infer_video import (
        extract_frames, extract_keypoints, add_velocity_channels,
        resize_temporal, LABELS, DEFAULT_T, DEFAULT_N_FRAMES,
    )

    sess, err = load_onnx_session()
    if sess is None:
        raise RuntimeError(f"Sesión ONNX no disponible: {err}")

    TARGET_T = current_app.config.get("TARGET_T", DEFAULT_T)
    N_FRAMES = current_app.config.get("N_FRAMES", DEFAULT_N_FRAMES)

    print("[ONNX] 🎬 Frames...")
    frames, total_orig, fps_orig = extract_frames(video_path, N_FRAMES)

    if len(frames) == 0:
        raise ValueError("No se pudieron extraer frames.")

    print("[ONNX] 🧍 Keypoints...")
    seq_raw = extract_keypoints(frames, yolo_model)

    print("[ONNX] ⚙️ Procesamiento...")
    seq_5ch = add_velocity_channels(seq_raw)
    seq_T = resize_temporal(seq_5ch, TARGET_T)

    # 🔥 FIX: adaptar dtype al modelo ONNX
    input_info = sess.get_inputs()[0]
    input_type = input_info.type

    print("[ONNX] 🎯 Usando dtype según modelo:", input_type)

    if "float16" in input_type:
        x = seq_T[np.newaxis].astype(np.float16)
    else:
        x = seq_T[np.newaxis].astype(np.float32)

    input_name = input_info.name

    print("[ONNX] 🚀 Inferencia...")
    logits_np = sess.run(None, {input_name: x})[0][0]

    # Softmax
    e = np.exp(logits_np - logits_np.max())
    probs = e / e.sum()

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    conf_pct = confidence * 100

    if conf_pct >= 90:
        diagnosis = ("muy_segura", "Predicción muy segura.")
    elif conf_pct >= 70:
        diagnosis = ("confiable", "Predicción confiable.")
    elif conf_pct >= 55:
        diagnosis = ("incierta", "Predicción incierta.")
    else:
        diagnosis = ("muy_incierta", "Predicción muy incierta.")

    return {
        "class_id": pred_class,
        "class_label": LABELS[pred_class],
        "confidence": round(confidence, 4),
        "probs": {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "logits": [round(float(v), 4) for v in logits_np],
        "n_frames": len(frames),
        "total_frames": total_orig,
        "fps": round(fps_orig, 2),
        "diagnosis_level": diagnosis[0],
        "diagnosis_message": diagnosis[1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 🌐 RUTAS
# ─────────────────────────────────────────────────────────────────────────────
@onnx_bp.route("/")
def index():
    sess, err = load_onnx_session()
    onnx_ok = sess is not None
    yolo_ok = current_app.config.get("YOLO_MODEL") is not None

    return render_template(
        "onnx.html",
        onnx_ok=onnx_ok,
        onnx_err=err,
        yolo_ok=yolo_ok,
        onnx_model=os.path.basename(
            current_app.config.get("ONNX_MODEL_PATH", "model.onnx")
        ),
    )


@onnx_bp.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No se recibió archivo."}), 400

    file = request.files["video"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Archivo inválido."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"onnx_{uuid.uuid4().hex}.{ext}"

    upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    save_path = os.path.join(upload_folder, filename)
    file.save(save_path)

    yolo_model = current_app.config.get("YOLO_MODEL")

    try:
        result = run_onnx_inference(save_path, yolo_model)

        from database.db import save_prediction
        save_prediction(result, file.filename, model_type="onnx")

        return jsonify({"success": True, "result": result})

    except Exception as e:
        print("[ONNX ERROR ❌]", str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)