"""
blueprints/onnx_bp_32.py
══════════════════════════════════════════════════════════════════════════════
Blueprint — Inferencia con modelo ONNX (FP32)
MODIFICADO:
  - /predict ahora genera video con keypoints igual que el modelo .pt
  - El video se sirve a través de /history/view/<id> (con range requests)
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


def load_onnx_session():
    """Carga (o reutiliza) la sesión ONNX."""
    if hasattr(load_onnx_session, "_session"):
        return load_onnx_session._session
    try:
        import onnxruntime as ort
        model_path = current_app.config.get("ONNX_MODEL_PATH", "model.onnx")
        if not os.path.exists(model_path):
            return None, f"Modelo ONNX no encontrado: {model_path}"
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        load_onnx_session._session = (sess, None)
        return sess, None
    except ImportError:
        return None, "onnxruntime no está instalado. Ejecuta: pip install onnxruntime"
    except Exception as e:
        return None, str(e)


def run_onnx_inference(video_path: str, yolo_model) -> dict:
    """
    Pipeline idéntico al de app.py pero con modelo ONNX en lugar de .pt.
    """
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

    frames, total_orig, fps_orig = extract_frames(video_path, N_FRAMES)
    if len(frames) == 0:
        raise ValueError("No se pudieron extraer frames del video.")

    seq_raw = extract_keypoints(frames, yolo_model)
    seq_5ch = add_velocity_channels(seq_raw)
    seq_T   = resize_temporal(seq_5ch, TARGET_T)

    x = seq_T[np.newaxis].astype(np.float32)
    input_name = sess.get_inputs()[0].name
    logits_np  = sess.run(None, {input_name: x})[0][0]

    e = np.exp(logits_np - logits_np.max())
    probs = e / e.sum()

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    conf_pct   = confidence * 100

    if conf_pct >= 90:
        diagnosis = ("muy_segura",   "Predicción muy segura — el modelo discrimina con claridad.")
    elif conf_pct >= 70:
        diagnosis = ("confiable",    "Predicción confiable.")
    elif conf_pct >= 55:
        diagnosis = ("incierta",     "Predicción incierta (zona ambigua).")
    else:
        diagnosis = ("muy_incierta", "Predicción muy incierta.")

    return {
        "class_id":          pred_class,
        "class_label":       LABELS[pred_class],
        "confidence":        round(confidence, 4),
        "probs":             {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "logits":            [round(float(v), 4) for v in logits_np],
        "n_frames":          len(frames),
        "total_frames":      total_orig,
        "fps":               round(fps_orig, 2),
        "diagnosis_level":   diagnosis[0],
        "diagnosis_message": diagnosis[1],
        # Devolvemos también frames y fps para generar el video KP
        "_frames":           frames,
        "_fps_orig":         fps_orig,
    }


def _generate_kp_video(frames: list, result: dict, yolo_model, fps_orig: float) -> str | None:
    """
    Genera el video con keypoints superpuestos.
    Lógica idéntica a la de app_extended.py (predict_extended).
    Retorna el nombre del archivo generado, o None si falla.
    """
    import sys
    import cv2

    main_dir = current_app.config.get("MAIN_APP_DIR", ".")
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    from infer_video import (
        CONF_THRESHOLD, SKELETON_EDGES, PERSON_COLORS, LABEL_COLORS,
    )

    KP_VIDEO_FOLDER = current_app.config.get("KP_VIDEO_FOLDER", "kp_videos")

    try:
        if len(frames) == 0 or yolo_model is None:
            return None

        kp_video_name = f"kp_onnx_{uuid.uuid4().hex}.mp4"
        kp_video_path = os.path.join(KP_VIDEO_FOLDER, kp_video_name)

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(kp_video_path, fourcc, max(fps_orig, 10), (w, h))

        label = result["class_label"]
        conf  = result["confidence"] * 100
        color = LABEL_COLORS[result["class_id"]]  # (B, G, R)

        for frame in frames:
            vis = frame.copy()
            results_yolo = yolo_model(frame, verbose=False, conf=CONF_THRESHOLD)
            for det in results_yolo:
                if det.keypoints is None:
                    continue
                kps      = det.keypoints.xy.cpu().numpy()   # [N_persons, 17, 2]
                kps_conf = (
                    det.keypoints.conf.cpu().numpy()
                    if det.keypoints.conf is not None
                    else None
                )
                for pi, kp in enumerate(kps[:2]):
                    pc = PERSON_COLORS[pi % len(PERSON_COLORS)]
                    for ji, (x, y) in enumerate(kp):
                        if kps_conf is not None and kps_conf[pi][ji] < 0.3:
                            continue
                        cv2.circle(vis, (int(x), int(y)), 4, pc, -1)
                    for (a, b) in SKELETON_EDGES:
                        if a < len(kp) and b < len(kp):
                            xa, ya = int(kp[a][0]), int(kp[a][1])
                            xb, yb = int(kp[b][0]), int(kp[b][1])
                            if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                                cv2.line(vis, (xa, ya), (xb, yb), pc, 2)

            # Etiqueta superpuesta
            cv2.rectangle(vis, (0, 0), (w, 36), (0, 0, 0), -1)
            cv2.putText(
                vis,
                f"{label}  {conf:.1f}%",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color[::-1], 2,
            )
            writer.write(vis)

        writer.release()
        return kp_video_name

    except Exception as kp_err:
        print(f"[ONNX KP Video] ⚠️  No se pudo generar video con keypoints: {kp_err}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Rutas
# ─────────────────────────────────────────────────────────────────────────────

@onnx_bp.route("/")
def index():
    sess, err = load_onnx_session()
    onnx_ok   = sess is not None
    yolo_ok   = current_app.config.get("YOLO_MODEL") is not None
    return render_template(
        "onnx.html",
        onnx_ok=onnx_ok,
        onnx_err=err,
        yolo_ok=yolo_ok,
        onnx_model=os.path.basename(current_app.config.get("ONNX_MODEL_PATH", "model.onnx")),
    )


@onnx_bp.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo."}), 400

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Archivo inválido o formato no soportado."}), 400

    ext           = file.filename.rsplit(".", 1)[1].lower()
    filename      = f"onnx_{uuid.uuid4().hex}.{ext}"
    upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
    save_path     = os.path.join(upload_folder, filename)
    file.save(save_path)

    yolo_model = current_app.config.get("YOLO_MODEL")

    try:
        result = run_onnx_inference(save_path, yolo_model)

        # Extraer datos internos y limpiar el dict antes de guardarlo
        frames   = result.pop("_frames", [])
        fps_orig = result.pop("_fps_orig", 10.0)

        # ── Generar video con keypoints (igual que modelo .pt) ──────────────
        kp_video_name = _generate_kp_video(frames, result, yolo_model, fps_orig)

        # ── Guardar en historial con model_type='onnx' ──────────────────────
        from database.db import save_prediction
        pred_id = save_prediction(
            result, file.filename,
            model_type="onnx",
            keypoints_video=kp_video_name,
        )

        response = {"success": True, "result": result, "prediction_id": pred_id}
        if kp_video_name:
            response["kp_video_url"]    = f"/history/view/{pred_id}"
            response["kp_download_url"] = f"/history/download/{pred_id}"

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)