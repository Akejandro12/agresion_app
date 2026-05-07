"""
app_extended.py
══════════════════════════════════════════════════════════════════════════════
Extensión de la aplicación Flask principal.
IMPORTA el app.py original sin modificarlo y agrega:
  1. Historial (SQLite)
  2. Videos con keypoints (descarga/visualización)
  3. Reportes y Estadísticas
  4. Centro de Alertas
  5. Centro de Ayuda
  6. /api/preview  — conversión rápida a H.264 para preview en el navegador

Uso:  python app_extended.py  →  http://localhost:5000
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import uuid
import subprocess
import warnings
import platform
import shutil
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Path al directorio del sistema principal (app.py original)
# ─────────────────────────────────────────────────────────────────────────────
MAIN_APP_DIR = os.environ.get(
    "MAIN_APP_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agresion_app", "agresion_app"),
)

sys.path.insert(0, MAIN_APP_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
#  Importar la aplicación Flask del sistema principal
# ─────────────────────────────────────────────────────────────────────────────
os.chdir(MAIN_APP_DIR)

from app import (
    app,
    onnx_session,
    yolo_model,
    device,
    run_inference,
    UPLOAD_FOLDER,
    ALLOWED_EXT
)

# ─────────────────────────────────────────────────────────────────────────────
#  Configuraciones adicionales
# ─────────────────────────────────────────────────────────────────────────────
KP_VIDEO_FOLDER      = os.path.join(MAIN_APP_DIR, "kp_videos")
PREVIEW_TEMP_FOLDER  = os.path.join(MAIN_APP_DIR, "preview_tmp")

os.makedirs(KP_VIDEO_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_TEMP_FOLDER, exist_ok=True)

app.config["KP_VIDEO_FOLDER"]     = KP_VIDEO_FOLDER
app.config["YOLO_MODEL"]          = yolo_model
app.config["MAIN_APP_DIR"]        = MAIN_APP_DIR
app.config["PREVIEW_TEMP_FOLDER"] = PREVIEW_TEMP_FOLDER

# ─────────────────────────────────────────────────────────────────────────────
#  ffmpeg Detection (Windows + Docker/Linux)
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if platform.system() == "Windows":
    # Usa ffmpeg.exe local del proyecto
    _FFMPEG_EXE = os.path.join(_THIS_DIR, "ffmpeg.exe")
else:
    # Usa ffmpeg instalado en Linux/Docker
    _FFMPEG_EXE = shutil.which("ffmpeg")


def _ffmpeg_available() -> bool:
    """
    Compatible con:
      ✔ Windows local
      ✔ Docker Linux
      ✔ Ubuntu
      ✔ WSL
    """

    global _FFMPEG_EXE

    if not _FFMPEG_EXE:
        print("[Init] ❌ ffmpeg no encontrado")
        return False

    try:
        result = subprocess.run(
            [_FFMPEG_EXE, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            print(f"[Init] ✅ ffmpeg encontrado: {_FFMPEG_EXE}")
            return True

    except Exception as e:
        print(f"[Init] ❌ Error verificando ffmpeg: {e}")

    print("[Init] ❌ ffmpeg no disponible")
    return False


FFMPEG_OK = _ffmpeg_available()


# ─────────────────────────────────────────────────────────────────────────────
#  Re-encoding H264
# ─────────────────────────────────────────────────────────────────────────────
def _reencode_h264(src_path: str, dst_path: str, preview: bool = False) -> bool:
    """
    Convierte videos a H264 compatible con navegador.
    """

    scale_filter = (
        "scale=-2:480"
        if preview
        else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )

    preset = "ultrafast" if preview else "fast"
    crf    = "28" if preview else "23"

    try:
        command = [
            _FFMPEG_EXE,
            "-y",
            "-i", src_path,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", crf,
            "-vf", scale_filter,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            dst_path,
        ]

        print("\n[FFmpeg] ▶ Ejecutando:")
        print(" ".join(command))

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print(f"\n[FFmpeg] ❌ Error:\n{result.stderr[:1000]}")
            return False

        print(f"[FFmpeg] ✅ Conversión exitosa: {dst_path}")
        return True

    except Exception as e:
        print(f"[FFmpeg] ❌ Excepción: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Inicializar base de datos
# ─────────────────────────────────────────────────────────────────────────────
from database.db import init_db
init_db()

# ─────────────────────────────────────────────────────────────────────────────
#  Registrar blueprints
# ─────────────────────────────────────────────────────────────────────────────
from blueprints.history_bp import history_bp
from blueprints.stats_bp   import stats_bp
from blueprints.alerts_bp  import alerts_bp
from blueprints.help_bp    import help_bp

app.register_blueprint(history_bp)
app.register_blueprint(stats_bp)
app.register_blueprint(alerts_bp)
app.register_blueprint(help_bp)

# ─────────────────────────────────────────────────────────────────────────────
#  Ruta /api/preview
# ─────────────────────────────────────────────────────────────────────────────
from flask import request, jsonify, send_file
import threading
import time


@app.route("/api/preview", methods=["POST"])
def api_preview():

    if not FFMPEG_OK:
        return jsonify({"error": "ffmpeg no disponible"}), 503

    if "video" not in request.files:
        return jsonify({"error": "No se recibió archivo"}), 400

    file = request.files["video"]

    ext = (
        file.filename.rsplit(".", 1)[1].lower()
        if "." in file.filename
        else "mp4"
    )

    uid = uuid.uuid4().hex

    src_path = os.path.join(
        PREVIEW_TEMP_FOLDER,
        f"src_{uid}.{ext}"
    )

    dst_path = os.path.join(
        PREVIEW_TEMP_FOLDER,
        f"prev_{uid}.mp4"
    )

    file.save(src_path)

    if not _reencode_h264(src_path, dst_path, preview=True):

        for p in (src_path, dst_path):
            if os.path.exists(p):
                os.remove(p)

        return jsonify({"error": "Error al convertir el video"}), 500

    if os.path.exists(src_path):
        os.remove(src_path)

    def _cleanup(path, delay=300):
        time.sleep(delay)

        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    threading.Thread(
        target=_cleanup,
        args=(dst_path,),
        daemon=True
    ).start()

    return send_file(
        dst_path,
        mimetype="video/mp4",
        as_attachment=False,
        download_name="preview.mp4",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Monkey-patch de /predict
# ─────────────────────────────────────────────────────────────────────────────
def predict_extended():

    from database.db import save_prediction

    from infer_video import (
        extract_frames,
        DEFAULT_N_FRAMES,
        CONF_THRESHOLD,
        SKELETON_EDGES,
        PERSON_COLORS,
        LABEL_COLORS,
    )

    import cv2

    if "video" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo."}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"error": "El nombre del archivo está vacío."}), 400

    ext = (
        file.filename.rsplit(".", 1)[1].lower()
        if "." in file.filename
        else ""
    )

    if ext not in ALLOWED_EXT:
        return jsonify({
            "error": f"Formato no soportado. Usa: {', '.join(ALLOWED_EXT)}"
        }), 400

    uid       = uuid.uuid4().hex
    filename  = f"{uid}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    file.seek(0)
    file.save(save_path)

    kp_video_name = None

    try:

        # ─────────────────────────────────────────────────────────────────────
        #  Inferencia ONNX
        # ─────────────────────────────────────────────────────────────────────
        result = run_inference(save_path)

        # ─────────────────────────────────────────────────────────────────────
        #  Video con keypoints
        # ─────────────────────────────────────────────────────────────────────
        try:

            kp_frames, _, fps_orig = extract_frames(
                save_path,
                DEFAULT_N_FRAMES
            )

            if len(kp_frames) > 0 and yolo_model is not None:

                raw_name = f"raw_{uid}.mp4"
                raw_path = os.path.join(KP_VIDEO_FOLDER, raw_name)

                h, w = kp_frames[0].shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    raw_path,
                    fourcc,
                    max(fps_orig, 10),
                    (w, h)
                )

                label = result["class_label"]
                conf  = result["confidence"] * 100
                color = LABEL_COLORS[result["class_id"]]

                for frame in kp_frames:

                    vis = frame.copy()

                    results_yolo = yolo_model(
                        frame,
                        verbose=False,
                        conf=CONF_THRESHOLD
                    )

                    for det in results_yolo:

                        if det.keypoints is None:
                            continue

                        kps = det.keypoints.xy.cpu().numpy()

                        kps_conf = (
                            det.keypoints.conf.cpu().numpy()
                            if det.keypoints.conf is not None
                            else None
                        )

                        for pi, kp in enumerate(kps[:2]):

                            pc = PERSON_COLORS[
                                pi % len(PERSON_COLORS)
                            ]

                            for ji, (x, y) in enumerate(kp):

                                if (
                                    kps_conf is not None
                                    and kps_conf[pi][ji] < 0.3
                                ):
                                    continue

                                cv2.circle(
                                    vis,
                                    (int(x), int(y)),
                                    4,
                                    pc,
                                    -1
                                )

                            for (a, b) in SKELETON_EDGES:

                                if a < len(kp) and b < len(kp):

                                    xa, ya = int(kp[a][0]), int(kp[a][1])
                                    xb, yb = int(kp[b][0]), int(kp[b][1])

                                    if (
                                        xa > 0 and ya > 0
                                        and xb > 0 and yb > 0
                                    ):
                                        cv2.line(
                                            vis,
                                            (xa, ya),
                                            (xb, yb),
                                            pc,
                                            2
                                        )

                    cv2.rectangle(
                        vis,
                        (0, 0),
                        (w, 36),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        vis,
                        f"{label}  {conf:.1f}%",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        color[::-1],
                        2,
                    )

                    writer.write(vis)

                writer.release()

                # ─────────────────────────────────────────────────────────────
                #  Re-encode H264
                # ─────────────────────────────────────────────────────────────
                final_name = f"kp_{uid}.mp4"
                final_path = os.path.join(
                    KP_VIDEO_FOLDER,
                    final_name
                )

                if (
                    FFMPEG_OK
                    and _reencode_h264(
                        raw_path,
                        final_path,
                        preview=False
                    )
                ):

                    os.remove(raw_path)

                    kp_video_name = final_name

                    print(
                        f"[KP Video] ✅ H264 listo: {final_name}"
                    )

                else:

                    os.rename(raw_path, final_path)

                    kp_video_name = final_name

                    print(
                        f"[KP Video] ⚠️ Guardado sin re-encoding"
                    )

        except Exception as kp_err:

            print(
                f"[KP Video] ❌ Error generando video: {kp_err}"
            )

            kp_video_name = None

        # ─────────────────────────────────────────────────────────────────────
        #  Guardar DB
        # ─────────────────────────────────────────────────────────────────────
        pred_id = save_prediction(
            result,
            file.filename,
            model_type="onnx",
            keypoints_video=kp_video_name,
        )

        response = {
            "success": True,
            "result": result,
            "prediction_id": pred_id
        }

        if kp_video_name:
            response["kp_video_url"] = (
                f"/history/view/{pred_id}"
            )

            response["kp_download_url"] = (
                f"/history/download/{pred_id}"
            )

        return jsonify(response)

    except Exception as e:

        return jsonify({"error": str(e)}), 500

    finally:

        if os.path.exists(save_path):
            os.remove(save_path)


app.view_functions["predict"] = predict_extended


# ─────────────────────────────────────────────────────────────────────────────
#  Context processor
# ─────────────────────────────────────────────────────────────────────────────
@app.context_processor
def inject_alert_count():

    try:

        from database.db import get_alerts

        unread = get_alerts(unread_only=True)

        return {
            "unread_alert_count": len(unread)
        }

    except Exception:

        return {
            "unread_alert_count": 0
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n[Extended] 🚀 Iniciando app...")

    print(f"[Extended] 📂 MAIN_APP_DIR       : {MAIN_APP_DIR}")
    print(f"[Extended] 📂 KP_VIDEO_FOLDER    : {KP_VIDEO_FOLDER}")
    print(f"[Extended] 📂 PREVIEW_TEMP       : {PREVIEW_TEMP_FOLDER}")

    print(
        f"[Extended] 🎬 ffmpeg            : "
        f"{_FFMPEG_EXE if FFMPEG_OK else 'NO DISPONIBLE'}"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )