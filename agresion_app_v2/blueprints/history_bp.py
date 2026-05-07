"""
blueprints/history_bp.py
══════════════════════════════════════════════════════════════════════════════
Blueprint — Historial de predicciones
MODIFICADO: /view/<id> ahora sirve el video con soporte de Range Requests
para que el elemento <video> de HTML pueda reproducirlo correctamente en el
navegador sin necesidad de descargarlo (Chrome, Firefox, Safari).
══════════════════════════════════════════════════════════════════════════════
"""
import os
from flask import (
    Blueprint, render_template, request, jsonify,
    send_file, abort, Response, current_app,
)

history_bp = Blueprint("history", __name__, url_prefix="/history")


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: servidor de video con Range Requests
#  El navegador necesita poder pedir rangos de bytes (p.ej. "bytes=0-")
#  para que el reproductor <video> funcione correctamente.
#  Flask/send_from_directory no implementa esto por defecto.
# ─────────────────────────────────────────────────────────────────────────────
def _serve_video_with_ranges(file_path: str) -> Response:
    """
    Sirve un archivo de video con soporte completo de HTTP Range Requests.
    Esto permite que el elemento <video> de HTML5 reproduzca el archivo
    sin tener que descargarlo completo primero.
    """
    if not os.path.exists(file_path):
        abort(404)

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range", None)

    # ── Sin cabecera Range → respuesta completa (200) ──────────────────────
    if not range_header:
        def generate_full():
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 256)  # 256 KB por chunk
                    if not chunk:
                        break
                    yield chunk

        resp = Response(
            generate_full(),
            status=200,
            mimetype="video/mp4",
            direct_passthrough=True,
        )
        resp.headers["Content-Length"] = file_size
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    # ── Con cabecera Range → respuesta parcial (206) ────────────────────────
    # Formato: "bytes=inicio-fin"  (fin es opcional)
    range_val = range_header.replace("bytes=", "")
    parts = range_val.split("-")
    byte_start = int(parts[0]) if parts[0] else 0
    byte_end   = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1

    # Validar rango
    byte_end = min(byte_end, file_size - 1)
    if byte_start > byte_end:
        abort(416)  # Range Not Satisfiable

    length = byte_end - byte_start + 1

    def generate_range():
        with open(file_path, "rb") as f:
            f.seek(byte_start)
            remaining = length
            while remaining > 0:
                chunk_size = min(1024 * 256, remaining)
                data = f.read(chunk_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    resp = Response(
        generate_range(),
        status=206,
        mimetype="video/mp4",
        direct_passthrough=True,
    )
    resp.headers["Content-Range"]  = f"bytes {byte_start}-{byte_end}/{file_size}"
    resp.headers["Content-Length"] = length
    resp.headers["Accept-Ranges"]  = "bytes"
    resp.headers["Cache-Control"]  = "no-cache"
    return resp


# ─────────────────────────────────────────────────────────────────────────────
#  Rutas (idénticas a la versión original excepto /view/<id>)
# ─────────────────────────────────────────────────────────────────────────────

@history_bp.route("/")
def index():
    from database.db import get_history
    page        = int(request.args.get("page", 1))
    filter_cls  = request.args.get("filter", "all")
    per_page    = 15
    rows, total = get_history(page, per_page, filter_cls)
    total_pages = max(1, (total + per_page - 1) // per_page)
    return render_template(
        "history.html",
        rows=rows,
        page=page,
        total_pages=total_pages,
        total=total,
        filter_cls=filter_cls,
    )


@history_bp.route("/api")
def api():
    from database.db import get_history
    page       = int(request.args.get("page", 1))
    filter_cls = request.args.get("filter", "all")
    rows, total = get_history(page, 15, filter_cls)
    return jsonify({"rows": rows, "total": total})


@history_bp.route("/delete/<int:pred_id>", methods=["DELETE"])
def delete(pred_id):
    from database.db import delete_prediction
    kv_path = delete_prediction(pred_id)
    if kv_path:
        full = os.path.join(current_app.config.get("KP_VIDEO_FOLDER", ""), kv_path)
        if os.path.exists(full):
            os.remove(full)
    return jsonify({"ok": True})


@history_bp.route("/download/<int:pred_id>")
def download_kp_video(pred_id):
    """Descarga el video con keypoints anotados."""
    from database.db import get_prediction_by_id
    row = get_prediction_by_id(pred_id)
    if not row or not row.get("keypoints_video"):
        abort(404)
    folder = current_app.config.get("KP_VIDEO_FOLDER", "")
    path   = os.path.join(folder, row["keypoints_video"])
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"keypoints_{pred_id}.mp4")


@history_bp.route("/view/<int:pred_id>")
def view_kp_video(pred_id):
    """
    Sirve el video con keypoints para reproducción en el navegador.
    MODIFICADO: usa _serve_video_with_ranges() en lugar de send_from_directory()
    para que el elemento <video> de HTML5 pueda cargar y saltar en el video.
    """
    from database.db import get_prediction_by_id
    row = get_prediction_by_id(pred_id)
    if not row or not row.get("keypoints_video"):
        abort(404)
    folder    = current_app.config.get("KP_VIDEO_FOLDER", "")
    file_path = os.path.join(folder, row["keypoints_video"])
    return _serve_video_with_ranges(file_path)