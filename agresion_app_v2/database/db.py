"""
database/db.py
══════════════════════════════════════════════════════════════════════════════
Módulo de base de datos SQLite para historial, alertas y estadísticas.
NO modifica ningún archivo del sistema principal.
══════════════════════════════════════════════════════════════════════════════
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "agresion_data.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crea todas las tablas si no existen."""
    conn = get_connection()
    c = conn.cursor()

    # ── Historial de predicciones ────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            filename        TEXT    NOT NULL,
            class_id        INTEGER NOT NULL,
            class_label     TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            prob_no_agresivo REAL   NOT NULL,
            prob_agresivo   REAL    NOT NULL,
            n_frames        INTEGER,
            total_frames    INTEGER,
            fps             REAL,
            diagnosis_level TEXT,
            diagnosis_message TEXT,
            logit0          REAL,
            logit1          REAL,
            model_type      TEXT    DEFAULT 'pt',
            keypoints_video TEXT    DEFAULT NULL
        )
    """)

    # ── Centro de alertas ────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id   INTEGER REFERENCES predictions(id),
            timestamp       TEXT    NOT NULL,
            level           TEXT    NOT NULL,
            message         TEXT    NOT NULL,
            read            INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] ✅ Base de datos inicializada:", DB_PATH)


def save_prediction(result: dict, filename: str, model_type: str = "pt", keypoints_video: str = None) -> int:
    """Guarda una predicción y genera alertas automáticamente."""
    conn = get_connection()
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    probs = result.get("probs", {})
    c.execute("""
        INSERT INTO predictions
            (timestamp, filename, class_id, class_label, confidence,
             prob_no_agresivo, prob_agresivo, n_frames, total_frames, fps,
             diagnosis_level, diagnosis_message, logit0, logit1, model_type, keypoints_video)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts,
        filename,
        result.get("class_id", 0),
        result.get("class_label", ""),
        result.get("confidence", 0),
        probs.get("No agresivo", 0),
        probs.get("Agresivo", 0),
        result.get("n_frames"),
        result.get("total_frames"),
        result.get("fps"),
        result.get("diagnosis_level"),
        result.get("diagnosis_message"),
        result.get("logits", [0, 0])[0] if result.get("logits") else 0,
        result.get("logits", [0, 0])[1] if result.get("logits") else 0,
        model_type,
        keypoints_video,
    ))
    pred_id = c.lastrowid

    # Generar alerta si es agresivo o confianza baja
    alerts_to_insert = []
    if result.get("class_id") == 1:
        conf_pct = result.get("confidence", 0) * 100
        level = "critical" if conf_pct >= 70 else "warning"
        msg = (
            f"Comportamiento AGRESIVO detectado en «{filename}» "
            f"con {conf_pct:.1f}% de confianza."
        )
        alerts_to_insert.append((pred_id, ts, level, msg))

    diag = result.get("diagnosis_level", "")
    if diag in ("incierta", "muy_incierta"):
        alerts_to_insert.append((
            pred_id, ts, "info",
            f"Predicción de baja confianza en «{filename}»: {result.get('diagnosis_message', '')}",
        ))

    for a in alerts_to_insert:
        c.execute(
            "INSERT INTO alerts (prediction_id, timestamp, level, message) VALUES (?, ?, ?, ?)",
            a,
        )

    conn.commit()
    conn.close()
    return pred_id


def get_history(page: int = 1, per_page: int = 20, filter_class: str = "all"):
    conn = get_connection()
    c = conn.cursor()
    offset = (page - 1) * per_page

    where = ""
    params_count = []
    params_data  = []

    if filter_class == "aggressive":
        where = "WHERE class_id = 1"
        params_count = []
        params_data  = [per_page, offset]
    elif filter_class == "safe":
        where = "WHERE class_id = 0"
        params_count = []
        params_data  = [per_page, offset]
    else:
        params_data  = [per_page, offset]

    total = c.execute(f"SELECT COUNT(*) FROM predictions {where}", params_count).fetchone()[0]
    rows  = c.execute(
        f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params_data,
    ).fetchall()

    conn.close()
    return [dict(r) for r in rows], total


def get_statistics():
    conn = get_connection()
    c = conn.cursor()

    total       = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    aggressive  = c.execute("SELECT COUNT(*) FROM predictions WHERE class_id=1").fetchone()[0]
    safe        = total - aggressive
    avg_conf    = c.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
    avg_conf_agg = c.execute("SELECT AVG(confidence) FROM predictions WHERE class_id=1").fetchone()[0] or 0
    avg_conf_safe = c.execute("SELECT AVG(confidence) FROM predictions WHERE class_id=0").fetchone()[0] or 0

    # Por día (últimos 14 días)
    by_day = c.execute("""
        SELECT DATE(timestamp) as day,
               SUM(CASE WHEN class_id=1 THEN 1 ELSE 0 END) as agresivos,
               SUM(CASE WHEN class_id=0 THEN 1 ELSE 0 END) as seguros,
               COUNT(*) as total
        FROM predictions
        WHERE timestamp >= DATE('now', '-14 days')
        GROUP BY day ORDER BY day ASC
    """).fetchall()

    # Niveles de diagnóstico
    diag_counts = c.execute("""
        SELECT diagnosis_level, COUNT(*) as cnt
        FROM predictions GROUP BY diagnosis_level
    """).fetchall()

    # Modelos usados
    model_counts = c.execute("""
        SELECT model_type, COUNT(*) as cnt
        FROM predictions GROUP BY model_type
    """).fetchall()

    conn.close()
    return {
        "total": total,
        "aggressive": aggressive,
        "safe": safe,
        "pct_aggressive": round(aggressive / total * 100, 1) if total else 0,
        "avg_conf": round(avg_conf * 100, 1),
        "avg_conf_agg": round(avg_conf_agg * 100, 1),
        "avg_conf_safe": round(avg_conf_safe * 100, 1),
        "by_day": [dict(r) for r in by_day],
        "diag_counts": [dict(r) for r in diag_counts],
        "model_counts": [dict(r) for r in model_counts],
    }


def get_alerts(unread_only: bool = False):
    conn = get_connection()
    c = conn.cursor()
    where = "WHERE a.read=0" if unread_only else ""
    rows = c.execute(f"""
        SELECT a.*, p.filename FROM alerts a
        LEFT JOIN predictions p ON a.prediction_id = p.id
        {where}
        ORDER BY a.id DESC LIMIT 100
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_alerts_read(alert_ids: list = None):
    conn = get_connection()
    c = conn.cursor()
    if alert_ids:
        placeholders = ",".join("?" * len(alert_ids))
        c.execute(f"UPDATE alerts SET read=1 WHERE id IN ({placeholders})", alert_ids)
    else:
        c.execute("UPDATE alerts SET read=1")
    conn.commit()
    conn.close()


def delete_prediction(pred_id: int):
    conn = get_connection()
    c = conn.cursor()
    # Get keypoints_video path before deleting
    row = c.execute("SELECT keypoints_video FROM predictions WHERE id=?", (pred_id,)).fetchone()
    kv_path = row["keypoints_video"] if row else None
    c.execute("DELETE FROM alerts WHERE prediction_id=?", (pred_id,))
    c.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
    conn.commit()
    conn.close()
    return kv_path


def get_prediction_by_id(pred_id: int):
    conn = get_connection()
    c = conn.cursor()
    row = c.execute("SELECT * FROM predictions WHERE id=?", (pred_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
