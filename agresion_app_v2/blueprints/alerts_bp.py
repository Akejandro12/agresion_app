"""
blueprints/alerts_bp.py
══════════════════════════════════════════════════════════════════════════════
Blueprint — Centro de Alertas
══════════════════════════════════════════════════════════════════════════════
"""
from flask import Blueprint, render_template, jsonify, request

alerts_bp = Blueprint("alerts", __name__, url_prefix="/alerts")


@alerts_bp.route("/")
def index():
    from database.db import get_alerts
    alerts = get_alerts()
    unread = [a for a in alerts if not a["read"]]
    return render_template("alerts.html", alerts=alerts, unread_count=len(unread))


@alerts_bp.route("/api")
def api():
    from database.db import get_alerts
    unread_only = request.args.get("unread") == "1"
    alerts = get_alerts(unread_only)
    return jsonify({"alerts": alerts, "total": len(alerts)})


@alerts_bp.route("/api/count")
def count():
    from database.db import get_alerts
    unread = get_alerts(unread_only=True)
    return jsonify({"count": len(unread)})


@alerts_bp.route("/read", methods=["POST"])
def mark_read():
    from database.db import mark_alerts_read
    data = request.get_json(silent=True) or {}
    ids  = data.get("ids", None)
    mark_alerts_read(ids)
    return jsonify({"ok": True})
