"""
blueprints/stats_bp.py
══════════════════════════════════════════════════════════════════════════════
Blueprint — Reportes y Estadísticas
══════════════════════════════════════════════════════════════════════════════
"""
from flask import Blueprint, render_template, jsonify

stats_bp = Blueprint("stats", __name__, url_prefix="/stats")


@stats_bp.route("/")
def index():
    from database.db import get_statistics
    stats = get_statistics()
    return render_template("stats.html", stats=stats)


@stats_bp.route("/api")
def api():
    from database.db import get_statistics
    return jsonify(get_statistics())