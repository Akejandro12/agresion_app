"""
blueprints/help_bp.py
══════════════════════════════════════════════════════════════════════════════
Blueprint — Centro de Ayuda
══════════════════════════════════════════════════════════════════════════════
"""
from flask import Blueprint, render_template

help_bp = Blueprint("help", __name__, url_prefix="/help")


@help_bp.route("/")
def index():
    return render_template("help.html")
