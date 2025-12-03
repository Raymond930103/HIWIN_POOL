from pathlib import Path
from flask import Flask, redirect, url_for
import os
from flask_login import login_required, current_user

# Support running as a module (-m webapp.app) and as a script (python webapp/app.py)
try:
    from .config import Config  # when imported as package
    from .database import db
    from .auth import auth_bp, init_login
    from .game import game_bp
except ImportError:  # direct script execution fallback
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from webapp.config import Config
    from webapp.database import db
    from webapp.auth import auth_bp, init_login
    from webapp.game import game_bp


def create_app(config_object=None):
    app = Flask(__name__, static_folder="static", template_folder="templates")
    if config_object is None:
        config_object = Config
    app.config.from_object(config_object)

    db.init_app(app)
    init_login(app)

    with app.app_context():
        db.create_all()

    app.register_blueprint(auth_bp)
    app.register_blueprint(game_bp)

    @app.route("/")
    def index():
        return redirect(url_for("game.home"))

    return app


if __name__ == "__main__":
    app = create_app()
    # Allow overriding host/port/debug via environment for external access (e.g., ZeroTier)
    host = os.getenv("WEB_HOST", "0.0.0.0")
    try:
        port = int(os.getenv("WEB_PORT", "8000"))
    except Exception:
        port = 8000
    debug_val = os.getenv("WEB_DEBUG", "true").strip().lower()
    debug = debug_val in ("1", "true", "yes", "on")
    # threaded=True helps concurrent MJPEG streaming and API handling
    app.run(host=host, port=port, debug=debug, threaded=True)
