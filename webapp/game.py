import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from flask import Blueprint, render_template, request, jsonify, url_for, current_app
import re
from flask_login import login_required, current_user

from .config import Config
from .database import db
from .models import Game, Shot, User
from .robot import capture_and_plan, compute_arm_payload, send_to_robot, perform_handshaked_strike
from .render import render_image
from .camera import camera_streamer

import sys
import time
from pathlib import Path as P
ROOT = P(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.core.billiard_api import compute_shot
from main.run_shot import get_last_plan_shot_error


game_bp = Blueprint("game", __name__)


@game_bp.route("/")
@login_required
def home():
    return render_template("game.html", user=current_user)


@game_bp.route("/camera")
@login_required
def camera_page():
    return render_template("camera.html")


@game_bp.route("/video_feed")
@login_required
def video_feed():
    from flask import Response

    def _mjpeg_stream():
        boundary = b"--frame\r\n"
        headers = b"Content-Type: image/jpeg\r\n\r\n"
        while True:
            frame = camera_streamer.get_jpeg()
            if frame is None:
                # Backoff a bit if not available
                time.sleep(0.1)
                continue
            yield boundary + headers + frame + b"\r\n"
            # ~15 fps
            time.sleep(1/15)
    return Response(_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@game_bp.route("/api/camera/release", methods=["POST"])
@login_required
def camera_release():
    try:
        camera_streamer.release()
        return jsonify({"released": True})
    except Exception as e:
        return jsonify({"released": False, "error": str(e)}), 500


def _type_id(b: dict) -> Optional[int]:
    t = b.get("type")
    if t is None:
        return None
    s = str(t)
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _balls_remaining(balls: list) -> int:
    return sum(1 for b in balls if (tid := _type_id(b)) is not None and tid != 0)


def _current_turn(game: Game) -> str:
    """Compute whose turn it is without extra DB columns.

    Rules:
    - If no shots yet: 'robot' when order=='first', else 'human'.
    - After a robot strike (last.just_capture == False): next is 'human'.
    - After a human confirmation/capture (last.just_capture == True): next is 'robot'.
    """
    last = game.shots[-1] if game.shots else None
    if not last:
        return "robot" if game.order == "first" else "human"
    return "robot" if last.just_capture else "human"


@game_bp.route("/api/start_game", methods=["POST"])
@login_required
def start_game():
    data = request.get_json(force=True)
    mode = data.get("mode")
    difficulty = data.get("difficulty")
    order = data.get("order")
    if mode not in ("9-ball", "10-ball"):
        return jsonify({"error": "invalid mode"}), 400
    if difficulty not in ("low", "medium", "high"):
        return jsonify({"error": "invalid difficulty"}), 400
    if order not in ("first", "second"):
        return jsonify({"error": "invalid order"}), 400

    game = Game(user_id=current_user.id, mode=mode, difficulty=difficulty, order=order, status="active")
    db.session.add(game)
    db.session.commit()
    return jsonify({
        "game_id": game.id,
        "current_turn": _current_turn(game),
    })


@game_bp.route("/api/capture_only", methods=["POST"])
@login_required
def capture_only():
    game_id = request.json.get("game_id")
    game = Game.query.get(game_id)
    if not game or game.user_id != current_user.id:
        return jsonify({"error": "game not found"}), 404

    # Enforce turn: capture_only is for ending HUMAN turn
    if _current_turn(game) != "human":
        return jsonify({"error": "not human turn"}), 400

    balls_data, result = capture_and_plan(target="min", intrinsics_path=Config.INTRINSICS_PATH)
    plan_error = None if result is not None else get_last_plan_shot_error()

    # Prepare shot record
    step = (game.shots[-1].step_number + 1) if game.shots else 1
    # Save under Flask static folder and return URL under /static
    image_dir = Path(current_app.static_folder) / "renders"

    # Build render inputs if possible
    angle_deg, cue_xy = (result or (None, None))
    cue = cue_xy if cue_xy else (0.1, 0.1)
    # pick target: smallest non-zero id
    balls = balls_data.get("balls", [])
    nonzeros = [b for b in balls if (tid := _type_id(b)) is not None and tid != 0]
    target_b = None
    if nonzeros:
        target_b = min(nonzeros, key=lambda b: (_type_id(b) if _type_id(b) is not None else 99))

    def _get_xy(b):
        x = b.get("cx_cm", b.get("x_cm"))
        y = b.get("cy_cm", b.get("y_cm"))
        return x / 100.0, y / 100.0

    if target_b and any(k in target_b for k in ("cx_cm", "x_cm")):
        target = _get_xy(target_b)
        blockers = [_get_xy(b) for b in balls if b not in (target_b,) and (_type_id(b) or 0) != 0]
        # Compute shot info for visualization
        info = compute_shot(cue, target, blockers)
        # Build labels: show YOLO ball numbers at their detected coords
        labels = []
        for b in balls:
            t = b.get("type")
            if t is None:
                continue
            x = b.get("cx_cm", b.get("x_cm"))
            y = b.get("cy_cm", b.get("y_cm"))
            if x is None or y is None:
                continue
            labels.append((str(t), (x/100.0, y/100.0)))
        img_path = render_image(cue, target, blockers, info, image_dir, labels=labels)
        rel_img = url_for("static", filename=f"renders/{img_path.name}")
    else:
        rel_img = None

    shot = Shot(
        game_id=game.id,
        step_number=step,
        balls_json=json.dumps(balls_data, ensure_ascii=False),
        angle_deg=angle_deg or None,
        cue_x_m=(cue_xy[0] if cue_xy else None),
        cue_y_m=(cue_xy[1] if cue_xy else None),
        image_path=rel_img,
        just_capture=True,
    )
    db.session.add(shot)

    # End game if only cue remains after player's shot
    balls = balls_data.get("balls", [])
    remaining = _balls_remaining(balls)
    if remaining == 0:
        game.status = "ended"
        game.ended_at = datetime.utcnow()

    db.session.commit()

    return jsonify({
        "step": step,
        "balls_remaining": remaining,
        "image_path": rel_img,
        "angle_deg": angle_deg,
        "cue_xy": cue_xy,
        "plan_error": plan_error,
        "next_turn": _current_turn(game),
    })


@game_bp.route("/api/strike", methods=["POST"])
@login_required
def strike():
    game_id = request.json.get("game_id")
    game = Game.query.get(game_id)
    if not game or game.user_id != current_user.id:
        return jsonify({"error": "game not found"}), 404

    # Enforce turn: strike is for ROBOT turn
    if _current_turn(game) != "robot":
        return jsonify({"error": "not robot turn"}), 400

    # Follow robot server protocol: connect→wait MOVING→capture→send→wait DONE
    step = (game.shots[-1].step_number + 1) if game.shots else 1
    # Save under Flask static folder and return URL under /static
    image_dir = Path(current_app.static_folder) / "renders"

    angle_deg = None
    cue_xy = None
    payload = None
    sent = False
    reply = None
    if Config.SEND_TO_ROBOT:
        # Perform the full handshake session which also captures and plans
        balls_data, result, payload, reply = perform_handshaked_strike(
            target="min", intrinsics_path=Config.INTRINSICS_PATH
        )
        sent = payload is not None
        if result is not None:
            angle_deg, cue_xy = result
    else:
        # Compute-only path without robot I/O
        balls_data, result = capture_and_plan(target="min", intrinsics_path=Config.INTRINSICS_PATH)
        if result is not None:
            angle_deg, cue_xy = result
            payload = compute_arm_payload(angle_deg, cue_xy)
    plan_error = None if result is not None else get_last_plan_shot_error()

    # Build visualization image when possible
    balls = balls_data.get("balls", [])
    cue_b = next((b for b in balls if _type_id(b) == 0), None)
    nonzeros = [b for b in balls if (tid := _type_id(b)) is not None and tid != 0]
    def _get_xy(b):
        x = b.get("cx_cm", b.get("x_cm"))
        y = b.get("cy_cm", b.get("y_cm"))
        return x / 100.0, y / 100.0
    if cue_b and nonzeros:
        cue = _get_xy(cue_b)
        target_b = min(nonzeros, key=lambda b: (_type_id(b) if _type_id(b) is not None else 99))
        target = _get_xy(target_b)
        blockers = [_get_xy(b) for b in balls if b not in (cue_b, target_b)]
        info = compute_shot(cue, target, blockers)
        labels = []
        for b in balls:
            t = b.get("type")
            if t is None:
                continue
            x = b.get("cx_cm", b.get("x_cm"))
            y = b.get("cy_cm", b.get("y_cm"))
            if x is None or y is None:
                continue
            labels.append((str(t), (x/100.0, y/100.0)))
        img_path = render_image(cue, target, blockers, info, image_dir, labels=labels)
        rel_img = url_for("static", filename=f"renders/{img_path.name}")
    else:
        rel_img = None

    shot = Shot(
        game_id=game.id,
        step_number=step,
        balls_json=json.dumps(balls_data, ensure_ascii=False),
        angle_deg=angle_deg,
        cue_x_m=(cue_xy[0] if cue_xy else None),
        cue_y_m=(cue_xy[1] if cue_xy else None),
        image_path=rel_img,
        just_capture=False,
    )
    db.session.add(shot)

    # End game if only cue remains
    remaining = _balls_remaining(balls)
    if remaining == 0:
        game.status = "ended"
        game.ended_at = datetime.utcnow()

    db.session.commit()

    return jsonify({
        "step": step,
        "balls_remaining": remaining,
        "image_path": rel_img,
        "angle_deg": angle_deg,
        "cue_xy": cue_xy,
        "payload": payload,
        "sent": sent,
        "reply": reply,
        "plan_error": plan_error,
        "game_status": game.status,
        "next_turn": _current_turn(game),
    })


@game_bp.route("/api/game_status/<int:game_id>")
@login_required
def game_status(game_id: int):
    game = Game.query.get(game_id)
    if not game or game.user_id != current_user.id:
        return jsonify({"error": "game not found"}), 404
    last = game.shots[-1] if game.shots else None
    return jsonify({
        "game": {
            "id": game.id,
            "mode": game.mode,
            "difficulty": game.difficulty,
            "order": game.order,
            "status": game.status,
            "started_at": game.started_at.isoformat(),
            "ended_at": game.ended_at.isoformat() if game.ended_at else None,
            "current_turn": _current_turn(game),
        },
        "last_shot": {
            "step": last.step_number,
            "image_path": last.image_path,
            "angle_deg": last.angle_deg,
            "cue_x_m": last.cue_x_m,
            "cue_y_m": last.cue_y_m,
        } if last else None
    })


@game_bp.route("/scoreboard")
@login_required
def scoreboard():
    # Basic template; frontend JS fetches stats
    return render_template("scoreboard.html")


@game_bp.route("/api/stats/me")
@login_required
def stats_me():
    return _stats_for_user(current_user.id)


@game_bp.route("/api/stats/user/<int:user_id>")
@login_required
def stats_user(user_id: int):
    return _stats_for_user(user_id)


def _stats_for_user(user_id: int):
    games = Game.query.filter_by(user_id=user_id).all()
    series = []
    all_shots = Shot.query.join(Game, Shot.game_id == Game.id).filter(Game.user_id == user_id).order_by(Shot.timestamp).all()

    # Line series: balls remaining per step in each game
    for g in games:
        pts = []
        for s in g.shots:
            balls = json.loads(s.balls_json).get("balls", [])
            pts.append({"x": s.step_number, "y": _balls_remaining(balls)})
        series.append({"game_id": g.id, "points": pts})

    # Leaderboard metrics for this user
    total_games = len(games)
    total_shots = len(all_shots)

    return jsonify({
        "series": series,
        "totals": {"games": total_games, "shots": total_shots},
    })


@game_bp.route("/api/leaderboard")
@login_required
def leaderboard():
    # Simple leaderboard by total shots (lower is better) and total games
    users = User.query.all()
    ranks = []
    for u in users:
        games = Game.query.filter_by(user_id=u.id).count()
        shots = Shot.query.join(Game, Shot.game_id == Game.id).filter(Game.user_id == u.id).count()
        ranks.append({"user": u.username, "games": games, "shots": shots})
    # Sort: more games desc, fewer shots asc
    ranks.sort(key=lambda r: (-r["games"], r["shots"]))
    return jsonify({"leaderboard": ranks})
