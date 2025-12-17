import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Dict, Optional, Tuple

from flask import Blueprint, render_template, request, jsonify, url_for, current_app
import re
from flask_login import login_required, current_user
from sqlalchemy.orm import joinedload
from sqlalchemy import func

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

AVG_SERIES_CACHE_TTL_SEC = 300  # five minutes
_avg_series_cache: Dict[Tuple, Tuple[float, list]] = {}
_avg_series_cache_lock = RLock()


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


def _parse_date_arg(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def _parse_bool_arg(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_limit_arg(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _extract_stats_query_args(args) -> Dict[str, Optional[object]]:
    raw_from = args.get("from")
    raw_to = args.get("to")
    date_from = _parse_date_arg(raw_from)
    date_to = _parse_date_arg(raw_to)
    limit_games = _parse_limit_arg(args.get("limit_games"))
    only_last = _parse_bool_arg(args.get("only_last"))

    return {
        "date_from": date_from,
        "date_to": (date_to + timedelta(days=1)) if date_to else None,
        "limit_games": limit_games,
        "only_last": only_last,
        "raw_from": raw_from if date_from else None,
        "raw_to": raw_to if date_to else None,
    }


def _avg_series_cache_key(user_id: int, filters: Dict[str, Optional[object]]) -> Tuple:
    return (
        user_id,
        filters.get("raw_from"),
        filters.get("raw_to"),
        filters.get("limit_games"),
        filters.get("only_last"),
    )


def _get_cached_avg_series(cache_key: Tuple):
    with _avg_series_cache_lock:
        cached = _avg_series_cache.get(cache_key)
        if not cached:
            return None
        timestamp, payload = cached
        if time.time() - timestamp > AVG_SERIES_CACHE_TTL_SEC:
            # Expired
            _avg_series_cache.pop(cache_key, None)
            return None
        return payload


def _set_cached_avg_series(cache_key: Tuple, payload):
    with _avg_series_cache_lock:
        _avg_series_cache[cache_key] = (time.time(), payload)


def _collect_game_data(game: Game, step_accumulator: Optional[defaultdict] = None) -> Dict[str, object]:
    points = []
    if not game.shots:
        return {
            "points": points,
            "last_step": 0,
            "first_timestamp": None,
            "last_timestamp": None,
            "duration_seconds": None,
            "shot_count": 0,
        }

    first_ts = game.shots[0].timestamp
    last_ts = None
    last_step = 0

    for shot in game.shots:
        try:
            payload = json.loads(shot.balls_json or "{}")
        except (TypeError, json.JSONDecodeError):
            payload = {}
        balls = payload.get("balls", [])
        remaining = _balls_remaining(balls)

        point = {"x": shot.step_number, "y": remaining}
        if shot.timestamp:
            point["t"] = shot.timestamp.isoformat()
            if first_ts:
                point["elapsed_seconds"] = (shot.timestamp - first_ts).total_seconds()
        points.append(point)

        last_step = shot.step_number
        last_ts = shot.timestamp or last_ts

        if step_accumulator is not None:
            agg = step_accumulator[shot.step_number]
            agg["sum"] += remaining
            agg["count"] += 1

    duration_seconds = None
    if first_ts and last_ts:
        duration_seconds = (last_ts - first_ts).total_seconds()

    return {
        "points": points,
        "last_step": last_step if points else 0,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "duration_seconds": duration_seconds,
        "shot_count": len(points),
    }


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
    session_error = None
    if Config.SEND_TO_ROBOT:
        # Perform the full handshake session which also captures and plans
        balls_data, result, payload, reply, session_error = perform_handshaked_strike(
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
    plan_error = None if result is not None else (session_error or get_last_plan_shot_error())

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

    # Record shot; set just_capture=False to pass turn to human next
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

    # Provide a human-readable status for the frontend
    if sent:
        status_msg = session_error or "已送出座標，等待機器人完成"
    else:
        status_msg = (plan_error or session_error or "找不到可行路徑，換玩家回合")

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
        "error_msg": session_error,
        "game_status": game.status,
        "next_turn": _current_turn(game),
        "status_msg": status_msg,
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


@game_bp.route("/api/keep_turn", methods=["POST"])
@login_required
def keep_turn():
    """Force keeping the current striker's turn.

    Implementation detail: we append a minimal Shot row with `just_capture`
    set so that `_current_turn(game)` stays the same side as before.
    - If it's robot's turn now -> set last.just_capture=True (robot still next)
    - If it's human's turn now -> set last.just_capture=False (human still next)
    This avoids DB schema changes.
    """
    game_id = request.json.get("game_id")
    game = Game.query.get(game_id)
    if not game or game.user_id != current_user.id:
        return jsonify({"error": "game not found"}), 404
    if game.status == "ended":
        return jsonify({"error": "game already ended"}), 400

    # Determine current side and create a no-op shot to preserve the turn
    current = _current_turn(game)
    last = game.shots[-1] if game.shots else None
    step = (last.step_number + 1) if last else 1
    balls_json = last.balls_json if last else json.dumps({"balls": []})

    # Choose just_capture to preserve the same side next
    preserve_robot = (current == "robot")
    just_capture = True if preserve_robot else False

    shot = Shot(
        game_id=game.id,
        step_number=step,
        balls_json=balls_json,
        angle_deg=None,
        cue_x_m=None,
        cue_y_m=None,
        image_path=(last.image_path if last else None),
        just_capture=just_capture,
    )
    db.session.add(shot)
    db.session.commit()

    # Compute remaining from balls_json we saved
    try:
        balls = json.loads(balls_json).get("balls", [])
        remaining = _balls_remaining(balls)
    except Exception:
        remaining = None

    return jsonify({
        "step": step,
        "balls_remaining": remaining,
        "next_turn": _current_turn(game),
        "kept": True,
    })


@game_bp.route("/api/switch_turn", methods=["POST"])
@login_required
def switch_turn():
    """Switch the side to play next (toggle current turn).

    Achieved by appending a minimal Shot with `just_capture` set to
    make `_current_turn(game)` return the opposite side.
    """
    game_id = request.json.get("game_id")
    game = Game.query.get(game_id)
    if not game or game.user_id != current_user.id:
        return jsonify({"error": "game not found"}), 404
    if game.status == "ended":
        return jsonify({"error": "game already ended"}), 400

    current = _current_turn(game)
    desired = "robot" if current == "human" else "human"
    # just_capture=True => next is robot, False => next is human
    just_capture = True if desired == "robot" else False

    last = game.shots[-1] if game.shots else None
    step = (last.step_number + 1) if last else 1
    balls_json = last.balls_json if last else json.dumps({"balls": []})

    shot = Shot(
        game_id=game.id,
        step_number=step,
        balls_json=balls_json,
        angle_deg=None,
        cue_x_m=None,
        cue_y_m=None,
        image_path=(last.image_path if last else None),
        just_capture=just_capture,
    )
    db.session.add(shot)
    db.session.commit()

    try:
        balls = json.loads(balls_json).get("balls", [])
        remaining = _balls_remaining(balls)
    except Exception:
        remaining = None

    return jsonify({
        "step": step,
        "balls_remaining": remaining,
        "next_turn": _current_turn(game),
        "switched": True,
    })


@game_bp.route("/scoreboard")
@login_required
def scoreboard():
    # Basic template; frontend JS fetches stats
    return render_template("scoreboard.html")


@game_bp.route("/api/stats/me")
@login_required
def stats_me():
    filters = _extract_stats_query_args(request.args)
    return _stats_for_user(current_user.id, filters)


@game_bp.route("/api/stats/user/<int:user_id>")
@login_required
def stats_user(user_id: int):
    filters = _extract_stats_query_args(request.args)
    return _stats_for_user(user_id, filters)


@game_bp.route("/api/stats/summary")
@login_required
def stats_summary():
    return _stats_summary_for_user(current_user.id)


def _stats_for_user(user_id: int, filters: Dict[str, Optional[object]]):
    user_exists = User.query.get(user_id)
    if not user_exists:
        return jsonify({"error": "user not found"}), 404

    query = Game.query.filter(Game.user_id == user_id)
    if filters.get("date_from"):
        query = query.filter(Game.started_at >= filters["date_from"])
    if filters.get("date_to"):
        query = query.filter(Game.started_at < filters["date_to"])

    query = query.order_by(Game.started_at.desc())
    limit = 1 if filters.get("only_last") else filters.get("limit_games")
    if limit:
        query = query.limit(limit)

    games = query.options(joinedload(Game.shots)).all()

    cache_key = _avg_series_cache_key(user_id, filters)
    avg_series = _get_cached_avg_series(cache_key)
    step_accumulator = defaultdict(lambda: {"sum": 0.0, "count": 0}) if avg_series is None else None

    series = []
    game_summaries = []
    total_shots = 0
    completed_games = 0

    for game in games:
        game_payload = _collect_game_data(game, step_accumulator)
        points = game_payload["points"]
        total_shots += game_payload["shot_count"]
        if game.status == "ended":
            completed_games += 1

        series.append({
            "game_id": game.id,
            "points": points,
            "started_at": game.started_at.isoformat() if game.started_at else None,
            "ended_at": game.ended_at.isoformat() if game.ended_at else None,
            "status": game.status,
            "mode": game.mode,
            "difficulty": game.difficulty,
        })

        game_summaries.append({
            "game_id": game.id,
            "total_steps": game_payload["last_step"],
            "shot_count": game_payload["shot_count"],
            "completed": game.status == "ended",
            "started_at": game.started_at.isoformat() if game.started_at else None,
            "ended_at": game.ended_at.isoformat() if game.ended_at else None,
            "duration_seconds": game_payload["duration_seconds"],
            "last_updated_at": game_payload["last_timestamp"].isoformat() if game_payload["last_timestamp"] else None,
            "mode": game.mode,
            "difficulty": game.difficulty,
        })

    if step_accumulator is not None:
        avg_series = []
        for step, stats in sorted(step_accumulator.items()):
            count = stats["count"]
            if not count:
                continue
            avg_series.append({"x": step, "y": stats["sum"] / count})
        _set_cached_avg_series(cache_key, avg_series)
    elif avg_series is None:
        avg_series = []

    applied_limit = 1 if filters.get("only_last") else filters.get("limit_games")

    return jsonify({
        "series": series,
        "avg_series": avg_series,
        "game_summaries": game_summaries,
        "totals": {
            "games": len(games),
            "completed_games": completed_games,
            "shots": total_shots,
        },
        "filters": {
            "from": filters.get("raw_from"),
            "to": filters.get("raw_to"),
            "limit_games": applied_limit,
            "only_last": bool(filters.get("only_last")),
        },
    })


def _stats_summary_for_user(user_id: int):
    user_exists = User.query.get(user_id)
    if not user_exists:
        return jsonify({"error": "user not found"}), 404

    total_games = db.session.query(func.count(Game.id)).filter(Game.user_id == user_id).scalar() or 0
    total_shots = (
        db.session.query(func.count(Shot.id))
        .join(Game, Shot.game_id == Game.id)
        .filter(Game.user_id == user_id)
        .scalar()
        or 0
    )

    avg_steps = (total_shots / total_games) if total_games else 0.0
    trend = _build_summary_trend(user_id)

    return jsonify({
        "total_games": total_games,
        "total_shots": total_shots,
        "avg_steps_per_game": avg_steps,
        "trend": trend,
    })


def _build_summary_trend(user_id: int) -> Dict[str, list]:
    today = datetime.utcnow().date()
    start_30 = today - timedelta(days=29)
    start_dt = datetime.combine(start_30, datetime.min.time())

    games = (
        Game.query.filter(Game.user_id == user_id, Game.started_at >= start_dt)
        .options(joinedload(Game.shots))
        .all()
    )

    per_day = {}
    for game in games:
        if not game.started_at:
            continue
        day = game.started_at.date()
        if day < start_30:
            continue
        stats = per_day.setdefault(day, {"games": 0, "steps": 0})
        stats["games"] += 1
        last_step = game.shots[-1].step_number if game.shots else 0
        stats["steps"] += last_step

    span = (today - start_30).days + 1
    trend_30 = []
    for offset in range(span):
        day = start_30 + timedelta(days=offset)
        stats = per_day.get(day, {"games": 0, "steps": 0})
        games_count = stats["games"]
        steps = stats["steps"]
        avg_steps = (steps / games_count) if games_count else 0.0
        trend_30.append({
            "date": day.isoformat(),
            "games": games_count,
            "steps": steps,
            "avg_steps": avg_steps,
        })

    trend_7 = trend_30[-7:] if len(trend_30) >= 7 else trend_30[:]
    return {"7d": trend_7, "30d": trend_30}


@game_bp.route("/api/leaderboard")
@login_required
def leaderboard():
    sort_key = request.args.get("sort", "").lower()

    users = User.query.all()
    game_counts = dict(
        db.session.query(Game.user_id, func.count(Game.id))
        .group_by(Game.user_id)
        .all()
    )
    shot_counts = dict(
        db.session.query(Game.user_id, func.count(Shot.id))
        .join(Game, Shot.game_id == Game.id)
        .group_by(Game.user_id)
        .all()
    )
    completed_counts = dict(
        db.session.query(Game.user_id, func.count(Game.id))
        .filter(Game.status == "ended")
        .group_by(Game.user_id)
        .all()
    )

    ranks = []
    for user in users:
        games = game_counts.get(user.id, 0)
        shots = shot_counts.get(user.id, 0)
        avg_steps = (shots / games) if games else None
        completed_games = completed_counts.get(user.id, 0)
        ranks.append({
            "user": user.username,
            "games": games,
            "shots": shots,
            "avg_steps_per_game": avg_steps,
            # Winner data is not currently tracked; expose null for UI fallback.
            "win_rate": None,
            "completed_games": completed_games,
        })

    if sort_key == "avg_steps":
        ranks.sort(key=lambda r: (
            float("inf") if r["avg_steps_per_game"] is None else r["avg_steps_per_game"],
            -r["games"],
            r["shots"],
        ))
    else:
        ranks.sort(key=lambda r: (-r["games"], r["shots"]))

    return jsonify({"leaderboard": ranks, "sort": "avg_steps" if sort_key == "avg_steps" else "games"})
