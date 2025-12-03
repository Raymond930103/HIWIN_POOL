import json
from datetime import datetime, timedelta

import pytest
from werkzeug.security import generate_password_hash
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.app import create_app
from webapp.config import Config
from webapp.database import db
from webapp.models import Game, Shot, User


class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    SEND_TO_ROBOT = False


@pytest.fixture
def app(tmp_path):
    TestConfig.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path/'test.db'}"
    application = create_app(TestConfig)
    with application.app_context():
        db.drop_all()
        db.create_all()
    yield application
    with application.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_client(app, client):
    with app.app_context():
        user = User(username="tester", password_hash=generate_password_hash("pass"))
        db.session.add(user)
        db.session.commit()
        user_id = user.id
    with client.session_transaction() as session:
        session["_user_id"] = str(user_id)
        session["_fresh"] = True
    return client, user_id


def _balls_payload(count: int) -> str:
    balls = [{"type": idx + 1} for idx in range(count)]
    return json.dumps({"balls": balls})


def _create_game(user_id: int, started_at: datetime, remaining_sequence, status="ended"):
    game = Game(
        user_id=user_id,
        mode="9-ball",
        difficulty="low",
        order="first",
        status=status,
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=30),
    )
    db.session.add(game)
    db.session.flush()

    for idx, remaining in enumerate(remaining_sequence, start=1):
        shot = Shot(
            game_id=game.id,
            step_number=idx,
            timestamp=started_at + timedelta(minutes=idx * 5),
            balls_json=_balls_payload(remaining),
            just_capture=False,
        )
        db.session.add(shot)

    db.session.commit()
    return game.id


def test_stats_me_avg_series_and_limit(app, auth_client):
    client, user_id = auth_client
    with app.app_context():
        base = datetime(2024, 1, 1, 10, 0, 0)
        _create_game(user_id, base, [5, 3, 0])
        game2_id = _create_game(user_id, base + timedelta(days=1), [7, 2, 1])

    resp = client.get("/api/stats/me?limit_games=2")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["series"]) == 2
    avg_map = {point["x"]: point["y"] for point in data["avg_series"]}
    assert avg_map[1] == pytest.approx(6.0)
    assert avg_map[2] == pytest.approx(2.5)
    assert avg_map[3] == pytest.approx(0.5)
    assert data["totals"]["games"] == 2
    assert data["totals"]["shots"] == 6

    resp_latest = client.get("/api/stats/me?limit_games=1")
    assert resp_latest.status_code == 200
    latest = resp_latest.get_json()
    assert len(latest["series"]) == 1
    latest_game = latest["series"][0]
    assert latest_game["game_id"] == game2_id
    avg_latest = {point["x"]: point["y"] for point in latest["avg_series"]}
    assert avg_latest[1] == pytest.approx(7.0)
    assert avg_latest[2] == pytest.approx(2.0)
    assert avg_latest[3] == pytest.approx(1.0)


def test_stats_me_date_filters(app, auth_client):
    client, user_id = auth_client
    with app.app_context():
        base = datetime(2024, 1, 1, 12, 0, 0)
        _create_game(user_id, base, [6, 4])
        game_recent_id = _create_game(user_id, base + timedelta(days=2), [5, 1])

    resp = client.get("/api/stats/me?from=2024-01-03")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["series"]) == 1
    assert data["series"][0]["game_id"] == game_recent_id

    resp_range = client.get("/api/stats/me?from=2024-01-02&to=2024-01-02")
    assert resp_range.status_code == 200
    data_range = resp_range.get_json()
    assert data_range["series"] == []
    assert data_range["avg_series"] == []
    assert data_range["totals"]["games"] == 0
