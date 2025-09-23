from pathlib import Path


class Config:
    # Change in production
    SECRET_KEY = "dev-secret-change-me"
    # SQLite DB in repo folder
    DATA_DIR = Path(__file__).resolve().parent
    DB_PATH = DATA_DIR / "app.db"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Robot send toggle: if False, only compute and record
    SEND_TO_ROBOT = True

    # Intrinsics path for capture (optional)
    INTRINSICS_PATH = "main/vision/intrinsics.yaml"

    # Camera settings for live preview
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1920  # e.g., 1280
    CAMERA_HEIGHT = 1080  # e.g., 720
    # Downscale large frames for streaming to keep UI smooth
    CAMERA_STREAM_MAX_WIDTH = 1280
