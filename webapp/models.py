from datetime import datetime
from flask_login import UserMixin
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import relationship

from .database import db


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    games = relationship("Game", back_populates="user")


class Game(db.Model):
    __tablename__ = "games"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    mode = Column(String(16), nullable=False)         # '9-ball' | '10-ball'
    difficulty = Column(String(16), nullable=False)   # 'low' | 'medium' | 'high'
    order = Column(String(16), nullable=False)        # 'first' | 'second'
    status = Column(String(16), default="active")    # 'active' | 'ended'
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="games")
    shots = relationship("Shot", back_populates="game", order_by="Shot.step_number")


class Shot(db.Model):
    __tablename__ = "shots"
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    step_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    balls_json = Column(Text, nullable=False)    # JSON string of balls snapshot
    angle_deg = Column(Float, nullable=True)
    cue_x_m = Column(Float, nullable=True)
    cue_y_m = Column(Float, nullable=True)
    image_path = Column(String(255), nullable=True)   # static file path for visualization
    just_capture = Column(Boolean, default=False)     # true if capture-only

    game = relationship("Game", back_populates="shots")

