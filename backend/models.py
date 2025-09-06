from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
import enum
from .database import Base


class MatchMode(str, enum.Enum):
    nine = "9"
    ten = "10"


class Difficulty(str, enum.Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class Turn(str, enum.Enum):
    player = "player"
    ai = "ai"


class ShotResult(str, enum.Enum):
    pocket = "pocket"
    miss = "miss"
    foul = "foul"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    display_name = Column(String, nullable=False)

    matches = relationship("Match", back_populates="user")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mode = Column(Enum(MatchMode), nullable=False)
    difficulty = Column(Enum(Difficulty), nullable=False)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    result = Column(String)

    user = relationship("User", back_populates="matches")
    rounds = relationship("Round", back_populates="match")


class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    turn = Column(Enum(Turn), nullable=False)
    shot_result = Column(Enum(ShotResult))
    state_json = Column(String)

    match = relationship("Match", back_populates="rounds")
