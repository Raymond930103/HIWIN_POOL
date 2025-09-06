from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr
from .models import MatchMode, Difficulty, Turn, ShotResult

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    display_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    display_name: str
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MatchCreate(BaseModel):
    mode: MatchMode
    difficulty: Difficulty

class RoundCreate(BaseModel):
    turn: Turn
    shot_result: ShotResult
    state_json: str

class RoundOut(BaseModel):
    id: int
    turn: Turn
    shot_result: ShotResult
    state_json: str
    class Config:
        orm_mode = True

class MatchOut(BaseModel):
    id: int
    mode: MatchMode
    difficulty: Difficulty
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    result: Optional[str]
    rounds: List[RoundOut] = []
    class Config:
        orm_mode = True
