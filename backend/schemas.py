from pydantic import BaseModel, EmailStr
from typing import Optional, Literal


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class JobControl(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["idle", "queued", "running", "success", "failed", "stopped"]
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    last_log: str = ""
    exit_code: Optional[int] = None
