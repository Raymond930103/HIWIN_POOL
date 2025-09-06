from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import models, database
from .auth import router as auth_router
from .matches import router as match_router
from .robot import router as robot_router

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or ["*"] during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(match_router)
app.include_router(robot_router)