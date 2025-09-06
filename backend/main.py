from fastapi import FastAPI

from . import models, database
from .auth import router as auth_router
from .matches import router as match_router
from .robot import router as robot_router

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.include_router(auth_router)
app.include_router(match_router)
app.include_router(robot_router)
