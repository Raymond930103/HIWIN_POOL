import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import auth, pipeline
from .auth import get_current_user
from .schemas import JobControl

app = FastAPI()

origins = [os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)


@app.post("/pipeline/run")
async def run_pipeline(user=Depends(get_current_user)):
    try:
        return pipeline.start_job()
    except HTTPException as exc:
        if exc.status_code == 409:
            return JSONResponse(status_code=409, content=exc.detail)
        raise


@app.get("/pipeline/status")
async def get_pipeline_status(job_id: str, user=Depends(get_current_user)):
    return pipeline.get_status(job_id)


@app.post("/pipeline/stop")
async def stop_pipeline(payload: JobControl, user=Depends(get_current_user)):
    return pipeline.stop_job(payload.job_id)


@app.get("/healthz")
async def healthz():
    return {"ok": True}
