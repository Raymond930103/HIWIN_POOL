from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .auth import get_current_user
from main.communicate import tcp_communicate

router = APIRouter(prefix="/robot", tags=["robot"])

robot_state = {"status": "idle", "last_error": ""}


class ShotCmd(BaseModel):
    angle: float
    power: float
    spin: float = 0.0


@router.post("/shoot")
def shoot(cmd: ShotCmd, user=Depends(get_current_user)):
    robot_state["status"] = "busy"
    success, resp = tcp_communicate.send_shot(cmd.angle, cmd.power, cmd.spin)
    if success:
        robot_state["status"] = "idle"
        return {"success": True, "response": resp}
    else:
        robot_state["status"] = "error"
        robot_state["last_error"] = resp
        return {"success": False, "response": resp}


@router.get("/status")
def status():
    return robot_state


@router.post("/stop")
def stop(user=Depends(get_current_user)):
    success, resp = tcp_communicate.send_stop()
    if success:
        robot_state["status"] = "idle"
    else:
        robot_state["status"] = "error"
        robot_state["last_error"] = resp
    return {"success": success, "response": resp}
