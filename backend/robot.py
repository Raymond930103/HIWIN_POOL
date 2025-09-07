from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .auth import get_current_user
from main.communicate import tcp_communicate
from main.vision.yoloball import capture_balls
from main.plan_shot import plan_shot_from_json

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


@router.post("/auto")
def auto_shoot(user=Depends(get_current_user)):
    """Run full pipeline: capture → plan → shoot."""
    robot_state["status"] = "busy"
    try:
        json_path, _ = capture_balls(
            wait_sec=3, show=False, intrinsics_path="main/vision/intrinsics.yaml"
        )
        if not json_path:
            raise RuntimeError("capture failed")

        result = plan_shot_from_json(json_path, "min", show=False)
        if result is None:
            raise RuntimeError("planning failed")

        angle_deg, _ = result
        power = 0.8  # default power

        success, resp = tcp_communicate.send_shot(angle_deg, power, 0.0)
        if success:
            robot_state["status"] = "idle"
            return {"success": True, "angle": angle_deg, "power": power, "response": resp}
        else:
            robot_state["status"] = "error"
            robot_state["last_error"] = resp
            return {"success": False, "response": resp}
    except Exception as e:
        robot_state["status"] = "error"
        robot_state["last_error"] = str(e)
        return {"success": False, "response": str(e)}


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
