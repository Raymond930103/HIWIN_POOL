import threading
import time
from typing import Callable


def run_once_or_loop(stop_event: threading.Event, on_log: Callable[[str], None]) -> int:
    """
    Simplified automatic shot pipeline.

    This function performs a sequence of mocked steps. Each step checks
    ``stop_event`` so the caller can request termination. Messages are
    reported via ``on_log``.
    """
    try:
        on_log("connecting to robot controller")
        for _ in range(5):
            if stop_event.is_set():
                on_log("stopped during connect")
                return 1
            time.sleep(0.5)
        steps = ["MOVING received", "capturing", "detecting", "solving", "sending result"]
        for step in steps:
            if stop_event.is_set():
                on_log("stop requested")
                return 1
            on_log(step)
            time.sleep(1)
        on_log("pipeline finished")
        return 0
    except Exception as exc:  # noqa: BLE001
        on_log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    evt = threading.Event()
    def printer(msg: str) -> None:
        print(msg)
    code = run_once_or_loop(evt, printer)
    raise SystemExit(code)
