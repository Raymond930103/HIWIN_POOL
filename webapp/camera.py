import threading
import time
from typing import Optional

import cv2

try:
    from .config import Config
except ImportError:  # fallback if run as script
    from webapp.config import Config


class CameraStreamer:
    def __init__(self, index: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    def _try_open(self, width: Optional[int], height: Optional[int]):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            return None
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Probe one frame to ensure stream works at this size
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
        return None

    def _open(self):
        # Try requested size first, then common fallbacks
        tried = set()
        candidates = []
        if self.width or self.height:
            candidates.append((self.width, self.height))
        # Add common HD/SD sizes
        for wh in ((1920, 1080), (1280, 720), (640, 480), (None, None)):
            if wh not in candidates:
                candidates.append(wh)
        for w, h in candidates:
            key = (w or -1, h or -1)
            if key in tried:
                continue
            tried.add(key)
            cap = self._try_open(w, h)
            if cap is not None:
                # Record actual size for reference
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return cap
        return None

    def _ensure_open(self):
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                # Try reopen
                self._release_locked()
                self._cap = self._open()

    def _release_locked(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def release(self):
        with self._lock:
            self._release_locked()

    def get_jpeg(self) -> Optional[bytes]:
        self._ensure_open()
        with self._lock:
            if self._cap is None:
                return None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Reopen on failure
                self._release_locked()
                return None
            # Optional downscale for smooth streaming
            try:
                from .config import Config as _Cfg
                max_w = getattr(_Cfg, 'CAMERA_STREAM_MAX_WIDTH', None)
            except Exception:
                max_w = None
            if max_w:
                h, w = frame.shape[:2]
                if w > max_w:
                    scale = max_w / float(w)
                    new_size = (int(w * scale), int(h * scale))
                    frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return None
            return bytes(buf)


# Global singleton streamer used by routes
camera_streamer = CameraStreamer(
    index=getattr(Config, 'CAMERA_INDEX', 0),
    width=getattr(Config, 'CAMERA_WIDTH', None),
    height=getattr(Config, 'CAMERA_HEIGHT', None),
)
