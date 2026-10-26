"""
Timer Manager - parallel countdown timers with callbacks.

Each timer runs in its own daemon thread. When a timer finishes it calls
`on_expire(timer_id, label)` so the UI and the voice engine can alert the user.
"""

import threading
import time
import platform
from typing import Callable, Optional, Dict


class TimerNode:
    def __init__(self, duration_seconds: float, timer_id: str, label: str,
                 on_expire: Optional[Callable[[str, str], None]] = None):
        self.duration = float(duration_seconds)
        self.remaining = float(duration_seconds)
        self.timer_id = timer_id
        self.label = label
        self.on_expire = on_expire
        self.created_at = time.time()
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._countdown, daemon=True)
        self._thread.start()

    def _countdown(self):
        while True:
            with self._lock:
                if not self._running:
                    return
                if self.remaining <= 0:
                    self._running = False
                    expired = True
                else:
                    self.remaining -= 0.25
                    expired = False
            if expired:
                self._alert()
                if self.on_expire:
                    try:
                        self.on_expire(self.timer_id, self.label)
                    except Exception:
                        pass
                return
            time.sleep(0.25)

    def _alert(self):
        """Short system beep (best effort)."""
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 400)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass

    def stop(self):
        with self._lock:
            self._running = False

    def reset(self, seconds: float):
        with self._lock:
            self.remaining = float(seconds)
            self.duration = float(seconds)
            self._running = True
        self._thread = threading.Thread(target=self._countdown, daemon=True)
        self._thread.start()

    def status(self) -> dict:
        with self._lock:
            recent = 0.0
            if self.created_at:
                recent = max(0.0, time.time() - self.created_at)
            return {
                "id": self.timer_id,
                "label": self.label,
                "remaining": round(max(0.0, self.remaining), 1),
                "total": round(self.duration, 1),
                "elapsed": round(min(self.duration, recent), 1),
                "running": self._running,
            }


class TimerManager:
    def __init__(self):
        self.timers: Dict[str, TimerNode] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self.on_expire: Optional[Callable[[str, str], None]] = None

    def _next_id(self) -> str:
        self._counter += 1
        return f"timer_{self._counter}"

    def create(self, seconds: float, label: str = "") -> str:
        with self._lock:
            tid = self._next_id()
            node = TimerNode(seconds, tid, label, on_expire=self._handle_expire)
            self.timers[tid] = node
        node.start()
        return tid

    def _handle_expire(self, timer_id: str, label: str):
        with self._lock:
            self.timers.pop(timer_id, None)
        if self.on_expire:
            try:
                self.on_expire(timer_id, label)
            except Exception:
                pass

    def cancel(self, timer_id: str) -> bool:
        with self._lock:
            node = self.timers.pop(timer_id, None)
        if node:
            node.stop()
            return True
        return False

    def cancel_all(self):
        with self._lock:
            nodes = list(self.timers.values())
            self.timers.clear()
        for n in nodes:
            n.stop()

    def reset(self, timer_id: str, seconds: float) -> bool:
        with self._lock:
            node = self.timers.get(timer_id)
        if node:
            node.reset(seconds)
            return True
        return False

    def list(self) -> list:
        with self._lock:
            nodes = list(self.timers.values())
        return sorted((n.status() for n in nodes), key=lambda t: t["id"])

    def get(self, timer_id: str):
        with self._lock:
            return self.timers.get(timer_id)


# Global timer manager
timers = TimerManager()
