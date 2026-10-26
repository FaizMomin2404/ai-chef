"""
Cooking State Manager - tracks the guided cooking session.

The session holds the recipe, where the user is, what they have finished,
which preparation tasks are confirmed, the timers that are running, and a
short conversation history. State is updated only from explicit user
confirmations (or a completed timer), never just because the AI spoke.
"""

from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List
import json
import re
import time


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "do": 2, "teen": 3, "char": 4, "chaar": 4, "panch": 5,
    "paanch": 5, "chhe": 6, "che": 6, "saat": 7, "aath": 8,
    "nau": 9, "das": 10,
}


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().split())


def estimate_task_seconds(step: str) -> int:
    """Estimate how long a step should take before asking about progress."""
    low = (step or "").lower()
    minutes = 0.0
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*)?"
        r"(minutes?|mins?|hours?|hrs?)",
        low,
    ):
        low_v = float(m.group(1))
        high_v = float(m.group(2)) if m.group(2) else low_v
        value = max(low_v, high_v)
        if m.group(3).startswith(("hour", "hr")):
            value *= 60
        minutes += value
    if minutes > 0:
        return int(max(60, min(1800, minutes * 60)))
    if any(p in low for p in ("until golden", "until soft", "until tender",
                              "until done", "until the oil separates")):
        return 240
    if any(p in low for p in ("boil", "simmer", "cook", "bake", "roast")):
        return 300
    return 180


@dataclass
class CookingSession:
    """State for one cooking session."""

    selected_dish: Optional[str] = None
    servings: int = 2
    ingredients: List[dict] = field(default_factory=list)   # [{"name":..., "quantity":...}]
    steps: List[str] = field(default_factory=list)
    current_step_index: int = 0
    completed_steps: Set[int] = field(default_factory=set)
    active_timers: Dict[str, dict] = field(default_factory=dict)
    last_user_response: Optional[str] = None
    session_started: bool = False
    finished: bool = False
    stopped: bool = False
    awaiting: str = "dish"   # dish | variant | servings | confirm | cooking | confirm_skip | finished | stopped

    # Rich context
    conversation: List[dict] = field(default_factory=list)   # [{"role","text","ts"}]
    prepared: Set[str] = field(default_factory=set)          # ingredient names the user confirmed
    tasks: Dict[str, dict] = field(default_factory=dict)     # per-step task records
    conditions: Dict[str, str] = field(default_factory=dict) # e.g. {"water": "not hot yet"}
    next_action: Optional[str] = None
    missing_ingredients: List[str] = field(default_factory=list)
    duration_minutes: int = 0
    safety: List[str] = field(default_factory=list)
    recipe_source: str = "ai"                                # ai | fallback
    started_at: Optional[float] = None
    last_activity: float = field(default_factory=time.time)
    paused: bool = False
    total_time_minutes: int = 0
    prep_time_minutes: int = 0
    cook_time_minutes: int = 0

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def current_step(self) -> Optional[str]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def progress_percent(self) -> int:
        if not self.steps:
            return 0
        return round((len(self.completed_steps) / len(self.steps)) * 100)

    @property
    def has_recipe(self) -> bool:
        return bool(self.selected_dish) and bool(self.steps)

    @property
    def pending_steps(self) -> List[int]:
        return [i for i in range(len(self.steps)) if i not in self.completed_steps]

    @property
    def current_task(self) -> Optional[dict]:
        task = self.tasks.get(str(self.current_step_index))
        if task is None:
            return None
        return task

    @property
    def elapsed_seconds(self) -> int:
        if not self.started_at:
            return 0
        return int(time.time() - self.started_at)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _build_tasks(self):
        self.tasks = {}
        for idx, step in enumerate(self.steps):
            self.tasks[str(idx)] = {
                "step_index": idx,
                "label": step[:90],
                "status": "pending",      # pending | started | done
                "started_at": None,
                "completed_at": None,
                "ask_after": estimate_task_seconds(step),
                "asked": 0,
                "last_ask": None,
                "note": "",
            }

    def prepare(self, dish: str, servings: int, ingredients: List[dict],
                steps: List[str], duration_minutes: int = 0,
                prep_time_minutes: int = 0, cook_time_minutes: int = 0,
                total_time_minutes: int = 0,
                safety: Optional[List[str]] = None, source: str = "ai",
                missing: Optional[List[str]] = None):
        """Load a recipe and wait for the user to confirm the ingredients."""
        self.selected_dish = dish
        self.servings = max(1, int(servings))
        self.ingredients = ingredients or []
        self.steps = [s.strip() for s in steps if s and s.strip()]
        self.current_step_index = 0
        self.completed_steps = set()
        self.active_timers = {}
        self.last_user_response = None
        self.session_started = False
        self.finished = False
        self.stopped = False
        self.awaiting = "confirm"
        self.conversation = []
        self.prepared = set()
        self.conditions = {}
        self.missing_ingredients = list(missing or [])
        self.duration_minutes = int(duration_minutes or 0)
        self.total_time_minutes = int(total_time_minutes or 0)
        self.prep_time_minutes = int(prep_time_minutes or 0)
        self.cook_time_minutes = int(cook_time_minutes or 0)
        self.safety = list(safety or [])
        self.recipe_source = source
        self.started_at = None
        self.last_activity = time.time()
        self.paused = False
        self.next_action = self.steps[0] if self.steps else None
        self._build_tasks()

    def begin(self):
        """Start guided cooking after the user says they are ready."""
        self.session_started = True
        self.stopped = False
        self.awaiting = "cooking"
        self.started_at = time.time()
        self.last_activity = time.time()
        self.next_action = self.current_step

    def start(self, dish: str, servings: int, ingredients: List[dict], steps: List[str]):
        """Backwards-compatible shortcut: prepare and begin immediately."""
        self.prepare(dish, servings, ingredients, steps)
        self.begin()

    def reset(self):
        """Clear the session completely."""
        self.__init__()

    def end(self, stopped: bool = False):
        """Mark the dish as finished, or the session as stopped early."""
        self.finished = True
        self.stopped = stopped
        self.session_started = False
        self.awaiting = "stopped" if stopped else "finished"
        self.paused = False

    # ------------------------------------------------------------------
    # Conversation and context
    # ------------------------------------------------------------------
    def record_message(self, role: str, text: str):
        text = (text or "").strip()
        if not text:
            return
        self.conversation.append({"role": role, "text": text[:500], "ts": time.time()})
        self.conversation = self.conversation[-30:]
        self.last_activity = time.time()

    def set_condition(self, key: str, value: str):
        self.conditions[key] = value
        self.last_activity = time.time()

    def mark_prepared(self, name: str):
        name = _normalise(name)
        if name:
            self.prepared.add(name)
        self.last_activity = time.time()

    def is_prepared(self, name: str) -> bool:
        n = _normalise(name)
        return any(p == n or p in n or n in p for p in self.prepared)

    # ------------------------------------------------------------------
    # Task tracking (only confirmed by the user)
    # ------------------------------------------------------------------
    def task_for_step(self, index: Optional[int] = None) -> Optional[dict]:
        idx = self.current_step_index if index is None else index
        return self.tasks.get(str(idx))

    def start_task(self, index: Optional[int] = None, note: str = "") -> Optional[dict]:
        task = self.task_for_step(index)
        if task is None:
            return None
        if task["status"] == "done":
            return task
        task["status"] = "started"
        task["started_at"] = task["started_at"] or time.time()
        task["note"] = note or task.get("note", "")
        self.last_activity = time.time()
        return task

    def complete_task(self, index: Optional[int] = None) -> Optional[dict]:
        task = self.task_for_step(index)
        if task is None:
            return None
        task["status"] = "done"
        task["completed_at"] = time.time()
        self.last_activity = time.time()
        return task

    def reset_task(self, index: int) -> Optional[dict]:
        task = self.tasks.get(str(index))
        if task is None:
            return None
        task["status"] = "pending"
        task["started_at"] = None
        task["completed_at"] = None
        task["asked"] = 0
        task["last_ask"] = None
        return task

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def next_step(self) -> Optional[str]:
        """Mark the current step done and move on. Returns the new step text."""
        if not self.steps:
            return None
        self.completed_steps.add(self.current_step_index)
        self.complete_task(self.current_step_index)
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self.next_action = self.current_step
            self.last_activity = time.time()
            return self.current_step
        self.finished = True
        self.session_started = False
        self.awaiting = "finished"
        self.next_action = None
        return self.steps[-1]

    def previous_step(self) -> Optional[str]:
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.completed_steps.discard(self.current_step_index)
            self.reset_task(self.current_step_index)
            self.next_action = self.current_step
            self.finished = False
            self.awaiting = "cooking"
            self.last_activity = time.time()
        return self.current_step

    def go_to(self, step_number: int) -> bool:
        """Go to a 1-based step number."""
        idx = step_number - 1
        if 0 <= idx < len(self.steps):
            self.current_step_index = idx
            self.next_action = self.current_step
            return True
        return False

    def repeat(self) -> Optional[str]:
        return self.current_step

    def previous_instruction(self) -> Optional[str]:
        if not self.steps:
            return None
        idx = self.current_step_index
        if idx > 0:
            return self.steps[idx - 1]
        return self.steps[0]

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------
    def add_timer(self, timer_id: str, seconds: float, label: str = ""):
        self.active_timers[timer_id] = {
            "total": round(seconds, 1),
            "label": label,
            "created_at": time.time(),
        }
        self.last_activity = time.time()

    def remove_timer(self, timer_id: str):
        self.active_timers.pop(timer_id, None)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def _task_view(self) -> dict:
        now = time.time()
        out = {}
        for key, task in self.tasks.items():
            view = dict(task)
            if task.get("started_at") and task["status"] == "started":
                view["elapsed_seconds"] = int(now - task["started_at"])
            out[key] = view
        return out

    def to_dict(self) -> dict:
        return {
            "dish": self.selected_dish,
            "servings": self.servings,
            "ingredients": self.ingredients,
            "steps": self.steps,
            "current_step": self.current_step_index,
            "current_step_text": self.current_step,
            "completed": sorted(self.completed_steps),
            "pending": self.pending_steps,
            "step_count": self.step_count,
            "progress_percent": self.progress_percent,
            "session_started": self.session_started,
            "finished": self.finished,
            "stopped": self.stopped,
            "awaiting": self.awaiting,
            "paused": self.paused,
            "timers": self.active_timers,
            "prepared": sorted(self.prepared),
            "tasks": self._task_view(),
            "conditions": self.conditions,
            "next_action": self.next_action,
            "missing_ingredients": self.missing_ingredients,
            "duration_minutes": self.duration_minutes,
            "total_time_minutes": self.total_time_minutes,
            "prep_time_minutes": self.prep_time_minutes,
            "cook_time_minutes": self.cook_time_minutes,
            "safety": self.safety,
            "recipe_source": self.recipe_source,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
            "history_count": len(self.conversation),
        }

    def to_persisted(self) -> dict:
        data = self.to_dict()
        data.pop("tasks", None)
        data["tasks"] = {
            k: {**v, "started_at": v.get("started_at"),
                "completed_at": v.get("completed_at")}
            for k, v in self.tasks.items()
        }
        data["completed_steps"] = sorted(self.completed_steps)
        data["prepared"] = sorted(self.prepared)
        data["active_timers"] = {}
        data["conversation"] = self.conversation[-20:]
        data["saved_at"] = time.time()
        data["version"] = 1
        return data

    def load_persisted(self, data: dict) -> bool:
        if not isinstance(data, dict) or not data.get("steps"):
            return False
        if (data.get("finished") or data.get("stopped")
                or data.get("awaiting") in ("finished", "stopped")):
            return False
        saved_at = data.get("saved_at")
        if saved_at is not None:
            try:
                if time.time() - float(saved_at) > 12 * 60 * 60:
                    print("[Session] Saved session is more than 12 hours old; starting fresh.")
                    return False
            except Exception:
                pass
        self.selected_dish = data.get("dish")
        self.servings = int(data.get("servings") or 2)
        self.ingredients = data.get("ingredients") or []
        self.steps = data.get("steps") or []
        self.current_step_index = int(data.get("current_step") or 0)
        self.completed_steps = set(data.get("completed_steps") or data.get("completed") or [])
        self.active_timers = {}
        self.session_started = bool(data.get("session_started"))
        self.finished = bool(data.get("finished"))
        self.stopped = bool(data.get("stopped"))
        self.awaiting = data.get("awaiting") or "cooking"
        self.paused = False
        self.conversation = data.get("conversation") or []
        self.prepared = set(data.get("prepared") or [])
        self.tasks = {k: v for k, v in (data.get("tasks") or {}).items()}
        if not self.tasks:
            self._build_tasks()
        self.conditions = data.get("conditions") or {}
        self.next_action = data.get("next_action") or self.current_step
        self.missing_ingredients = data.get("missing_ingredients") or []
        self.duration_minutes = int(data.get("duration_minutes") or 0)
        self.safety = data.get("safety") or []
        self.recipe_source = data.get("recipe_source") or "ai"
        self.started_at = data.get("started_at")
        self.last_activity = time.time()
        return self.has_recipe

    # ------------------------------------------------------------------
    # Voice helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _number(text: str) -> Optional[int]:
        text = text.strip().lower()
        if text.isdigit():
            return int(text)
        return _NUMBER_WORDS.get(text)

    @classmethod
    def parse_servings(cls, text: str) -> Optional[int]:
        """Extract a serving count like 'for 4', '4 people', 'chaar log'."""
        low = (text or "").lower()
        m = re.search(
            r"(\d+)\s*(?:serving|servings|people|person|persons|portion|portions|log|logo|ke liye)",
            low,
        )
        if m:
            return int(m.group(1))
        m = re.search(r"(?:for|feeds)\s+(\d+)", low)
        if m:
            return int(m.group(1))
        m = re.search(
            r"(?:for|feeds)\s+(one|two|three|four|five|six|seven|eight|nine|ten|"
            r"do|teen|char|chaar|panch|paanch|chhe|saat|aath|nau|das)\b",
            low,
        )
        if m:
            return cls._number(m.group(1))
        m = re.search(
            r"(one|two|three|four|five|six|seven|eight|nine|ten|do|teen|char|chaar|"
            r"panch|paanch|chhe|saat|aath|nau|das)\s*(?:log|logo|people|persons)\b",
            low,
        )
        if m:
            return cls._number(m.group(1))
        return None

    @staticmethod
    def parse_timer_minutes(text: str) -> Optional[int]:
        """Extract minutes from 'set a timer for 10 minutes' or 'two minutes'."""
        low = (text or "").lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", low)
        if m:
            return max(1, int(float(m.group(1))))
        m = re.search(
            r"(one|two|three|four|five|six|seven|eight|nine|ten|do|teen|char|chaar|"
            r"panch|paanch|chhe|saat|aath|nau|das)\s*(?:minutes?|mins?)\b",
            low,
        )
        if m:
            return _NUMBER_WORDS.get(m.group(1))
        return None

    @staticmethod
    def detect_task_cue(text: str) -> Optional[str]:
        """Return 'started', 'done' or 'not_ready' for a cooking statement."""
        low = _normalise(text)
        if any(p in low for p in (
            "not yet", "not hot", "not boiling", "still cold", "still raw",
            "not done", "not ready", "nahi hua", "abhi nahi", "thanda hai",
        )):
            return "not_ready"
        if any(p in low for p in (
            "i put", "i've put", "i have put", "put it on", "on the stove",
            "on the flame", "started heating", "is heating", "i started",
            "gas on", "placed it", "chadhaya", "chadha diya",
        )):
            return "started"
        if any(p in low for p in (
            "done", "finished", "chopped", "cut the", "boiled", "golden",
            "soft", "tender", "ready", "ho gaya", "ho gayi", "kat gaye",
        )):
            return "done"
        return None


# Global session shared by the app
session = CookingSession()
SESSION_FILE = "session_state.json"


def save_session(s: CookingSession, path: str) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s.to_persisted(), f)
        return True
    except Exception as e:
        print(f"[Session] Could not save state: {e}")
        return False


def load_session(s: CookingSession, path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return s.load_persisted(data)
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[Session] Could not load state: {e}")
        return False
