"""
Recipe generation support: prompt building, deterministic validation,
serving-size scaling, dish matching, and built-in dish-specific fallbacks.

The language model is never trusted on its own. Every recipe is checked
against the requested dish before it can reach the cooking session. If the
model is unavailable or returns something unrelated, a tested fallback for
that exact dish is used. If no fallback exists, the app says so honestly
instead of inventing a recipe for a different dish.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class RecipeError(Exception):
    """Base recipe failure."""


class RecipeValidationError(RecipeError):
    """The model returned something that is not a usable recipe."""


class RecipeUnavailable(RecipeError):
    """No reliable recipe can be produced for this dish."""


MAX_SERVINGS = 20
MIN_STEPS = 4
MAX_STEPS = 15


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------
def smart_title(text: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in (text or "").split())


def normalize_name(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


# Words that describe a style rather than a different dish.
_STYLE_WORDS = {
    "easy", "simple", "quick", "homemade", "home", "made", "style", "indian",
    "restaurant", "best", "tasty", "delicious", "authentic", "basic", "classic",
    "one", "pot", "pan", "instant", "pressure", "cooker", "spicy", "mild",
    "creamy", "dry", "gravy", "fresh", "recipe", "dish", "meal", "sauce", "masala",
}

# A requested dish can carry a dietary/variant word that must not silently
# change (for example "veg biryani" must never return chicken biryani).
_VARIANT_WORDS = {
    "chicken": "chicken", "murgh": "chicken",
    "mutton": "mutton", "lamb": "mutton", "goat": "mutton",
    "paneer": "paneer",
    "veg": "veg", "vegetable": "veg", "veggie": "veg", "sabzi": "veg",
    "egg": "egg", "anda": "egg",
    "fish": "fish", "prawn": "fish", "shrimp": "fish",
    "white sauce": "white", "white": "white", "cream": "white",
    "creamy": "white", "cheese": "white", "alfredo": "white",
    "red sauce": "red", "red": "red", "tomato": "red",
    "arrabbiata": "red", "marinara": "red",
    "aloo": "aloo", "potato": "aloo",
    "plain": "plain", "sada": "plain",
    "fried": "fried", "pulao": "pulao", "pilaf": "pulao",
}
_VARIANT_TOKEN_SET = {word for key in _VARIANT_WORDS for word in key.split()}
_TOKEN_ALIASES = {"kanda": "onion", "chawal": "rice"}


def _variant_tags(dish: str) -> set:
    text = " " + normalize_name(dish) + " "
    return {tag for word, tag in _VARIANT_WORDS.items() if f" {word} " in text}


def _singular(word: str) -> str:
    """Crude singular form so 'brownie'/'brownies' compare equal."""
    w = (word or "").lower()
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _content_tokens(dish: str) -> set:
    """Meaningful dish words, ignoring style and dietary variant words."""
    tokens = [t for t in _core_tokens(dish) if t not in _VARIANT_TOKEN_SET]
    return {_singular(_TOKEN_ALIASES.get(t, t)) for t in tokens}

_FAMILY_KEYS: List[Tuple[str, str]] = [
    ("pav bhaji", "pav_bhaji"),
    ("fried rice", "fried_rice"),
    ("paneer butter masala", "paneer"),
    ("butter chicken", "chicken_curry"),
    ("aloo paratha", "paratha"),
    ("chicken biryani", "biryani"),
    ("veg biryani", "biryani"),
    ("vegetable biryani", "biryani"),
    ("biryani", "biryani"),
    ("biriyani", "biryani"),
    ("pasta", "pasta"),
    ("spaghetti", "pasta"),
    ("penne", "pasta"),
    ("macaroni", "pasta"),
    ("pancake", "pancakes"),
    ("omelette", "omelette"),
    ("omelet", "omelette"),
    ("veg pulao", "pulao"),
    ("pulao", "pulao"),
    ("poha", "poha"),
    ("upma", "upma"),
    ("khichdi", "khichdi"),
    ("dal", "dal"),
    ("daal", "dal"),
    ("paratha", "paratha"),
    ("paneer", "paneer"),
    ("chole", "chole"),
    ("chana", "chole"),
    ("rajma", "rajma"),
    ("maggi", "maggi"),
    ("noodles", "noodles"),
    ("chow mein", "noodles"),
    ("fried rice", "fried_rice"),
    ("curry", "curry"),
    ("tea", "tea"),
    # Dishes that are not in the built-in database but still get semantic
    # validation, so an unrelated AI recipe can never masquerade as them.
    ("idli", "idli"),
    ("dosa", "dosa"),
    ("shawarma", "shawarma"),
    ("falafel", "falafel"),
    ("hummus", "hummus"),
    ("pizza", "pizza"),
    ("risotto", "risotto"),
    ("lasagna", "lasagna"),
    ("dumpling", "dumplings"),
    ("cake", "cake"),
    ("brownie", "brownie"),
    ("sandwich", "sandwich"),
    ("burger", "burger"),
    ("kheer", "kheer"),
    ("soup", "soup"),
    ("salad", "salad"),
]


def family_of(dish: str) -> Optional[str]:
    """Map a dish name to a coarse family used for mismatch checks."""
    text = " " + normalize_name(dish) + " "
    for key, family in _FAMILY_KEYS:
        if f" {key} " in text or text.strip().startswith(key):
            return family
    return None


def _core_tokens(dish: str) -> List[str]:
    return [t for t in normalize_name(dish).split()
            if len(t) > 2 and t not in _STYLE_WORDS]


def dish_matches(requested: str, returned: str) -> bool:
    """True when `returned` is the same dish and variant as `requested`."""
    if not requested or not returned:
        return False
    req_family = family_of(requested)
    got_family = family_of(returned)
    if req_family and got_family and req_family != got_family:
        return False

    req_tags = _variant_tags(requested)
    got_tags = _variant_tags(returned)
    if req_tags and got_tags and not (req_tags & got_tags):
        return False
    # Shared style tags (pulao, fried, ...) must not hide a real dietary or
    # sauce conflict such as "chicken pulao" vs "vegetable pulao".
    for group in ({"chicken", "mutton", "fish", "egg", "paneer", "veg"},
                  {"white", "red"}):
        req_group = req_tags & group
        got_group = got_tags & group
        if req_group and got_group and not (req_group & got_group):
            return False

    req_tokens = _content_tokens(requested)
    got_tokens = _content_tokens(returned)
    if req_tokens and got_tokens:
        # A generic returned name ("Biryani") is allowed for a specific
        # request ("chicken biryani"); the ingredient check catches it later.
        if not (req_tokens <= got_tokens or got_tokens <= req_tokens):
            return False
    return True


# ----------------------------------------------------------------------
# Ingredient / quantity handling
# ----------------------------------------------------------------------
_COUNTABLE = {"cup", "cups", "clove", "cloves", "onion", "onions", "tomato",
              "tomatoes", "potato", "potatoes", "packet", "packets", "pav"}
_INVARIANT = {"g", "kg", "ml", "l", "tbsp", "tsp", "pinch", "pcs", "inch"}

_QUANTITY_UNITS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon",
    "teaspoons", "tsp", "g", "gram", "grams", "kg", "kilogram", "kilograms",
    "ml", "milliliter", "milliliters", "l", "liter", "liters", "litre",
    "litres", "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds",
    "clove", "cloves", "can", "cans", "tin", "tins", "packet", "packets",
    "pinch", "pinches", "bunch", "bunches", "sprig", "sprigs", "slice",
    "slices", "piece", "pieces", "inch", "inches", "cm", "handful",
    "handfuls", "stick", "sticks", "pod", "pods", "pav", "ball", "balls",
    "head", "heads",
    "stalk", "stalks", "leaf", "leaves", "sachet", "sachets", "container",
    "containers", "bottle", "bottles", "jar", "jars", "cube", "cubes",
    "dash", "dashes", "drop", "drops", "scoop", "scoops", "knob", "knobs",
    "large", "medium", "small", "whole", "big",
}
_QUANTITY_PHRASES = {
    "to taste", "as needed", "as required", "for garnish", "for serving",
    "for frying", "for deep frying", "optional", "a pinch", "pinch",
    "a handful", "handful", "a few", "few",
}


def _quantity_ok(quantity: str) -> bool:
    """Reject obvious model typos such as '1 teass' before they reach a user."""
    q = " ".join((quantity or "").lower().split())
    if not q:
        return False
    if q in _QUANTITY_PHRASES:
        return True
    if any(p in q for p in ("to taste", "as needed", "as required", "optional")):
        return True
    q = re.sub(r"^(?:about|approx\.?|approximately|around)\s+", "", q)
    m = re.match(
        r"(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?)",
        q,
    )
    if not m:
        return False
    rest = q[m.end():].strip()
    if not rest:
        return True
    words = re.findall(r"[a-z]+", rest)
    return any(w in _QUANTITY_UNITS for w in words)


def format_amount(value: float) -> str:
    if value >= 20:
        rounded = int(round(value / 5.0) * 5)
        return str(rounded)
    if abs(value - round(value)) < 0.02:
        return str(int(round(value)))
    if abs(value * 2 - round(value * 2)) < 0.02:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def quantity_text(amount: float, unit: str) -> str:
    unit = (unit or "").strip()
    if not unit:
        return format_amount(amount)
    if unit.lower() in _COUNTABLE and amount > 1.02:
        if not unit.lower().endswith("s"):
            unit = unit + "s"
    return f"{format_amount(amount)} {unit}".strip()


def split_quantity(text: str) -> Tuple[str, str]:
    """Split '2 cups rice' into ('2 cups', 'rice'). Keeps odd strings intact."""
    text = (text or "").strip()
    m = re.match(r"^(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*([a-zA-Z]+)?\s*(.*)$", text)
    if not m:
        return "", text
    amount, unit, rest = m.group(1), m.group(2) or "", m.group(3) or ""
    known = {"cup", "cups", "g", "kg", "ml", "l", "tbsp", "tsp", "pinch",
             "clove", "cloves", "packet", "packets", "slice", "slices"}
    if unit.lower() in known and rest:
        return f"{amount} {unit}".strip(), rest
    # "2 onions" -> quantity 2, name onions
    return amount, (f"{unit} {rest}".strip() if unit else rest)


def scale_quantity(quantity: str, factor: float) -> str:
    """Scale a display quantity such as '200 g' or '1.5 cups'."""
    if not quantity or factor == 1:
        return quantity
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(.*)$", quantity)
    if not m:
        return quantity  # "to taste", "a pinch", "as needed"
    amount = float(m.group(1))
    unit = m.group(2)
    return f"{format_amount(amount * factor)} {unit}".strip()


# ----------------------------------------------------------------------
# Recipe validation
# ----------------------------------------------------------------------
_REQUIRED_MARKERS = {
    "biryani": (["rice", "basmati"], []),
    "pasta": (["pasta", "spaghetti", "penne", "macaroni", "noodle"], []),
    "pancakes": (["flour", "maida", "atta"], []),
    "omelette": (["egg"], []),
    "poha": (["poha", "flattened rice"], []),
    "upma": (["semolina", "rava", "sooji"], []),
    "khichdi": (["rice", "dal", "moong"], []),
    "pulao": (["rice", "basmati"], []),
    "paneer": (["paneer"], []),
    "paratha": (["flour", "atta", "maida"], []),
    "pav_bhaji": (["pav", "potato"], []),
    "chole": (["chickpea", "chana", "chole"], []),
    "rajma": (["rajma", "kidney bean"], []),
    "maggi": (["maggi", "noodle"], []),
    "fried_rice": (["rice"], []),
    "dal": (["dal", "daal", "lentil"], []),
    "curry": ([], []),
    # Dishes generated by the LLM (not built in) still get a light semantic
    # check so a recipe for a completely different dish is rejected.
    "idli": (["rice", "idli", "batter", "rava", "semolina", "urad", "lentil"], []),
    "dosa": (["rice", "dosa", "batter", "flour", "rava", "semolina", "lentil", "dal"], []),
    "shawarma": (["chicken", "paneer", "chickpea", "mushroom", "lamb", "mutton",
                  "beef", "soy", "tofu"], []),
    "falafel": (["chickpea", "chana", "fava", "bean", "lentil"], []),
    "hummus": (["chickpea", "chana"], []),
    "pizza": (["flour", "maida", "dough", "base", "bread"], []),
    "risotto": (["rice", "arborio"], []),
    "lasagna": (["pasta", "lasagna", "sheet", "flour"], []),
    "dumplings": (["flour", "wrapper", "dough", "maida"], []),
    "cake": (["flour", "maida", "cocoa", "cake mix"], []),
    "brownie": (["chocolate", "cocoa", "flour", "maida"], []),
    "sandwich": (["bread"], []),
    "burger": (["bun", "bread"], []),
    "kheer": (["milk", "rice", "vermicelli", "semolina", "rava"], []),
}

# A few families need a recognisable cooking technique in the steps. The list
# is deliberately broad so legitimate regional variations still pass.
_REQUIRED_STEP_WORDS = {
    "idli": ("steam", "steamer", "idli maker"),
    "shawarma": ("marin", "grill", "roast", "pan", "oven", "bake", "skillet"),
    "falafel": ("fry", "bake", "air fry", "oven"),
    "hummus": ("blend", "process", "mix", "grind"),
    "pizza": ("bake", "oven", "grill"),
    "risotto": ("stir", "simmer", "stock", "broth"),
    "dumplings": ("steam", "boil", "fry", "pan"),
    "cake": ("bake", "oven"),
    "brownie": ("bake", "oven"),
    "kheer": ("simmer", "boil", "cook", "stir"),
}

_FORBIDDEN_MARKERS = {
    "biryani": ["pasta", "spaghetti", "penne", "macaroni", "lasagna"],
    "pulao": ["pasta", "spaghetti", "penne", "macaroni", "lasagna"],
    "pasta": ["basmati", "biryani masala"],
    "pancakes": ["pasta", "spaghetti"],
    "omelette": ["pasta", "spaghetti"],
    "idli": ["pasta", "spaghetti", "bread"],
    "dosa": ["pasta", "spaghetti"],
    "shawarma": ["pasta", "spaghetti"],
    "pizza": ["biryani masala", "basmati"],
    "cake": ["biryani masala", "soy sauce"],
    "brownie": ["biryani masala", "soy sauce"],
}


def _ingredient_names(ingredients: List[dict]) -> str:
    return " ".join(normalize_name(i.get("name", "")) for i in ingredients)


def _has_word(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _variant_ingredients_ok(requested: str, ingredients: List[dict],
                            steps: List[str]) -> Tuple[bool, str]:
    """Make sure the ingredients honour the requested dietary/variant words."""
    tags = _variant_tags(requested)
    if not tags:
        return True, ""
    text = " " + _ingredient_names(ingredients) + " "

    if "veg" in tags:
        bad = re.search(
            r"\b(chicken|mutton|lamb|beef|pork|fish|prawn|shrimp|meat|egg)\b",
            text,
        )
        if bad:
            return False, f"vegetarian recipe contains {bad.group(1)}"

    checks = [
        ("chicken", ("chicken",), "chicken"),
        ("mutton", ("mutton", "lamb", "goat"), "mutton or lamb"),
        ("paneer", ("paneer",), "paneer"),
        ("egg", ("egg",), "egg"),
        ("fish", ("fish", "prawn", "shrimp"), "fish"),
        ("aloo", ("aloo", "potato"), "potato"),
    ]
    for tag, words, label in checks:
        if tag in tags and not _has_word(text, *words):
            return False, f"recipe does not contain the requested {label}"

    if "white" in tags:
        if not (_has_word(text, "cream", "creamy", "milk", "butter", "cheese")
                or "white sauce" in text):
            return False, "white-sauce recipe has no creamy ingredient"
    if "red" in tags:
        if not (_has_word(text, "tomato") or "red sauce" in text):
            return False, "red-sauce recipe has no tomato"
    if "plain" in tags and _has_word(text, "aloo", "potato", "paneer"):
        return False, "plain paratha contains a filling"

    return True, ""


def _extra_tokens_ok(requested: str, returned: str, ingredients: List[dict],
                     steps: List[str]) -> Tuple[bool, str]:
    """Generic AI dish names must still include the specific request words."""
    extra = _content_tokens(requested) - _content_tokens(returned)
    if not extra:
        return True, ""
    text = normalize_name(_ingredient_names(ingredients))
    missing = sorted(t for t in extra if t not in text)
    if missing:
        return False, f"recipe does not mention {', '.join(missing)}"
    return True, ""


def estimate_duration(steps: List[str]) -> int:
    """Estimate minutes from explicit timings in the steps."""
    minutes = 0.0
    for step in steps:
        for m in re.finditer(
            r"(\d+(?:\.\d+)?)\s*(?:(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*)?"
            r"(minutes?|mins?|hours?|hrs?)",
            step.lower(),
        ):
            low = float(m.group(1))
            high = float(m.group(2)) if m.group(2) else low
            value = max(low, high)
            if m.group(3).startswith(("hour", "hr")):
                value *= 60
            minutes += value
    if minutes <= 0:
        return 30
    # Waiting-heavy preparations (fermentation, marination, soaking) are real
    # elapsed time, so they must not be truncated to a short cook time.
    return int(max(5, min(600, round(minutes + 10))))


def default_safety(steps: List[str]) -> List[str]:
    text = " ".join(steps).lower()
    notes = [
        "Never leave hot oil or an open flame unattended.",
        "Use dry hands and a dry cloth when handling hot pans.",
        "Turn off the heat before moving a hot pot.",
    ]
    if "oil" in text or "fry" in text:
        notes.append("Pat food dry before adding it to hot oil to avoid splashing.")
    if "boil" in text or "water" in text:
        notes.append("Pour boiling water slowly and keep your face away from the steam.")
    if "chicken" in text or "meat" in text or "mutton" in text:
        notes.append("Wash your hands, knife, and board after touching raw meat.")
        notes.append("Cook meat until the juices run clear and the inside is no longer pink.")
    return notes


def _clean_steps(raw) -> List[str]:
    if isinstance(raw, str):
        raw = [line for line in re.split(r"[\n\r]+", raw) if line.strip()]
    if not isinstance(raw, list):
        raise RecipeValidationError("steps must be a list")
    steps = []
    for item in raw:
        duration = None
        heat = ""
        if isinstance(item, dict):
            duration = (item.get("duration_minutes") or item.get("minutes")
                        or item.get("time_minutes") or item.get("duration"))
            heat = str(item.get("heat") or item.get("heat_level") or "").strip()
            item = item.get("text") or item.get("step") or item.get("instruction") or ""
        text = str(item or "").strip()
        text = re.sub(r"^\s*(?:step\s*)?\d+\s*[:.\-–)]\s*", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        # Preserve structured timing/heat metadata as readable text so the
        # timer/step-duration logic and the UI keep working unchanged.
        if duration and not re.search(r"\b\d+\s*(?:min|minute|hour|hr)", text, re.I):
            try:
                minutes = int(float(str(duration).split()[0]))
                if 0 < minutes <= 600:
                    text += f" (about {minutes} minutes)"
            except Exception:
                pass
        if heat and heat.lower() not in ("", "none", "n/a") \
                and "heat" not in text.lower():
            text += f" (heat: {heat})"
        steps.append(text)
    return steps


def _clean_ingredients(raw) -> List[dict]:
    if isinstance(raw, dict):
        raw = [{"name": k, "quantity": v} for k, v in raw.items()]
    if not isinstance(raw, list):
        raise RecipeValidationError("ingredients must be a list")
    out: List[dict] = []
    seen = set()
    for item in raw:
        if isinstance(item, str):
            qty, name = split_quantity(item)
            item = {"name": name, "quantity": qty}
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("item") or "").strip()
        qty = item.get("quantity", item.get("amount", ""))
        unit = str(item.get("unit") or "").strip()
        note = str(item.get("preparation") or item.get("notes") or
                   item.get("note") or "").strip()
        if isinstance(qty, (int, float)):
            qty = str(qty)
        qty = str(qty or "").strip()
        if not name:
            continue
        if unit and unit.lower() not in qty.lower():
            qty = f"{qty} {unit}".strip() if qty else unit
        if not qty:
            qty = "as needed"
        if not _quantity_ok(qty):
            raise RecipeValidationError(
                f"ingredient {name!r} has an unclear quantity {qty!r}"
            )
        key = normalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        entry = {"name": name[:80], "quantity": qty[:40]}
        if note:
            entry["note"] = note[:80]
        out.append(entry)
    return out


def validate_recipe(data: dict, requested_dish: str, requested_servings: int) -> dict:
    """Validate/normalise a model recipe or raise RecipeValidationError."""
    if not isinstance(data, dict):
        raise RecipeValidationError("recipe is not a JSON object")
    if isinstance(data.get("recipe"), dict):
        data = data["recipe"]

    dish = str(data.get("dish") or data.get("name") or requested_dish).strip()
    if not dish:
        raise RecipeValidationError("recipe has no dish name")
    if not dish_matches(requested_dish, dish):
        raise RecipeValidationError(
            f"recipe is for {dish!r}, not {requested_dish!r}"
        )

    ingredients = _clean_ingredients(data.get("ingredients"))
    if len(ingredients) < 2:
        raise RecipeValidationError("recipe needs at least two ingredients")

    steps = _clean_steps(data.get("steps") or data.get("instructions"))
    if len(steps) < MIN_STEPS:
        raise RecipeValidationError(f"recipe needs at least {MIN_STEPS} steps")
    if len(steps) > MAX_STEPS:
        raise RecipeValidationError(f"recipe has too many steps ({len(steps)})")
    if any(len(s) < 5 for s in steps):
        raise RecipeValidationError("recipe has an empty or unusable step")

    try:
        servings = int(float(str(data.get("servings", requested_servings)).strip().split()[0]))
    except Exception:
        servings = requested_servings
    servings = max(1, min(MAX_SERVINGS, servings))

    # If the model produced a different batch size, rescale the quantities.
    factor = requested_servings / servings if servings else 1.0
    if abs(factor - 1.0) > 0.01:
        for ing in ingredients:
            ing["quantity"] = scale_quantity(ing["quantity"], factor)
    servings = requested_servings

    family = family_of(requested_dish) or family_of(dish)
    if family in _REQUIRED_MARKERS:
        required, _ = _REQUIRED_MARKERS[family]
        names = _ingredient_names(ingredients)
        if required and not any(r in names for r in required):
            raise RecipeValidationError(
                f"{family} recipe is missing a required ingredient ({', '.join(required)})"
            )
    if family in _FORBIDDEN_MARKERS:
        names = _ingredient_names(ingredients)
        for bad in _FORBIDDEN_MARKERS[family]:
            if bad in names:
                raise RecipeValidationError(
                    f"{family} recipe contains unrelated ingredient {bad!r}"
                )

    variant_ok, reason = _variant_ingredients_ok(requested_dish, ingredients, steps)
    if not variant_ok:
        raise RecipeValidationError(reason)

    extra_ok, reason = _extra_tokens_ok(requested_dish, dish, ingredients, steps)
    if not extra_ok:
        raise RecipeValidationError(reason)

    if family in _REQUIRED_STEP_WORDS:
        joined = " ".join(steps).lower()
        if not any(w in joined for w in _REQUIRED_STEP_WORDS[family]):
            raise RecipeValidationError(
                f"{family} recipe is missing the expected cooking technique"
            )

    def _field(*names) -> Optional[int]:
        for name in names:
            value = data.get(name)
            if value is None or value == "":
                continue
            try:
                minutes = int(float(str(value).split()[0]))
            except Exception:
                continue
            if 0 <= minutes <= 600:
                return minutes
        return None

    estimated = estimate_duration(steps)
    prep = _field("preparation_time_minutes", "prep_time_minutes")
    cook = _field("active_cooking_time_minutes", "cook_time_minutes",
                  "cooking_time_minutes")
    wait = _field("waiting_time_minutes", "resting_time_minutes")
    total = _field("total_elapsed_time_minutes", "total_time_minutes")
    duration = _field("duration_minutes", "total_time_minutes")

    if duration is None:
        duration = total or (prep or 0) + (cook or 0) + (wait or 0) or estimated
    if total is None:
        total = max(duration, (prep or 0) + (cook or 0) + (wait or 0))
    if prep is None:
        prep = max(0, round(duration * 0.25))
    if cook is None:
        cook = max(0, duration - prep)
    if wait is None:
        wait = max(0, total - prep - cook)

    safety = data.get("safety") or data.get("safety_notes") or []
    if isinstance(safety, str):
        safety = [safety]
    safety = [str(s).strip() for s in safety if str(s).strip()]
    if not safety:
        safety = default_safety(steps)

    return {
        "dish": smart_title(dish),
        "servings": servings,
        "ingredients": ingredients,
        "steps": steps,
        "duration_minutes": max(5, min(600, duration)),
        "prep_time_minutes": max(0, min(600, prep)),
        "cook_time_minutes": max(0, min(600, cook)),
        "waiting_time_minutes": max(0, min(600, wait)),
        "total_time_minutes": max(5, min(600, total)),
        "safety": safety[:6],
        "source": "ai",
        "source_type": "ai_generated",
        "validation_status": "validated",
    }


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------
def build_recipe_prompt(dish: str, servings: int,
                        context: Optional[str] = None) -> str:
    context_block = ""
    if context and context.strip():
        context_block = (
            "\nREFERENCE NOTES (retrieved culinary knowledge; use only facts that "
            "genuinely match the requested dish, ignore anything unrelated):\n"
            f"{context.strip()[:1500]}\n"
        )
    return (
        "You are a precise, practical recipe writer for a cooking teacher. "
        f"Create a real, complete recipe for exactly this dish: {dish!r} "
        f"for {servings} servings.\n"
        "Return ONLY one valid JSON object, no prose and no markdown, with this shape:\n"
        '{"dish":"...","cuisine":"...","servings":' + str(servings) + ','
        '"ingredients":[{"name":"item","quantity":"2","unit":"cups",'
        '"preparation":"chopped (optional)"}],'
        '"steps":[{"text":"one clear action","duration_minutes":5,'
        '"heat":"medium"}],'
        '"preparation_time_minutes":15,"active_cooking_time_minutes":25,'
        '"waiting_time_minutes":0,"total_elapsed_time_minutes":40,'
        '"safety":["..."]}\n'
        f"{context_block}"
        "HARD RULES:\n"
        f"1. The recipe MUST be for {dish!r}. Never return a recipe for another dish.\n"
        f"2. All ingredient quantities must be scaled for exactly {servings} servings; "
        "use real units (g, ml, cups, tbsp, tsp, pieces).\n"
        "3. Give 5 to 10 ordered steps. Every step is one clear action and must be "
        "necessary for this exact dish (preparation, cooking, finishing).\n"
        "4. Include the defining ingredients and technique of the dish. Examples: "
        "idli needs a rice/urad batter, fermentation and steaming; shawarma needs a "
        "marinated meat or vegetarian filling plus a cooking method and assembly; "
        "pancakes need flour, a liquid, mixing and pan cooking. These are examples, "
        "not a whitelist - support any legitimate regional variation.\n"
        "5. Keep preparation, active cooking and waiting time separate. Fermentation, "
        "marination, resting and soaking belong in waiting time. Use 0 when not needed. "
        "total_elapsed_time_minutes should reflect realistic overlap, not a blind sum.\n"
        "6. Give each step a realistic duration_minutes and a heat level where it "
        "applies (low, medium, high, or empty for no heat).\n"
        "7. Use the ingredient and spice names that actually belong to this cuisine. "
        "For Indian dishes use Indian names (basmati rice, ghee, garam masala).\n"
        "8. Add 2 to 4 safety notes about hot oil, boiling water, raw meat, steam, "
        "or hot pans when relevant.\n"
        "9. Do not invent a restaurant-only method; it must work in a normal home "
        "kitchen with common equipment.\n"
        "10. Do not claim the recipe is tested or cite a source."
    )


def build_repair_prompt(dish: str, servings: int, bad_output: str,
                        reason: str = "") -> str:
    reason_block = f"The specific problem was: {reason}.\n" if reason else ""
    return (
        f"Your previous answer was not a valid recipe for {dish!r} for {servings} servings. "
        "It may have named a different dish, used unrelated ingredients, missed an "
        "essential step, or contained malformed fields.\n"
        f"{reason_block}"
        f"Previous answer: {bad_output[:800]}\n"
        "Now return ONLY valid JSON for the requested dish, using the exact schema "
        "requested earlier. Do not include any other dish."
    )


# ----------------------------------------------------------------------
# Built-in, dish-specific fallback recipes
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RecipeTemplate:
    key: str            # matching key
    dish: str           # display name
    base_servings: int
    ingredients: List[Tuple[float, str, str]]  # amount, unit, name
    steps: List[str]
    duration_minutes: int
    safety: List[str] = field(default_factory=list)
    prep_time_minutes: int = 0
    cook_time_minutes: int = 0
    waiting_time_minutes: int = 0


_TEMPLATES: List[RecipeTemplate] = [
    RecipeTemplate(
        "chicken biryani", "Chicken Biryani", 4,
        [
            (500, "g", "chicken, bone-in"),
            (2, "cup", "basmati rice"),
            (1, "cup", "plain yogurt"),
            (2, "", "onions, sliced"),
            (2, "", "tomatoes, chopped"),
            (1, "tbsp", "ginger-garlic paste"),
            (2, "tsp", "biryani masala"),
            (0.5, "tsp", "turmeric"),
            (1, "tsp", "red chilli powder"),
            (3, "tbsp", "ghee or oil"),
            (1, "tsp", "salt"),
            (3, "cup", "water"),
            (0.5, "cup", "mint and coriander leaves"),
        ],
        [
            "Wash the basmati rice and soak it in water for 20 minutes. Wash the chicken and pat it dry.",
            "Heat the ghee in a heavy pot. Fry the sliced onions until golden brown, then set half aside for the topping.",
            "Add ginger-garlic paste, chicken, yogurt, biryani masala, turmeric, chilli powder and salt. Cook on medium heat for 10 minutes, stirring now and then.",
            "Add the chopped tomatoes and cook until the oil separates, about 5 minutes.",
            "Drain the rice and layer it over the chicken. Add the water and half the mint, then cover with a tight lid.",
            "Cook on the lowest heat for 20 minutes. Do not lift the lid while the rice steams.",
            "Turn off the heat and rest for 5 minutes. Fluff gently with a fork and serve with the reserved onions and mint.",
        ],
        60,
        [
            "Wash your hands, knife, and board after touching raw chicken.",
            "Cook the chicken until the juices run clear and the inside is no longer pink.",
            "Do not leave the pot on high heat unattended.",
            "Open the lid away from your face; the steam is very hot.",
        ],
    ),
    RecipeTemplate(
        "veg biryani", "Vegetable Biryani", 4,
        [
            (2, "cup", "basmati rice"),
            (3, "cup", "mixed vegetables, chopped"),
            (1, "cup", "plain yogurt"),
            (2, "", "onions, sliced"),
            (2, "", "tomatoes, chopped"),
            (1, "tbsp", "ginger-garlic paste"),
            (2, "tsp", "biryani masala"),
            (0.5, "tsp", "turmeric"),
            (3, "tbsp", "ghee or oil"),
            (1, "tsp", "salt"),
            (3, "cup", "water"),
            (0.5, "cup", "mint and coriander leaves"),
        ],
        [
            "Wash the basmati rice and soak it in water for 20 minutes.",
            "Heat the ghee in a heavy pot and fry the onions until golden. Set half aside.",
            "Add ginger-garlic paste, vegetables, yogurt, biryani masala, turmeric and salt. Cook for 8 minutes.",
            "Add the tomatoes and cook until they soften, about 4 minutes.",
            "Drain the rice and layer it over the vegetables. Add water and half the mint, then cover tightly.",
            "Cook on the lowest heat for 18 to 20 minutes without opening the lid.",
            "Rest for 5 minutes, fluff gently and serve with the reserved onions and mint.",
        ],
        45,
        [
            "Do not leave the pot on high heat unattended.",
            "Open the lid away from your face; the steam is very hot.",
        ],
    ),
    RecipeTemplate(
        "pasta", "Tomato Pasta", 2,
        [
            (200, "g", "pasta"),
            (3, "cup", "water"),
            (1, "tbsp", "salt"),
            (1, "tbsp", "olive oil"),
            (1, "cup", "tomato sauce"),
            (2, "clove", "garlic, chopped"),
            (0.5, "tsp", "dried oregano"),
            (2, "tbsp", "grated cheese"),
        ],
        [
            "Fill a large pot with the water and bring it to a rolling boil.",
            "Add the salt and carefully slide in the pasta. Stir once so it does not stick.",
            "Boil for the time on the packet, usually 8 to 10 minutes, until just tender.",
            "While the pasta cooks, heat the olive oil in a pan and cook the garlic for 30 seconds.",
            "Add the tomato sauce and oregano. Simmer gently for 5 minutes.",
            "Reserve a little pasta water, then drain the pasta in a colander.",
            "Toss the pasta with the sauce, loosen with pasta water if needed, and finish with cheese.",
        ],
        25,
        [
            "Pour boiling water slowly and keep your face away from the steam.",
            "Use oven gloves when lifting the hot pot.",
        ],
    ),
    RecipeTemplate(
        "white sauce pasta", "Creamy White Sauce Pasta", 2,
        [
            (200, "g", "pasta"),
            (3, "cup", "water"),
            (1, "tbsp", "salt"),
            (2, "tbsp", "butter"),
            (2, "tbsp", "all-purpose flour"),
            (1.5, "cup", "milk"),
            (3, "tbsp", "grated cheese"),
            (0.25, "tsp", "black pepper"),
        ],
        [
            "Boil the water with salt in a large pot and cook the pasta until just tender.",
            "While it cooks, melt the butter in a pan on low heat.",
            "Add the flour and stir for 1 minute, without letting it brown.",
            "Pour in the milk little by little, whisking so no lumps form. Simmer until thick.",
            "Stir in the cheese, pepper and a pinch of salt.",
            "Drain the pasta, add it to the sauce and toss gently.",
            "Serve hot, with extra cheese if you like.",
        ],
        25,
        [
            "Pour boiling water slowly and keep your face away from the steam.",
            "Whisk hot sauce carefully; it can splash.",
        ],
    ),
    RecipeTemplate(
        "pancakes", "Pancakes", 2,
        [
            (1, "cup", "all-purpose flour"),
            (1, "cup", "milk"),
            (1, "", "egg"),
            (1, "tbsp", "sugar"),
            (1, "tsp", "baking powder"),
            (1, "pinch", "salt"),
            (1, "tbsp", "butter or oil"),
        ],
        [
            "Whisk the flour, sugar, baking powder and salt in a bowl.",
            "Beat in the egg and milk until the batter is smooth.",
            "Heat a non-stick pan on medium and add a little butter.",
            "Pour a small ladle of batter into the pan.",
            "Cook until bubbles form on top, about 2 minutes.",
            "Flip carefully with a spatula and cook 1 more minute.",
            "Serve warm with your favourite topping.",
        ],
        20,
        [
            "Keep your hand away from the hot pan while flipping.",
            "Do not leave hot oil or butter unattended.",
        ],
        5,
        12,
    ),
    RecipeTemplate(
        "omelette", "Masala Omelette", 2,
        [
            (3, "", "eggs"),
            (2, "tbsp", "milk"),
            (1, "tsp", "butter"),
            (0.5, "", "onion, finely chopped"),
            (1, "pinch", "salt"),
            (1, "pinch", "black pepper"),
        ],
        [
            "Crack the eggs into a bowl and beat with the milk, salt and pepper.",
            "Stir in the chopped onion.",
            "Melt the butter in a non-stick pan over medium-low heat.",
            "Pour in the egg mixture and let it set for about a minute.",
            "Pull the edges inward with a spatula so uncooked egg runs underneath.",
            "When mostly set but still soft, fold the omelette in half.",
            "Slide it onto a plate and serve immediately.",
        ],
        10,
        [
            "Keep the pan handle turned inward so it cannot be knocked over.",
            "Do not leave the hot pan unattended.",
        ],
    ),
    RecipeTemplate(
        "poha", "Kanda Poha", 2,
        [
            (2, "cup", "poha (flattened rice)"),
            (1, "", "onion, chopped"),
            (1, "", "potato, diced"),
            (1, "tsp", "mustard seeds"),
            (10, "", "curry leaves"),
            (0.5, "tsp", "turmeric"),
            (1, "tbsp", "oil"),
            (1, "tsp", "salt"),
            (0.5, "", "lemon"),
        ],
        [
            "Rinse the poha in a sieve under running water until soft, then leave it to drain.",
            "Heat the oil in a pan and add the mustard seeds. Wait for them to pop.",
            "Add the curry leaves, onion and potato. Cook until the potato is soft, about 6 minutes.",
            "Stir in the turmeric and salt.",
            "Add the drained poha and toss gently on low heat for 3 minutes.",
            "Turn off the heat, squeeze over the lemon and mix once.",
            "Serve warm, with a little coriander if you have it.",
        ],
        20,
        [
            "Keep your face back when the mustard seeds pop in hot oil.",
            "Turn off the heat before squeezing the lemon.",
        ],
    ),
    RecipeTemplate(
        "upma", "Rava Upma", 2,
        [
            (1, "cup", "semolina (rava)"),
            (2.5, "cup", "water"),
            (1, "", "onion, chopped"),
            (1, "tsp", "mustard seeds"),
            (8, "", "curry leaves"),
            (1, "", "green chilli, chopped"),
            (1, "tbsp", "oil"),
            (1, "tsp", "salt"),
        ],
        [
            "Dry-roast the semolina in a pan on low heat for 3 minutes, until it smells nutty. Set it aside.",
            "Heat the oil in the same pan and add the mustard seeds, curry leaves and green chilli.",
            "Add the onion and cook until soft, about 3 minutes.",
            "Pour in the water and add the salt. Bring it to a boil.",
            "Lower the heat and slowly add the roasted semolina, stirring continuously.",
            "Cover and cook for 3 minutes, then fluff with a fork.",
            "Serve hot with a spoon of ghee if you like.",
        ],
        20,
        [
            "Add semolina slowly to hot water; it can bubble and splash.",
            "Do not leave the pan on high heat unattended.",
        ],
    ),
    RecipeTemplate(
        "khichdi", "Moong Dal Khichdi", 2,
        [
            (0.5, "cup", "rice"),
            (0.5, "cup", "moong dal"),
            (2.5, "cup", "water"),
            (0.5, "tsp", "turmeric"),
            (1, "tsp", "cumin seeds"),
            (1, "tbsp", "ghee"),
            (1, "tsp", "salt"),
        ],
        [
            "Rinse the rice and moong dal together until the water runs clear.",
            "Heat the ghee in a pressure cooker or heavy pot and add the cumin seeds.",
            "Add the rice, dal, turmeric, salt and water. Stir once.",
            "Close the pressure cooker and cook for 3 whistles on medium heat. If using a pot, cover and simmer for 25 minutes.",
            "Turn off the heat. Wait for the pressure to release naturally before opening.",
            "Stir gently and add a little hot water if it is too thick.",
            "Serve warm with yogurt or pickle.",
        ],
        30,
        [
            "Never open a pressure cooker by force; wait for the pressure to drop.",
            "Keep your face away from the steam when you open the lid.",
        ],
    ),
    RecipeTemplate(
        "veg pulao", "Vegetable Pulao", 4,
        [
            (1.5, "cup", "basmati rice"),
            (1.5, "cup", "mixed vegetables (carrot, peas, beans)"),
            (1, "", "onion, sliced"),
            (2, "tbsp", "ghee or oil"),
            (1, "tsp", "cumin seeds"),
            (1, "", "bay leaf"),
            (2, "", "cloves"),
            (0.5, "tsp", "turmeric"),
            (1, "tsp", "salt"),
            (2.5, "cup", "water"),
            (0.25, "cup", "coriander leaves"),
        ],
        [
            "Rinse the basmati rice until the water runs clear and soak it for 15 minutes.",
            "Heat the ghee in a heavy pot and add the cumin seeds, bay leaf and cloves.",
            "Add the sliced onion and cook until soft and lightly golden.",
            "Add the mixed vegetables and turmeric. Stir for 2 minutes.",
            "Drain the rice and add it with the salt. Stir gently for 1 minute.",
            "Pour in the water and bring it to a boil.",
            "Cover, lower the heat and cook for 12 minutes without stirring.",
            "Turn off the heat and let it rest covered for 5 minutes. Fluff with a fork and add coriander.",
        ],
        30,
        [
            "Keep your face away from the steam when you open the lid.",
            "Do not leave the hot pot unattended.",
        ],
    ),
    RecipeTemplate(
        "dal", "Dal Tadka", 4,
        [
            (1, "cup", "toor dal"),
            (3, "cup", "water"),
            (1, "", "onion, chopped"),
            (2, "", "tomatoes, chopped"),
            (2, "clove", "garlic, chopped"),
            (1, "tsp", "cumin seeds"),
            (0.5, "tsp", "turmeric"),
            (1, "tbsp", "ghee"),
            (1, "tsp", "salt"),
            (0.25, "cup", "coriander leaves"),
        ],
        [
            "Rinse the dal and boil it with the water, turmeric and salt until soft, about 20 minutes.",
            "Mash the dal lightly with the back of a spoon.",
            "Heat the ghee in a small pan and add the cumin seeds.",
            "Add the garlic and onion and fry until golden.",
            "Add the tomatoes and cook until they soften, about 4 minutes.",
            "Pour the tempering into the dal and stir well.",
            "Simmer for 3 minutes and finish with coriander leaves.",
        ],
        35,
        [
            "Hot dal can bubble and splash; stir gently.",
            "Turn off the heat before moving the hot pot.",
        ],
    ),
    RecipeTemplate(
        "paneer", "Paneer Butter Masala", 3,
        [
            (250, "g", "paneer, cubed"),
            (3, "", "tomatoes, chopped"),
            (1, "", "onion, chopped"),
            (2, "tbsp", "butter"),
            (0.5, "cup", "cream"),
            (1, "tsp", "ginger-garlic paste"),
            (1, "tsp", "garam masala"),
            (0.5, "tsp", "red chilli powder"),
            (1, "tsp", "salt"),
            (0.25, "cup", "coriander leaves"),
        ],
        [
            "Simmer the tomatoes and onion with a little water for 10 minutes, then blend into a smooth puree.",
            "Melt the butter in a pan and add the ginger-garlic paste. Cook for 30 seconds.",
            "Pour in the puree and cook on medium heat for 8 minutes, until it darkens.",
            "Add the garam masala, chilli powder and salt.",
            "Stir in the cream and simmer gently for 2 minutes.",
            "Add the paneer cubes and warm them for 3 minutes without boiling hard.",
            "Finish with coriander leaves and serve with roti or rice.",
        ],
        30,
        [
            "Blend hot puree carefully; keep the lid vented.",
            "Do not let the cream boil over.",
        ],
    ),
    RecipeTemplate(
        "paratha", "Aloo Paratha", 3,
        [
            (2, "cup", "whole wheat flour (atta)"),
            (3, "", "potatoes, boiled"),
            (1, "", "onion, finely chopped"),
            (1, "", "green chilli, chopped"),
            (1, "tsp", "cumin seeds"),
            (2, "tbsp", "ghee"),
            (1, "tsp", "salt"),
            (0.75, "cup", "water"),
        ],
        [
            "Knead the flour, salt and water into a soft dough. Cover it and rest for 15 minutes.",
            "Mash the boiled potatoes with the onion, green chilli, cumin and salt.",
            "Divide the dough into balls. Flatten one, put a spoon of filling in the centre and seal it.",
            "Roll it gently into a flat paratha, keeping the filling inside.",
            "Heat a tawa or non-stick pan on medium and cook the paratha for 2 minutes.",
            "Flip it, brush with ghee and cook until golden spots appear on both sides.",
            "Serve hot with yogurt or pickle.",
        ],
        35,
        [
            "Keep your fingers away from the hot tawa when flipping.",
            "Do not leave the flame on high while rolling the next paratha.",
        ],
    ),
    RecipeTemplate(
        "pav bhaji", "Pav Bhaji", 4,
        [
            (4, "", "pav buns"),
            (3, "", "potatoes, boiled"),
            (1, "cup", "green peas"),
            (2, "", "tomatoes, chopped"),
            (1, "", "onion, chopped"),
            (1, "", "capsicum, chopped"),
            (2, "tbsp", "butter"),
            (1, "tbsp", "pav bhaji masala"),
            (1, "tsp", "salt"),
            (0.5, "", "lemon"),
        ],
        [
            "Mash the boiled potatoes and peas together.",
            "Melt half the butter in a pan and cook the onion until soft.",
            "Add the capsicum and tomatoes and cook for 5 minutes.",
            "Add the pav bhaji masala, salt and the mashed vegetables. Mix well.",
            "Cook on low heat for 8 minutes, adding a splash of water if it sticks.",
            "Toast the pav buns with the remaining butter until golden.",
            "Serve the bhaji with the pav, a squeeze of lemon and chopped onion.",
        ],
        40,
        [
            "Mashed vegetables can bubble and splash; stir gently.",
            "Keep your hand away from the hot pan while toasting the pav.",
        ],
    ),
    RecipeTemplate(
        "fried rice", "Vegetable Fried Rice", 2,
        [
            (2, "cup", "cooked rice, cooled"),
            (1, "cup", "mixed vegetables, chopped"),
            (2, "", "spring onions, chopped"),
            (1, "tbsp", "soy sauce"),
            (1, "tbsp", "oil"),
            (2, "clove", "garlic, chopped"),
            (0.5, "tsp", "salt"),
            (0.25, "tsp", "black pepper"),
        ],
        [
            "Heat the oil in a wide pan or wok on high heat.",
            "Add the garlic and spring onions and stir for 30 seconds.",
            "Add the mixed vegetables and cook for 3 minutes, keeping them crisp.",
            "Add the cold cooked rice and toss well.",
            "Sprinkle the soy sauce, salt and pepper over the rice and mix.",
            "Cook for 2 more minutes, then taste and adjust the salt.",
            "Serve immediately while hot.",
        ],
        20,
        [
            "Use a dry cloth to hold the hot wok handle.",
            "Do not leave hot oil unattended on high heat.",
        ],
    ),
    RecipeTemplate(
        "maggi", "Masala Maggi", 1,
        [
            (1, "packet", "maggi noodles"),
            (1.5, "cup", "water"),
            (1, "tsp", "butter"),
            (1, "", "tomato, chopped"),
            (0.25, "tsp", "salt"),
        ],
        [
            "Heat the butter in a small pan and cook the tomato for 2 minutes.",
            "Add the water and bring it to a boil.",
            "Add the noodles and the tastemaker. Stir once.",
            "Cook for 3 minutes on medium heat, until the noodles are soft.",
            "Taste and add a little salt only if needed.",
            "Turn off the heat and rest for 1 minute before serving.",
        ],
        10,
        [
            "Use a long spoon; the small pan will be hot.",
            "Do not leave the pan on the flame after the noodles are done.",
        ],
    ),
    RecipeTemplate(
        "chole", "Chole Masala", 4,
        [
            (1.5, "cup", "chickpeas, soaked overnight"),
            (2, "", "onions, chopped"),
            (2, "", "tomatoes, chopped"),
            (1, "tbsp", "chana masala"),
            (1, "tsp", "cumin seeds"),
            (2, "tbsp", "oil"),
            (1, "tsp", "salt"),
            (0.25, "cup", "coriander leaves"),
        ],
        [
            "Drain the soaked chickpeas and boil them with fresh water until soft, about 25 minutes.",
            "Heat the oil in a pan and add the cumin seeds.",
            "Add the onions and cook until golden.",
            "Add the tomatoes and cook until they soften, about 5 minutes.",
            "Stir in the chana masala and salt, then add the boiled chickpeas with a little cooking water.",
            "Simmer for 10 minutes so the flavours come together.",
            "Finish with coriander leaves and serve with rice or roti.",
        ],
        50,
        [
            "Beans and hot gravy can splash; stir gently.",
            "Turn off the heat before moving the pot.",
        ],
    ),
    RecipeTemplate(
        "rajma", "Rajma Curry", 4,
        [
            (1.5, "cup", "kidney beans, soaked overnight"),
            (2, "", "onions, chopped"),
            (2, "", "tomatoes, chopped"),
            (1, "tsp", "ginger-garlic paste"),
            (1, "tsp", "cumin seeds"),
            (1, "tsp", "garam masala"),
            (2, "tbsp", "oil"),
            (1, "tsp", "salt"),
            (0.25, "cup", "coriander leaves"),
        ],
        [
            "Drain the soaked kidney beans and boil them until very soft, about 30 minutes.",
            "Heat the oil in a pan and add the cumin seeds.",
            "Add the onions and cook until golden, then add the ginger-garlic paste.",
            "Add the tomatoes and cook until they break down, about 6 minutes.",
            "Stir in the garam masala and salt.",
            "Add the beans with some cooking water and simmer for 12 minutes.",
            "Mash a few beans against the pan to thicken the gravy, then finish with coriander.",
        ],
        50,
        [
            "Hot bean gravy can bubble and splash; stir gently.",
            "Turn off the heat before moving the pot.",
        ],
    ),
    RecipeTemplate(
        "chicken curry", "Chicken Curry", 3,
        [
            (500, "g", "chicken, bone-in"),
            (2, "", "onions, chopped"),
            (2, "", "tomatoes, chopped"),
            (1, "tbsp", "ginger-garlic paste"),
            (0.5, "tsp", "turmeric"),
            (1, "tsp", "red chilli powder"),
            (1, "tsp", "garam masala"),
            (2, "tbsp", "oil"),
            (1, "tsp", "salt"),
            (0.25, "cup", "coriander leaves"),
        ],
        [
            "Wash the chicken and pat it dry. Heat the oil in a heavy pan.",
            "Add the onions and cook until golden brown.",
            "Add the ginger-garlic paste and cook for 1 minute.",
            "Add the tomatoes, turmeric, chilli powder and salt. Cook until the oil separates, about 6 minutes.",
            "Add the chicken and turn it in the masala until coated.",
            "Add half a cup of hot water, cover and simmer for 20 minutes, until the chicken is cooked through.",
            "Stir in the garam masala, finish with coriander and serve hot.",
        ],
        45,
        [
            "Wash your hands, knife, and board after touching raw chicken.",
            "Cook the chicken until the juices run clear and the inside is no longer pink.",
            "Turn off the heat before moving the hot pan.",
        ],
    ),
    RecipeTemplate(
        "mixed vegetable curry", "Mixed Vegetable Curry", 4,
        [
            (3, "cup", "mixed vegetables, chopped"),
            (2, "", "onions, chopped"),
            (2, "", "tomatoes, chopped"),
            (1, "tbsp", "ginger-garlic paste"),
            (1, "tsp", "cumin seeds"),
            (1, "tsp", "garam masala"),
            (0.5, "tsp", "turmeric"),
            (2, "tbsp", "oil"),
            (1, "tsp", "salt"),
            (0.25, "cup", "coriander leaves"),
        ],
        [
            "Heat the oil in a pan and add the cumin seeds.",
            "Add the onions and cook until golden, about 5 minutes.",
            "Add the ginger-garlic paste and cook for 1 minute.",
            "Add the tomatoes, turmeric and salt. Cook until the tomatoes soften.",
            "Add the mixed vegetables and stir to coat them in the masala.",
            "Add half a cup of hot water, cover and simmer for 12 minutes, until the vegetables are tender.",
            "Stir in the garam masala, finish with coriander and serve with roti or rice.",
        ],
        35,
        [
            "Hot curry can bubble and splash; stir gently.",
            "Turn off the heat before moving the hot pan.",
        ],
    ),
    RecipeTemplate(
        "kanda poha", "Kanda Poha", 2,
        [
            (2, "cup", "flattened rice (poha)"),
            (1, "", "onion, finely chopped"),
            (2, "tbsp", "peanuts"),
            (1, "tbsp", "olive oil or ghee"),
            (1, "tsp", "mustard seeds"),
            (1, "tsp", "turmeric powder"),
            (8-10, "", "curry leaves"),
            (1, "", "green chilli, finely chopped"),
            (1, "tsp", "salt"),
            (1, "tsp", "sugar"),
            (2, "cup", "water"),
        ],
        [
            "Rinse the poha in water just until soft, drain and set aside.",
            "Heat oil in a kadhai, add mustard seeds and peanuts, fry until they splutter.",
            "Add chopped onion, green chilli and curry leaves, fry until translucent.",
            "Add the poha, turmeric and salt, toss gently for 2 minutes.",
            "Add a pinch of sugar and fresh coriander, mix well.",
            "Serve hot with lemon wedges.",
        ],
        20,
        [
            "Wash hands before handling food.",
            "Do not over-fry the poha or it turns soggy.",
        ],
    ),
    RecipeTemplate(
        "batata poha", "Batata Poha", 2,
        [
            (2, "cup", "flattened rice (poha)"),
            (1, "", "potato, diced small"),
            (1, "", "onion, finely chopped"),
            (2, "tbsp", "peanuts"),
            (1, "tbsp", "olive oil or ghee"),
            (1, "tsp", "mustard seeds"),
            (1, "tsp", "turmeric powder"),
            (8-10, "", "curry leaves"),
            (1, "", "green chilli, finely chopped"),
            (1, "tsp", "salt"),
            (2, "cup", "water"),
        ],
        [
            "Rinse the poha in water just until soft, drain and set aside.",
            "Heat oil in a kadhai, add mustard seeds and peanuts, fry until they splutter.",
            "Add diced potato and fry until partially cooked.",
            "Add chopped onion, green chilli and curry leaves, fry until translucent.",
            "Add the poha, turmeric and salt, toss gently for 2 minutes.",
            "Add a pinch of sugar and fresh coriander, mix well.",
            "Serve hot with lemon wedges.",
        ],
        20,
        [
            "Wash hands before handling food.",
            "Do not over-fry the poha or it turns soggy.",
        ],
    ),
    RecipeTemplate(
        "sabudana khichdi", "Sabudana Khichdi", 4,
        [
            (1, "cup", "sabudana (tapioca pearls)"),
            (2, "cup", "potato, diced small"),
            (1, "", "peanut, crushed"),
            (1, "", "rocky sendha namak (rock salt) to taste"),
            (8-10, "", "curry leaves"),
            (1, "", "green chilli, finely chopped"),
            (1, "tbsp", "ghee or oil"),
            (2, "cup", "water"),
            (1, "", "lemon juice, to taste"),
            (1, "tsp", "sugar"),
        ],
        [
            "Wash the sabudana and soak in water for 4 hours or overnight.",
            "Drain excess water. Heat ghee in a kadhai, add curry leaves and green chilli.",
            "Add diced potato and fry partially.",
            "Add drained sabudana and crushed peanuts, stir gently until translucent.",
            "Add rock salt and sugar, mix well.",
            "Add lemon juice and fresh coriander, serve hot.",
        ],
        25,
        [
            "Wash hands before handling food.",
            "Do not over-stir or sabudana turns mushy.",
        ],
    ),
    RecipeTemplate(
        "misal pav", "Misal Pav", 4,
        [
            (2, "cup", "matki (moth beans) sprouts"),
            (1, "cup", "öl or ghee"),
            (1, "tbsp", "mustard seeds"),
            (1, "tbsp", "urad dal"),
            (1, "tbsp", "chana dal"),
            (8-10, "", "curry leaves"),
            (1, "", "dry red chilli"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "red chilli powder"),
            (1, "tsp", "salt"),
            (1, "tsp", "goda masala"),
            (2, "cup", "water"),
            (1, "", "onion, chopped (for topping)"),
            (1, "", "coriander leaves (for topping)"),
            (4, "pav", "bread rolls"),
        ],
        [
            "Cook matki sprouts in water until soft, about 30 minutes.",
            "Heat ghee, add mustard seeds, urad dal, chana dal, dry red chilli, curry leaves and fry.",
            "Add turmeric, red chilli powder, goda masala and salt, fry for 1 minute.",
            "Add water and sprouts, simmer for 15 minutes until thick.",
            "For the tarap: temper additional ghee with mustard seeds, broken red chilli and curry leaves.",
            "Serve hot matki mixture with pav, topped with chopped onion, coriander and tarap.",
        ],
        40,
        [
            "Wash hands before handling sprouts.",
            "Goda masala gives the authentic flavour; do not substitute lightly.",
        ],
    ),
    RecipeTemplate(
        "pithla bhakri", "Pithla Bhakri", 3,
        [
            (1, "cup", "besan (gram flour)"),
            (3, "cup", "water"),
            (1, "tbsp", "olive oil or ghee"),
            (1, "tsp", "mustard seeds"),
            (1, "tsp", "cumin seeds"),
            (8-10, "", "curry leaves"),
            (1, "", "green chilli, finely chopped"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "salt"),
        ],
        [
            "Mix besan with water to a smooth lump-free batter.",
            "Heat ghee in a kadhai, add mustard seeds, cumin seeds and fry until they splutter.",
            "Add curry leaves, green chilli and fry briefly.",
            "Pour the batter into the kadhai, stir continuously on medium heat.",
            "Keep stirring until the mixture thickens and leaves the sides of the pan.",
            "Add salt and adjust consistency with water if needed.",
            "Serve hot with bhakri (millet flatbread) or roti.",
        ],
        25,
        [
            "Wash hands before handling food.",
            "Stir continuously to avoid lumps forming.",
        ],
    ),
    RecipeTemplate(
        "varan bhaat", "Varan Bhaat", 4,
        [
            (1, "cup", "toor dal (pigeon pea dal)"),
            (3, "cup", "water"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "salt"),
            (1, "tbsp", "ghee"),
            (1, "tsp", "cumin seeds"),
            (8-10, "", "curry leaves"),
            (1, "", "dry red chilli"),
            (1, "", "onion, sliced (optional)"),
            (1, "tsp", "garam masala"),
        ],
        [
            "Pressure cook or boil toor dal with water, turmeric and salt until soft.",
            "Mash the dal well. Heat ghee in a small tadka pan, add cumin seeds, curry leaves and dry red chilli.",
            "Pour the tadka over the dal, add garam masala, mix well.",
            "Serve hot dal with steamed rice (bhaat).",
        ],
        30,
        [
            "Wash hands before handling food.",
            "Do not leave the pressure cooker unattended.",
        ],
    ),
    RecipeTemplate(
        "thalipeeth", "Thalipeeth", 4,
        [
            (2, "cup", "bhajri flour (millet flour)"),
            (1, "cup", "jowar flour (sorghum flour)"),
            (0.5, "cup", "wheat flour"),
            (1, "", "onion, finely chopped"),
            (1, "", "green chilli, finely chopped"),
            (1, "", "ginger, grated"),
            (1, "tbsp", "ghee"),
            (1, "tsp", "cumin seeds"),
            (8-10, "", "curry leaves"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "salt"),
            (1, "", "yogurt or buttermilk, as needed for consistency"),
        ],
        [
            "Mix all dry flours with onion, green chilli, ginger, cumin seeds, curry leaves, turmeric and salt.",
            "Add yogurt or buttermilk gradually to form a soft dough.",
            "Divide dough into portions, shape into thick discs.",
            "Heat a tawa, add a little ghee.",
            "Cook thalipeeth on medium heat until both sides are golden brown and cooked through.",
            "Serve hot with white butter or curd.",
        ],
        35,
        [
            "Wash hands before handling food.",
            "Keep heat medium; if too high, thalipeeth burns before cooking through.",
        ],
    ),
    RecipeTemplate(
        "bharli vangi", "Bharli Vangi", 4,
        [
            (4-6, "small", "brinjal (eggplant)"),
            (2, "tbsp", "peanut powder"),
            (1, "tbsp", "goda masala"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "red chilli powder"),
            (1, "tsp", "salt"),
            (1, "tbsp", "ghee"),
            (1, "", "onion, finely chopped"),
            (1, "", "garlic, chopped"),
            (8-10, "", "curry leaves"),
            (1, "", "green chilli, finely chopped"),
        ],
        [
            "Slit the brinjals into quarters, keeping the base intact.",
            "Roast peanuts, grind to powder.",
            "Mix peanut powder with goda masala, turmeric, red chilli powder and salt.",
            "Fill the brinjals with the masala mixture.",
            "Heat ghee in a kadhai, add chopped onion, garlic, curry leaves and green chilli, fry until golden.",
            "Add the filled brinjals, cover and cook on low heat until brinjals are soft, turning occasionally.",
            "Garnish with fresh coriander and serve hot with roti or rice.",
        ],
        40,
        [
            "Wash hands before handling food.",
            "Do not open the lid too often while cooking; steam escapes and cooking takes longer.",
        ],
    ),
    RecipeTemplate(
        "zunka", "Zunka", 3,
        [
            (1, "cup", "besan (gram flour)"),
            (2, "cup", "water"),
            (1, "tbsp", "ghee or oil"),
            (1, "tsp", "mustard seeds"),
            (1, "tsp", "cumin seeds"),
            (8-10, "", "curry leaves"),
            (1, "", "green chilli, finely chopped"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "salt"),
        ],
        [
            "Mix besan with water to a smooth lump-free batter.",
            "Heat ghee in a kadhai, add mustard seeds, cumin seeds and fry until they splutter.",
            "Add curry leaves, green chilli and fry briefly.",
            "Pour the batter into the kadhai, stir continuously on medium heat.",
            "Keep stirring until the mixture thickens and leaves the sides of the pan.",
            "Add salt and serve hot with bhakri or roti.",
        ],
        20,
        [
            "Wash hands before handling food.",
            "Stir continuously to avoid lumps forming.",
        ],
    ),
    RecipeTemplate(
        "puran poli", "Puran Poli", 6,
        [
            (1, "cup", "whole wheat flour (atta)"),
            (0.5, "cup", "besan (gram flour)"),
            (0.75, "cup", "jaggery"),
            (0.25, "cup", "chana dal (split Bengal gram)"),
            (1, "tbsp", "ghee, for dough"),
            (3, "tbsp", "ghee, for cooking"),
            (1, "tsp", "cardamom powder"),
            (1, "tsp", "nutmeg powder"),
            (1, "", "orange food colour, optional"),
        ],
        [
            "Cook chana dal with water until soft, drain and mash.",
            "Add jaggery, cardamom powder, nutmeg powder and cook until thick and rolling.",
            "Make a soft dough with wheat flour, ghee and water.",
            "Divide dough into portions, roll into thick puris.",
            "Place a spoonful of stuffing in the centre, seal and roll gently.",
            "Cook on a hot tawa with ghee until both sides are golden brown.",
            "Serve hot with milk or shrikhand.",
        ],
        45,
        [
            "Wash hands before handling food.",
            "Do not over-stuff; the poli may burst while cooking.",
        ],
    ),
    RecipeTemplate(
        "kothimbir vadi", "Kothimbir Vadi", 4,
        [
            (2, "cup", "besan (gram flour)"),
            (1, "cup", "rice flour"),
            (1, "bunch", "fresh coriander (dhaniya) leaves, finely chopped"),
            (1, "", "green chilli, finely chopped"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "salt"),
            (1, "tbsp", "yogurt or buttermilk"),
            (2, "tbsp", "olive oil or ghee"),
        ],
        [
            "Mix besan, rice flour, coriander, green chilli, turmeric and salt.",
            "Add yogurt or buttermilk to form a thick batter.",
            "Heat oil in a kadhai, drop small portions of batter and deep-fry until golden and crisp.",
            "Drain on absorbent paper and serve hot with chutney.",
        ],
        30,
        [
            "Wash hands before handling food.",
            "Do not overcrowd the kadhai; fry in batches.",
        ],
    ),
    RecipeTemplate(
        "batata bhaji", "Batata Bhaji", 4,
        [
            (4, "", "potato, diced small"),
            (1, "", "onion, finely chopped"),
            (1, "", "tomato, diced"),
            (2, "tbsp", "olive oil or ghee"),
            (1, "tsp", "mustard seeds"),
            (1, "tsp", "cumin seeds"),
            (8-10, "", "curry leaves"),
            (1, "", "green chilli, finely chopped"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "salt"),
            (1, "tbsp", "fresh coriander"),
        ],
        [
            "Heat oil in a kadhai, add mustard seeds and cumin seeds, let them splutter.",
            "Add chopped onion and fry until translucent.",
            "Add diced potato and fry for 3-4 minutes.",
            "Add diced tomato, green chilli and curry leaves, fry for 2 minutes.",
            "Add turmeric and salt, mix well and cover for 5 minutes.",
            "Garnish with fresh coriander and serve hot with roti or rice.",
        ],
        25,
        [
            "Wash hands before handling food.",
            "Do not uncovered-cook too long or potatoes turn mushy.",
        ],
    ),
    RecipeTemplate(
        "amti", "Amti", 4,
        [
            (1, "cup", "toor dal (pigeon pea dal)"),
            (3, "cup", "water"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "sambar powder"),
            (1, "tsp", "tamarind pulp"),
            (1, "tsp", "salt"),
            (1, "tbsp", "ghee"),
            (1, "tsp", "mustard seeds"),
            (1, "tsp", "cumin keys"),
            (8-10, "", "curry leaves"),
            (1, "", "dry red chilli"),
            (1, "", "jaggery, a pinch"),
        ],
        [
            "Pressure cook toor dal with water, turmeric and salt until soft, about 3 whistles.",
            "Extract tamarind pulp by soaking in warm water and squeezing.",
            "Heat ghee in a kadhai, add mustard seeds, cumin seeds, curry leaves and dry red chilli, fry briefly.",
            "Add tamarind pulp, sambar powder and salt, simmer for 5 minutes.",
            "Add the cooked dal, adjust consistency with water, simmer for 10 minutes.",
            "Add a pinch of jaggery and garnish with fresh coriander.",
            "Serve hot with rice or roti.",
        ],
        35,
        [
            "Wash hands before handling food.",
            "Do not boil the amti too long after adding the dal or it turns sour.",
        ],
    ),
    # --- Other Indian dishes ---
    RecipeTemplate(
        "egg bhurji", "Egg Bhurji", 3,
        [
            (4, "", "eggs"),
            (2, "tbsp", "olive oil or ghee"),
            (1, "", "onion, finely chopped"),
            (1, "", "tomato, finely chopped"),
            (1, "tsp", "ginger-garlic paste"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "red chilli powder"),
            (1, "tsp", "salt"),
            (8-10, "", "curry leaves"),
        ],
        [
            "Beat eggs lightly with a pinch of salt.",
            "Heat oil or ghee in a kadhai, add curry leaves and fry briefly.",
            "Add ginger-garlic paste, onion and tomato, fry until soft.",
            "Add turmeric, red chilli powder and salt, stir for 1 minute.",
            "Pour in the beaten eggs, scramble gently until just set.",
            "Garnish with fresh coriander and serve with roti or bread.",
        ],
        20,
        [
            "Wash hands before handling food.",
            "Do not over-scramble or eggs turn rubbery.",
        ],
    ),
    RecipeTemplate(
        "jeera rice", "Jeera Rice", 2,
        [
            (1, "cup", "basmati rice"),
            (1.5, "cup", "water"),
            (1, "tsp", "cumin seeds"),
            (1, "tbsp", "ghee or butter"),
            (1, "tsp", "salt"),
        ],
        [
            "Rinse basmati rice and soak for 20 minutes. Drain well.",
            "Heat ghee in a kadhai, add cumin seeds and let them splutter.",
            "Add drained rice, salt and 1.5 cups water, stir gently.",
            "Cover and cook on low heat until rice is tender and water is absorbed.",
            "Fluff gently with a fork and serve hot.",
        ],
        15,
        [
            "Wash hands before handling food.",
            "Do not lift the lid while rice is cooking; steam escapes.",
        ],
    ),
    RecipeTemplate(
        "dosa", "Dosa", 2,
        [
            (1, "cup", "rice"),
            (0.5, "cup", "urad dal"),
            (0.25, "tsp", "fenugreek seeds"),
            (1, "", "water, as needed for batter"),
            (1, "tsp", "salt"),
        ],
        [
            "Wash rice and urad dal separately, soak for 4-5 hours.",
            "Grind to a smooth batter, add salt, ferment overnight.",
            "Heat a tawa, pour a ladle of batter, spread in circular motion.",
            "Drizzle little oil around the edges, cook until crisp and golden.",
            "Serve hot with sambar and chutney.",
        ],
        30,
        [
            "Wash hands before handling food.",
            "Batter consistency is key; too thick gives hard dosa, too thin spreads too much.",
        ],
    ),
    RecipeTemplate(
        "idli", "Idli", 2,
        [
            (1, "cup", "rice"),
            (0.5, "cup", "urad dal"),
            (0.25, "tsp", "fenugreek seeds"),
            (1.25, "cup", "water, as needed for batter"),
            (1, "tsp", "salt"),
        ],
        [
            "Wash rice and urad dal separately, then soak them with fenugreek seeds for 4 to 5 hours.",
            "Grind the soaked rice and dal with a little water to a smooth, pourable batter.",
            "Stir in salt, cover, and ferment in a warm place for 8 to 10 hours until the batter rises.",
            "Lightly grease the idli moulds and pour the batter into each mould, leaving a little space.",
            "Steam for 12 to 15 minutes, then rest for 2 minutes before unmoulding.",
            "Serve hot with sambar and coconut chutney.",
        ],
        25,
        [
            "Wash hands before handling food.",
            "Do not open the steamer too early; idlis may be undercooked.",
            "Use a cloth or mitts when removing the hot steamer plate.",
        ],
        20,
        15,
        480,
    ),
    RecipeTemplate(
        "chicken shawarma", "Chicken Shawarma", 4,
        [
            (600, "g", "boneless chicken thighs"),
            (4, "", "pita or flatbreads"),
            (1, "cup", "plain yogurt"),
            (3, "clove", "garlic, crushed"),
            (2, "tbsp", "lemon juice"),
            (2, "tbsp", "olive oil"),
            (1, "tsp", "ground cumin"),
            (1, "tsp", "paprika"),
            (0.5, "tsp", "ground black pepper"),
            (1, "tsp", "salt"),
            (1, "", "onion, thinly sliced"),
            (2, "", "tomatoes, sliced"),
            (2, "cup", "shredded cabbage or lettuce"),
            (0.5, "cup", "tahini or garlic yogurt sauce"),
        ],
        [
            "Mix the yogurt, garlic, lemon juice, olive oil, cumin, paprika, pepper and salt in a bowl.",
            "Add the chicken thighs, coat well, cover and marinate for at least 30 minutes (or overnight).",
            "Heat a grill pan or heavy skillet over medium-high heat until it is hot.",
            "Cook the chicken for 6 to 7 minutes per side, until browned and the juices run clear.",
            "Rest the chicken on a board for 5 minutes, then slice it into thin strips.",
            "Warm the pita breads in the pan for about 20 seconds on each side.",
            "Fill each pita with chicken, onion, tomato and cabbage, then drizzle with tahini or garlic yogurt sauce.",
            "Roll tightly, wrap the base in paper, and serve immediately.",
        ],
        45,
        [
            "Wash your hands, knife, and board after touching raw chicken.",
            "Cook the chicken until the juices run clear and the inside is no longer pink.",
            "Do not leave hot oil on the stove unattended.",
        ],
        20,
        15,
    ),
    RecipeTemplate(
        "sambar", "Sambar", 4,
        [
            (1, "cup", "toor dal"),
            (2, "cup", "tamarind pulp"),
            (2, "cup", "mixed vegetables, chopped (carrot, beans, drumstick)"),
            (1, "tsp", "tamarind powder"),
            (1, "tsp", "sambar powder"),
            (1, "tsp", "turmeric powder"),
            (1, "tsp", "salt"),
            (2, "tbsp", "ghee or oil"),
            (1, "tsp", "mustard seeds"),
            (1, "tsp", "cumin seeds"),
            (8-10, "", "curry leaves"),
            (1, "", "dry red chilli"),
        ],
        [
            "Pressure cook toor dal with turmeric until soft.",
            "Extract tamarind pulp by soaking in warm water and squeezing.",
            "Heat ghee in a kadhai, add mustard seeds, cumin seeds, curry leaves and dry red chilli and fry.",
            "Add tamarind pulp, sambar powder, turmeric and salt, simmer for 5 minutes.",
            "Add chopped vegetables and cook until tender.",
            "Add the cooked dal, simmer for 10 minutes until combined.",
            "Garnish with fresh coriander and serve hot with rice or idli.",
        ],
        40,
        [
            "Wash hands before handling food.",
            "Do not squeeze tamarind pulp with bare hands; it stains.",
        ],
    ),
    RecipeTemplate(
        "roti", "Roti / Chapati", 4,
        [
            (3, "cup", "whole wheat flour (atta)"),
            (1, "cup", "water, warm"),
            (1, "tsp", "salt"),
            (1, "tbsp", "olive oil or ghee, for cooking"),
        ],
        [
            "Mix flour and salt. Add warm water gradually and knead into a soft dough.",
            "Cover and rest for 20 minutes.",
            "Divide into equal portions, roll each into a thin circle.",
            "Heat a tawa, cook roti on medium heat until brown spots appear.",
            "Flip and cook the other side, apply ghee if desired.",
            "Keep warm in a roti basket or cloth.",
        ],
        20,
        [
            "Wash hands before handling food.",
            "Keep dough soft; if too stiff, roti turns hard.",
        ],
    ),
]


def _canonical_name(dish: str) -> str:
    text = normalize_name(dish)
    text = text.replace("biriyani", "biryani")
    text = text.replace("vegetable biryani", "veg biryani")
    text = text.replace("veggie biryani", "veg biryani")
    text = re.sub(r"\b(recipe|dish|meal)\b", " ", text)
    return " ".join(text.split())


def _find_template(dish: str) -> Optional[RecipeTemplate]:
    """Find a tested fallback only when it is the same dish and variant."""
    text = _canonical_name(dish)
    if not text:
        return None
    req_tokens = _content_tokens(text)

    best: Optional[RecipeTemplate] = None
    best_len = 0
    for tpl in _TEMPLATES:
        for key in (normalize_name(tpl.key), normalize_name(tpl.dish)):
            key = _canonical_name(key)
            if not key:
                continue
            if text == key or text.endswith(" " + key):
                if req_tokens <= _content_tokens(tpl.dish) and dish_matches(text, tpl.dish):
                    if len(key) > best_len:
                        best, best_len = tpl, len(key)
    if best is not None:
        return best

    # Bare generic names are only used by direct callers; the app asks a
    # clarifying question before reaching this point.
    if text in {"biryani", "curry", "pasta", "paratha"}:
        family = family_of(text)
        for tpl in _TEMPLATES:
            if family_of(tpl.dish) == family:
                return tpl

    # One unambiguous, variant-compatible family template is safe.
    family = family_of(text)
    if family:
        candidates = [
            tpl for tpl in _TEMPLATES
            if family_of(tpl.dish) == family
            and req_tokens <= _content_tokens(tpl.dish)
            and dish_matches(text, tpl.dish)
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


def build_fallback(dish: str, servings: int) -> Optional[dict]:
    """Return a tested, dish-specific fallback, or None (never a wrong dish)."""
    tpl = _find_template(dish)
    if tpl is None:
        return None

    servings = max(1, min(MAX_SERVINGS, int(servings)))
    factor = servings / tpl.base_servings
    ingredients = [
        {"name": name, "quantity": quantity_text(amount * factor, unit)}
        for amount, unit, name in tpl.ingredients
    ]
    return {
        "dish": tpl.dish,
        "servings": servings,
        "ingredients": ingredients,
        "steps": list(tpl.steps),
        "duration_minutes": tpl.duration_minutes,
        "prep_time_minutes": tpl.prep_time_minutes,
        "cook_time_minutes": tpl.cook_time_minutes,
        "waiting_time_minutes": tpl.waiting_time_minutes,
        "total_time_minutes": (
            tpl.prep_time_minutes + tpl.cook_time_minutes + tpl.waiting_time_minutes
        ) or tpl.duration_minutes,
        "safety": list(tpl.safety) or default_safety(tpl.steps),
        "source": "fallback",
        "source_type": "builtin",
        "validation_status": "builtin",
    }


# ----------------------------------------------------------------------
# Ambiguous dishes
# ----------------------------------------------------------------------
_AMBIGUOUS = {
    "biryani": "Chicken or vegetable biryani?",
    "biriyani": "Chicken or vegetable biryani?",
    "pasta": "Red sauce or white sauce pasta?",
    "curry": "Which curry: chicken, paneer, or mixed vegetable?",
    "paratha": "Aloo, paneer, or plain paratha?",
    "rice": "Fried rice, pulao, or plain rice?",
}

_VARIANT_HINTS = {
    "biryani": ("chicken", "murgh", "veg", "vegetable", "paneer", "mutton",
                "lamb", "egg", "fish", "prawn", "non veg", "nonveg"),
    "biriyani": ("chicken", "murgh", "veg", "vegetable", "paneer", "mutton",
                 "lamb", "egg", "fish", "prawn", "non veg", "nonveg"),
    "pasta": ("white", "red", "tomato", "cream", "cheese", "alfredo",
              "arrabbiata", "marinara", "pesto"),
    "curry": ("chicken", "paneer", "veg", "vegetable", "mixed", "mutton",
              "egg", "fish", "dal"),
    "paratha": ("aloo", "potato", "paneer", "plain", "sada", "gobi", "mooli"),
    "rice": ("fried", "pulao", "pilaf", "plain", "steamed", "jeera", "lemon"),
}


def clarify_dish(dish: str) -> Optional[str]:
    key = normalize_name(dish)
    if key in _AMBIGUOUS:
        return _AMBIGUOUS[key]
    for base, question in _AMBIGUOUS.items():
        if not re.search(rf"\b{re.escape(base)}\b", key):
            continue
        hints = _VARIANT_HINTS.get(base, ())
        if not any(h in key for h in hints):
            return question
    return None


def ambiguous_base(dish: str) -> Optional[str]:
    """Return the base word ('biryani') for a phrased request ('biryani recipe')."""
    key = normalize_name(dish)
    for base in _AMBIGUOUS:
        if re.search(rf"\b{re.escape(base)}\b", key):
            return base
    return None


def apply_variant(base: str, answer: str) -> Optional[str]:
    """Combine a base dish with the user's clarifying answer."""
    base_key = ambiguous_base(base) or normalize_name(base)
    ans = normalize_name(answer)
    if base_key in ("biryani", "biriyani"):
        if any(w in ans for w in ("chicken", "murgh", "non veg", "nonveg")):
            return "chicken biryani"
        if any(w in ans for w in ("mutton", "lamb", "goat")):
            return "mutton biryani"
        if any(w in ans for w in ("egg", "anda")):
            return "egg biryani"
        if any(w in ans for w in ("fish", "prawn", "shrimp")):
            return "fish biryani"
        if "paneer" in ans:
            return "paneer biryani"
        if any(w in ans for w in ("veg", "vegetable", "sabzi")):
            return "veg biryani"
        return None
    if base_key == "pasta":
        if any(w in ans for w in ("white", "cream", "alfredo", "cheese")):
            return "white sauce pasta"
        if any(w in ans for w in ("red", "tomato", "arrabbiata", "marinara")):
            return "pasta"
        if "pesto" in ans:
            return "pesto pasta"
        return None
    if base_key == "curry":
        if "chicken" in ans:
            return "chicken curry"
        if "paneer" in ans:
            return "paneer butter masala"
        if "mutton" in ans:
            return "mutton curry"
        if any(w in ans for w in ("egg", "anda")):
            return "egg curry"
        if any(w in ans for w in ("fish", "prawn", "shrimp")):
            return "fish curry"
        if any(w in ans for w in ("veg", "vegetable", "mixed")):
            return "mixed vegetable curry"
        return None
    if base_key == "paratha":
        if any(w in ans for w in ("aloo", "potato")):
            return "aloo paratha"
        if "paneer" in ans:
            return "paneer paratha"
        if any(w in ans for w in ("plain", "sada", "no filling")):
            return "plain paratha"
        return None
    if base_key == "rice":
        if "fried" in ans:
            return "fried rice"
        if any(w in ans for w in ("pulao", "pilaf")):
            return "pulao"
        if "jeera" in ans or "cumin" in ans:
            return "jeera rice"
        if "lemon" in ans:
            return "lemon rice"
        if any(w in ans for w in ("plain", "steamed", "simple")):
            return "steamed rice"
        return None
    return None


# ----------------------------------------------------------------------
# Ingredient-first suggestions (deterministic first, LLM can improve)
# ----------------------------------------------------------------------
_INGREDIENT_RULES = [
    (("potato", "aloo"), "Aloo paratha", ["potatoes", "flour"], ["ghee"]),
    (("paneer",), "Paneer butter masala", ["paneer"], ["cream", "butter"]),
    (("rice", "chawal"), "Vegetable fried rice", ["rice"], ["soy sauce"]),
    (("pasta", "spaghetti", "penne"), "Tomato pasta", ["pasta"], ["tomato sauce"]),
    (("bread", "pav"), "Pav bhaji", ["pav"], ["butter", "pav bhaji masala"]),
    (("egg", "anda"), "Masala omelette", ["eggs"], ["butter"]),
    (("chickpea", "chana", "chole"), "Chole masala", ["chickpeas"], ["onion", "tomato"]),
    (("rajma", "kidney bean"), "Rajma curry", ["rajma"], ["onion", "tomato"]),
    (("poha",), "Kanda poha", ["poha"], ["lemon"]),
    (("semolina", "rava", "sooji"), "Rava upma", ["rava"], ["curry leaves"]),
    (("dal", "lentil"), "Dal tadka", ["dal"], ["ghee", "cumin"]),
    (("chicken",), "Chicken curry", ["chicken"], ["onion", "tomato", "ginger-garlic"]),
    (("tomato",), "Tomato pasta", ["tomato"], ["pasta"]),
]


def suggest_from_ingredients(text: str) -> List[dict]:
    low = normalize_name(text)
    out: List[dict] = []
    seen = set()
    for keys, dish, uses, missing in _INGREDIENT_RULES:
        if any(k in low for k in keys) and dish not in seen:
            seen.add(dish)
            out.append({"dish": dish, "uses": uses, "missing": missing})
        if len(out) >= 3:
            break
    if not out:
        out = [
            {"dish": "Masala omelette", "uses": ["eggs"], "missing": []},
            {"dish": "Kanda poha", "uses": ["poha"], "missing": ["lemon"]},
            {"dish": "Tomato pasta", "uses": ["pasta"], "missing": ["tomato sauce"]},
        ]
    return out


def parse_json_loose(text: str) -> Optional[dict]:
    """Parse JSON that may be wrapped in prose or code fences."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None
