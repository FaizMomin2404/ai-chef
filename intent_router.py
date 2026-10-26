"""
Intent Router for AI Chef.

Classifies user messages into intents so that normal conversation
("Hi, how are you?") is not mistakenly routed into recipe generation.

The router uses pattern matching with Hinglish/English awareness and
an optional LLM-based fallback for ambiguous cases. It never fabricates
a cooking context where none exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class IntentResult:
    intent: str
    confidence: float  # 0.0 - 1.0
    dish: Optional[str] = None
    servings: Optional[int] = None
    timer_minutes: Optional[int] = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------
GENERAL_CHAT = "GENERAL_CHAT"
RECIPE_INFORMATION = "RECIPE_INFORMATION"
RECIPE_REQUEST = "RECIPE_REQUEST"
INGREDIENT_BASED_SUGGESTION = "INGREDIENT_BASED_SUGGESTION"
START_COOKING = "START_COOKING"
COOKING_FOLLOW_UP = "COOKING_FOLLOW_UP"
TIMER_COMMAND = "TIMER_COMMAND"
REPEAT_OR_CLARIFY = "REPEAT_OR_CLARIFY"
UNSAFE_OR_UNSUPPORTED_REQUEST = "UNSAFE_OR_UNSUPPORTED_REQUEST"

_ALL_INTENTS = {
    GENERAL_CHAT,
    RECIPE_INFORMATION,
    RECIPE_REQUEST,
    INGREDIENT_BASED_SUGGESTION,
    START_COOKING,
    COOKING_FOLLOW_UP,
    TIMER_COMMAND,
    REPEAT_OR_CLARIFY,
    UNSAFE_OR_UNSUPPORTED_REQUEST,
}

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

# Greetings and social conversation — never cooking
_GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "namaste", "namaskar", "sup", "yo", "hi there", "hello there",
    "how are you", "how r u", "kaisa hai", "kya haal", "hal kaisa",
    "what's up", "whats up", "kya chal raha", "kya haal hai",
    "goodnight", "good night", "good bye", "bye", "alvida",
    "thank you", "thanks", "dhanyavad", "shukriya",
    "sorry", "excuse me", "maaf kijiye",
}

# A cooking request must win over social openers such as "hi, make idli".
_COOKING_SIGNAL_RE = re.compile(
    r"\b(?:make|cook|prepare|banao|bana\s*do|banana|pakana|banaye|banayenge)\b"
    r"|\brecipe\s+for\b"
)

# Questions about capabilities or general knowledge
_CAPABILITY_PHRASES = {
    "what can you do", "what do you do", "your capabilities",
    "who are you", "are you", "tell me about yourself",
    "help me", "how do i use", "how to use this app",
    "instructions", "guide me", "what is ai chef",
}

# Jokes and light conversation
_JOKE_PHRASES = {
    "tell me a joke", "joke", "funny", "make me laugh",
    "something funny", "joke sunao", "joke batao",
}

# Dish information questions — user wants to know about a dish, not cook it
_DISH_INFO_PHRASES = {
    "what is", "what's", "tell me about", "explain",
    "kya hai", "baare mein batao", "details", "origin",
    "history of", "about the dish", "meaning of",
}

# Recipe request — user explicitly wants to cook
_RECIPE_REQUEST_PHRASES = {
    "i want to make", "i want to cook", "how do i make",
    "how to cook", "how to make", "recipe for",
    "make me", "cook for me", "banana hai", "banana chahiye",
    "pakana hai", "banate hain", "banayenge", "banaye",
    "i'll make", "i will make", "lets make", "let's make",
    "chahiye mujhe", "mujhe banana hai", "mujhe pakana hai",
}

# "I want X" patterns where X might be a dish name — checked with dish detection
_WANT_DISH_PHRASES = {
    "i want", "i want a", "i want an", "i would like", "i'd like",
}

# Ingredient-first — user lists what they have
_INGREDIENT_PHRASES = {
    "i have", "i've got", "i have got", "mujhe paas hai",
    "mere paas", "ingredients", "what can i make",
    "kya bana sakta", "kya bana sakte hain", "suggest",
    "recommend", "kya banau", "kya pakau",
}

# Cooking follow-up — during an active session
_COOKING_COMMANDS = {
    "next", "next step", "aage", "aage badho", "move on",
    "previous", "back", "pichle", "pichla", "repeat",
    "repeat step", "repeat the last", "dobara", "fir se",
    "what next", "kya karein", "ab kya", "ab kya karein",
    "what should i do", "current step", "step number",
    "how much time", "kitna time", "time left", "remaining",
    "timer", "pause timer", "resume timer", "stop timer",
    "reset timer", "add too much", "too much salt", "too much water",
    "missing ingredient", "dont have", "do not have", "nahi hai",
    "explain in english", "explain that in english", "angrezi mein",
    "speak slowly", "bol thoda slow", "slowly", "slow down",
    "i added", "i put", "i have put", "put it on", "on the stove",
    "chopped", "cut", "diced", "sliced", "washed",
    "ready", "start cooking", "begin", "shuru", "chalo",
}

# Timer commands
_TIMER_PHRASES = {
    "set timer", "start timer", "timer for", "minutes",
    "minute", "mins", "min", "timer", "countdown",
}

# Unsafe requests
_UNSAFE_PHRASES = {
    "leave it unattended", "leave alone", "walk away",
    "dont watch", "do not watch", "ignore the fire",
    "leave the stove", "leave the gas", "leave the oil",
    "leave the pan", "leave the pot",
}

# Dish names the system knows (for intent disambiguation).
# This list is intentionally broad but not exhaustive: it only helps the
# router recognise a dish quickly. Unknown dishes are still routed to the
# recipe path via cooking-verb detection and handled by LLM generation.
_KNOWN_DISH_WORDS = {
    # Indian
    "biryani", "biriyani", "pasta", "poha", "upma", "khichdi",
    "dal", "daal", "paneer", "paratha", "chole", "rajma",
    "misal", "pav bhaji", "sabudana", "thalipeeth", "bharli vangi",
    "zunka", "puran poli", "kothimbir vadi", "batata bhaji", "amti",
    "egg bhurji", "jeera rice", "dosa", "idli", "sambar",
    "roti", "chapati", "curry", "omelette", "pancakes", "pancake",
    "maggi", "fried rice", "pulao", "kheer", "halwa", "samosa",
    # Middle Eastern
    "shawarma", "falafel", "hummus", "kebab", "kofta", "tabbouleh",
    "pita", "baklava",
    # Italian
    "pizza", "risotto", "lasagna", "gnocchi", "carbonara", "ravioli",
    # East / South-East Asian
    "noodles", "dumplings", "spring rolls", "chow mein", "manchurian",
    "ramen", "sushi", "stir fry", "pad thai",
    # Western
    "sandwich", "salad", "soup", "burger", "steak", "waffles",
    "brownie", "cake", "pudding", "pie", "cookie", "muffin",
}


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _extract_number(text: str) -> Optional[int]:
    """Extract a small integer from text."""
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        return int(m.group(1))
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "do": 2, "teen": 3, "char": 4, "chaar": 4,
        "panch": 5, "paanch": 5, "chhe": 6, "che": 6,
        "saat": 7, "aath": 8, "nau": 9, "das": 10,
    }
    for w, n in words.items():
        if re.search(rf"\b{w}\b", text):
            return n
    return None


def _contains_dish_name(text: str) -> Optional[str]:
    """Return a known dish name found in text, or None."""
    for word in sorted(_KNOWN_DISH_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", text):
            return word
    return None


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def classify_intent(text: str) -> IntentResult:
    """
    Classify a user message into an intent.

    This is a fast, dependency-free classifier. It uses keyword matching
    with English and Hinglish awareness. For ambiguous cases it returns
    GENERAL_CHAT with low confidence so the caller can ask a clarification.
    """
    low = normalize(text)

    if not low:
        return IntentResult(GENERAL_CHAT, 0.0, reason="empty message")

    # 1. Greetings / social conversation. A cooking request in the same
    #    sentence ("hi, make idli") must not be swallowed by the greeting.
    for g in _GREETINGS:
        if re.search(rf"\b{re.escape(g)}\b", low):
            if _COOKING_SIGNAL_RE.search(low):
                break
            return IntentResult(GENERAL_CHAT, 0.9, reason=f"greeting: {g}")

    # 2. Capability questions (but not "help me make idli")
    for c in _CAPABILITY_PHRASES:
        if re.search(rf"\b{re.escape(c)}\b", low):
            if _COOKING_SIGNAL_RE.search(low) and c in ("help me",):
                break
            return IntentResult(GENERAL_CHAT, 0.85, reason=f"capability: {c}")

    # 3. Jokes
    for j in _JOKE_PHRASES:
        if re.search(rf"\b{re.escape(j)}\b", low):
            return IntentResult(GENERAL_CHAT, 0.85, reason=f"joke: {j}")

    # 4. Timer commands
    for t in _TIMER_PHRASES:
        if re.search(rf"\b{re.escape(t)}\b", low):
            minutes = _extract_number(low)
            return IntentResult(
                TIMER_COMMAND, 0.8,
                timer_minutes=minutes,
                reason=f"timer: {t}",
            )

    # 5. Unsafe requests
    for u in _UNSAFE_PHRASES:
        if re.search(rf"\b{re.escape(u)}\b", low):
            return IntentResult(
                UNSAFE_OR_UNSUPPORTED_REQUEST, 0.9,
                reason=f"unsafe: {u}",
            )

    # 6. Cooking commands
    for c in _COOKING_COMMANDS:
        if re.search(rf"\b{re.escape(c)}\b", low):
            # Distinguish "repeat" from general chat
            if c in ("repeat", "repeat step", "repeat the last", "dobara", "fir se"):
                return IntentResult(REPEAT_OR_CLARIFY, 0.8, reason=f"repeat: {c}")
            if c in ("pause timer", "resume timer", "stop timer", "reset timer",
                     "set timer", "start timer", "timer for"):
                minutes = _extract_number(low)
                return IntentResult(
                    TIMER_COMMAND, 0.8,
                    timer_minutes=minutes,
                    reason=f"timer command: {c}",
                )
            return IntentResult(
                COOKING_FOLLOW_UP, 0.75,
                reason=f"cooking command: {c}",
            )

    # 7. Dish information questions ("what is biryani?", "explain poha")
    for phrase in _DISH_INFO_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            dish = _contains_dish_name(low)
            if dish:
                return IntentResult(
                    RECIPE_INFORMATION, 0.85,
                    dish=dish,
                    reason=f"dish info: {phrase} {dish}",
                )
            return IntentResult(
                RECIPE_INFORMATION, 0.6,
                reason=f"dish info generic: {phrase}",
            )

    # 8. Ingredient-first suggestions
    for phrase in _INGREDIENT_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            # Check if there's actually an ingredient mentioned
            return IntentResult(
                INGREDIENT_BASED_SUGGESTION, 0.75,
                reason=f"ingredient-first: {phrase}",
            )

    # 9. Explicit recipe requests ("i want to cook biryani")
    for phrase in _RECIPE_REQUEST_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            dish = _contains_dish_name(low)
            if dish:
                servings = _extract_number(low)
                return IntentResult(
                    RECIPE_REQUEST, 0.9,
                    dish=dish,
                    servings=servings,
                    reason=f"recipe request: {phrase} {dish}",
                )
            return IntentResult(
                RECIPE_REQUEST, 0.6,
                reason=f"recipe request generic: {phrase}",
            )

    # 9b. "I want X" where X is a known dish name → recipe request
    for phrase in _WANT_DISH_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            dish = _contains_dish_name(low)
            if dish:
                servings = _extract_number(low)
                return IntentResult(
                    RECIPE_REQUEST, 0.85,
                    dish=dish,
                    servings=servings,
                    reason=f"want dish: {phrase} {dish}",
                )

    # 9c. "make X" / "cook X" / "prepare X". A known dish gives a confident
    #     result; an unknown dish is still a recipe request (the caller extracts
    #     the name and the LLM generates it).
    for verb in ("make", "cook", "prepare", "whip up"):
        if re.search(rf"\b{verb}\s+", low):
            dish = _contains_dish_name(low)
            servings = _extract_number(low)
            if dish:
                return IntentResult(
                    RECIPE_REQUEST, 0.85,
                    dish=dish,
                    servings=servings,
                    reason=f"verb dish: {verb} {dish}",
                )
            obj = re.sub(
                rf"^.*?\b{re.escape(verb)}\s+(?:me\s+|us\s+)?(?:a\s+|an\s+|some\s+|the\s+)?",
                "", low,
            ).strip()
            if len(obj) >= 3 and obj not in {
                "something", "anything", "food", "dinner", "lunch",
                "breakfast", "a dish", "dish", "meal", "it", "this", "that",
            }:
                return IntentResult(
                    RECIPE_REQUEST, 0.75,
                    servings=servings,
                    reason=f"verb object: {verb} {obj[:40]}",
                )

    # 10. If the text contains a known dish name but no cooking verb,
    #     it could be information or a request — lean toward information
    dish = _contains_dish_name(low)
    if dish:
        # Check if there's a cooking verb nearby
        cooking_verbs = {"make", "cook", "prepare", "banana", "pakana",
                         "banate", "banayenge", "banao", "bana", "banado",
                         "banaye"}
        if not any(v in low for v in cooking_verbs):
            return IntentResult(
                RECIPE_INFORMATION, 0.7,
                dish=dish,
                reason=f"dish mentioned without cooking verb: {dish}",
            )
        return IntentResult(
            RECIPE_REQUEST, 0.8,
            dish=dish,
            servings=_extract_number(low),
            reason=f"dish with cooking verb: {dish}",
        )

    # 11. Default: general chat
    return IntentResult(GENERAL_CHAT, 0.5, reason="default fallback")


# ---------------------------------------------------------------------------
# Conversational responses (no LLM needed for common cases)
# ---------------------------------------------------------------------------

def conversational_response(text: str, intent: IntentResult) -> str:
    """
    Generate a natural conversational response without invoking the LLM
    for common greetings and questions. Returns None if the LLM should
    handle it.
    """
    low = normalize(text)

    # "How are you?" should get a real answer even when it follows a greeting.
    if "how are you" in low or "how r u" in low or "kaisa hai" in low or "kya haal" in low:
        return ("I'm doing well, thank you! Ready to help you cook. "
                "What dish are you thinking about today?")

    # Greetings
    if any(re.search(rf"\b{re.escape(g)}\b", low) for g in ("hi", "hello", "hey", "namaste", "namaskar")):
        return ("Hi! I'm AI Chef, your cooking companion. "
                "I can help you with Indian recipes like poha, biryani, "
                "dal, and more. What would you like to cook today, "
                "or do you have a question about a dish?")

    if any(re.search(rf"\b{re.escape(g)}\b", low) for g in ("thank you", "thanks", "dhanyavad", "shukriya")):
        return ("You're welcome! Happy cooking. What can I help you with next?")

    if any(re.search(rf"\b{re.escape(g)}\b", low) for g in ("bye", "goodbye", "good night", "goodnight", "alvida")):
        return ("Goodbye! Come back anytime when you're ready to cook.")

    if any(re.search(rf"\b{re.escape(g)}\b", low) for g in ("sorry", "excuse me", "maaf kijiye")):
        return ("No problem at all. How can I help you in the kitchen today?")

    # Capabilities
    if any(c in low for c in ("what can you do", "your capabilities", "what is ai chef")):
        return ("I'm AI Chef, your voice cooking companion. I can help you:\n"
                "\u2022 Find and validate Indian and other recipes\n"
                "\u2022 Guide you step by step while you cook\n"
                "\u2022 Set and manage cooking timers\n"
                "\u2022 Answer questions about dishes and ingredients\n"
                "\u2022 Suggest recipes from ingredients you have\n"
                "Just tell me what you'd like to cook, or ask me anything!")

    # Jokes
    if any(j in low for j in ("tell me a joke", "joke", "funny", "make me laugh")):
        return ("Why did the onion break up with the garlic? "
                "Because it couldn't handle the pressure cooker! "
                "\U0001F600 What shall we cook instead?")

    # "What is X" without a known dish name
    if any(p in low for p in ("what is", "what's", "tell me about", "explain")):
        return None  # Let the LLM handle it

    return None