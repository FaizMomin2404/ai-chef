---
title: AI Chef
emoji: 🍳
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Hands-free AI cooking teacher with validated recipes
---

# AI Chef — Voice Cooking Companion

AI Chef is a hands-free cooking teacher. The default local mode runs entirely on
your machine: Ollama for recipes, faster-whisper for speech recognition, and
Piper (with a Windows SAPI fallback) for speech. An optional public deployment
keeps local Ollama for development and uses a free hosted model for the website,
so the site can stay online while your laptop is off.

No paid API keys are required for the supported configurations. See
[`DEPLOYMENT.md`](DEPLOYMENT.md) for the zero-cost public architecture and its
honest trade-offs.

## Quick start

1. Install the Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Install and start Ollama, then pull the recipe model:

   ```powershell
   ollama serve
   ollama pull phi3:3.8b
   ```

3. Start the app:

   ```powershell
   python main.py
   ```

   Or double-click `run.bat`.

4. Open <http://127.0.0.1:8000>. The first start takes roughly 15–30 seconds
   while Piper and Whisper load.

## What the screen tells you

The top-right status is deliberately split so the indicators cannot contradict
each other:

| Indicator | Meaning |
|---|---|
| `Connected` / `Reconnecting…` | The browser is connected to the FastAPI backend. |
| `AI ready` / `AI offline` | The configured language model (local Ollama or hosted provider) answered and the model is present. Built-in recipes still work when this is offline. |
| `Voice ready` / `Voice off` | Piper/SAPI can play speech, or the browser will speak (`Voice ready (browser)`). Replies are always shown on screen too. |
| Mic pill (`Hands-Free Off`, `Hands-Free On — Listening`, `Hands-Free On — Processing`, `Hands-Free On — AI Speaking`, `Hands-Free On — Reconnecting`, `Microphone Permission Required`, `Microphone Unavailable`) | The real microphone state. It never says “listening” unless the browser recognition session actually started, and transient restarts show **Reconnecting**. |

The recipe panel shows the dish, servings, progress, a large **Now** card with
the current action, ingredient check-marks, the full ordered steps, safety
notes, and live timers.

## Cooking flow

1. Ask for a dish (“I want to make chicken biryani for four people”) or list
   ingredients (“I have potatoes, onions and tomatoes”).
2. Ambiguous dishes ask first: `biryani` → chicken or vegetable; `pasta` → red
   or white sauce; `curry`, `paratha`, and `rice` are clarified the same way.
3. If no serving count was given, the chef asks for one.
4. The recipe is **validated before display**. The dish family, required
   ingredients (for example rice for biryani), dietary variant (veg/chicken/
   paneer), sauce type, and quantities are checked. One repair attempt is made
   if the model returns something wrong.
5. If the AI is unavailable or still wrong, a dish-specific built-in recipe is
   used only when it matches the request. Otherwise the app says it cannot
   build a reliable recipe; it never silently substitutes another dish.
6. Check the ingredient list, then press **▶ Start cooking** (or say “ready”).
7. Follow the **Now** card and say **next** when the step is done. Instructions
   do not mark a task complete by themselves.

`session_state.json` keeps an unfinished session for 12 hours so a server
restart does not lose your place. Delete that file to start fresh. **New dish**
also clears it.

## Universal recipe engine

Any dish name is accepted, not just the built-in ones. Requests such as *idli*,
*chicken shawarma*, *falafel*, *pizza*, or *paneer biryani* go through the same
pipeline:

1. Intent routing separates conversation from cooking follow-ups and recipe
   requests (English and Hinglish).
2. The dish and serving count are extracted from the sentence — never the whole
   sentence — and serving quantities are scaled.
3. Local RAG documents are retrieved only when they genuinely mention the dish;
   unrelated neighbours are discarded.
4. The configured language model generates the missing recipe, grounded in that
   context when present.
5. The result is validated for dish match, defining ingredients, cooking
   technique, step order, quantities, and time fields. One bounded repair
   attempt is made; otherwise the app returns an honest error or a matching
   built-in recipe. It never silently substitutes a different dish.

Recipes include preparation, active cooking, waiting (fermentation,
marination, soaking), and realistic total elapsed time, plus per-step timing
and heat where the model supplies them. Sources are labelled
`ai_generated`/`validated` or `builtin`; AI output is never described as tested.

## Voice interaction

### Hands-free (browser)

Click **🎧 Hands-free** for continuous recognition. It uses the browser's Web
Speech API, so Chrome/Edge needs internet and microphone permission.

Hands-Free stays on until you click it off. When the browser ends a recognition
session (Chrome does this after silence or a final result), the app reconnects
on its own with a bounded exponential backoff (0.8s → 6s). It pauses only while
the chef is speaking and resumes when speech finishes. Transient `no-speech`,
`aborted`, and `network` errors never turn the mode off; only denied permission,
a missing microphone, or an unsupported browser do.

The mic pill shows the real state: **Hands-Free On — Listening / Processing /
AI Speaking / Reconnecting**, **Hands-Free Off**, **Microphone Permission
Required**, or **Microphone Unavailable**.

### Fully offline push-to-talk

Click **🎤 Talk 6s (offline)** to record six seconds and transcribe locally with
Whisper. This path works without internet. Both paths send the same text
commands to the same cooking engine.

The **Mic** row lets you choose an input device and click **Test level** while
speaking. A silent saved device is ignored at startup, so the app falls back to
the Windows default microphone instead of listening to silence.

### Speech controls

- **⏹ Voice** stops the chef's speech immediately (also available as
  `POST /api/voice/stop`).
- **Pause** / **Resume** pauses the session (`pause`, `resume`, or
  “wait two minutes”).
- **Stop** ends the session and clears timers.
- **New dish** resets the recipe and keeps the microphone state honest.

If speech output is unavailable, the UI says so and keeps showing replies as
text. It never claims audio played when it did not.

## Commands (English / Hinglish)

**Navigation:** `next`, `next step`, `nxtf` / `nxt` (common typos),
`previous`, `go back`, `repeat`, `repeat the previous step`,
`what should I do next?`, `aage kya`, `let's finish this dish` (continues from
the real current step and lists the remaining steps in order).

**Task updates:** “the onions are chopped”, “I put the water on the stove”,
“the water is not hot yet”, “I added too much salt”.

**Ingredients:** “I don't have tomatoes”, “skip the paneer”, “I have paneer
instead”.

**Timers and breaks:** “set a timer for 10 minutes”, “wait two minutes”,
“cancel timer”, “resume”.

**Session:** `ready` / `start`, `pause`, `resume`, `stop`,
`I'm done`, `new dish`.

All commands can be typed in the message box if the microphone is unavailable.

## Safety and cooking honesty

- Safety notes cover hot oil, boiling water, pressure cookers, raw meat, and
  unattended heating. The chef refuses or redirects unsafe shortcuts.
- Timers and elapsed time are **not** proof that food is cooked. The chef asks
  you to check doneness, especially for chicken, seafood, and pressure-cooked
  dishes.
- The app cannot see your food and never claims to have looked at it.
- Raw-meat handling and cross-contamination warnings are included where
  relevant.

## Public deployment (laptop can be off)

The public configuration is selected only through server-side environment
variables; local Ollama is not modified. The repository is ready to push as a
Hugging Face **Docker Space** (`sdk: docker`, `app_port: 7860`), but Hugging Face
currently requires a paid PRO plan to create Docker Spaces. Static Spaces are
free but cannot run this FastAPI/WebSocket backend. See
[`DEPLOYMENT.md`](DEPLOYMENT.md) for the verified free-tier decision, exact
Space setup, and the zero-cost Render fallback.

| Layer | Local default | HF Docker Space (when eligible) |
|---|---|---|
| App hosting | `127.0.0.1:8000` | Docker on port `7860` (HTTPS) |
| Language model | Ollama `phi3:3.8b` | Groq Free `openai/gpt-oss-20b` |
| Speech input | Browser Web Speech or local Whisper | Browser Web Speech |
| Speech output | Piper/SAPI | Browser `speechSynthesis` |
| Sessions | One persisted local session | Isolated per WebSocket; memory only |

```powershell
# Local, unchanged
python main.py

# Hosted provider locally (optional)
$env:AI_PROVIDER="hosted"
$env:HOSTED_LLM_BASE_URL="https://api.groq.com/openai/v1"
$env:HOSTED_LLM_MODEL="openai/gpt-oss-20b"
$env:HOSTED_LLM_API_KEY="<your-key>"
python main.py
```

The `Dockerfile` uses UID 1000, listens on `0.0.0.0:7860`, installs only
`requirements-render.txt`, and reads `GROQ_API_KEY` from the Space's **Secrets**
(never from the image or browser). `render.yaml` remains an optional free
fallback. The provider adapter (`llm_provider.py`) supports Ollama, Groq, and
any OpenAI-compatible endpoint. It has bounded retries/timeouts, validates
responses, redacts secrets from errors, and never silently falls back to a paid
provider. With `STT_MODE=browser`/`TTS_MODE=browser`, no torch, Whisper, or
Piper model is loaded on the server.

The website is **not** using your laptop's Ollama model. Any live deployment is
also subject to the chosen host's cold starts and Groq's shared free daily
limits; details, exact setup steps, and rollback are in
[`DEPLOYMENT.md`](DEPLOYMENT.md).

## Automated tests

The test suite mocks Ollama, Piper, Whisper, and timers, so it does not need
network access, a microphone, or speakers:

```powershell
python -m pytest -q
```

The **133** tests cover recipe validation (including the biryani-vs-pasta bug,
veg-vs-chicken mismatches, unclear model quantities, idli, pancakes for three,
and complete chicken shawarma), serving scaling, session task confirmation,
navigation, repeat/previous/next, `nxtf`-style typos, finish-from-current-step,
full ingredients+method requests, timers, AI failure, TTS/browser-voice fallback,
barge-in, microphone errors, persistence, multi-user isolation, deployment
configuration, and non-contradictory UI status markup.

They also cover the universal recipe engine: natural-language dish extraction,
unseen-dish generation, Hinglish requests, general conversation staying
conversational, certificate questions not advancing a step, ingredient/step/time
validation, plural dish names, bounded JSON repair, honest provider failures,
RAG relevance filtering, and one-shot full-pipeline requests. Provider tests
exercise Ollama defaults, hosted configuration, missing/rejected tokens,
timeouts, health checks, and the guarantee that no secret reaches `/health`.

`tests/handsfree_lifecycle.test.js` runs the real Hands-Free code extracted from
`static/index.html` in Node with a mocked `SpeechRecognition` and fake timers.
It verifies auto-restart, transient-error recovery, backoff bounds, terminal
permission errors, no duplicate recognition instances, no duplicate utterances,
and resuming after AI speech. It is executed automatically by pytest when Node
is installed (`node tests/handsfree_lifecycle.test.js` runs it directly).

## Project structure

```
D:\GEN\
├── main.py              # Starts FastAPI/uvicorn on HOST:PORT (local default 127.0.0.1:8000)
├── voice_pipeline.py    # WebSocket router, voice I/O, recipe/session orchestration
├── llm_provider.py      # Ollama/Groq/OpenAI-compatible adapter with safe errors
├── app_config.py        # Deployment, speech, CORS, and multi-user environment config
├── recipe_engine.py     # Prompts, JSON parsing, semantic validation, fallbacks
├── intent_router.py     # Conversation vs recipe vs cooking-command routing
├── rag_system.py        # Local knowledge retrieval (Chroma/FAISS/in-memory)
├── index_knowledge.py   # Builds the knowledge index
├── knowledge/           # Curated recipe/cooking documents (JSONL)
├── state_manager.py     # Cooking session, tasks, timers, conversation, persistence
├── timer_manager.py     # Parallel countdown timers
├── static/index.html    # UI, orb state machine, Hands-Free lifecycle, browser voice
├── tests/               # pytest suite + Node Hands-Free lifecycle test
├── requirements.txt     # Local (Ollama + local speech models)
├── requirements-render.txt # Lightweight public deployment
├── render.yaml          # Render Free Blueprint
├── Dockerfile           # HF Docker Space image (UID 1000, port 7860)
├── DEPLOYMENT.md        # HF/Render deployment, free-tier limits, rollback
├── .env.example         # Placeholder-only environment template
├── run.bat
├── install.bat
└── README.md
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `AI offline` | Local: start Ollama (`ollama serve`) and pull `phi3:3.8b`. Hosted: check the Space/Render logs and the `GROQ_API_KEY` Secret; the badge refreshes automatically. |
| Hosted site is slow on first visit | Free hosting can cold-start after idling. Wait for the wake-up; this is a free-tier limitation. |
| Hosted quota error / HTTP 429 | The Groq free daily/minute limit is shared by visitors. Wait or rotate the key; the app never charges a paid provider. |
| Hosted microphone button is disabled | The public deployment uses browser speech recognition; allow microphone permission and use Chrome/Edge. Typing always works. |
| A recipe request says it cannot build a reliable recipe | The AI failed and there is no safe built-in recipe for that exact dish. Try again, choose a supported dish, or make the AI status ready first. |
| Microphone level is 0 | Windows Settings → System → Sound → Input: choose the right mic, unmute it, and raise the volume. Use **Test level** and pick a device whose bar moves. |
| Saved microphone is silent | It is probed and ignored automatically; choose a working device in the **Mic** row. |
| Hands-free says “Reconnecting” | Browser speech recognition needs internet. The app retries automatically with backoff; use **🎤 Talk 6s (offline)** if it stays offline. |
| Hands-free says “Microphone Permission Required” | Allow microphone access for this site in the browser, then click **🎧 Hands-free** again. Denied permission is never auto-retried. |
| No sound | Check speakers/headphones and the `Voice ready` indicator. Without Piper, the Windows SAPI fallback is used if `pyttsx3` is installed. |
| First reply is slow | CPU-only `phi3:3.8b` inference takes time. Built-in recipes are instant once the AI is unavailable. |

## Limitations

- CPU-only inference is slower than cloud models.
- Browser hands-free mode depends on Chrome/Edge's online speech service;
  offline push-to-talk is the reliable local path.
- Whisper can mishear noisy speech; the transcript is always shown so you can
  correct it.
- The app cannot see, smell, or taste the food; doneness checks remain your
  responsibility.
- The safety layer is rule-based, not a substitute for adult supervision around
  hot oil, open flames, or pressure cookers.
- The public deployment uses a free hosted model, not this laptop's Ollama. It
  is subject to Render cold starts and Groq's shared free quota; see
  [`DEPLOYMENT.md`](DEPLOYMENT.md).

Happy cooking!
