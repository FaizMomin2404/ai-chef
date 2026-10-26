# Hugging Face Docker Space image (also usable on Render-like hosts).
# Hugging Face runs Docker containers as UID 1000 and expects the app on 7860.
FROM python:3.12-slim

RUN useradd -m -u 1000 user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=7860 \
    AI_PROVIDER=groq \
    GROQ_MODEL=openai/gpt-oss-20b \
    MULTI_USER=1 \
    SESSION_PERSISTENCE=0 \
    STT_MODE=browser \
    TTS_MODE=browser \
    TTS_ENABLED=1

WORKDIR /home/user/app
RUN chown user:user /home/user/app

COPY --chown=user requirements-render.txt ./
USER user
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-render.txt

COPY --chown=user . .

EXPOSE 7860
# Cheap liveness probe. It deliberately does NOT call the hosted AI provider,
# so monitoring cannot consume the free daily request quota.
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4)"

# GROQ_API_KEY must be supplied as a Space Secret (or host environment variable).
# There is deliberately no key, token, or model weight baked into this image.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers"]
