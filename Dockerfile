FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY web_app.py web_game.py web_agents.py game.py heuristic_agent.py expectimax_agent.py ./
COPY web/ ./web/
COPY results/canonical/expectimax_depth_2_seeds/ ./results/canonical/expectimax_depth_2_seeds/
COPY results/canonical/heuristic_strategy_v1_seeds/ ./results/canonical/heuristic_strategy_v1_seeds/

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
