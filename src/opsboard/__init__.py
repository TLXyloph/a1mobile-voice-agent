"""The unified operations board — one screen, projected, during a live call.

    .venv/bin/python -m uvicorn src.opsboard.app:app --port 8130

Everything on it is either read off disk (`evidence/*.json`) or pushed into
`src.opsboard.registry.OPS` by the components doing the real work. The board
computes nothing it could get wrong and marks nothing done.
"""

from src.opsboard.registry import OPS, OpsRegistry

__all__ = ["OPS", "OpsRegistry"]
