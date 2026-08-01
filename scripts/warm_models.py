"""Pre-download every model that would otherwise stall you at 9am.

Run once, offline-safe afterwards. Each step is independent: a failure in one
model must not prevent the others from caching.
"""

from __future__ import annotations

import sys
import traceback


def step(label: str, fn) -> bool:
    print(f"\n=== {label} ===", flush=True)
    try:
        fn()
    except Exception:  # noqa: BLE001 - we want every step attempted
        traceback.print_exc()
        print(f"FAILED: {label}", flush=True)
        return False
    print(f"OK: {label}", flush=True)
    return True


def silero_vad() -> None:
    from livekit.plugins import silero

    silero.VAD.load()


def turn_detector() -> None:
    # Local end-of-utterance model. Cloud `inference.TurnDetector` needs a
    # LiveKit Cloud key; this local copy keeps us running without one.
    #
    # Note: instantiating the model requires a live job context, so we go
    # through the plugin's download hook instead of constructing it.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from livekit.plugins.turn_detector import EOUPlugin, multilingual

        EOUPlugin(multilingual._EUORunnerMultilingual).download_files()


def whisper() -> None:
    # Independent transcription of call recordings. We never trust the agent's
    # own transcript when producing a receipt - this is the second opinion.
    import mlx_whisper
    import numpy as np

    mlx_whisper.transcribe(
        np.zeros(16000, dtype=np.float32),
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    )


if __name__ == "__main__":
    results = {
        "silero-vad": step("Silero VAD", silero_vad),
        "turn-detector": step("Turn detector (multilingual)", turn_detector),
        "whisper-turbo": step("Whisper large-v3-turbo (MLX)", whisper),
    }
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(results.values()) else 1)
