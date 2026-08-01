"""Independent transcription of call audio.

The agent's own STT stream is not evidence: it is the same pipeline whose
conclusions we are trying to check, and it has already been shaped by the
agent's expectations. Re-transcribing the recording with a separate local model
gives a second opinion that no prompt can influence.

Runs on-device via MLX, so it works on venue wifi and costs nothing per call.
Model is pre-cached by scripts/warm_models.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.verify.receipts import Channel, Evidence

logger = logging.getLogger("verify.transcribe")

MODEL = "mlx-community/whisper-large-v3-turbo"


def transcribe(audio_path: str | Path, *, language: str | None = None) -> dict[str, Any]:
    """Transcribe a recording locally. Returns the raw whisper result."""
    import mlx_whisper

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"No recording at {path}")

    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=MODEL,
        language=language,
        word_timestamps=True,
    )
    logger.info("transcribed %s (%d segments)", path.name, len(result.get("segments", [])))
    return result


def save_transcript(audio_path: str | Path, out_dir: str | Path) -> Path:
    """Transcribe and persist alongside the recording, for the judge."""
    result = transcribe(audio_path)
    out = Path(out_dir) / f"{Path(audio_path).stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.get("text", ""))
    return out


def corroborate(
    claim: Any,
    audio_path: str | Path,
    *,
    must_contain: list[str],
) -> bool:
    """Check the recording actually contains what the agent says it heard.

    This catches the specific failure the rules punish: an agent that reports
    "they confirmed the booking" when the recording contains no such thing.

    A miss attaches CONTRADICTING evidence rather than staying silent, because
    here absence is meaningful - we have the full audio, so if the words are not
    in it, they were not said.
    """
    result = transcribe(audio_path)
    text = result.get("text", "").lower()

    missing = [t for t in must_contain if t.lower() not in text]
    transcript_path = Path(audio_path).with_suffix(".txt")

    if missing:
        claim.attach_evidence(
            Evidence(
                channel=Channel.INDEPENDENT_TRANSCRIPT,
                summary=(
                    f"Recording does not contain {missing}. The agent's account "
                    "is not supported by the audio."
                ),
                supports=False,
                artifact_path=str(transcript_path),
                raw={"missing": missing, "transcript": text[:2000]},
            )
        )
        logger.warning("claim %s CONTRADICTED: missing %s", claim.id, missing)
        return False

    claim.attach_evidence(
        Evidence(
            channel=Channel.INDEPENDENT_TRANSCRIPT,
            summary=f"Recording independently confirms: {', '.join(must_contain)}",
            artifact_path=str(transcript_path),
            raw={"transcript": text[:2000]},
        )
    )
    return True
