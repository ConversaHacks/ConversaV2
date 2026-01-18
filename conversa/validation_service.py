"""
ValidationService - AI Post-Processing

Validates recorded audio using Gemini API to filter out false positives.
Moves recordings to approved/rejected folders based on validation results.
"""

import shutil
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Tuple

from .config import config
from .gemini_client import GeminiClient


class ValidationService:
    """
    AI-powered validation service for recorded audio.

    Takes recorded audio and sends to Gemini API to verify human voice.
    Moves recordings to approved/rejected folders based on results.
    Auto-approves on API errors (fail-safe).
    """

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini_client = gemini_client or GeminiClient()

        # Paths
        self.pending_dir = config.paths.pending_dir
        self.approved_dir = config.paths.approved_dir
        self.rejected_dir = config.paths.rejected_dir

        # Processing queue
        self._queue: Queue[Tuple[str, str]] = Queue()

        # State
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start validation service."""
        if self._is_running:
            return

        self._is_running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        print("[ValidationService] Started")

    def stop(self):
        """Stop validation service."""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        print("[ValidationService] Stopped")

    def queue_for_validation(self, video_path: str, audio_path: str):
        """Add a recording to the validation queue."""
        self._queue.put((video_path, audio_path))
        print(f"[ValidationService] Queued for validation: {Path(video_path).name}")

    def _process_loop(self):
        """Main processing loop (runs in separate thread)."""
        while self._is_running:
            try:
                # Wait for item with timeout
                video_path, audio_path = self._queue.get(timeout=1.0)
                self._validate_recording(video_path, audio_path)
                self._queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"[ValidationService] Error in processing loop: {e}")

    def _validate_recording(self, video_path: str, audio_path: str):
        """Validate a single recording."""
        video_path = Path(video_path)
        audio_path = Path(audio_path)

        print(f"[ValidationService] Validating: {video_path.name}")

        # Check if files exist
        if not video_path.exists():
            print(f"[ValidationService] Video file not found: {video_path}")
            return

        # Read audio file
        audio_data = None
        if audio_path.exists():
            try:
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
            except Exception as e:
                print(f"[ValidationService] Error reading audio: {e}")

        # Validate with Gemini
        result = None
        if audio_data:
            result = self.gemini_client.validate_audio(audio_data)

        # Handle result
        if result is None:
            # API error - auto-approve (fail-safe)
            print(f"[ValidationService] API error, auto-approving: {video_path.name}")
            self._move_to_approved(video_path, audio_path)
        elif result:
            # Voice detected - approve
            print(f"[ValidationService] Voice detected, approving: {video_path.name}")
            self._move_to_approved(video_path, audio_path)
        else:
            # No voice - reject
            print(f"[ValidationService] No voice detected, rejecting: {video_path.name}")
            self._move_to_rejected(video_path, audio_path)

    def _move_to_approved(self, video_path: Path, audio_path: Path):
        """Move recording to approved folder."""
        self._move_files(video_path, audio_path, self.approved_dir)

    def _move_to_rejected(self, video_path: Path, audio_path: Path):
        """Move recording to rejected folder."""
        self._move_files(video_path, audio_path, self.rejected_dir)

    def _move_files(self, video_path: Path, audio_path: Path, target_dir: Path):
        """Move video and audio files to target directory."""
        try:
            if video_path.exists():
                shutil.move(str(video_path), str(target_dir / video_path.name))

            if audio_path.exists():
                shutil.move(str(audio_path), str(target_dir / audio_path.name))

        except Exception as e:
            print(f"[ValidationService] Error moving files: {e}")

    def process_pending(self):
        """Process all recordings in pending folder."""
        print("[ValidationService] Processing pending recordings...")

        for video_path in self.pending_dir.glob("*.mp4"):
            # Find matching audio file
            audio_path = video_path.with_suffix(".wav")
            self.queue_for_validation(str(video_path), str(audio_path))

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._is_running
