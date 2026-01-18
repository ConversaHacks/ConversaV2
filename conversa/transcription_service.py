"""
TranscriptionService - Snowflake AI Transcription

Transcribes recorded audio using Snowflake AI_TRANSCRIBE with speaker diarization.
"""

import shutil
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Tuple, List, Callable

from .config import config
from test_snowflake import transcribe_audio as snowflake_transcribe


class TranscriptionService:
    """
    Transcription service using Snowflake AI_TRANSCRIBE.

    Takes recorded audio, uploads to Snowflake stage, and transcribes
    with speaker diarization. Returns segments with speaker labels.
    """

    def __init__(self, on_transcription: Optional[Callable[[str, str, List], None]] = None):
        """
        Args:
            on_transcription: Callback when transcription completes.
                              Receives (video_path, audio_path, segments)
                              where segments is list of (speaker, start_s, end_s, text)
        """
        self.on_transcription = on_transcription

        # Paths
        self.pending_dir = config.paths.pending_dir
        self.processed_dir = config.paths.approved_dir  # Reuse approved as processed

        # Processing queue
        self._queue: Queue[Tuple[str, str]] = Queue()

        # State
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start transcription service."""
        if self._is_running:
            return

        self._is_running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        print("[TranscriptionService] Started")

    def stop(self):
        """Stop transcription service."""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        print("[TranscriptionService] Stopped")

    def queue_for_transcription(self, video_path: str, audio_path: str):
        """Add a recording to the transcription queue."""
        self._queue.put((video_path, audio_path))
        print(f"[TranscriptionService] Queued: {Path(audio_path).name}")

    def _process_loop(self):
        """Main processing loop (runs in separate thread)."""
        while self._is_running:
            try:
                video_path, audio_path = self._queue.get(timeout=1.0)
                self._transcribe_recording(video_path, audio_path)
                self._queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"[TranscriptionService] Error in processing loop: {e}")

    def _transcribe_recording(self, video_path: str, audio_path: str):
        """Transcribe a single recording using Snowflake."""
        video_path = Path(video_path)
        audio_path = Path(audio_path)

        print(f"[TranscriptionService] Transcribing: {audio_path.name}")

        # Check if audio file exists
        if not audio_path.exists():
            print(f"[TranscriptionService] Audio file not found: {audio_path}")
            return

        # Transcribe with Snowflake (raw audio - Snowflake handles noise well)
        segments = snowflake_transcribe(str(audio_path), verbose=True)

        if segments is None:
            print(f"[TranscriptionService] Transcription failed: {audio_path.name}")
            return

        # Print transcript
        print()
        print("=" * 60)
        print("TRANSCRIPT (with speaker diarization):")
        print("=" * 60)

        if not segments:
            print("No speech detected")
        else:
            for speaker, start_s, end_s, text in segments:
                print(f"[{start_s:6.2f} - {end_s:6.2f}] {speaker}: {text}")

        print("=" * 60)
        print()

        # Call callback if provided (for further actions)
        if self.on_transcription and segments:
            self.on_transcription(str(video_path), str(audio_path), segments)

        # Move files to processed folder
        self._move_to_processed(video_path, audio_path)

    def _move_to_processed(self, video_path: Path, audio_path: Path):
        """Move recording to processed folder."""
        try:
            if video_path.exists():
                shutil.move(str(video_path), str(self.processed_dir / video_path.name))

            if audio_path.exists():
                shutil.move(str(audio_path), str(self.processed_dir / audio_path.name))

        except Exception as e:
            print(f"[TranscriptionService] Error moving files: {e}")

    def process_pending(self):
        """Process all recordings in pending folder."""
        print("[TranscriptionService] Processing pending recordings...")

        for video_path in self.pending_dir.glob("*.mp4"):
            audio_path = video_path.with_suffix(".wav")
            self.queue_for_transcription(str(video_path), str(audio_path))

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._is_running
