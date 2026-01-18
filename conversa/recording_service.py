"""
RecordingService - Video + Audio Recording

Maintains a rolling buffer of video frames and records synchronized
video + audio when triggered by VAD commands.
"""

import threading
import time
import datetime
from collections import deque
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from .config import config
from .audio_capture import AudioCapture
from .action_service import VADCommand


class RecordingService:
    """
    Video and audio recording service.

    Maintains a 15-second rolling buffer of video frames.
    When recording starts, includes the pre-buffer.
    Exports synchronized video + audio files.
    """

    def __init__(self, audio_capture: AudioCapture):
        self.audio_capture = audio_capture

        # Video settings
        self.fps = config.recording.fps
        self.width = config.recording.video_width
        self.height = config.recording.video_height
        self.pre_buffer_duration = config.recording.pre_buffer_duration

        # Calculate buffer size
        self._pre_buffer_frames = int(self.pre_buffer_duration * self.fps)

        # Buffers
        self._frame_buffer: deque = deque(maxlen=self._pre_buffer_frames)
        self._recording_frames: list = []

        # State
        self._is_running = False
        self._is_recording = False
        self._capture: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

        # Threads
        self._capture_thread: Optional[threading.Thread] = None

        # Output paths
        self.output_dir = config.paths.pending_dir

        # Recording start time
        self._recording_start_time: Optional[float] = None

    def start(self):
        """Start video capture."""
        if self._is_running:
            return

        # Open webcam
        self._capture = cv2.VideoCapture(0)
        if not self._capture.isOpened():
            raise RuntimeError("Failed to open webcam")

        # Set resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)

        # Get actual settings
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

        self._is_running = True

        # Start capture thread
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        print(f"[RecordingService] Started (resolution={actual_width}x{actual_height}, fps={actual_fps})")

    def stop(self):
        """Stop video capture."""
        self._is_running = False

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._capture:
            self._capture.release()
            self._capture = None

        print("[RecordingService] Stopped")

    def _capture_loop(self):
        """Main video capture loop (runs in separate thread)."""
        frame_interval = 1.0 / self.fps
        last_frame_time = time.time()

        while self._is_running:
            current_time = time.time()

            # Maintain frame rate
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue

            last_frame_time = current_time

            # Capture frame
            ret, frame = self._capture.read()
            if not ret:
                continue

            with self._lock:
                # Always add to rolling buffer
                self._frame_buffer.append((current_time, frame.copy()))

                # If recording, add to recording buffer
                if self._is_recording:
                    self._recording_frames.append((current_time, frame.copy()))

    def handle_command(self, command: VADCommand):
        """Handle VAD command (START/STOP)."""
        if command == VADCommand.START:
            self.start_recording()
        elif command == VADCommand.STOP:
            self.stop_recording()

    def start_recording(self):
        """Start recording video and audio."""
        with self._lock:
            if self._is_recording:
                return

            self._is_recording = True
            self._recording_start_time = time.time()

            # Copy pre-buffer frames to recording buffer
            self._recording_frames = list(self._frame_buffer)

            # Start audio recording
            self.audio_capture.start_recording()

            print(f"[RecordingService] Recording started (pre-buffer: {len(self._recording_frames)} frames)")

    def stop_recording(self) -> Optional[Tuple[str, str]]:
        """Stop recording and save files."""
        with self._lock:
            if not self._is_recording:
                return None

            self._is_recording = False

            # Get audio data
            audio_data = self.audio_capture.stop_recording()

            # Get video frames
            frames = self._recording_frames.copy()
            self._recording_frames = []

            if not frames:
                print("[RecordingService] No frames to save")
                return None

        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.output_dir / f"recording_{timestamp}.mp4"
        audio_path = self.output_dir / f"recording_{timestamp}.wav"

        # Save files
        success = self._save_recording(frames, audio_data, video_path, audio_path)

        if success:
            print(f"[RecordingService] Saved: {video_path.name}")
            return str(video_path), str(audio_path)
        return None

    def _save_recording(
        self,
        frames: list,
        audio_data: bytes,
        video_path: Path,
        audio_path: Path
    ) -> bool:
        """Save video and audio files."""
        try:
            # Save video
            if frames:
                # Get frame dimensions from first frame
                _, first_frame = frames[0]
                height, width = first_frame.shape[:2]

                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    self.fps,
                    (width, height)
                )

                for _, frame in frames:
                    writer.write(frame)

                writer.release()

            # Save audio
            if audio_data:
                self.audio_capture.save_wav(audio_data, str(audio_path))

            return True

        except Exception as e:
            print(f"[RecordingService] Error saving recording: {e}")
            return False

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame for preview."""
        with self._lock:
            if self._frame_buffer:
                _, frame = self._frame_buffer[-1]
                return frame.copy()
            return None

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self._lock:
            return self._is_recording

    @property
    def recording_duration(self) -> float:
        """Get current recording duration in seconds."""
        with self._lock:
            if self._is_recording and self._recording_start_time:
                return time.time() - self._recording_start_time
            return 0.0
