"""
AudioCapture - Microphone Handler

Uses PyAudio for cross-platform audio capture with ring buffers
for VAD analysis and recording.
"""

import threading
import struct
import wave
import io
from collections import deque
from typing import Optional, Callable
import numpy as np

try:
    import pyaudio
except ImportError:
    raise ImportError("pyaudio is required. Install with: pip install pyaudio")

from .config import config


class AudioCapture:
    """
    Handles microphone input with ring buffers for VAD and recording.

    Maintains two buffers:
    - Ring buffer for VAD analysis (~23 seconds)
    - Recording buffer that grows during active recording
    """

    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        chunk_size: int = None,
        pre_buffer_seconds: float = None,
    ):
        self.sample_rate = sample_rate or config.vad.sample_rate
        self.channels = channels or config.vad.channels
        self.chunk_size = chunk_size or config.vad.chunk_size
        self.pre_buffer_seconds = pre_buffer_seconds or config.recording.audio_pre_buffer

        # PyAudio instance
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

        # Calculate buffer sizes
        self._chunks_per_second = self.sample_rate / self.chunk_size
        self._vad_buffer_size = int(23 * self._chunks_per_second)  # ~23 seconds for VAD
        self._pre_buffer_chunks = int(self.pre_buffer_seconds * self._chunks_per_second)

        # Buffers
        self._vad_buffer: deque = deque(maxlen=self._vad_buffer_size)
        self._pre_buffer: deque = deque(maxlen=self._pre_buffer_chunks)
        self._recording_buffer: list = []

        # State
        self._is_running = False
        self._is_recording = False
        self._lock = threading.Lock()

        # Callback for new audio data (used by VAD)
        self._audio_callback: Optional[Callable[[np.ndarray], None]] = None

    def start(self):
        """Start audio capture from microphone."""
        if self._is_running:
            return

        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._stream_callback,
        )

        self._is_running = True
        self._stream.start_stream()
        print(f"[AudioCapture] Started (sample_rate={self.sample_rate}, chunk_size={self.chunk_size})")

    def stop(self):
        """Stop audio capture."""
        self._is_running = False

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

        print("[AudioCapture] Stopped")

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback - called for each audio chunk."""
        if not self._is_running:
            return (None, pyaudio.paComplete)

        # Convert to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        # Normalize to [-1, 1]
        audio_data = audio_data / 32768.0

        with self._lock:
            # Always add to VAD buffer
            self._vad_buffer.append(audio_data.copy())

            # Always add to pre-buffer (rolling)
            self._pre_buffer.append(in_data)

            # If recording, add to recording buffer
            if self._is_recording:
                self._recording_buffer.append(in_data)

        # Call audio callback if set (for VAD analysis)
        if self._audio_callback:
            self._audio_callback(audio_data)

        return (None, pyaudio.paContinue)

    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function for audio data (used by VAD)."""
        self._audio_callback = callback

    def start_recording(self):
        """Start recording audio, including pre-buffer."""
        with self._lock:
            if self._is_recording:
                return

            self._is_recording = True
            # Copy pre-buffer to recording buffer
            self._recording_buffer = list(self._pre_buffer)
            print(f"[AudioCapture] Recording started (pre-buffer: {len(self._recording_buffer)} chunks)")

    def stop_recording(self) -> bytes:
        """Stop recording and return audio data as bytes."""
        with self._lock:
            if not self._is_recording:
                return b""

            self._is_recording = False
            audio_data = b"".join(self._recording_buffer)
            self._recording_buffer = []
            print(f"[AudioCapture] Recording stopped ({len(audio_data)} bytes)")
            return audio_data

    def get_latest_chunk(self) -> Optional[np.ndarray]:
        """Get the most recent audio chunk for VAD analysis."""
        with self._lock:
            if self._vad_buffer:
                return self._vad_buffer[-1].copy()
            return None

    def create_wav_bytes(self, audio_data: bytes) -> bytes:
        """Convert raw audio data to WAV format."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)
        return wav_buffer.getvalue()

    def save_wav(self, audio_data: bytes, filepath: str) -> bool:
        """Save raw audio data to a WAV file."""
        try:
            with wave.open(filepath, "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            return True
        except Exception as e:
            print(f"[AudioCapture] Error saving WAV: {e}")
            return False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording
