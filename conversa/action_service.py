"""
ActionService - Voice Activity Detection (VAD)

Captures audio from microphone or WebRTC stream continuously and analyzes
energy levels to detect speech. Sends START/STOP commands to trigger recording.
"""

import threading
import time
from enum import Enum
from typing import Callable, Optional, TYPE_CHECKING
import numpy as np

from .config import config
from .audio_capture import AudioCapture

if TYPE_CHECKING:
    from .webrtc_capture import WebRTCCapture


class VADCommand(Enum):
    """Commands sent by VAD when state changes."""
    START = "START"
    STOP = "STOP"


class ActionService:
    """
    Voice Activity Detection service.

    Analyzes audio energy levels and zero-crossing rate to detect speech.
    Uses adaptive threshold that adjusts to background noise.
    Supports both local microphone and WebRTC audio sources.
    """

    def __init__(
        self,
        audio_capture: AudioCapture,
        on_command: Optional[Callable[[VADCommand], None]] = None,
        webrtc_capture: Optional["WebRTCCapture"] = None,
    ):
        self.audio_capture = audio_capture
        self.on_command = on_command
        self._webrtc_capture = webrtc_capture
        self._use_webrtc = webrtc_capture is not None

        # VAD parameters from config
        self.zcr_max = config.vad.zero_cross_rate_max

        # Frame counters - adjust for WebRTC sample rate if needed
        if self._use_webrtc:
            # WebRTC typically uses 48kHz
            self._sample_rate = 48000
            self._chunk_size = 960  # 20ms at 48kHz
            # WebRTC audio - higher threshold to ignore static noise
            self.zcr_min = 0.01
            self.energy_threshold = 0.015  # Higher to filter out static
        else:
            self._sample_rate = config.vad.sample_rate
            self._chunk_size = config.vad.chunk_size
            self.zcr_min = config.vad.zero_cross_rate_min
            self.energy_threshold = config.vad.energy_threshold

        self._chunks_per_second = self._sample_rate / self._chunk_size
        self.speech_confirm_frames = int(config.vad.speech_confirm_time * self._chunks_per_second)
        self.silence_threshold_frames = int(config.vad.silence_threshold_time * self._chunks_per_second)

        # State
        self._is_running = False
        self._is_speech_active = False
        self._speech_frames = 0
        self._silence_frames = 0

        # Adaptive threshold
        self._noise_level = self.energy_threshold
        self._noise_alpha = 0.995  # Slow adaptation for background noise

        # WebRTC audio buffer for VAD analysis
        self._webrtc_audio_buffer: list = []
        self._webrtc_vad_recording = False

        # Thread
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Start VAD analysis."""
        if self._is_running:
            return

        self._is_running = True

        if self._use_webrtc:
            # Start WebRTC audio capture for VAD
            self._start_webrtc_vad()
            print(f"[ActionService] Started with WebRTC audio (speech_confirm={self.speech_confirm_frames} frames, "
                  f"silence_threshold={self.silence_threshold_frames} frames)")
        else:
            # Set local audio callback
            self.audio_capture.set_audio_callback(self._analyze_audio)
            print(f"[ActionService] Started with local mic (speech_confirm={self.speech_confirm_frames} frames, "
                  f"silence_threshold={self.silence_threshold_frames} frames)")

        # Start analysis thread
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop VAD analysis."""
        self._is_running = False
        self._webrtc_vad_recording = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        print("[ActionService] Stopped")

    def _start_webrtc_vad(self):
        """Start capturing WebRTC audio for VAD analysis."""
        self._webrtc_vad_recording = True
        self._webrtc_audio_buffer = []

    def _run(self):
        """Main VAD loop (runs in separate thread)."""
        if self._use_webrtc:
            self._run_webrtc_vad()
        else:
            # For local mic, just sleep as audio comes via callback
            while self._is_running:
                time.sleep(0.01)

    def _run_webrtc_vad(self):
        """Poll WebRTC audio and analyze for VAD."""
        while self._is_running and self._webrtc_capture:
            # Get latest audio chunk for VAD analysis
            audio_bytes = self._webrtc_capture.get_vad_audio_chunk()

            if audio_bytes:
                # Convert bytes to numpy array
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

                # Normalize to float [-1, 1] for analysis
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Analyze the audio chunk
                self._analyze_audio(audio_float)

            time.sleep(0.01)  # ~10ms polling interval

    def _analyze_audio(self, audio_data: np.ndarray):
        """Analyze audio chunk for voice activity."""
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(audio_data ** 2))

        # Calculate zero-crossing rate
        zcr = self._calculate_zcr(audio_data)

        # Update adaptive noise threshold
        if not self._is_speech_active:
            self._noise_level = (self._noise_alpha * self._noise_level +
                                (1 - self._noise_alpha) * energy)

        # Adaptive threshold: noise level + margin
        adaptive_threshold = max(self.energy_threshold, self._noise_level * 2)

        # Check if speech is detected
        is_speech = (
            energy > adaptive_threshold and
            self.zcr_min <= zcr <= self.zcr_max
        )

        with self._lock:
            if is_speech:
                self._speech_frames += 1
                self._silence_frames = 0

                # Confirm speech after threshold
                if not self._is_speech_active and self._speech_frames >= self.speech_confirm_frames:
                    self._is_speech_active = True
                    self._send_command(VADCommand.START)
            else:
                self._silence_frames += 1

                # Stop after silence threshold (only if speech was active)
                if self._is_speech_active and self._silence_frames >= self.silence_threshold_frames:
                    self._is_speech_active = False
                    self._speech_frames = 0
                    self._send_command(VADCommand.STOP)

    def _calculate_zcr(self, audio_data: np.ndarray) -> float:
        """Calculate zero-crossing rate of audio data."""
        if len(audio_data) < 2:
            return 0.0
        signs = np.sign(audio_data)
        signs[signs == 0] = 1  # Treat zeros as positive
        zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return zero_crossings / len(audio_data)

    def _send_command(self, command: VADCommand):
        """Send command to callback."""
        print(f"[ActionService] {command.value} (energy_threshold={self._noise_level:.4f})")
        if self.on_command:
            self.on_command(command)

    @property
    def is_speech_active(self) -> bool:
        """Check if speech is currently detected."""
        with self._lock:
            return self._is_speech_active

    @property
    def current_threshold(self) -> float:
        """Get current adaptive threshold."""
        return max(self.energy_threshold, self._noise_level * 2)
