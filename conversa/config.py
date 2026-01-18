"""
Configuration settings for Conversa.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VADConfig:
    """Voice Activity Detection settings."""
    sample_rate: int = 44100
    channels: int = 1
    chunk_size: int = 1024  # Samples per frame (~23ms at 44100Hz)
    energy_threshold: float = 0.01  # Adaptive threshold base
    zero_cross_rate_min: float = 0.02
    zero_cross_rate_max: float = 0.5
    speech_confirm_frames: int = 4  # ~100ms of speech to confirm
    silence_threshold_frames: int = 40  # ~1s of silence to stop
    speech_confirm_time: float = 0.3  # seconds
    silence_threshold_time: float = 3.0  # seconds


@dataclass
class RecordingConfig:
    """Recording settings."""
    fps: int = 30
    pre_buffer_duration: float = 15.0  # seconds of video buffer
    audio_pre_buffer: float = 2.0  # seconds captured before speech
    video_width: int = 640
    video_height: int = 480


@dataclass
class GeminiConfig:
    """Gemini API settings."""
    api_key: str = ""
    base_url: str = "https://generativelanguage.googleapis.com"
    model: str = "gemini-2.0-flash"
    use_proxy: bool = False
    max_audio_bytes: int = 882000  # ~10 seconds of audio
    max_tokens: int = 1024

    def __post_init__(self):
        # Load from environment variables
        self.api_key = os.getenv("GEMINI_API_KEY", self.api_key)
        self.base_url = os.getenv("GEMINI_BASE_URL", self.base_url)
        self.model = os.getenv("GEMINI_MODEL", self.model)
        self.use_proxy = os.getenv("GEMINI_USE_PROXY", "false").lower() == "true"


@dataclass
class PathConfig:
    """Output path settings."""
    base_dir: Path = Path("output")
    pending_dir: Path = None
    approved_dir: Path = None
    rejected_dir: Path = None

    def __post_init__(self):
        self.pending_dir = self.base_dir / "pending"
        self.approved_dir = self.base_dir / "approved"
        self.rejected_dir = self.base_dir / "rejected"

    def ensure_directories(self):
        """Create output directories if they don't exist."""
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.approved_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration container."""
    vad: VADConfig = None
    recording: RecordingConfig = None
    gemini: GeminiConfig = None
    paths: PathConfig = None

    def __post_init__(self):
        self.vad = self.vad or VADConfig()
        self.recording = self.recording or RecordingConfig()
        self.gemini = self.gemini or GeminiConfig()
        self.paths = self.paths or PathConfig()
        self.paths.ensure_directories()


# Global configuration instance
config = Config()
