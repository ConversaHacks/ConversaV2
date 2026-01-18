"""
GeminiClient - API Communication

Handles communication with Gemini API for audio transcription.
Uses Gemini generateContent API format.
"""

import base64
from dataclasses import dataclass
from typing import Optional, Dict, Any
import requests

from .config import config


@dataclass
class TranscriptResult:
    """Result from audio transcription."""
    has_speech: bool
    transcript: str = ""


class GeminiClient:
    """
    Client for Gemini API communication.

    Uses Gemini generateContent API (/v1beta/models/{model}:generateContent).
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        self.api_key = api_key or config.gemini.api_key
        self.base_url = base_url or config.gemini.base_url
        self.model = model or config.gemini.model
        self.max_tokens = config.gemini.max_tokens
        self.max_audio_bytes = config.gemini.max_audio_bytes

        # Session for connection pooling
        self._session = requests.Session()

    def transcribe_audio(self, audio_data: bytes) -> Optional[TranscriptResult]:
        """
        Send audio to Gemini API to transcribe speech.

        Args:
            audio_data: WAV audio data as bytes

        Returns:
            TranscriptResult with has_speech and transcript, None on error
        """
        if not self.api_key:
            print("[GeminiClient] No API key configured, skipping transcription")
            return TranscriptResult(has_speech=True, transcript="[No API key - auto-approved]")

        # Truncate audio if too long
        if len(audio_data) > self.max_audio_bytes:
            audio_data = audio_data[:self.max_audio_bytes]
            print(f"[GeminiClient] Audio truncated to {self.max_audio_bytes} bytes")

        try:
            return self._transcribe_api(audio_data)
        except Exception as e:
            print(f"[GeminiClient] API error: {e}")
            return None

    def _transcribe_api(self, audio_data: bytes) -> Optional[TranscriptResult]:
        """Use Gemini generateContent API format."""
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "contents": [{
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "audio/mpeg",
                            "data": audio_b64
                        }
                    },
                    {
                        "text": "Transcribe this audio with speaker diarization. Label each speaker (Speaker 1, Speaker 2, etc.) and format as:\nSpeaker 1: [text]\nSpeaker 2: [text]\n\nIf there is only one speaker, still label them. If there is no speech or just noise, respond with exactly: [NO_SPEECH]"
                    }
                ]
            }]
        }

        response = self._session.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            print(f"[GeminiClient] API returned status {response.status_code}: {response.text}")
            return None

        return self._parse_response(response.json())

    def _parse_response(self, response_data: Dict[str, Any]) -> Optional[TranscriptResult]:
        """Parse Gemini generateContent API response."""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return None

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return None

            text = parts[0].get("text", "").strip()
            return self._extract_transcript(text)

        except Exception as e:
            print(f"[GeminiClient] Error parsing response: {e}")
            return None

    def _extract_transcript(self, text: str) -> TranscriptResult:
        """Extract transcript from response text."""
        if not text or text == "[NO_SPEECH]" or "no speech" in text.lower():
            return TranscriptResult(has_speech=False, transcript="")

        return TranscriptResult(has_speech=True, transcript=text)

    def close(self):
        """Close the session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
