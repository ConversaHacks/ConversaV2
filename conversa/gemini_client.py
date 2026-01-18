"""
GeminiClient - API Communication

Handles communication with Gemini API for audio validation.
Supports both native Gemini API and OpenAI-compatible proxy mode.
"""

import base64
import json
from typing import Optional, Dict, Any
import requests

from .config import config


class GeminiClient:
    """
    Client for Gemini API communication.

    Two modes:
    - Native Gemini API: Direct to Google's API
    - Proxy Mode: Uses OpenAI chat completions format (for custom proxies)
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        use_proxy: bool = None,
    ):
        self.api_key = api_key or config.gemini.api_key
        self.base_url = base_url or config.gemini.base_url
        self.model = model or config.gemini.model
        self.use_proxy = use_proxy if use_proxy is not None else config.gemini.use_proxy
        self.max_tokens = config.gemini.max_tokens
        self.max_audio_bytes = config.gemini.max_audio_bytes

        # Session for connection pooling
        self._session = requests.Session()

    def validate_audio(self, audio_data: bytes) -> Optional[bool]:
        """
        Send audio to Gemini API to validate if it contains human voice.

        Args:
            audio_data: WAV audio data as bytes

        Returns:
            True if voice detected, False if no voice, None on error
        """
        if not self.api_key:
            print("[GeminiClient] No API key configured, auto-approving")
            return True

        # Truncate audio if too long
        if len(audio_data) > self.max_audio_bytes:
            audio_data = audio_data[:self.max_audio_bytes]
            print(f"[GeminiClient] Audio truncated to {self.max_audio_bytes} bytes")

        try:
            if self.use_proxy:
                return self._validate_proxy_mode(audio_data)
            else:
                return self._validate_native_mode(audio_data)
        except Exception as e:
            print(f"[GeminiClient] API error: {e}")
            return None

    def _validate_native_mode(self, audio_data: bytes) -> Optional[bool]:
        """Use native Gemini API format."""
        # Encode audio to base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        # Build request
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        payload = {
            "contents": [{
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": audio_b64
                        }
                    },
                    {
                        "text": 'Does this audio contain human voice or speech? Reply ONLY with a JSON object: {"v":true} if voice is present, or {"v":false} if no voice is present.'
                    }
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": 0.1
            }
        }

        response = self._session.post(
            url,
            params=params,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            print(f"[GeminiClient] API returned status {response.status_code}: {response.text}")
            return None

        return self._parse_response(response.json())

    def _validate_proxy_mode(self, audio_data: bytes) -> Optional[bool]:
        """Use OpenAI chat completions format (for proxies)."""
        # Encode audio to base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        # Build request in OpenAI format
        url = f"{self.base_url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "text",
                        "text": 'Does this audio contain human voice or speech? Reply ONLY with a JSON object: {"v":true} if voice is present, or {"v":false} if no voice is present.'
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
            print(f"[GeminiClient] Proxy API returned status {response.status_code}: {response.text}")
            return None

        return self._parse_proxy_response(response.json())

    def _parse_response(self, response_data: Dict[str, Any]) -> Optional[bool]:
        """Parse native Gemini API response."""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return None

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return None

            text = parts[0].get("text", "")
            return self._extract_voice_result(text)

        except Exception as e:
            print(f"[GeminiClient] Error parsing response: {e}")
            return None

    def _parse_proxy_response(self, response_data: Dict[str, Any]) -> Optional[bool]:
        """Parse OpenAI chat completions format response."""
        try:
            choices = response_data.get("choices", [])
            if not choices:
                return None

            message = choices[0].get("message", {})
            text = message.get("content", "")
            return self._extract_voice_result(text)

        except Exception as e:
            print(f"[GeminiClient] Error parsing proxy response: {e}")
            return None

    def _extract_voice_result(self, text: str) -> Optional[bool]:
        """Extract voice detection result from response text."""
        text = text.strip()

        # Try to parse as JSON
        try:
            # Handle various JSON formats
            if "{" in text:
                # Extract JSON from text
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]
                data = json.loads(json_str)

                # Check for 'v' key
                if "v" in data:
                    return bool(data["v"])
                # Check for 'voice' key
                if "voice" in data:
                    return bool(data["voice"])

        except json.JSONDecodeError:
            pass

        # Fallback: look for true/false in text
        text_lower = text.lower()
        if "true" in text_lower or "yes" in text_lower:
            return True
        if "false" in text_lower or "no" in text_lower:
            return False

        print(f"[GeminiClient] Could not parse response: {text}")
        return None

    def close(self):
        """Close the session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
