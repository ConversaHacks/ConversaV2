"""
GeminiVideoClient - Video + Audio Analysis with Gemini 2.5 Pro

Sends video files to Gemini for multimodal analysis including:
- Speaker identification and diarization
- Visual description of participants
- Transcript with speaker mapping
"""

import base64
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests

from .config import config


@dataclass
class SpeakerSegment:
    """A segment of speech from one speaker."""
    speaker_id: str  # e.g., "speaker_1"
    speaker_description: str  # e.g., "woman with dark hair, blue shirt"
    speaker_name: str = ""  # Name if mentioned in conversation (e.g., "Sarah")
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""


@dataclass
class VideoAnalysisResult:
    """Result from video analysis."""
    success: bool
    participants: List[Dict[str, str]] = field(default_factory=list)  # [{id, description, name}]
    segments: List[SpeakerSegment] = field(default_factory=list)
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    suggested_title: str = ""
    error: str = ""


class GeminiVideoClient:
    """
    Client for Gemini 2.5 Pro video analysis.

    Sends video files for multimodal analysis with speaker identification.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or config.gemini.api_key
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL") or config.gemini.base_url
        # Use gemini-2.5-pro for video analysis
        self.model = model or os.getenv("GEMINI_VIDEO_MODEL", "gemini-2.5-pro")

        # Session for connection pooling
        self._session = requests.Session()

    def analyze_video(
        self,
        video_path: str,
        known_participants: Optional[List[Dict[str, str]]] = None,
        transcript_segments: Optional[List[tuple]] = None
    ) -> VideoAnalysisResult:
        """
        Analyze a video file with Gemini using key frames + transcript.

        Args:
            video_path: Path to video file (MP4)
            known_participants: Optional list of known people with descriptions
            transcript_segments: Optional list of (speaker, start, end, text) from Snowflake

        Returns:
            VideoAnalysisResult with participants, transcript segments, and analysis
        """
        video_path = Path(video_path)

        if not video_path.exists():
            return VideoAnalysisResult(success=False, error=f"Video file not found: {video_path}")

        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"[GeminiVideoClient] Analyzing video: {video_path.name} ({file_size_mb:.1f}MB)")

        try:
            # Use key frames approach - much faster and more reliable
            return self._analyze_with_keyframes(video_path, known_participants, transcript_segments)
        except Exception as e:
            print(f"[GeminiVideoClient] Error: {e}")
            return VideoAnalysisResult(success=False, error=str(e))

    def _extract_keyframes(self, video_path: Path, max_frames: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key frames from video using OpenCV.

        Args:
            video_path: Path to video
            max_frames: Maximum number of frames to extract

        Returns:
            List of {timestamp: float, image_b64: str}
        """
        import cv2

        frames = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[GeminiVideoClient] Could not open video: {video_path}")
            return frames

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Calculate frame interval to get evenly spaced frames
        if duration <= 0:
            return frames

        # Get frames at regular intervals
        interval_seconds = max(duration / max_frames, 1.0)
        timestamps = [i * interval_seconds for i in range(max_frames) if i * interval_seconds < duration]

        print(f"[GeminiVideoClient] Extracting {len(timestamps)} frames from {duration:.1f}s video")

        for ts in timestamps:
            frame_num = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                # Resize to reduce size (max 640px width)
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    frame = cv2.resize(frame, (640, int(height * scale)))

                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_b64 = base64.b64encode(buffer).decode('utf-8')

                frames.append({
                    'timestamp': ts,
                    'image_b64': img_b64
                })

        cap.release()
        print(f"[GeminiVideoClient] Extracted {len(frames)} frames")
        return frames

    def _analyze_with_keyframes(
        self,
        video_path: Path,
        known_participants: Optional[List[Dict[str, str]]],
        transcript_segments: Optional[List[tuple]]
    ) -> VideoAnalysisResult:
        """Analyze video using key frames + transcript."""
        # Extract key frames
        frames = self._extract_keyframes(video_path, max_frames=8)

        if not frames:
            return VideoAnalysisResult(success=False, error="Could not extract frames from video")

        # Build the prompt with transcript
        prompt = self._build_keyframes_prompt(known_participants, transcript_segments)

        # Build parts: images first, then text prompt
        parts = []

        for i, frame in enumerate(frames):
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": frame['image_b64']
                }
            })

        parts.append({"text": prompt})

        # API request
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192
            }
        }

        print(f"[GeminiVideoClient] Sending {len(frames)} frames to {self.model}...")
        start_time = time.time()

        response = self._session.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        elapsed = time.time() - start_time
        print(f"[GeminiVideoClient] Response received in {elapsed:.1f}s")

        if response.status_code != 200:
            error_msg = f"API returned status {response.status_code}: {response.text[:500]}"
            print(f"[GeminiVideoClient] {error_msg}")
            return VideoAnalysisResult(success=False, error=error_msg)

        return self._parse_response(response.json())

    def _build_keyframes_prompt(
        self,
        known_participants: Optional[List[Dict[str, str]]],
        transcript_segments: Optional[List[tuple]]
    ) -> str:
        """Build prompt for key frames analysis."""
        known_context = ""
        if known_participants:
            known_list = "\n".join([
                f"- {p['name']}: {p.get('description', 'No description')}"
                for p in known_participants
            ])
            known_context = f"""
Known participants (match these if you recognize them):
{known_list}
"""

        transcript_context = ""
        if transcript_segments:
            lines = []
            for speaker, start, end, text in transcript_segments:
                lines.append(f"[{start:.1f}s - {end:.1f}s] {speaker}: {text}")
            transcript_context = f"""
TRANSCRIPT (from audio analysis):
{chr(10).join(lines)}
"""

        return f"""I'm showing you {8} frames from a video recording of a conversation.

{known_context}
{transcript_context}

Based on these frames and the transcript, identify the people visible and match them to the speakers in the transcript.

CRITICAL: Pay close attention to when people are addressed by name or introduce themselves!
- If the transcript shows "My name is Seamoon" or "Oh, my name is Kevin", use those names
- If someone says "Hey Sarah" or "Thanks David", that tells you the other person's name

Provide your response as valid JSON with this exact structure:
{{
    "participants": [
        {{"id": "speaker_1", "name": "Seamoon", "description": "Person with dark hair, wearing blue shirt"}},
        {{"id": "speaker_2", "name": "Kevin", "description": "Person with glasses, wearing grey sweater"}}
    ],
    "segments": [
        {{"speaker_id": "speaker_1", "start": 0.0, "end": 2.5, "text": "What they said"}},
        {{"speaker_id": "speaker_2", "start": 2.7, "end": 5.0, "text": "What they said"}}
    ],
    "summary": "2-3 sentence summary of the conversation",
    "key_points": ["Key point 1", "Key point 2", "Key point 3"],
    "suggested_title": "Short descriptive title for this conversation"
}}

Important:
1. Extract names from the transcript - use them in the participants list
2. Map SPEAKER_00, SPEAKER_01 etc from the transcript to the people you see in the frames
3. Include physical descriptions so speakers can be matched visually
4. Use the transcript text for the segments
5. Return ONLY valid JSON, no other text"""

    def _compress_video(self, video_path: Path, max_size_mb: float = 15.0) -> Path:
        """
        Compress video to reduce file size for API upload.

        Args:
            video_path: Original video path
            max_size_mb: Target max size in MB

        Returns:
            Path to compressed video (or original if already small enough)
        """
        file_size_mb = video_path.stat().st_size / (1024 * 1024)

        if file_size_mb <= max_size_mb:
            return video_path

        print(f"[GeminiVideoClient] Compressing video from {file_size_mb:.1f}MB...")

        # Create temp file for compressed video
        temp_dir = tempfile.gettempdir()
        compressed_path = Path(temp_dir) / f"compressed_{video_path.name}"

        # Calculate target bitrate (aim for ~80% of max size to have headroom)
        # Get video duration first
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, timeout=30
            )
            duration = float(result.stdout.strip())
        except Exception:
            duration = 60  # Assume 60 seconds if we can't get duration

        # Target bitrate in kbps (size_in_kb * 8 / duration_in_seconds)
        target_size_kb = max_size_mb * 0.8 * 1024
        target_bitrate = int((target_size_kb * 8) / duration)
        target_bitrate = max(target_bitrate, 200)  # Minimum 200kbps

        # Compress with ffmpeg
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(video_path),
                "-c:v", "libx264",
                "-b:v", f"{target_bitrate}k",
                "-maxrate", f"{target_bitrate * 2}k",
                "-bufsize", f"{target_bitrate * 2}k",
                "-vf", "scale=640:-2",  # Scale to 640px width, maintain aspect
                "-c:a", "aac",
                "-b:a", "64k",  # Lower audio bitrate
                "-ac", "1",  # Mono audio
                str(compressed_path)
            ], capture_output=True, timeout=120, check=True)

            new_size_mb = compressed_path.stat().st_size / (1024 * 1024)
            print(f"[GeminiVideoClient] Compressed to {new_size_mb:.1f}MB")
            return compressed_path

        except subprocess.CalledProcessError as e:
            print(f"[GeminiVideoClient] Compression failed: {e.stderr.decode()[:200]}")
            return video_path
        except FileNotFoundError:
            print("[GeminiVideoClient] ffmpeg not found, sending original video")
            return video_path

    def _analyze_with_api(
        self,
        video_path: Path,
        known_participants: Optional[List[Dict[str, str]]]
    ) -> VideoAnalysisResult:
        """Send video to Gemini API and parse response."""
        # Compress video if needed
        video_to_send = self._compress_video(video_path)

        # Read and encode video
        with open(video_to_send, "rb") as f:
            video_data = f.read()

        video_b64 = base64.b64encode(video_data).decode("utf-8")

        # Clean up temp file if we compressed
        if video_to_send != video_path and video_to_send.exists():
            try:
                video_to_send.unlink()
            except Exception:
                pass

        # Determine mime type
        suffix = video_path.suffix.lower()
        mime_type = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo"
        }.get(suffix, "video/mp4")

        # Build prompt
        prompt = self._build_analysis_prompt(known_participants)

        # API request
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
                            "mimeType": mime_type,
                            "data": video_b64
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192
            }
        }

        print(f"[GeminiVideoClient] Sending request to {self.model}...")
        start_time = time.time()

        response = self._session.post(
            url,
            headers=headers,
            json=payload,
            timeout=120  # Video analysis can take time
        )

        elapsed = time.time() - start_time
        print(f"[GeminiVideoClient] Response received in {elapsed:.1f}s")

        if response.status_code != 200:
            error_msg = f"API returned status {response.status_code}: {response.text[:500]}"
            print(f"[GeminiVideoClient] {error_msg}")
            return VideoAnalysisResult(success=False, error=error_msg)

        return self._parse_response(response.json())

    def _build_analysis_prompt(
        self,
        known_participants: Optional[List[Dict[str, str]]]
    ) -> str:
        """Build the analysis prompt for Gemini."""
        known_context = ""
        if known_participants:
            known_list = "\n".join([
                f"- {p['name']}: {p.get('description', 'No description')}"
                for p in known_participants
            ])
            known_context = f"""
Known participants (match these if you recognize them):
{known_list}
"""

        return f"""Analyze this video recording of a conversation. Identify all speakers and transcribe what they say.

{known_context}

CRITICAL: Pay close attention to when people are addressed by name or introduce themselves!
- If someone says "Hey Sarah" or "Thanks David", that tells you the other person's name
- If someone says "I'm John" or "My name is Elena", that's their name
- Use these spoken names to identify participants

Provide your response as valid JSON with this exact structure:
{{
    "participants": [
        {{"id": "speaker_1", "name": "Sarah", "description": "Woman with dark hair, blue shirt, left side of frame"}},
        {{"id": "speaker_2", "name": "", "description": "Man with glasses, grey sweater, right side of frame"}}
    ],
    "segments": [
        {{"speaker_id": "speaker_1", "start": 0.0, "end": 2.5, "text": "What they said"}},
        {{"speaker_id": "speaker_2", "start": 2.7, "end": 5.0, "text": "What they said"}}
    ],
    "summary": "2-3 sentence summary of the conversation",
    "key_points": ["Key point 1", "Key point 2", "Key point 3"],
    "suggested_title": "Short descriptive title for this conversation"
}}

Important instructions:
1. Use consistent speaker IDs (speaker_1, speaker_2, etc.) throughout
2. **IMPORTANT**: Extract names from the conversation! If someone calls a person "Sarah" or "David", use that as their name
3. If a name is mentioned for a speaker, include it in the "name" field (leave empty if unknown)
4. Provide accurate timestamps in seconds
5. Be specific in physical descriptions so speakers can be matched visually
6. Include all spoken content, not just summaries
7. Return ONLY valid JSON, no other text"""

    def _parse_response(self, response_data: Dict[str, Any]) -> VideoAnalysisResult:
        """Parse Gemini API response."""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return VideoAnalysisResult(success=False, error="No response candidates")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return VideoAnalysisResult(success=False, error="No response parts")

            text = parts[0].get("text", "").strip()

            # Try to parse as JSON
            # Sometimes the response has markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)

            # Parse participants
            participants = data.get("participants", [])

            # Parse segments
            segments = []
            for seg in data.get("segments", []):
                # Find description and name for this speaker
                speaker_desc = ""
                speaker_name = ""
                for p in participants:
                    if p.get("id") == seg.get("speaker_id"):
                        speaker_desc = p.get("description", "")
                        speaker_name = p.get("name", "")
                        break

                segments.append(SpeakerSegment(
                    speaker_id=seg.get("speaker_id", "unknown"),
                    speaker_description=speaker_desc,
                    speaker_name=speaker_name,
                    start_time=float(seg.get("start", 0)),
                    end_time=float(seg.get("end", 0)),
                    text=seg.get("text", "")
                ))

            return VideoAnalysisResult(
                success=True,
                participants=participants,
                segments=segments,
                summary=data.get("summary", ""),
                key_points=data.get("key_points", []),
                suggested_title=data.get("suggested_title", "Conversation")
            )

        except json.JSONDecodeError as e:
            print(f"[GeminiVideoClient] JSON parse error: {e}")
            print(f"[GeminiVideoClient] Raw response: {text[:500]}")
            return VideoAnalysisResult(
                success=False,
                error=f"Failed to parse response as JSON: {e}"
            )
        except Exception as e:
            print(f"[GeminiVideoClient] Parse error: {e}")
            return VideoAnalysisResult(success=False, error=str(e))

    def close(self):
        """Close the session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def analyze_conversation_video(
    video_path: str,
    known_people: Optional[List[Dict[str, str]]] = None
) -> VideoAnalysisResult:
    """
    Convenience function to analyze a conversation video.

    Args:
        video_path: Path to video file
        known_people: Optional list of known people with descriptions

    Returns:
        VideoAnalysisResult
    """
    with GeminiVideoClient() as client:
        return client.analyze_video(video_path, known_people)
