#!/usr/bin/env python3
"""
Test script for Gemini audio transcription.
"""

from pathlib import Path
from conversa.gemini_client import GeminiClient

# Hardcoded test file - change this path as needed
TEST_AUDIO_FILE = "output/approved/recording.mp3"


def main():
    audio_path = Path(TEST_AUDIO_FILE)

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        print("\nAvailable files in output/approved/:")
        approved_dir = Path("output/approved")
        if approved_dir.exists():
            for f in approved_dir.glob("*.wav"):
                print(f"  {f}")
        return

    print(f"Testing with: {audio_path}")
    print(f"File size: {audio_path.stat().st_size} bytes")
    print()

    # Read audio file
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Create client and transcribe
    client = GeminiClient()
    print(f"Using API: {client.base_url}/v1/chat/completions")
    print(f"Model: {client.model}")
    print()

    print("Sending to Gemini...")
    result = client.transcribe_audio(audio_data)

    print()
    print("=" * 50)
    print("RESULT:")
    print("=" * 50)

    if result is None:
        print("ERROR: API returned None (check logs above)")
    else:
        print(f"Has speech: {result.has_speech}")
        print(f"Transcript: {result.transcript}")

    client.close()


if __name__ == "__main__":
    main()