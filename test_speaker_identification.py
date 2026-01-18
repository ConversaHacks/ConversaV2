#!/usr/bin/env python3
"""
Test script for the speaker identification pipeline.

This script tests:
1. Gemini video analysis (without face recognition for now)
2. Backend API integration
3. Full pipeline from video to conversation creation
"""

import os
import sys
from pathlib import Path

# Add conversa to path
sys.path.insert(0, str(Path(__file__).parent))

from conversa.gemini_video_client import GeminiVideoClient, analyze_conversation_video
from conversa.speaker_identification_service import (
    SpeakerIdentificationService,
    process_and_upload_recording
)


def test_gemini_client():
    """Test the Gemini video client."""
    print("\n" + "="*60)
    print("TEST 1: Gemini Video Client")
    print("="*60)

    client = GeminiVideoClient()
    print(f"API Key: {'*' * 10 + client.api_key[-4:] if client.api_key else 'Not set'}")
    print(f"Base URL: {client.base_url}")
    print(f"Model: {client.model}")

    # Check for test video
    test_videos = list(Path("output/pending").glob("*.mp4"))
    if not test_videos:
        test_videos = list(Path("output/approved").glob("*.mp4"))

    if not test_videos:
        print("\nNo test videos found in output/pending or output/approved")
        print("Please record a video first or place a test video in output/pending/")
        return None

    test_video = test_videos[0]
    print(f"\nTesting with: {test_video}")

    result = client.analyze_video(str(test_video))

    if result.success:
        print("\n✓ Video analysis successful!")
        print(f"\nParticipants ({len(result.participants)}):")
        for p in result.participants:
            print(f"  - {p.get('id')}: {p.get('description')}")

        print(f"\nSegments ({len(result.segments)}):")
        for seg in result.segments[:5]:  # Show first 5
            print(f"  [{seg.start_time:.1f}-{seg.end_time:.1f}] {seg.speaker_id}: {seg.text[:50]}...")

        print(f"\nSummary: {result.summary[:200]}...")
        print(f"\nKey Points: {result.key_points}")
        print(f"\nSuggested Title: {result.suggested_title}")
    else:
        print(f"\n✗ Video analysis failed: {result.error}")

    client.close()
    return result


def test_backend_api():
    """Test the backend API connection."""
    print("\n" + "="*60)
    print("TEST 2: Backend API Connection")
    print("="*60)

    import requests

    base_url = "http://localhost:8000/api/v1"

    # Test people endpoint
    try:
        response = requests.get(f"{base_url}/people/", timeout=5)
        if response.status_code == 200:
            people = response.json()
            print(f"\n✓ People endpoint working - Found {len(people)} people")
            for p in people[:3]:
                has_face = p.get('has_face_data', False)
                print(f"  - {p['name']} (met {p['met_count']}x) {'[has face data]' if has_face else ''}")
        else:
            print(f"\n✗ People endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"\n✗ Backend not reachable: {e}")
        print("Make sure the backend is running: cd ConversaBE && source venv/bin/activate && uvicorn app.main:app --reload")
        return False

    # Test face matching endpoint
    try:
        # Send a dummy embedding to test the endpoint exists
        dummy_embedding = ["0.1"] * 128
        response = requests.post(
            f"{base_url}/people/match-face",
            json={"face_embedding": dummy_embedding, "threshold": 0.6},
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Face matching endpoint working")
            print(f"  Matched: {result.get('matched', False)}")
        else:
            print(f"\n✗ Face matching endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"\n✗ Face matching test failed: {e}")

    return True


def test_full_pipeline():
    """Test the full speaker identification pipeline."""
    print("\n" + "="*60)
    print("TEST 3: Full Speaker Identification Pipeline")
    print("="*60)

    # Find test video
    test_videos = list(Path("output/pending").glob("*.mp4"))
    if not test_videos:
        test_videos = list(Path("output/approved").glob("*.mp4"))

    if not test_videos:
        print("\nNo test videos found")
        return

    video_path = test_videos[0]
    audio_path = video_path.with_suffix(".wav")

    if not audio_path.exists():
        print(f"\nAudio file not found: {audio_path}")
        # Try without audio
        audio_path = video_path

    print(f"\nProcessing: {video_path.name}")

    service = SpeakerIdentificationService()

    try:
        result = service.process_recording(
            str(video_path),
            str(audio_path),
            location="Test Location"
        )

        if result.success:
            print("\n✓ Processing successful!")
            print(f"\nTitle: {result.title}")
            print(f"Date: {result.date}")
            print(f"Location: {result.location}")

            print(f"\nParticipants ({len(result.participants)}):")
            for p in result.participants:
                name = p.person_name or p.description or "Unknown"
                status = "matched" if p.person_id else "new"
                print(f"  - {name} ({status})")

            print(f"\nTranscript Preview:")
            for seg in result.segments[:5]:
                speaker = seg.person_name or seg.speaker_id
                print(f"  [{seg.start_time:.1f}s] {speaker}: {seg.text[:50]}...")

            # Ask if user wants to upload
            print("\n" + "-"*40)
            upload = input("Upload to backend? (y/n): ").strip().lower()

            if upload == 'y':
                conversation_id = service.send_to_backend(result)
                if conversation_id:
                    print(f"\n✓ Conversation created: {conversation_id}")
                else:
                    print("\n✗ Failed to create conversation")
        else:
            print(f"\n✗ Processing failed: {result.error}")

    finally:
        service.close()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SPEAKER IDENTIFICATION PIPELINE TEST")
    print("="*60)

    # Check environment
    print("\nEnvironment:")
    print(f"  GEMINI_API_KEY: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not set'}")
    print(f"  GEMINI_BASE_URL: {os.getenv('GEMINI_BASE_URL', 'Not set (using default)')}")
    print(f"  GEMINI_VIDEO_MODEL: {os.getenv('GEMINI_VIDEO_MODEL', 'Not set (using gemini-2.5-pro)')}")

    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("  .env file: Loaded")
    except ImportError:
        print("  .env file: python-dotenv not installed")

    # Run tests
    print("\n" + "-"*60)

    # Test 1: Backend API
    backend_ok = test_backend_api()

    if not backend_ok:
        print("\n⚠ Backend not available. Some tests will be skipped.")

    # Test 2: Gemini Client
    gemini_result = test_gemini_client()

    # Test 3: Full Pipeline (only if both work)
    if backend_ok and gemini_result and gemini_result.success:
        test_full_pipeline()
    else:
        print("\n⚠ Skipping full pipeline test (prerequisites not met)")

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
