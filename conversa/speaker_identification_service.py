"""
SpeakerIdentificationService - Full Pipeline for Speaker Recognition

Combines face detection, face matching, and Gemini video analysis to:
1. Extract faces from video
2. Match faces against known people in the database
3. Send video to Gemini for transcript with speaker diarization
4. Map speakers to identified people
5. Output structured data for the backend API
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests

from .config import config

# Try to import face recognition
try:
    from .face_service import FaceService, TrackedPerson, extract_faces_from_video, FACE_RECOGNITION_AVAILABLE
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    FaceService = None
    TrackedPerson = None

from .gemini_video_client import GeminiVideoClient, VideoAnalysisResult, SpeakerSegment


@dataclass
class IdentifiedParticipant:
    """A participant identified in the conversation."""
    speaker_id: str  # From Gemini (e.g., "speaker_1")
    person_id: Optional[str] = None  # From database if matched
    person_name: Optional[str] = None  # From database if matched
    description: str = ""  # Physical description from Gemini
    face_embedding: Optional[List[str]] = None  # For new person registration
    face_thumbnail_base64: Optional[str] = None  # For new person registration
    match_confidence: float = 0.0
    is_new_person: bool = True


@dataclass
class TranscriptSegment:
    """A segment of the transcript with speaker information."""
    speaker_id: str
    person_id: Optional[str]
    person_name: Optional[str]
    start_time: float
    end_time: float
    text: str


@dataclass
class ConversationData:
    """Full conversation data ready for backend API."""
    success: bool
    video_path: str
    audio_path: str
    participants: List[IdentifiedParticipant] = field(default_factory=list)
    segments: List[TranscriptSegment] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    date: str = ""
    location: str = ""
    full_transcript: str = ""
    error: str = ""


class SpeakerIdentificationService:
    """
    Service for identifying speakers in video recordings.

    Pipeline:
    1. Extract faces from video → face embeddings
    2. Match embeddings against known people in database
    3. Analyze video with Gemini → transcript + speaker descriptions
    4. Correlate Gemini speakers with face matches
    5. Output structured data for API
    """

    def __init__(self, backend_url: str = "http://localhost:8000/api/v1"):
        self.backend_url = backend_url
        self.gemini_client = GeminiVideoClient()

        if FACE_RECOGNITION_AVAILABLE:
            self.face_service = FaceService()
        else:
            self.face_service = None
            print("[SpeakerIdentificationService] Face recognition not available")

    def process_recording(
        self,
        video_path: str,
        audio_path: str,
        location: str = "Unknown Location",
        transcript_segments: Optional[List[tuple]] = None
    ) -> ConversationData:
        """
        Process a recording to identify speakers and generate conversation data.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            location: Where the conversation took place
            transcript_segments: Optional list of (speaker, start, end, text) from Snowflake

        Returns:
            ConversationData ready for backend API
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)

        if not video_path.exists():
            return ConversationData(
                success=False,
                video_path=str(video_path),
                audio_path=str(audio_path),
                error=f"Video file not found: {video_path}"
            )

        print(f"\n{'='*60}")
        print(f"[SpeakerIdentificationService] Processing: {video_path.name}")
        print(f"{'='*60}\n")

        # Step 1: Extract faces from video
        tracked_persons = []
        if self.face_service:
            print("[Step 1] Extracting faces from video...")
            tracked_persons = extract_faces_from_video(str(video_path), sample_interval=2.0)
            print(f"         Found {len(tracked_persons)} unique faces")
        else:
            print("[Step 1] Skipping face extraction (not available)")

        # Step 2: Match faces against known people
        print("\n[Step 2] Matching faces against known people...")
        face_matches = self._match_faces_to_database(tracked_persons)
        for match in face_matches:
            if match.matched_person_name:
                print(f"         Matched: {match.track_id} → {match.matched_person_name} ({match.match_confidence:.0%})")
            else:
                print(f"         No match: {match.track_id}")

        # Step 3: Get known participants for Gemini context
        known_participants = self._get_known_participants_context(face_matches)

        # Step 4: Analyze video with Gemini (using key frames + transcript)
        print("\n[Step 3] Analyzing with Gemini (key frames + transcript)...")
        analysis = self.gemini_client.analyze_video(
            str(video_path),
            known_participants,
            transcript_segments=transcript_segments
        )

        if not analysis.success:
            return ConversationData(
                success=False,
                video_path=str(video_path),
                audio_path=str(audio_path),
                error=f"Gemini analysis failed: {analysis.error}"
            )

        print(f"         Identified {len(analysis.participants)} speakers")
        print(f"         Generated {len(analysis.segments)} transcript segments")

        # Step 5: Correlate Gemini speakers with face matches
        print("\n[Step 4] Correlating speakers with face data...")
        participants = self._correlate_speakers(analysis, face_matches, tracked_persons)

        # Step 6: Build transcript segments
        segments = self._build_transcript_segments(analysis.segments, participants)

        # Step 7: Generate full transcript text
        full_transcript = self._generate_full_transcript(segments)

        # Step 8: Generate date string
        date_str = datetime.now().strftime("%b %d • %I:%M %p")

        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Title: {analysis.suggested_title}")
        print(f"Participants: {', '.join(p.person_name or p.description for p in participants)}")
        print(f"Segments: {len(segments)}")
        print(f"{'='*60}\n")

        return ConversationData(
            success=True,
            video_path=str(video_path),
            audio_path=str(audio_path),
            participants=participants,
            segments=segments,
            title=analysis.suggested_title,
            summary=analysis.summary,
            key_points=analysis.key_points,
            date=date_str,
            location=location,
            full_transcript=full_transcript
        )

    def _match_faces_to_database(
        self,
        tracked_persons: List[TrackedPerson]
    ) -> List[TrackedPerson]:
        """Match tracked faces against the database."""
        if not tracked_persons:
            return []

        for person in tracked_persons:
            try:
                # Call backend API to match face
                embedding_str = [str(x) for x in person.avg_embedding]

                response = requests.post(
                    f"{self.backend_url}/people/match-face",
                    json={
                        "face_embedding": embedding_str,
                        "threshold": 0.6
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    match_data = response.json()
                    if match_data.get("matched"):
                        person.matched_person_id = match_data.get("person_id")
                        person.matched_person_name = match_data.get("person_name")
                        person.match_confidence = match_data.get("confidence", 0)

            except Exception as e:
                print(f"[SpeakerIdentificationService] Face match error: {e}")

        return tracked_persons

    def _get_known_participants_context(
        self,
        face_matches: List[TrackedPerson]
    ) -> List[Dict[str, str]]:
        """Build known participants list for Gemini context."""
        known = []
        for person in face_matches:
            if person.matched_person_name:
                known.append({
                    "name": person.matched_person_name,
                    "description": f"Recognized from database (confidence: {person.match_confidence:.0%})"
                })
        return known

    def _correlate_speakers(
        self,
        analysis: VideoAnalysisResult,
        face_matches: List[TrackedPerson],
        tracked_persons: List[TrackedPerson]
    ) -> List[IdentifiedParticipant]:
        """
        Correlate Gemini's speaker IDs with face recognition results.

        Priority for identifying a person:
        1. Name extracted from conversation by Gemini (e.g., "Hey Sarah")
        2. Face recognition match from database
        3. Physical description from Gemini
        """
        participants = []

        for gemini_participant in analysis.participants:
            speaker_id = gemini_participant.get("id", "unknown")
            description = gemini_participant.get("description", "")
            gemini_name = gemini_participant.get("name", "")  # Name from conversation!

            # Try to find a matching face
            matched_face = None
            matched_person_id = None
            matched_person_name = None

            # Priority 1: If Gemini extracted a name from conversation, try to match to database
            if gemini_name:
                print(f"         [Name from conversation] {speaker_id}: '{gemini_name}'")
                # Try to find this person in the database by name
                db_match = self._find_person_by_name(gemini_name)
                if db_match:
                    matched_person_id = db_match.get("id")
                    matched_person_name = db_match.get("name")
                    print(f"         [Database match] Found: {matched_person_name} (id: {matched_person_id})")
                else:
                    # Use the name from conversation even if not in database
                    matched_person_name = gemini_name
                    print(f"         [New person] Will use name: {gemini_name}")

            # Priority 2: Check face recognition matches
            if not matched_person_id:
                for face in face_matches:
                    if face.matched_person_name:
                        # Check if name appears in description
                        if face.matched_person_name.lower() in description.lower():
                            matched_face = face
                            matched_person_id = face.matched_person_id
                            matched_person_name = face.matched_person_name
                            break

            # Priority 3: Correlate by order (heuristic)
            if not matched_face and face_matches:
                idx = int(speaker_id.split("_")[-1]) - 1 if "_" in speaker_id else 0
                if 0 <= idx < len(face_matches):
                    matched_face = face_matches[idx]
                    if matched_face.matched_person_id:
                        matched_person_id = matched_face.matched_person_id
                        matched_person_name = matched_face.matched_person_name

            # Use Gemini name if we still don't have a name
            if not matched_person_name and gemini_name:
                matched_person_name = gemini_name

            # Build participant
            participant = IdentifiedParticipant(
                speaker_id=speaker_id,
                person_id=matched_person_id,
                person_name=matched_person_name,
                description=description,
                match_confidence=matched_face.match_confidence if matched_face else 0,
                is_new_person=matched_person_id is None
            )

            # Add face data for new persons
            if participant.is_new_person and matched_face:
                participant.face_embedding = [str(x) for x in matched_face.avg_embedding]
                participant.face_thumbnail_base64 = self._get_thumbnail_base64(matched_face)
            elif participant.is_new_person and tracked_persons:
                # Use first unmatched face
                for tp in tracked_persons:
                    if not tp.matched_person_id:
                        participant.face_embedding = [str(x) for x in tp.avg_embedding]
                        participant.face_thumbnail_base64 = self._get_thumbnail_base64(tp)
                        break

            participants.append(participant)

        return participants

    def _find_person_by_name(self, name: str) -> Optional[Dict]:
        """Find a person in the database by name (fuzzy match)."""
        try:
            response = requests.get(f"{self.backend_url}/people/", timeout=10)
            if response.status_code == 200:
                people = response.json()
                name_lower = name.lower()
                for person in people:
                    person_name = person.get("name", "").lower()
                    # Check if the spoken name matches (first name or full name)
                    if name_lower == person_name or name_lower in person_name.split():
                        return person
                    # Also check if first name matches
                    if person_name.startswith(name_lower + " ") or person_name == name_lower:
                        return person
        except Exception as e:
            print(f"[SpeakerIdentificationService] Error finding person by name: {e}")
        return None

    def _get_thumbnail_base64(self, person: TrackedPerson) -> Optional[str]:
        """Get thumbnail as base64."""
        if person and person.best_thumbnail:
            import base64
            return base64.b64encode(person.best_thumbnail).decode('utf-8')
        return None

    def _build_transcript_segments(
        self,
        gemini_segments: List[SpeakerSegment],
        participants: List[IdentifiedParticipant]
    ) -> List[TranscriptSegment]:
        """Build transcript segments with person information."""
        # Build speaker lookup
        speaker_lookup = {p.speaker_id: p for p in participants}

        segments = []
        for seg in gemini_segments:
            participant = speaker_lookup.get(seg.speaker_id)

            segments.append(TranscriptSegment(
                speaker_id=seg.speaker_id,
                person_id=participant.person_id if participant else None,
                person_name=participant.person_name if participant else seg.speaker_description,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text
            ))

        return segments

    def _generate_full_transcript(self, segments: List[TranscriptSegment]) -> str:
        """Generate formatted full transcript."""
        lines = []
        for seg in segments:
            speaker = seg.person_name or f"Speaker {seg.speaker_id}"
            lines.append(f"{speaker}: {seg.text}")
        return "\n\n".join(lines)

    def send_to_backend(self, conversation_data: ConversationData) -> Optional[str]:
        """
        Send conversation data to the backend API.

        Returns the created conversation ID, or None on failure.
        """
        if not conversation_data.success:
            print(f"[SpeakerIdentificationService] Cannot send failed conversation")
            return None

        # Determine primary person (first participant with a person_id, or create new)
        primary_person_id = None
        for p in conversation_data.participants:
            if p.person_id:
                primary_person_id = p.person_id
                break

        # If no matched person, we need to create one or use the first participant
        if not primary_person_id and conversation_data.participants:
            first_participant = conversation_data.participants[0]

            # Create a new person if needed
            if first_participant.is_new_person:
                new_person = self._create_person(first_participant)
                if new_person:
                    primary_person_id = new_person.get("id")
                    first_participant.person_id = primary_person_id
                    first_participant.person_name = new_person.get("name")

        if not primary_person_id:
            print("[SpeakerIdentificationService] No primary person identified")
            return None

        # Update met_count for the primary person
        self._update_person_met(primary_person_id)

        # Build conversation payload
        participant_ids = [p.person_id for p in conversation_data.participants if p.person_id]

        payload = {
            "person_id": primary_person_id,
            "participants": participant_ids,
            "title": conversation_data.title,
            "date": conversation_data.date,
            "location": conversation_data.location,
            "summary": conversation_data.summary,
            "key_points": conversation_data.key_points,
            "full_transcript": conversation_data.full_transcript,
            "action_items": []  # Could extract from summary
        }

        try:
            response = requests.post(
                f"{self.backend_url}/conversations/",
                json=payload,
                timeout=10
            )

            if response.status_code in [200, 201]:
                result = response.json()
                conversation_id = result.get("id")
                print(f"[SpeakerIdentificationService] Created conversation: {conversation_id}")
                return conversation_id
            else:
                print(f"[SpeakerIdentificationService] API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"[SpeakerIdentificationService] Error sending to backend: {e}")
            return None

    def _create_person(self, participant: IdentifiedParticipant) -> Optional[Dict]:
        """Create a new person in the database."""
        # Generate a name from description or use generic
        name = self._extract_name_from_description(participant.description)
        if not name:
            name = f"Person ({participant.speaker_id})"

        payload = {
            "name": name,
            "role": "New Contact",
            "avatar_color": self._random_avatar_color(),
            "context": f"First met during a conversation. {participant.description}",
            "interests": [],
            "open_follow_ups": [],
            "physical_description": participant.description,
            "face_embedding": participant.face_embedding,
            "face_thumbnail_base64": participant.face_thumbnail_base64
        }

        try:
            response = requests.post(
                f"{self.backend_url}/people/",
                json=payload,
                timeout=10
            )

            if response.status_code in [200, 201]:
                return response.json()
            else:
                print(f"[SpeakerIdentificationService] Failed to create person: {response.text}")
                return None

        except Exception as e:
            print(f"[SpeakerIdentificationService] Error creating person: {e}")
            return None

    def _update_person_met(self, person_id: str):
        """Update a person's met_count and last_met."""
        try:
            # Get current person data
            response = requests.get(f"{self.backend_url}/people/{person_id}", timeout=10)
            if response.status_code != 200:
                return

            person = response.json()
            current_count = person.get("met_count", 0)

            # Update
            update_payload = {
                "met_count": current_count + 1,
                "last_met": datetime.now().strftime("%b %d")
            }

            requests.put(
                f"{self.backend_url}/people/{person_id}",
                json=update_payload,
                timeout=10
            )

        except Exception as e:
            print(f"[SpeakerIdentificationService] Error updating person: {e}")

    def _extract_name_from_description(self, description: str) -> Optional[str]:
        """Try to extract a name from the description."""
        # Simple heuristic: look for capitalized words that might be names
        words = description.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Check if it's not a common word
                common = ["The", "This", "That", "Person", "Man", "Woman", "With", "In", "On", "At"]
                if word not in common:
                    # Check next word for last name
                    if i + 1 < len(words) and words[i+1][0].isupper():
                        return f"{word} {words[i+1]}"
                    return word
        return None

    def _random_avatar_color(self) -> str:
        """Generate a random Tailwind avatar color."""
        import random
        colors = [
            "bg-indigo-200", "bg-emerald-200", "bg-orange-200",
            "bg-pink-200", "bg-purple-200", "bg-cyan-200",
            "bg-yellow-200", "bg-red-200", "bg-blue-200"
        ]
        return random.choice(colors)

    def close(self):
        """Clean up resources."""
        self.gemini_client.close()


def process_and_upload_recording(
    video_path: str,
    audio_path: str,
    location: str = "Unknown Location",
    backend_url: str = "http://localhost:8000/api/v1"
) -> Optional[str]:
    """
    Convenience function to process a recording and upload to backend.

    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        location: Where the conversation took place
        backend_url: Backend API URL

    Returns:
        Conversation ID if successful, None otherwise
    """
    service = SpeakerIdentificationService(backend_url)

    try:
        conversation_data = service.process_recording(video_path, audio_path, location)

        if conversation_data.success:
            return service.send_to_backend(conversation_data)
        else:
            print(f"[Error] Processing failed: {conversation_data.error}")
            return None

    finally:
        service.close()
