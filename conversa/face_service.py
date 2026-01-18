"""
FaceService - Face Detection and Recognition

Extracts faces from video frames, generates embeddings for recognition,
and matches faces against known people.
"""

import base64
import io
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[FaceService] face_recognition not available. Install with: pip install face_recognition")


@dataclass
class DetectedFace:
    """A face detected in a frame."""
    face_id: str  # Temporary ID for this detection session
    bbox: Tuple[int, int, int, int]  # (top, right, bottom, left)
    embedding: List[float]  # 128-dimensional face embedding
    thumbnail: bytes  # JPEG thumbnail of the face
    frame_time: float  # Timestamp in video


@dataclass
class TrackedPerson:
    """A person tracked across multiple frames."""
    track_id: str  # Consistent ID across frames
    embeddings: List[List[float]]  # All embeddings seen
    avg_embedding: List[float]  # Average embedding
    best_thumbnail: bytes  # Best quality thumbnail
    first_seen: float  # First timestamp
    last_seen: float  # Last timestamp
    frame_count: int  # Number of frames seen
    matched_person_id: Optional[str] = None  # ID from database if matched
    matched_person_name: Optional[str] = None  # Name from database if matched
    match_confidence: float = 0.0


class FaceService:
    """
    Face detection and recognition service.

    Extracts faces from video, generates embeddings, and tracks
    individuals across frames.
    """

    def __init__(self, detection_model: str = "hog"):
        """
        Args:
            detection_model: "hog" (faster, CPU) or "cnn" (more accurate, GPU)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            raise RuntimeError("face_recognition library not available")

        self.detection_model = detection_model
        self._tracked_persons: Dict[str, TrackedPerson] = {}
        self._next_track_id = 0
        self._embedding_threshold = 0.6  # Distance threshold for same person

    def detect_faces(self, frame: np.ndarray, frame_time: float = 0.0) -> List[DetectedFace]:
        """
        Detect faces in a single frame.

        Args:
            frame: BGR image from OpenCV
            frame_time: Timestamp of this frame

        Returns:
            List of DetectedFace objects
        """
        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame, model=self.detection_model)

        if not face_locations:
            return []

        # Get face encodings (embeddings)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_faces = []
        for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = location

            # Extract face thumbnail
            face_img = frame[top:bottom, left:right]
            thumbnail = self._create_thumbnail(face_img)

            detected_faces.append(DetectedFace(
                face_id=f"face_{i}_{frame_time:.2f}",
                bbox=location,
                embedding=encoding.tolist(),
                thumbnail=thumbnail,
                frame_time=frame_time
            ))

        return detected_faces

    def _create_thumbnail(self, face_img: np.ndarray, size: int = 128) -> bytes:
        """Create a JPEG thumbnail of a face."""
        if face_img.size == 0:
            return b""

        # Resize to square
        h, w = face_img.shape[:2]
        if h > w:
            face_img = face_img[(h-w)//2:(h-w)//2+w, :]
        elif w > h:
            face_img = face_img[:, (w-h)//2:(w-h)//2+h]

        face_img = cv2.resize(face_img, (size, size))

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

    def track_person(self, face: DetectedFace) -> str:
        """
        Track a detected face across frames.

        Args:
            face: Detected face to track

        Returns:
            Track ID (consistent across frames for same person)
        """
        # Find existing track with similar embedding
        best_match_id = None
        best_match_distance = float('inf')

        for track_id, person in self._tracked_persons.items():
            distance = self._embedding_distance(face.embedding, person.avg_embedding)
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_id = track_id

        if best_match_id and best_match_distance < self._embedding_threshold:
            # Update existing track
            person = self._tracked_persons[best_match_id]
            person.embeddings.append(face.embedding)
            person.avg_embedding = self._average_embedding(person.embeddings)
            person.last_seen = face.frame_time
            person.frame_count += 1

            # Update thumbnail if this one is better (larger face)
            if len(face.thumbnail) > len(person.best_thumbnail):
                person.best_thumbnail = face.thumbnail

            return best_match_id
        else:
            # Create new track
            track_id = f"person_{self._next_track_id}"
            self._next_track_id += 1

            self._tracked_persons[track_id] = TrackedPerson(
                track_id=track_id,
                embeddings=[face.embedding],
                avg_embedding=face.embedding,
                best_thumbnail=face.thumbnail,
                first_seen=face.frame_time,
                last_seen=face.frame_time,
                frame_count=1
            )

            return track_id

    def _embedding_distance(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate Euclidean distance between embeddings."""
        return np.linalg.norm(np.array(emb1) - np.array(emb2))

    def _average_embedding(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate average of multiple embeddings."""
        return np.mean(embeddings, axis=0).tolist()

    def process_video_frames(
        self,
        frames: List[Tuple[float, np.ndarray]],
        sample_interval: float = 1.0
    ) -> Dict[str, TrackedPerson]:
        """
        Process video frames to detect and track all people.

        Args:
            frames: List of (timestamp, frame) tuples
            sample_interval: Sample frames every N seconds (for efficiency)

        Returns:
            Dictionary of track_id -> TrackedPerson
        """
        self._tracked_persons = {}
        self._next_track_id = 0

        last_sample_time = -sample_interval

        for timestamp, frame in frames:
            # Sample frames at interval
            if timestamp - last_sample_time < sample_interval:
                continue
            last_sample_time = timestamp

            # Detect faces
            faces = self.detect_faces(frame, timestamp)

            # Track each face
            for face in faces:
                self.track_person(face)

        return self._tracked_persons

    def get_tracked_persons(self) -> List[TrackedPerson]:
        """Get all tracked persons sorted by first appearance."""
        persons = list(self._tracked_persons.values())
        persons.sort(key=lambda p: p.first_seen)
        return persons

    def get_embedding_for_api(self, person: TrackedPerson) -> List[str]:
        """Convert embedding to string format for API."""
        return [str(x) for x in person.avg_embedding]

    def get_thumbnail_base64(self, person: TrackedPerson) -> str:
        """Get thumbnail as base64 string."""
        return base64.b64encode(person.best_thumbnail).decode('utf-8')

    def clear_tracks(self):
        """Clear all tracked persons."""
        self._tracked_persons = {}
        self._next_track_id = 0


def extract_faces_from_video(
    video_path: str,
    sample_interval: float = 1.0
) -> List[TrackedPerson]:
    """
    Extract all unique faces from a video file.

    Args:
        video_path: Path to video file
        sample_interval: Sample frames every N seconds

    Returns:
        List of TrackedPerson objects
    """
    if not FACE_RECOGNITION_AVAILABLE:
        print("[FaceService] face_recognition not available")
        return []

    service = FaceService()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[FaceService] Failed to open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[FaceService] Processing video: {frame_count} frames @ {fps} fps")

    # Read frames
    frames = []
    frame_idx = 0
    sample_frame_interval = int(fps * sample_interval)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample at interval
        if frame_idx % sample_frame_interval == 0:
            timestamp = frame_idx / fps
            frames.append((timestamp, frame))

        frame_idx += 1

    cap.release()

    print(f"[FaceService] Sampled {len(frames)} frames for face detection")

    # Process frames
    service.process_video_frames(frames, sample_interval=0)  # Already sampled

    persons = service.get_tracked_persons()
    print(f"[FaceService] Found {len(persons)} unique people")

    return persons
