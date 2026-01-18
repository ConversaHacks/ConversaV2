"""
Conversa - Main Application

Conversation recording system that automatically detects speech,
records video+audio, and validates using AI (Gemini).
"""

import signal
import sys
import time
import cv2

from .config import config
from .audio_capture import AudioCapture
from .action_service import ActionService, VADCommand
from .recording_service import RecordingService
from .transcription_service import TranscriptionService


class ConversaApp:
    """
    Main Conversa application.

    Orchestrates all services:
    - AudioCapture: Microphone input
    - ActionService: Voice activity detection
    - RecordingService: Video + audio recording
    - ValidationService: AI-powered validation
    """

    def __init__(self, show_preview: bool = True, webrtc_url: str = None):
        self.show_preview = show_preview
        self.webrtc_url = webrtc_url

        # Initialize components
        self.audio_capture = AudioCapture()
        self.recording_service = RecordingService(self.audio_capture, webrtc_url=webrtc_url)
        self.transcription_service = TranscriptionService(
            on_transcription=self._on_transcription
        )

        # ActionService will be initialized in start() after WebRTC is connected
        self.action_service = None

        # State
        self._is_running = False

    def _on_transcription(self, video_path: str, audio_path: str, segments: list):
        """Handle transcription result - this is where you add actions."""
        # segments is list of (speaker, start_s, end_s, text)
        # Combine all text for now
        full_transcript = " ".join(text for _, _, _, text in segments)
        print(f"[Conversa] Transcript ready: {full_transcript[:100]}...")

        # TODO: Add your actions here based on the transcript
        # e.g., send to another service, trigger commands, etc.

    def _on_vad_command(self, command: VADCommand):
        """Handle VAD commands."""
        if command == VADCommand.START:
            self.recording_service.start_recording()
        elif command == VADCommand.STOP:
            result = self.recording_service.stop_recording()
            if result:
                video_path, audio_path = result
                self.transcription_service.queue_for_transcription(video_path, audio_path)

    def start(self):
        """Start all services."""
        print("=" * 50)
        print("CONVERSA - Conversation Recording System")
        print("=" * 50)
        print()

        # Ensure output directories exist
        config.paths.ensure_directories()

        # Start services
        self.audio_capture.start()
        self.recording_service.start()

        # Initialize ActionService AFTER RecordingService has started
        # This ensures WebRTC capture is available if using WebRTC mode
        webrtc_capture = self.recording_service._webrtc_capture if self.webrtc_url else None

        self.action_service = ActionService(
            self.audio_capture,
            on_command=self._on_vad_command,
            webrtc_capture=webrtc_capture
        )
        self.action_service.start()

        self.transcription_service.start()

        self._is_running = True

        print()
        print("Services started. Listening for speech...")
        print("Keys: [r] start recording | [s] stop recording | [q] quit")
        print()

    def stop(self):
        """Stop all services."""
        print()
        print("Shutting down...")

        self._is_running = False

        # Stop services in reverse order
        if self.action_service:
            self.action_service.stop()
        self.recording_service.stop()
        self.audio_capture.stop()
        self.transcription_service.stop()

        # Close preview window
        cv2.destroyAllWindows()

        print("Shutdown complete.")

    def run(self):
        """Main run loop with preview window."""
        self.start()

        try:
            while self._is_running:
                if self.show_preview:
                    # Get and display preview frame
                    frame = self.recording_service.get_preview_frame()
                    if frame is not None:
                        # Add status overlay
                        frame = self._add_status_overlay(frame)
                        cv2.imshow("Conversa Preview", frame)

                    # Check for keys
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Manual start recording
                        if not self.recording_service.is_recording:
                            print("[Manual] Starting recording...")
                            self.recording_service.start_recording()
                    elif key == ord('s'):
                        # Manual stop recording
                        if self.recording_service.is_recording:
                            print("[Manual] Stopping recording...")
                            result = self.recording_service.stop_recording()
                            if result:
                                video_path, audio_path = result
                                self.transcription_service.queue_for_transcription(video_path, audio_path)
                else:
                    # No preview - just sleep
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.stop()

    def _add_status_overlay(self, frame):
        """Add status information overlay to frame."""
        # Recording indicator
        if self.recording_service.is_recording:
            # Red circle for recording
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            duration = self.recording_service.recording_duration
            text = f"REC {duration:.1f}s"
            cv2.putText(frame, text, (55, 38), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 2)
        else:
            # Green circle for standby
            cv2.circle(frame, (30, 30), 15, (0, 255, 0), -1)
            cv2.putText(frame, "STANDBY", (55, 38), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)

        # Speech indicator
        if self.action_service and self.action_service.is_speech_active:
            cv2.putText(frame, "SPEECH DETECTED", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Transcription queue
        queue_size = self.transcription_service.queue_size
        if queue_size > 0:
            cv2.putText(frame, f"Transcribing: {queue_size}",
                       (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Keybind hint
        cv2.putText(frame, "[r] rec | [s] stop | [q] quit",
                   (frame.shape[1] - 200, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Conversa - Conversation Recording System")
    parser.add_argument("--no-preview", action="store_true",
                       help="Run without preview window")
    parser.add_argument("--process-pending", action="store_true",
                       help="Process pending recordings and exit")
    parser.add_argument("--webrtc", type=str, metavar="URL",
                       help="WebRTC stream URL (default: from config)")

    args = parser.parse_args()

    if args.process_pending:
        # Process pending recordings only
        print("Processing pending recordings...")
        transcription_service = TranscriptionService()
        transcription_service.start()
        transcription_service.process_pending()

        # Wait for queue to empty
        while transcription_service.queue_size > 0:
            time.sleep(0.5)

        transcription_service.stop()
        print("Done.")
    else:
        # Always use WebRTC
        webrtc_url = args.webrtc or config.dev.webrtc_url
        print(f"[Mode] WebRTC: {webrtc_url}")

        app = ConversaApp(show_preview=not args.no_preview, webrtc_url=webrtc_url)

        # Handle signals
        def signal_handler(sig, frame):
            app.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        app.run()


if __name__ == "__main__":
    main()
