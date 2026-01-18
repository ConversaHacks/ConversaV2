"""
WebRTC Capture - Remote Stream Handler

Connects to a go2rtc WebRTC stream using WHEP protocol.
"""

import asyncio
import threading
import time
from collections import deque
from typing import Optional, Callable
from urllib.parse import urlparse, parse_qs
import numpy as np

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
    import aiohttp
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


class WebRTCCapture:
    """
    WebRTC video capture from go2rtc using WHEP protocol.
    """

    def __init__(self, stream_url: str):
        if not WEBRTC_AVAILABLE:
            raise ImportError("aiortc and aiohttp are required for WebRTC. Install with: pip install aiortc aiohttp")

        self.stream_url = stream_url
        self._whep_url = self._build_whep_url(stream_url)
        self._pc: Optional[RTCPeerConnection] = None

        # Frame buffer
        self._frame_buffer: deque = deque(maxlen=30)
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # Audio buffer for recording
        self._audio_buffer: list = []
        self._audio_recording: bool = False
        self._audio_sample_rate: int = 48000  # WebRTC typically uses 48kHz
        self._audio_channels: int = 1
        self._audio_lock = threading.Lock()

        # VAD audio buffer (always captures latest chunk for analysis)
        self._vad_audio_chunk: bytes = b""
        self._vad_chunk_updated: bool = False

        # State
        self._is_running = False
        self._is_connected = False
        self._frame_count = 0
        self._audio_frame_count = 0

        # Async event loop
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        # Frame callback
        self._frame_callback: Optional[Callable[[np.ndarray], None]] = None

    def _build_whep_url(self, url: str) -> str:
        """Build WHEP endpoint URL from stream URL."""
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        src = params.get('src', ['cam'])[0]
        whep_url = f"{parsed.scheme}://{parsed.netloc}/api/webrtc?src={src}"
        print(f"[WebRTCCapture] WHEP URL: {whep_url}")
        return whep_url

    def start(self):
        """Start WebRTC capture in background thread."""
        if self._is_running:
            return

        self._is_running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

        # Wait for connection
        timeout = 15.0
        start = time.time()
        while not self._is_connected and time.time() - start < timeout:
            time.sleep(0.1)

        if self._is_connected:
            print(f"[WebRTCCapture] Connected to {self.stream_url}")
        else:
            print(f"[WebRTCCapture] Warning: Connection timeout")

    def stop(self):
        """Stop WebRTC capture."""
        self._is_running = False

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        print(f"[WebRTCCapture] Stopped (received {self._frame_count} frames)")

    def _run_async_loop(self):
        """Run the async event loop in a thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._connect())
            self._loop.run_forever()
        except Exception as e:
            print(f"[WebRTCCapture] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self._pc:
                self._loop.run_until_complete(self._pc.close())
            self._loop.close()

    async def _connect(self):
        """Connect to go2rtc WebRTC using WHEP."""
        # Configure with STUN server
        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        )
        self._pc = RTCPeerConnection(configuration=config)

        # Add transceivers for receiving media
        self._pc.addTransceiver("video", direction="recvonly")
        self._pc.addTransceiver("audio", direction="recvonly")

        @self._pc.on("track")
        def on_track(track):
            print(f"[WebRTCCapture] Received track: {track.kind}")
            if track.kind == "video":
                asyncio.ensure_future(self._receive_video(track))
            elif track.kind == "audio":
                asyncio.ensure_future(self._receive_audio(track))

        @self._pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = self._pc.connectionState
            print(f"[WebRTCCapture] Connection state: {state}")
            if state == "connected":
                self._is_connected = True
            elif state in ["failed", "closed", "disconnected"]:
                self._is_connected = False

        @self._pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"[WebRTCCapture] ICE state: {self._pc.iceConnectionState}")

        # Create and send offer
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        # POST SDP offer to go2rtc WHEP endpoint
        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/sdp"}
            print(f"[WebRTCCapture] Sending offer to {self._whep_url}")

            async with session.post(
                self._whep_url,
                data=self._pc.localDescription.sdp,
                headers=headers
            ) as response:
                print(f"[WebRTCCapture] Response status: {response.status}")

                if response.status in [200, 201]:
                    answer_sdp = await response.text()
                    print(f"[WebRTCCapture] Received answer ({len(answer_sdp)} bytes)")
                    await self._pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer_sdp, type="answer")
                    )
                else:
                    body = await response.text()
                    print(f"[WebRTCCapture] Error {response.status}: {body}")

    async def _receive_video(self, track):
        """Receive video frames from the track."""
        print("[WebRTCCapture] Starting video receive loop")
        while self._is_running:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                img = frame.to_ndarray(format="bgr24")

                self._frame_count += 1
                if self._frame_count % 30 == 0:
                    print(f"[WebRTCCapture] Received {self._frame_count} frames")

                with self._lock:
                    self._latest_frame = img.copy()
                    self._frame_buffer.append((time.time(), img.copy()))

                if self._frame_callback:
                    self._frame_callback(img)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self._is_running:
                    print(f"[WebRTCCapture] Frame error: {e}")
                break

    async def _receive_audio(self, track):
        """Receive audio frames from the track."""
        print("[WebRTCCapture] Starting audio receive loop")
        while self._is_running:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=1.0)

                # Update sample rate from actual frame
                self._audio_sample_rate = frame.sample_rate
                self._audio_channels = len(frame.layout.channels)

                self._audio_frame_count += 1
                if self._audio_frame_count % 100 == 0:
                    print(f"[WebRTCCapture] Received {self._audio_frame_count} audio frames "
                          f"(rate={self._audio_sample_rate}, channels={self._audio_channels})")

                # Get audio as s16 (16-bit signed int) format
                # frame.to_ndarray() returns shape (channels, samples) for planar formats
                audio_array = frame.to_ndarray()

                # Handle different array shapes
                if len(audio_array.shape) == 2:
                    # Planar format: (channels, samples) -> interleave to (samples, channels)
                    # Then flatten for mono or keep interleaved for stereo
                    if audio_array.shape[0] > 1:
                        # Multi-channel: transpose and flatten (interleave)
                        audio_array = audio_array.T.flatten()
                    else:
                        # Single channel: just flatten
                        audio_array = audio_array.flatten()

                # Convert to int16 if needed
                if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                    # Float audio is typically in range [-1.0, 1.0]
                    audio_array = np.clip(audio_array, -1.0, 1.0)
                    audio_array = (audio_array * 32767).astype(np.int16)
                elif audio_array.dtype != np.int16:
                    audio_array = audio_array.astype(np.int16)

                audio_bytes = audio_array.tobytes()

                with self._audio_lock:
                    # Always store latest chunk for VAD analysis
                    self._vad_audio_chunk = audio_bytes
                    self._vad_chunk_updated = True

                    # Store for recording if active
                    if self._audio_recording:
                        self._audio_buffer.append(audio_bytes)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self._is_running:
                    print(f"[WebRTCCapture] Audio error: {e}")
                    import traceback
                    traceback.print_exc()
                break

    def start_audio_recording(self):
        """Start recording audio from WebRTC stream."""
        with self._audio_lock:
            self._audio_buffer = []
            self._audio_recording = True
            print("[WebRTCCapture] Audio recording started")

    def stop_audio_recording(self) -> bytes:
        """Stop recording and return audio data as bytes."""
        with self._audio_lock:
            self._audio_recording = False
            audio_data = b"".join(self._audio_buffer)
            self._audio_buffer = []
            print(f"[WebRTCCapture] Audio recording stopped ({len(audio_data)} bytes)")
            return audio_data

    @property
    def audio_sample_rate(self) -> int:
        return self._audio_sample_rate

    @property
    def audio_channels(self) -> int:
        return self._audio_channels

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest video frame."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_buffered_frames(self) -> list:
        """Get all buffered frames with timestamps."""
        with self._lock:
            return list(self._frame_buffer)

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for new frames."""
        self._frame_callback = callback

    def get_vad_audio_chunk(self) -> Optional[bytes]:
        """Get the latest audio chunk for VAD analysis. Returns None if no new data."""
        with self._audio_lock:
            if self._vad_chunk_updated:
                self._vad_chunk_updated = False
                return self._vad_audio_chunk
            return None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_running(self) -> bool:
        return self._is_running