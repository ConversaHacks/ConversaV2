# Conversa

A conversation recording system that automatically detects speech and records video+audio, then validates recordings using AI (Gemini) to filter out false positives.

## Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  ActionService  │────▶│  RecordingService │────▶│  ValidationService │
│  (VAD Detection)│     │  (Video + Audio)  │     │  (Gemini API)      │
└─────────────────┘     └───────────────────┘     └────────────────────┘
        │                        │                         │
        ▼                        ▼                         ▼
   Microphone              Webcam + Mic              Gemini API
   (PyAudio)               (OpenCV)                  (HTTP POST)
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Gemini API key
```

### macOS Note
If you encounter issues installing PyAudio on macOS, install PortAudio first:
```bash
brew install portaudio
pip install pyaudio
```

## Usage

### Run the application
```bash
python main.py
```

### Run without preview window
```bash
python main.py --no-preview
```

### Process pending recordings only
```bash
python main.py --process-pending
```

## How It Works

1. **Voice Activity Detection (VAD)**: Continuously monitors microphone input, analyzing audio energy levels and zero-crossing rate to detect speech.

2. **Recording**: When speech is detected for 0.3s, recording starts (including 2s of pre-buffered audio and 15s of pre-buffered video). Recording stops after 3s of silence.

3. **Validation**: Recorded audio is sent to Gemini API to verify human voice presence. Recordings are moved to `approved/` or `rejected/` folders.

## Output Structure

```
output/
├── pending/    # Newly recorded, awaiting validation
├── approved/   # Validated as containing speech
└── rejected/   # Filtered out as noise/silence
```

## Configuration

Edit `conversa/config.py` to customize:

- **VAD Settings**: Energy threshold, speech confirm time, silence threshold
- **Recording Settings**: FPS, resolution, buffer durations
- **Gemini Settings**: API endpoint, model, proxy mode

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Gemini API key | (required) |
| `GEMINI_BASE_URL` | API base URL | `https://generativelanguage.googleapis.com` |
| `GEMINI_MODEL` | Model to use | `gemini-2.0-flash` |
| `GEMINI_USE_PROXY` | Use Anthropic format | `false` |
