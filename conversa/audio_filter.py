"""
Audio Filter - Voice-preserving noise reduction

Filters audio to keep voices audible while removing static/background noise.
Uses bandpass filtering for voice frequencies and spectral subtraction.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import io


def bandpass_filter(audio: np.ndarray, sample_rate: int,
                    low_freq: int = 200, high_freq: int = 4000) -> np.ndarray:
    """
    Apply bandpass filter to isolate voice frequencies.

    Human voice typically 85-255Hz (fundamental) but harmonics go up to 4kHz.
    We use 200-4000Hz to capture voice while removing low rumble and high static.
    """
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Ensure frequencies are valid
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))

    # Design butterworth bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')

    # Apply filter
    filtered = signal.filtfilt(b, a, audio)
    return filtered


def reduce_noise_spectral(audio: np.ndarray, sample_rate: int,
                          noise_reduce_factor: float = 0.7) -> np.ndarray:
    """
    Simple spectral noise reduction.

    Estimates noise from the first 0.5s and subtracts it from the signal.
    Preserves voice by using soft thresholding.
    """
    # Use first 0.5 seconds as noise profile (or less if audio is short)
    noise_samples = min(int(sample_rate * 0.5), len(audio) // 4)
    if noise_samples < 100:
        return audio  # Too short to process

    # FFT parameters
    n_fft = 2048
    hop_length = n_fft // 4

    # Compute STFT
    f, t, stft = signal.stft(audio, sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Estimate noise spectrum from beginning
    noise_stft = signal.stft(audio[:noise_samples], sample_rate,
                             nperseg=n_fft, noverlap=n_fft - hop_length)[2]
    noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

    # Spectral subtraction with soft thresholding
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Subtract noise floor, but don't go below a small threshold
    reduced_magnitude = np.maximum(magnitude - noise_reduce_factor * noise_magnitude,
                                   0.1 * magnitude)

    # Reconstruct
    reduced_stft = reduced_magnitude * np.exp(1j * phase)
    _, filtered = signal.istft(reduced_stft, sample_rate,
                               nperseg=n_fft, noverlap=n_fft - hop_length)

    # Match original length
    if len(filtered) > len(audio):
        filtered = filtered[:len(audio)]
    elif len(filtered) < len(audio):
        filtered = np.pad(filtered, (0, len(audio) - len(filtered)))

    return filtered


def normalize_audio(audio: np.ndarray, target_level: float = 0.8) -> np.ndarray:
    """Normalize audio to target level without clipping."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio * (target_level / max_val)
    return audio


def filter_audio(audio_bytes: bytes, sample_rate: int = 48000,
                 apply_bandpass: bool = True,
                 apply_noise_reduction: bool = True,
                 normalize: bool = True) -> bytes:
    """
    Main function to filter audio for voice clarity.

    Args:
        audio_bytes: Raw audio as bytes (int16 format)
        sample_rate: Sample rate of the audio
        apply_bandpass: Apply voice-frequency bandpass filter
        apply_noise_reduction: Apply spectral noise reduction
        normalize: Normalize output level

    Returns:
        Filtered audio as bytes (int16 format)
    """
    # Convert bytes to numpy array
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

    # Normalize to [-1, 1] range for processing
    audio = audio / 32768.0

    # Apply filters
    if apply_bandpass:
        audio = bandpass_filter(audio, sample_rate)

    if apply_noise_reduction:
        audio = reduce_noise_spectral(audio, sample_rate)

    if normalize:
        audio = normalize_audio(audio)

    # Convert back to int16
    audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    return audio.tobytes()


def filter_wav_file(input_path: str, output_path: str = None,
                    apply_bandpass: bool = True,
                    apply_noise_reduction: bool = True) -> str:
    """
    Filter a WAV file and save the result.

    Args:
        input_path: Path to input WAV file
        output_path: Path for output (default: input with _filtered suffix)

    Returns:
        Path to filtered file
    """
    if output_path is None:
        output_path = input_path.replace('.wav', '_filtered.wav')

    # Read WAV file
    sample_rate, audio = wavfile.read(input_path)

    # Handle stereo by converting to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Apply filters
    if apply_bandpass:
        audio = bandpass_filter(audio, sample_rate)

    if apply_noise_reduction:
        audio = reduce_noise_spectral(audio, sample_rate)

    # Normalize
    audio = normalize_audio(audio)

    # Convert to int16 and save
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)

    return output_path
