# stt - Speech-to-Text Module

Handles local speech transcription using faster-whisper.

## Components

### `whisper.py` - SpeechToText

Wraps faster-whisper for efficient local transcription.

**Key details:**
- Model loads on-demand (lazy loading) to save memory when idle
- Automatic device selection (CUDA if available, else CPU)
- Uses VAD filter for better transcription quality
- Expects 16kHz audio input

**Usage:**
```python
stt = SpeechToText(
    model_size="small",   # tiny, base, small, medium, large-v3
    device="auto",        # auto, cpu, cuda
    language="en",
)

transcript = stt.transcribe(audio_array, sample_rate=16000)
# transcript is a string

stt.unload_model()  # Free memory when not needed
```

## Model Sizes

| Model | Disk | RAM | Speed | Accuracy |
|-------|------|-----|-------|----------|
| tiny | 75 MB | ~273 MB | Fastest | Basic |
| base | 142 MB | ~388 MB | Fast | Good |
| small | 466 MB | ~852 MB | Medium | Better |
| medium | 1.5 GB | ~2.1 GB | Slow | Great |
| large-v3 | 2.9 GB | ~3.9 GB | Slowest | Best |

**Recommendation:** Use `small` for desktop workstations. It offers a good balance of speed and accuracy.

## Configuration

```yaml
stt:
  model_size: "small"    # Model to use
  device: "auto"         # auto, cpu, cuda
  compute_type: "auto"   # auto, int8, float16
  language: "en"         # Language code
```

## Performance Notes

- **First transcription** is slow (model loading)
- **Subsequent transcriptions** are fast
- Model stays loaded between transcriptions
- Call `unload_model()` to free ~500MB-2GB RAM

## Why faster-whisper?

- ~2x faster than whisper.cpp on CPU
- Better timestamp accuracy
- CTranslate2 backend with int8 quantization
- Python-native, easy to integrate
