# detection - Wake Word Detection Module

Handles wake word detection using openWakeWord with integrated Silero VAD.

## Components

### `wakeword.py` - WakeWordDetector

Wraps openWakeWord for wake word detection with voice activity detection.

**Key details:**
- Uses openWakeWord with built-in Silero VAD
- VAD reduces false positives from non-speech sounds
- Processes 80ms audio frames
- Returns (detected: bool, confidence: float)

**Usage:**
```python
detector = WakeWordDetector(
    model_path="hey_jarvis",  # Pre-trained model name
    threshold=0.5,            # Detection threshold (0-1)
    vad_threshold=0.5,        # VAD threshold
)

detected, confidence = detector.process_frame(audio_frame)
if detected:
    # Wake word was spoken!
    detector.reset()  # Prepare for next detection
```

## Available Wake Word Models

Pre-trained models (non-commercial license):
- `hey_jarvis`
- `alexa`
- `hey_mycroft`
- `ok_nabu`

Custom models can be trained - see openWakeWord documentation.

## Configuration

```yaml
wakeword:
  model: "hey_jarvis"    # Model name or path to .onnx file
  threshold: 0.5         # Higher = fewer false positives, may miss soft speech
  vad_threshold: 0.5     # VAD sensitivity
```

## How Detection Works

1. Audio frame (80ms) is fed to openWakeWord
2. Silero VAD confirms human speech is present
3. Wake word model checks for keyword
4. If confidence >= threshold AND VAD confirms speech → detection!
5. Model state is reset after detection

## Tuning Tips

- **Too many false positives:** Increase `threshold` (try 0.6-0.7)
- **Missing wake words:** Decrease `threshold` (try 0.3-0.4)
- **Background noise issues:** Increase `vad_threshold`
- **Speex noise suppression** is enabled by default on Linux
