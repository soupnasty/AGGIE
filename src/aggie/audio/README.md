# audio - Audio I/O Module

Handles all audio input and output for AGGIE.

## Components

### `capture.py` - AudioCapture
Continuous microphone capture using sounddevice.

**Key details:**
- Sample rate: 16kHz (required by wake word and STT)
- Frame size: 80ms (1280 samples) - optimal for openWakeWord
- Uses callback-based capture with asyncio queue for thread-safe transfer
- Async generator interface: `async for frame in capture.frames()`

**Usage:**
```python
capture = AudioCapture()
await capture.start()
async for frame in capture.frames():
    # frame is np.ndarray of int16
    process(frame)
await capture.stop()
```

### `playback.py` - AudioPlayback
Non-blocking audio playback with cancellation support.

**Key details:**
- Supports any sample rate (TTS typically outputs 22050Hz)
- Cancel mid-playback via `cancel()` method
- Returns `True` if completed, `False` if cancelled

**Usage:**
```python
playback = AudioPlayback()
completed = await playback.play(audio_data, sample_rate)
# From another task:
playback.cancel()  # Stops playback gracefully
```

### `buffer.py` - AudioRingBuffer
Ring buffer for pre-roll audio capture.

**Purpose:** Captures audio before wake word is fully detected, so the beginning of the user's utterance isn't lost.

**Key details:**
- Default: 1.5 seconds of pre-roll
- FIFO eviction when buffer is full
- Returns concatenated audio as single numpy array

**Usage:**
```python
buffer = AudioRingBuffer(duration_seconds=1.5)
buffer.append(frame)  # Called for every frame
pre_roll = buffer.get_all()  # Get buffered audio when wake word detected
```

## Audio Format

All audio in AGGIE uses:
- **Sample rate:** 16000 Hz
- **Channels:** 1 (mono)
- **Dtype:** int16 (PCM)
- **Frame duration:** 80ms (1280 samples)

Exception: TTS output may be different sample rate (typically 22050Hz), which is handled by the playback module.
