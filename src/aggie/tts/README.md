# tts - Text-to-Speech Module

Handles local speech synthesis using Piper TTS.

## Components

### `piper.py` - TextToSpeech

Wraps Piper for fast, high-quality local speech synthesis.

**Key details:**
- Uses Piper neural TTS models
- Lazy-loads voice model on first use
- Outputs int16 audio at model's native sample rate (typically 22050Hz)
- Falls back to CLI if Python bindings fail

**Usage:**
```python
tts = TextToSpeech(
    voice_model="en_US-lessac-medium",
    speaking_rate=1.0,
)

audio_data, sample_rate = tts.synthesize("Hello, how can I help?")
# audio_data is np.ndarray of int16
# sample_rate is typically 22050
```

## Voice Models

Piper voices are named: `{language}_{region}-{name}-{quality}`

Examples:
- `en_US-lessac-medium` - US English, Lessac voice, medium quality
- `en_GB-alba-medium` - British English, Alba voice
- `de_DE-thorsten-high` - German, Thorsten voice, high quality

**Quality levels:**
- `low` - Fastest, smaller model
- `medium` - Good balance (recommended)
- `high` - Best quality, slower

## Configuration

```yaml
tts:
  voice_model: "en_US-lessac-medium"
  speaking_rate: 1.0    # 0.5 = slow, 2.0 = fast
  use_cuda: false       # GPU acceleration
```

## Voice Installation

Voices are downloaded automatically on first use, or manually:

```bash
# List available voices
piper --list-voices

# Download a specific voice
piper --download-voice en_US-lessac-medium
```

Voices are stored in `~/.local/share/piper-voices/`.

## Output Format

- **Sample rate:** Model-dependent (usually 22050Hz)
- **Channels:** 1 (mono)
- **Dtype:** int16

The playback module handles the different sample rate automatically.

## Performance

- First synthesis is slow (model loading)
- Subsequent synthesis is near real-time
- Medium voices: ~100-200ms for a sentence on modern CPU
- Consider `use_cuda: true` for faster synthesis with GPU
