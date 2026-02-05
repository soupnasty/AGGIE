"""Main daemon orchestrator for AGGIE voice assistant."""

import asyncio
import logging
import signal
import time
from typing import Optional

import numpy as np

from .audio.buffer import AudioRingBuffer
from .audio.capture import AudioCapture
from .audio.cues import AudioCues, CueType
from .audio.playback import AudioPlayback
from .config import Config
from .context import SessionContext
from .detection.wakeword import WakeWordDetector
from .ipc.protocol import (
    Command,
    CommandType,
    ContextStatusResponse,
    DebugDumpResponse,
    ResponseStatus,
    SimpleResponse,
    StatusResponse,
)
from .ipc.server import IPCServer
from .llm.claude import APIError, ClaudeClient
from .state import State, StateMachine
from .stt.whisper import SpeechToText, detect_gpu
from .tts.piper import TextToSpeech
from .logging import set_current_state, get_debug_log_contents, get_debug_log_path

logger = logging.getLogger(__name__)


class AggieDaemon:
    """Main daemon orchestrating all voice assistant components.

    Manages the audio pipeline, wake word detection, STT, LLM,
    and TTS components. Handles IPC commands for control.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the daemon.

        Args:
            config: Configuration object.
        """
        self._config = config
        self._start_time = time.time()

        # State machine
        self._state_machine = StateMachine()
        self._state_machine.on_transition(self._on_state_change)
        self._muted = False

        # Audio components
        self._audio_capture = AudioCapture(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
        )
        self._audio_playback = AudioPlayback()
        self._audio_buffer = AudioRingBuffer(
            duration_seconds=config.audio.pre_roll_seconds,
            sample_rate=config.audio.sample_rate,
        )

        # Recording buffer for accumulating audio during LISTENING state
        self._recording_buffer: list[np.ndarray] = []
        self._recording_frames = 0
        self._silence_frames = 0

        # Processing components (lazy-loaded)
        self._wakeword: Optional[WakeWordDetector] = None
        self._stt: Optional[SpeechToText] = None
        self._llm: Optional[ClaudeClient] = None
        self._tts: Optional[TextToSpeech] = None

        # Audio cues (optional)
        self._audio_cues: Optional[AudioCues] = None
        if config.audio.cues_enabled:
            self._audio_cues = AudioCues()
            logger.info("Audio cues enabled")

        # Session context for multi-turn conversations
        self._session = SessionContext(
            soft_decay_minutes=config.session.soft_decay_minutes,
            hard_decay_minutes=config.session.hard_decay_minutes,
            max_session_tokens=config.session.max_session_tokens,
            summarizer_model=config.session.summarizer_model,
        )

        # IPC server
        self._ipc_server = IPCServer(
            socket_path=config.ipc.get_socket_path(),
            command_handler=self._handle_command,
        )

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()

        # Frame queue (filled by audio callback, processed by main loop)
        self._pending_frames: list[np.ndarray] = []
        self._frame_count = 0

        # Background playback task for interrupt support
        self._playback_task: Optional[asyncio.Task] = None

    def _on_state_change(self, old_state: State, new_state: State, context) -> None:
        """Sync state changes to the logging module for context injection."""
        set_current_state(new_state)

    def _audio_frame_callback(self, frame: np.ndarray) -> None:
        """Queue audio frames for processing in main thread."""
        self._frame_count += 1

        # Always buffer for pre-roll
        self._audio_buffer.append(frame)

        # Queue ALL frames for main thread processing
        self._pending_frames.append(frame.copy())

    def _ensure_wakeword(self) -> WakeWordDetector:
        """Ensure wake word detector is initialized."""
        if self._wakeword is None:
            self._wakeword = WakeWordDetector(
                model_path=self._config.wakeword.model,
                threshold=self._config.wakeword.threshold,
                vad_threshold=self._config.wakeword.vad_threshold,
            )
        return self._wakeword

    def _ensure_stt(self) -> SpeechToText:
        """Ensure STT is initialized."""
        if self._stt is None:
            self._stt = SpeechToText(
                model_size=self._config.stt.model_size,
                device=self._config.stt.device,
                compute_type=self._config.stt.compute_type,
                language=self._config.stt.language,
            )
        return self._stt

    def _ensure_llm(self) -> ClaudeClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = ClaudeClient(
                model=self._config.llm.model,
                max_tokens=self._config.llm.max_tokens,
                timeout=self._config.llm.timeout,
                max_retries=self._config.llm.max_retries,
            )
        return self._llm

    def _ensure_tts(self) -> TextToSpeech:
        """Ensure TTS is initialized."""
        if self._tts is None:
            self._tts = TextToSpeech(
                voice_model=self._config.tts.voice_model,
                speaking_rate=self._config.tts.speaking_rate,
                use_cuda=self._config.tts.use_cuda,
            )
        return self._tts

    async def _handle_command(self, command: Command):
        """Handle IPC commands.

        Args:
            command: Command to process.

        Returns:
            Response object.
        """
        if command.type == CommandType.STATUS:
            _, gpu_info = detect_gpu()
            return StatusResponse(
                status=ResponseStatus.OK,
                state=self._state_machine.state.name,
                muted=self._muted,
                uptime_seconds=time.time() - self._start_time,
                gpu=gpu_info,
            )

        elif command.type == CommandType.MUTE:
            self._muted = True
            await self._state_machine.transition(State.MUTED)
            self._recording_buffer = []
            return SimpleResponse(status=ResponseStatus.OK, message="Muted")

        elif command.type == CommandType.UNMUTE:
            self._muted = False
            await self._state_machine.transition(State.IDLE)
            return SimpleResponse(status=ResponseStatus.OK, message="Unmuted")

        elif command.type == CommandType.CANCEL:
            self._audio_playback.cancel()
            self._recording_buffer = []
            await self._state_machine.force_transition(State.IDLE)
            return SimpleResponse(status=ResponseStatus.OK, message="Cancelled")

        elif command.type == CommandType.SHUTDOWN:
            self._shutdown_event.set()
            return SimpleResponse(status=ResponseStatus.OK, message="Shutting down")

        elif command.type == CommandType.DEBUG_DUMP:
            log_path = get_debug_log_path()
            content = get_debug_log_contents(lines=200)
            line_count = content.count("\n")
            return DebugDumpResponse(
                status=ResponseStatus.OK,
                log_path=str(log_path),
                lines=line_count,
                content=content,
            )

        elif command.type == CommandType.CONTEXT_STATUS:
            ctx_status = self._session.get_status()
            return ContextStatusResponse(
                status=ResponseStatus.OK,
                turn_count=ctx_status["turn_count"],
                token_estimate=ctx_status["token_estimate"],
                silence_seconds=ctx_status["silence_seconds"],
                has_summary=ctx_status["has_summary"],
            )

        elif command.type == CommandType.CONTEXT_CLEAR:
            self._session.clear()
            return SimpleResponse(
                status=ResponseStatus.OK,
                message="Session context cleared",
            )

        return SimpleResponse(
            status=ResponseStatus.ERROR,
            message="Unknown command",
            error=f"Unknown command type: {command.type}",
        )

    async def _process_recording(self) -> None:
        """Process recorded audio: STT -> LLM -> TTS."""
        # Play "done listening" cue before processing
        await self._play_cue(CueType.DONE_LISTENING)
        await self._state_machine.transition(State.THINKING)

        try:
            # Concatenate all recorded audio
            if not self._recording_buffer:
                logger.warning("No audio recorded")
                await self._state_machine.transition(State.IDLE)
                return

            audio = np.concatenate(self._recording_buffer)
            self._recording_buffer = []
            self._recording_frames = 0

            # Transcribe
            stt = self._ensure_stt()
            transcript = stt.transcribe(audio, self._config.audio.sample_rate)

            if not transcript.strip():
                logger.info("Empty transcript, returning to idle")
                await self._state_machine.transition(State.IDLE)
                return

            logger.info(f"Transcript: {transcript}")

            # Check for context decay before processing
            self._session.check_decay()

            # Add user turn to session
            self._session.add_turn("user", transcript)

            # Get LLM response with full conversation context
            llm = self._ensure_llm()
            messages = self._session.build_messages()

            try:
                response_text = await llm.get_response(messages)
            except APIError as e:
                logger.error(f"Claude API failed: {e}")
                # Speak error message to user
                await self._speak_error("Sorry, I'm running into issues with Claude.")
                return

            if not response_text.strip():
                logger.warning("Empty LLM response")
                await self._state_machine.transition(State.IDLE)
                return

            # Add assistant turn to session
            self._session.add_turn("assistant", response_text)

            # Synthesize and play (non-blocking for interrupt support)
            await self._state_machine.transition(State.SPEAKING)
            tts = self._ensure_tts()
            audio_data, sample_rate = tts.synthesize(response_text)

            if len(audio_data) > 0:
                # Start playback as background task so main loop can detect interrupts
                self._playback_task = asyncio.create_task(
                    self._play_with_completion(audio_data, sample_rate)
                )
            else:
                await self._state_machine.force_transition(State.IDLE)

        except Exception as e:
            logger.error(f"Error processing recording: {e}", exc_info=True)
            await self._state_machine.force_transition(State.IDLE)

    async def _play_with_completion(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Play audio and transition to IDLE when done (unless interrupted).

        Args:
            audio_data: Audio samples to play.
            sample_rate: Sample rate of the audio.
        """
        try:
            completed = await self._audio_playback.play(audio_data, sample_rate)
            # Only transition to IDLE if still in SPEAKING state (not interrupted)
            if self._state_machine.state == State.SPEAKING:
                await self._state_machine.force_transition(State.IDLE)
            elif not completed:
                logger.debug("Playback was interrupted, not transitioning to IDLE")
        except Exception as e:
            logger.error(f"Error during playback: {e}")
            if self._state_machine.state == State.SPEAKING:
                await self._state_machine.force_transition(State.IDLE)

    async def _speak_error(self, message: str) -> None:
        """Speak an error message to the user."""
        try:
            await self._state_machine.transition(State.SPEAKING)
            tts = self._ensure_tts()
            audio_data, sample_rate = tts.synthesize(message)
            if len(audio_data) > 0:
                # Use background task for interrupt support
                self._playback_task = asyncio.create_task(
                    self._play_with_completion(audio_data, sample_rate)
                )
            else:
                await self._state_machine.force_transition(State.IDLE)
        except Exception as e:
            logger.error(f"Failed to speak error message: {e}")
            await self._state_machine.force_transition(State.IDLE)

    async def _play_cue(self, cue_type: CueType) -> None:
        """Play an audio feedback cue if enabled.

        Args:
            cue_type: Type of cue to play.
        """
        if self._audio_cues is None:
            return

        try:
            audio_data, sample_rate = self._audio_cues.get_cue(cue_type)
            if len(audio_data) > 0:
                await self._audio_playback.play(audio_data, sample_rate)
        except Exception as e:
            logger.warning(f"Failed to play audio cue: {e}")

    async def run(self) -> None:
        """Main daemon loop."""
        logger.info("Starting AGGIE daemon")

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        # Initialize wake word detector early
        self._ensure_wakeword()

        # Set up direct audio callback for low-latency wake word detection
        self._audio_capture.set_frame_callback(self._audio_frame_callback)

        # Start components
        await self._ipc_server.start()
        await self._audio_capture.start()

        logger.info("AGGIE daemon ready - listening for wake word")

        import time as _time
        last_log = _time.time()
        last_transcribe = _time.time()
        debug_audio_buffer: list[np.ndarray] = []

        try:
            while not self._shutdown_event.is_set():
                state = self._state_machine.state

                # Process pending frames
                frames_to_process = []
                while self._pending_frames:
                    frames_to_process.append(self._pending_frames.pop(0))

                if state == State.IDLE and not self._muted:
                    # Process frames for wake word detection
                    for frame in frames_to_process:
                        detected, confidence = self._wakeword.process_frame(frame)
                        debug_audio_buffer.append(frame.copy())

                        # Debug logging - periodically transcribe to show what's being said
                        now = _time.time()
                        if confidence > 0.1:
                            logger.info(f"[WAKEWORD] confidence={confidence:.3f} {'>>> DETECTED! <<<' if detected else ''}")
                        elif now - last_transcribe >= 3.0 and len(debug_audio_buffer) > 0:
                            # Transcribe last 3 seconds of audio to show what's being spoken
                            last_transcribe = now
                            try:
                                stt = self._ensure_stt()
                                audio = np.concatenate(debug_audio_buffer)
                                transcript = stt.transcribe(audio, self._config.audio.sample_rate)
                                if transcript.strip():
                                    logger.info(f"[DEBUG STT] Heard: \"{transcript.strip()}\"")
                                else:
                                    logger.debug(f"[DEBUG STT] (silence or unintelligible)")
                            except Exception as e:
                                logger.warning(f"[DEBUG STT] Transcription failed: {e}")
                            debug_audio_buffer.clear()
                            last_log = now

                        if detected:
                            logger.info("Wake word detected! Starting to listen...")
                            await self._play_cue(CueType.WAKE)
                            self._recording_buffer = [self._audio_buffer.get_all()]
                            self._silence_frames = 0
                            self._recording_frames = 0
                            await self._state_machine.transition(State.LISTENING)
                            break

                elif state == State.LISTENING:
                    # Accumulate recording
                    for frame in frames_to_process:
                        self._recording_buffer.append(frame)
                        self._recording_frames += 1

                        # Simple energy-based silence detection
                        energy = float(np.mean(frame.astype(np.float32) ** 2))
                        silence_threshold = self._config.audio.silence_threshold

                        if energy < silence_threshold:
                            self._silence_frames += 1
                        else:
                            self._silence_frames = 0

                    # Check for end conditions
                    frame_duration = AudioCapture.FRAME_DURATION_MS / 1000.0
                    silence_duration = self._silence_frames * frame_duration
                    recording_duration = self._recording_frames * frame_duration

                    if silence_duration >= self._config.audio.silence_duration:
                        logger.info(f"Silence detected ({silence_duration:.1f}s), processing speech")
                        await self._process_recording()
                    elif recording_duration >= self._config.audio.max_recording_duration:
                        logger.info(f"Max recording duration reached ({recording_duration:.1f}s)")
                        await self._process_recording()

                elif state == State.SPEAKING:
                    # Listen for wake word interrupt during playback
                    for frame in frames_to_process:
                        detected, confidence = self._wakeword.process_frame(frame)

                        if detected:
                            logger.info("Wake word detected during playback - interrupting!")
                            self._audio_playback.cancel()
                            await self._play_cue(CueType.WAKE)
                            self._recording_buffer = [self._audio_buffer.get_all()]
                            self._silence_frames = 0
                            self._recording_frames = 0
                            await self._state_machine.transition(State.LISTENING)
                            break

                # Small sleep to yield control
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)

        finally:
            # Cleanup
            logger.info("Shutting down AGGIE daemon")
            await self._audio_capture.stop()
            await self._ipc_server.stop()
            logger.info("AGGIE daemon stopped")
