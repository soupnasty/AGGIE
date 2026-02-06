"""Main daemon orchestrator for AGGIE voice assistant."""

import asyncio
import logging
import os
import signal
import time
from typing import AsyncIterator, Optional

import numpy as np

from .audio.buffer import AudioRingBuffer
from .audio.capture import AudioCapture
from .audio.cues import AudioCues, CueType
from .audio.playback import AudioPlayback
from .config import Config
from .context import ContextCompressor, ProjectState, PromptComposer
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
from .llm.sentence_buffer import sentences_from_stream
from .state import State, StateMachine
from .stt.whisper import SpeechToText, detect_gpu
from .tools import ToolRegistry
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

        # Tool registry + MCP manager
        self._tool_registry: Optional[ToolRegistry] = None
        self._mcp_manager = None
        if config.tools.enabled:
            if config.tools.mcp_servers:
                from .tools.mcp_manager import MCPManager
                self._mcp_manager = MCPManager()

            self._tool_registry = ToolRegistry(
                working_dir=os.path.expanduser(config.tools.working_dir),
                timeout=config.tools.timeout,
                max_output_chars=config.tools.max_output_chars,
                mcp_manager=self._mcp_manager,
            )

        # Three-tier context management
        self._project_state = ProjectState()
        self._compressor = ContextCompressor(
            state=self._project_state,
            compress_every=config.context.compress_every_turns,
            haiku_model=config.context.haiku_model,
            recent_window=config.context.recent_window,
        )
        self._composer = PromptComposer(state=self._project_state)

        # Voice phrases that trigger a full context clear
        self._clear_phrases = {
            "fresh start",
            "new conversation",
            "forget everything",
            "start over",
            "self destruct",
        }

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

        # Debug STT state (avoid transcribing AGGIE's own speech)
        self._debug_audio_buffer: list[np.ndarray] = []
        self._idle_since: float = 0.0

    def _on_state_change(self, old_state: State, new_state: State, context) -> None:
        """Sync state changes to the logging module for context injection."""
        set_current_state(new_state)
        if new_state == State.IDLE:
            self._idle_since = time.time()
            self._debug_audio_buffer.clear()

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
        """Ensure STT is initialized with model loaded."""
        if self._stt is None:
            self._stt = SpeechToText(
                model_size=self._config.stt.model_size,
                device=self._config.stt.device,
                compute_type=self._config.stt.compute_type,
                language=self._config.stt.language,
            )
            self._stt._ensure_model()
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
        """Ensure TTS is initialized with voice model loaded."""
        if self._tts is None:
            self._tts = TextToSpeech(
                voice_model=self._config.tts.voice_model,
                speaking_rate=self._config.tts.speaking_rate,
                use_cuda=self._config.tts.use_cuda,
            )
            self._tts._ensure_voice()
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
            ctx_status = self._project_state.get_status()
            return ContextStatusResponse(
                status=ResponseStatus.OK,
                turn_count=ctx_status["turn_count"],
                token_estimate=ctx_status["token_estimate"],
                silence_seconds=ctx_status["silence_seconds"],
                has_summary=ctx_status["has_history_summary"],
            )

        elif command.type == CommandType.CONTEXT_CLEAR:
            self._project_state.clear()
            return SimpleResponse(
                status=ResponseStatus.OK,
                message="Context cleared",
            )

        return SimpleResponse(
            status=ResponseStatus.ERROR,
            message="Unknown command",
            error=f"Unknown command type: {command.type}",
        )

    async def _process_recording(self) -> None:
        """Process recorded audio: STT -> LLM -> streaming TTS."""
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

            # Transcribe (batch - not streamable)
            stt = self._ensure_stt()
            transcript = stt.transcribe(audio, self._config.audio.sample_rate)

            if not transcript.strip():
                logger.info("Empty transcript, returning to idle")
                await self._state_machine.transition(State.IDLE)
                return

            logger.info(f"Transcript: {transcript}")

            # Check for context-clear voice command
            if self._is_clear_phrase(transcript):
                logger.info("Context clear phrase detected, resetting project state")
                self._project_state.clear()
                await self._speak_confirmation("Context cleared. Fresh start.")
                return

            # Add user turn to context
            self._project_state.add_turn("user", transcript)

            tts = self._ensure_tts()
            system_prompt, messages = self._composer.compose()
            await self._process_claude_response(tts, messages, system_prompt)

        except Exception as e:
            logger.error(f"Error processing recording: {e}", exc_info=True)
            await self._state_machine.force_transition(State.IDLE)

    async def _process_claude_response(
        self,
        tts: "TextToSpeech",
        messages: list[dict],
        system_prompt: Optional[str] = None,
    ) -> None:
        """Stream response from Claude through TTS."""
        llm = self._ensure_llm()
        try:
            await self._stream_response(llm, tts, messages, system_prompt)
        except APIError as e:
            logger.error(f"Claude API failed: {e}")
            await self._speak_error("Sorry, I'm running into issues with Claude.")

    TOOL_SYSTEM_PROMPT_ADDITION = (
        "\n\nYou have access to tools for interacting with the local machine. "
        "Use them when the user asks you to do something that requires local access "
        "(reading files, writing files, running code, git operations, etc.). "
        "Be resourceful — if your first approach doesn't work, try alternatives "
        "before giving up. Solve the problem, don't ask for clarification "
        "unless you've exhausted reasonable approaches. "
        "Keep spoken responses concise — summarize tool output rather than reading it verbatim."
    )

    async def _agent_text_stream(
        self,
        llm: ClaudeClient,
        messages: list[dict],
        system_prompt: Optional[str],
        tools: Optional[list[dict]],
        tool_registry: Optional[ToolRegistry],
        max_tokens: Optional[int],
        max_agent_turns: int,
    ) -> AsyncIterator[str]:
        """Async generator yielding text tokens across multiple API round trips.

        When Claude uses a tool, execution pauses text output, runs the command,
        sends results back, and resumes streaming. The downstream sentence buffer
        and TTS see a continuous token stream regardless of tool calls.

        Args:
            llm: Claude client.
            messages: Conversation messages.
            system_prompt: System prompt.
            tools: Tool definitions (None if tools disabled).
            tool_registry: Registry for executing tools.
            max_tokens: Max tokens per API call.
            max_agent_turns: Maximum tool-use round trips.

        Yields:
            Text tokens from Claude's response.
        """
        loop_messages = list(messages)

        for turn in range(max_agent_turns):
            async with llm.create_stream(
                loop_messages, system_prompt, tools, max_tokens
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final_message = await stream.get_final_message()

            if final_message.stop_reason != "tool_use":
                return

            # Yield a space so the sentence buffer can flush the last sentence
            # from this stream before the next API round begins
            yield " "

            # Build assistant content from the response
            assistant_content = []
            for block in final_message.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
            loop_messages.append({"role": "assistant", "content": assistant_content})

            # Execute tool calls and collect results
            tool_results = []
            for block in final_message.content:
                if block.type == "tool_use":
                    logger.info(f"Tool call: {block.name}({block.input})")
                    output = await tool_registry.execute(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        }
                    )
            loop_messages.append({"role": "user", "content": tool_results})

        # Max turns exhausted — make one final call WITHOUT tools
        # so Claude can summarize what it accomplished
        logger.warning(f"Agent loop hit max turns ({max_agent_turns}), requesting summary")
        async with llm.create_stream(
            loop_messages, system_prompt, tools=None, max_tokens=max_tokens
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_response(
        self,
        llm: ClaudeClient,
        tts: "TextToSpeech",
        messages: list[dict],
        system_prompt: Optional[str] = None,
    ) -> None:
        """Stream LLM response through TTS and playback.

        This creates a pipeline: Claude tokens -> sentences -> TTS -> playback
        Each stage runs concurrently for minimum latency.

        Args:
            llm: Claude client.
            tts: TTS engine.
            messages: Conversation messages to send to Claude.
            system_prompt: Optional system prompt override.
        """
        # Append tool instructions to system prompt when tools are available
        tools = self._tool_registry.get_tools() if self._tool_registry else None
        if tools and system_prompt:
            system_prompt += self.TOOL_SYSTEM_PROMPT_ADDITION
            # Append MCP server context files (e.g. Playwright design checklist)
            if self._mcp_manager:
                mcp_context = self._mcp_manager.get_context()
                if mcp_context:
                    system_prompt += "\n\n" + mcp_context

        # Accumulate full response for session context
        full_response: list[str] = []

        async def sentence_generator():
            """Generate sentences from Claude stream, accumulating full response."""
            if tools:
                max_tokens = self._config.tools.max_tokens
                token_stream = self._agent_text_stream(
                    llm,
                    messages,
                    system_prompt,
                    tools,
                    self._tool_registry,
                    max_tokens,
                    self._config.tools.max_agent_turns,
                )
            else:
                token_stream = llm.stream_response(messages, system_prompt=system_prompt)
            async for sentence in sentences_from_stream(token_stream):
                full_response.append(sentence)
                yield sentence

        # Transition to SPEAKING on first sentence
        first_sentence = True

        async def audio_generator():
            """Generate audio chunks, transitioning to SPEAKING on first chunk."""
            nonlocal first_sentence
            async for audio_data, sample_rate in tts.synthesize_stream(sentence_generator()):
                if first_sentence:
                    await self._state_machine.transition(State.SPEAKING)
                    first_sentence = False
                yield audio_data, sample_rate

        # Start streaming playback as background task for interrupt support
        self._playback_task = asyncio.create_task(
            self._streaming_playback_with_completion(audio_generator())
        )

        # Wait for playback to complete (or be interrupted)
        await self._playback_task

        # Save full response to context
        response_text = " ".join(full_response)
        if response_text.strip():
            self._project_state.add_turn("assistant", response_text)
            logger.info(f"Claude response ({len(full_response)} sentences): '{response_text[:80]}...'")
            # Fire-and-forget compression check
            self._compressor.maybe_trigger_compression()
        else:
            logger.warning("Empty LLM response")

    async def _streaming_playback_with_completion(
        self,
        audio_stream,
    ) -> None:
        """Play streaming audio and transition to IDLE when done.

        Args:
            audio_stream: Async iterator of (audio_data, sample_rate) tuples.
        """
        try:
            completed = await self._audio_playback.play_stream(audio_stream)
            if self._state_machine.state == State.SPEAKING:
                await self._state_machine.force_transition(State.IDLE)
            elif not completed:
                logger.debug("Streaming playback was interrupted")
        except Exception as e:
            logger.error(f"Error during streaming playback: {e}")
            if self._state_machine.state == State.SPEAKING:
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

    def _is_clear_phrase(self, transcript: str) -> bool:
        """Check if transcript matches a context-clear voice command."""
        normalized = transcript.strip().lower().rstrip(".!?")
        return normalized in self._clear_phrases

    async def _speak_confirmation(self, message: str) -> None:
        """Speak a short confirmation and return to IDLE."""
        try:
            await self._state_machine.transition(State.SPEAKING)
            tts = self._ensure_tts()
            audio_data, sample_rate = tts.synthesize(message)
            if len(audio_data) > 0:
                self._playback_task = asyncio.create_task(
                    self._play_with_completion(audio_data, sample_rate)
                )
            else:
                await self._state_machine.force_transition(State.IDLE)
        except Exception as e:
            logger.error(f"Failed to speak confirmation: {e}")
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

        # Load .env file before anything that needs env vars (e.g. MCP servers)
        from dotenv import load_dotenv
        env_path = os.path.expanduser("~/.config/aggie/.env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        # Preload models so first interaction isn't slow
        self._ensure_wakeword()
        self._ensure_stt()
        self._ensure_tts()

        # Set up direct audio callback for low-latency wake word detection
        self._audio_capture.set_frame_callback(self._audio_frame_callback)

        # Start components
        await self._ipc_server.start()
        await self._audio_capture.start()

        # Start MCP servers if configured
        if self._mcp_manager and self._config.tools.mcp_servers:
            try:
                await self._mcp_manager.start(self._config.tools.mcp_servers)
            except Exception as e:
                logger.warning(f"MCP manager failed to start: {e}")
                self._mcp_manager = None

        logger.info("AGGIE daemon ready - listening for wake word")

        import time as _time
        last_transcribe = _time.time()
        self._idle_since = _time.time()

        try:
            while not self._shutdown_event.is_set():
                state = self._state_machine.state

                # Process pending frames
                frames_to_process = []
                while self._pending_frames:
                    frames_to_process.append(self._pending_frames.pop(0))

                if state == State.IDLE and not self._muted:
                    # Cooldown: skip debug STT for 3s after entering IDLE
                    # to avoid transcribing AGGIE's own speech from speakers
                    now = _time.time()
                    idle_cooldown = (now - self._idle_since) < 3.0

                    # Process frames for wake word detection
                    for frame in frames_to_process:
                        detected, confidence = self._wakeword.process_frame(frame)

                        if not idle_cooldown:
                            self._debug_audio_buffer.append(frame.copy())

                        # Debug logging - periodically transcribe to show what's being said
                        now = _time.time()
                        if confidence > 0.1:
                            logger.info(f"[WAKEWORD] confidence={confidence:.3f} {'>>> DETECTED! <<<' if detected else ''}")
                        elif not idle_cooldown and now - last_transcribe >= 3.0 and len(self._debug_audio_buffer) > 0:
                            # Transcribe last 3 seconds of audio to show what's being spoken
                            last_transcribe = now
                            try:
                                stt = self._ensure_stt()
                                audio = np.concatenate(self._debug_audio_buffer)
                                transcript = stt.transcribe(audio, self._config.audio.sample_rate)
                                if transcript.strip():
                                    logger.info(f"[DEBUG STT] Heard: \"{transcript.strip()}\"")
                                else:
                                    logger.debug(f"[DEBUG STT] (silence or unintelligible)")
                            except Exception as e:
                                logger.warning(f"[DEBUG STT] Transcription failed: {e}")
                            self._debug_audio_buffer.clear()

                        if detected:
                            logger.info("Wake word detected! Starting to listen...")
                            await self._play_cue(CueType.WAKE)
                            self._recording_buffer = []
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

                        # Debug: log energy every 10 frames (~800ms)
                        if self._recording_frames % 10 == 0:
                            frame_duration_dbg = AudioCapture.FRAME_DURATION_MS / 1000.0
                            logger.debug(
                                f"[LISTENING] frame={self._recording_frames} "
                                f"energy={energy:.1f} threshold={silence_threshold:.1f} "
                                f"silence_frames={self._silence_frames} "
                                f"silence={self._silence_frames * frame_duration_dbg:.2f}s"
                            )

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
                            self._recording_buffer = []
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
            if self._mcp_manager:
                await self._mcp_manager.shutdown()
            await self._audio_capture.stop()
            await self._ipc_server.stop()
            logger.info("AGGIE daemon stopped")
