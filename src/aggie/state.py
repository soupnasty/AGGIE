"""State machine for AGGIE voice assistant."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class State(Enum):
    """Voice assistant operational states."""

    IDLE = auto()  # Listening for wake word only
    LISTENING = auto()  # Recording user speech after wake word
    THINKING = auto()  # Processing STT + LLM request
    SPEAKING = auto()  # Playing TTS response
    MUTED = auto()  # All audio processing paused


@dataclass
class StateContext:
    """Context data carried between state transitions."""

    audio_buffer: Optional[bytes] = None
    transcript: Optional[str] = None
    response_text: Optional[str] = None
    cancel_requested: bool = False


# Type alias for state transition callbacks
StateCallback = Callable[[State, State, StateContext], None]


class StateMachine:
    """Manages state transitions for the voice assistant.

    Thread-safe state machine with async transition support and
    listener callbacks for state changes.
    """

    # Valid state transitions
    VALID_TRANSITIONS: dict[State, set[State]] = {
        State.IDLE: {State.LISTENING, State.MUTED},
        State.LISTENING: {State.THINKING, State.IDLE, State.MUTED},
        State.THINKING: {State.SPEAKING, State.IDLE, State.MUTED},
        State.SPEAKING: {State.IDLE, State.MUTED},
        State.MUTED: {State.IDLE},
    }

    def __init__(self) -> None:
        self._state = State.IDLE
        self._context = StateContext()
        self._listeners: list[StateCallback] = []
        self._lock = asyncio.Lock()

    @property
    def state(self) -> State:
        """Current state (read-only)."""
        return self._state

    @property
    def context(self) -> StateContext:
        """Current context (read-only)."""
        return self._context

    @property
    def is_active(self) -> bool:
        """True if assistant is actively processing (not idle or muted)."""
        return self._state in {State.LISTENING, State.THINKING, State.SPEAKING}

    async def transition(
        self,
        new_state: State,
        context: Optional[StateContext] = None,
    ) -> bool:
        """Transition to a new state.

        Args:
            new_state: Target state to transition to.
            context: Optional new context data.

        Returns:
            True if transition succeeded, False if invalid transition.
        """
        async with self._lock:
            old_state = self._state

            # Check if transition is valid
            if new_state not in self.VALID_TRANSITIONS.get(old_state, set()):
                logger.warning(
                    f"Invalid state transition: {old_state.name} -> {new_state.name}"
                )
                return False

            # Update state and context
            self._state = new_state
            if context is not None:
                self._context = context

            logger.info(f"State transition: {old_state.name} -> {new_state.name}")

            # Notify listeners
            for listener in self._listeners:
                try:
                    result = listener(old_state, new_state, self._context)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in state listener: {e}")

            return True

    async def force_transition(
        self,
        new_state: State,
        context: Optional[StateContext] = None,
    ) -> None:
        """Force a transition to any state (for error recovery).

        Args:
            new_state: Target state.
            context: Optional new context data.
        """
        async with self._lock:
            old_state = self._state
            self._state = new_state
            if context is not None:
                self._context = context

            logger.warning(f"Forced state transition: {old_state.name} -> {new_state.name}")

            for listener in self._listeners:
                try:
                    result = listener(old_state, new_state, self._context)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in state listener: {e}")

    def on_transition(self, callback: StateCallback) -> None:
        """Register a callback for state transitions.

        Args:
            callback: Function called with (old_state, new_state, context).
        """
        self._listeners.append(callback)

    def reset_context(self) -> None:
        """Reset context to default values."""
        self._context = StateContext()
