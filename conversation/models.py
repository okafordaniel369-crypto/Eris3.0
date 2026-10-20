"""Shared data structures for Eris 3.0.

This module is the common language the components use to talk to each other.
It holds data shapes and small, predictable validation only. It has no NLU,
memory, personality or security logic, and it imports nothing from Eris.

Notes on the design:

* Every structure is a frozen dataclass. Collections are converted to tuples
  (and mappings to read-only views) when an object is created, so a list
  passed in cannot be changed afterwards.
* A frozen dataclass does NOT make authentication secure. AuthState describes
  an outcome; only security.py may be trusted to create an authenticated one.
  Identity, session identity and authentication are deliberately separate types.
* Controlled values (intents, actions, auth levels, ...) are enums, not strings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Mapping, Optional


# --------------------------------------------------------------------------
# Controlled values
# --------------------------------------------------------------------------

class Intent(Enum):
    """Broad purpose of a user message."""
    UNKNOWN = auto()
    GENERAL_QUESTION = auto()
    FACTUAL_QUESTION = auto()
    INSTRUCTION = auto()
    REQUEST = auto()
    GREETING = auto()
    FAREWELL = auto()
    THANKS = auto()
    APOLOGY = auto()
    CORRECTION = auto()
    OPINION = auto()
    PERSONAL_INFORMATION = auto()
    MEMORY_RECALL = auto()
    CAPABILITY_QUESTION = auto()
    IDENTITY_QUESTION = auto()
    EMOTIONAL_DISCLOSURE = auto()
    CLARIFICATION_RESPONSE = auto()


class DialogueAct(Enum):
    """Conversational function of a message (not the same as intent)."""
    UNKNOWN = auto()
    QUESTION = auto()
    STATEMENT = auto()
    GREETING = auto()
    FAREWELL = auto()
    REQUEST = auto()
    COMMAND = auto()
    THANKING = auto()
    APOLOGY = auto()
    CORRECTION = auto()
    CONFIRMATION = auto()
    DENIAL = auto()
    CLARIFICATION = auto()
    EXPRESSION_OF_EMOTION = auto()


class Understanding(Enum):
    """How well the analyzer believes it understood a message.

    Lets Brain tell understanding from guessing from not knowing.
    """
    UNDERSTOOD = auto()
    GUESSED = auto()
    NOT_UNDERSTOOD = auto()


class Sentiment(Enum):
    UNKNOWN = auto()
    NEGATIVE = auto()
    NEUTRAL = auto()
    POSITIVE = auto()
    MIXED = auto()


class EntityKind(Enum):
    PERSON = auto()
    PLACE = auto()
    TIME = auto()
    NUMBER = auto()
    TOPIC = auto()
    SELF_REFERENCE = auto()  # text is "assistant", "speaker" or "creator"
    OTHER = auto()


class ActionKind(Enum):
    """What Brain decided Eris should do."""
    UNKNOWN = auto()
    ANSWER = auto()
    CLARIFY = auto()
    RECALL = auto()
    ACKNOWLEDGE_CORRECTION = auto()
    ADMIT_UNCERTAINTY = auto()
    GREET = auto()
    FAREWELL = auto()
    ACKNOWLEDGE = auto()
    EMPATHIZE = auto()
    REFUSE = auto()
    REQUEST_AUTHENTICATION = auto()
    CONFIRM = auto()
    EXPLAIN = auto()


class StyleHint(Enum):
    """Delivery hints from Brain to Personality. They never change meaning."""
    SERIOUS = auto()
    PLAYFUL = auto()
    GENTLE = auto()
    CONCISE = auto()
    DETAILED = auto()
    FORMAL = auto()
    CASUAL = auto()


class Role(Enum):
    ASSISTANT = auto()
    OWNER = auto()
    USER = auto()
    MENTIONED = auto()


class Speaker(Enum):
    USER = auto()
    ERIS = auto()


class MemoryCategory(Enum):
    OWNER_PROFILE = auto()
    CONVERSATION_HISTORY = auto()
    OTHER_PEOPLE = auto()
    PROJECTS_KNOWLEDGE = auto()
    LEARNING = auto()


class AuthLevel(Enum):
    NONE = auto()
    OWNER = auto()


class AuthMethod(Enum):
    NONE = auto()
    PASSCODE = auto()
    RECOVERY = auto()


# --------------------------------------------------------------------------
# Small validation helpers (kept deliberately simple)
# --------------------------------------------------------------------------

def _text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_text(name: str, value: Any) -> Optional[str]:
    return None if value is None else _text(name, value)


def _unit(name: str, value: Any) -> float:
    """A number from 0.0 to 1.0 inclusive. Booleans and NaN are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    return float(value)


def _enum(name: str, value: Any, cls: type) -> Any:
    if not isinstance(value, cls):
        raise TypeError(f"{name} must be a {cls.__name__}, got {value!r}")
    return value


def _bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True or False, got {value!r}")
    return value


def _tuple_of(name: str, value: Any, item_type: type) -> tuple:
    """Accept a list or tuple of item_type and return it as a tuple."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, item_type):
            raise TypeError(f"{name} items must be {item_type.__name__}, got {item!r}")
    return tuple(value)


def _freeze(value: Any) -> Any:
    """Recursively make structured content read-only."""
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"content keys must be strings, got {key!r}")
            frozen[key] = _freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool, Enum)):
        return value
    raise TypeError(f"unsupported content value: {type(value).__name__}")


def _set(obj: Any, name: str, value: Any) -> None:
    """Assign during __post_init__ on a frozen dataclass."""
    object.__setattr__(obj, name, value)


# --------------------------------------------------------------------------
# Language analysis (produced by nlu.py)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    """Something found in a message. span is (start, end) in the raw text."""
    text: str
    kind: EntityKind
    span: Optional[tuple] = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _text("text", self.text)
        _enum("kind", self.kind, EntityKind)
        _set(self, "confidence", _unit("confidence", self.confidence))
        if self.span is not None:
            span = _tuple_of("span", self.span, int)
            if len(span) != 2 or any(isinstance(p, bool) for p in span):
                raise TypeError("span must be two integer positions (start, end)")
            start, end = span
            if start < 0 or end <= start:
                raise ValueError(f"invalid span {span!r}: need 0 <= start < end")
            _set(self, "span", span)


@dataclass(frozen=True)
class Analysis:
    """Structured interpretation of one user message.

    `analyzer` names the method that produced it (for example "rules-v1") so
    nothing downstream mistakes a rule-based guess for real understanding.
    """
    raw_text: str
    normalized_text: str
    intent: Intent = Intent.UNKNOWN
    dialogue_act: DialogueAct = DialogueAct.UNKNOWN
    entities: tuple = ()
    sentiment: Sentiment = Sentiment.UNKNOWN
    emotion_cues: tuple = ()
    confidence: float = 0.0
    understanding: Understanding = Understanding.NOT_UNDERSTOOD
    ambiguities: tuple = ()
    references: tuple = ()
    analyzer: str = "unspecified"

    def __post_init__(self) -> None:
        _text("raw_text", self.raw_text)
        _text("normalized_text", self.normalized_text)
        _enum("intent", self.intent, Intent)
        _enum("dialogue_act", self.dialogue_act, DialogueAct)
        _enum("sentiment", self.sentiment, Sentiment)
        _enum("understanding", self.understanding, Understanding)
        _text("analyzer", self.analyzer)
        _set(self, "confidence", _unit("confidence", self.confidence))
        _set(self, "entities", _tuple_of("entities", self.entities, Entity))
        _set(self, "emotion_cues", _tuple_of("emotion_cues", self.emotion_cues, str))
        _set(self, "ambiguities", _tuple_of("ambiguities", self.ambiguities, str))
        _set(self, "references", _tuple_of("references", self.references, str))
        if self.understanding is Understanding.UNDERSTOOD and self.intent is Intent.UNKNOWN:
            raise ValueError("an UNKNOWN intent cannot be marked UNDERSTOOD")
        for entity in self.entities:
            if entity.span is not None and entity.span[1] > len(self.raw_text):
                raise ValueError(f"entity span {entity.span!r} is outside the message text")


# --------------------------------------------------------------------------
# Session state (managed by conversation.py, never persistent memory)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    """One message in the current session. timestamp is epoch seconds."""
    speaker: Speaker
    text: str
    timestamp: float
    analysis: Optional[Analysis] = None

    def __post_init__(self) -> None:
        _enum("speaker", self.speaker, Speaker)
        _text("text", self.text)
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be a number of seconds")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError(f"invalid timestamp {self.timestamp!r}")
        _set(self, "timestamp", float(self.timestamp))
        if self.analysis is not None:
            _enum("analysis", self.analysis, Analysis)


@dataclass(frozen=True)
class Context:
    """Immediate conversation state for the current session."""
    recent_turns: tuple = ()
    current_topic: Optional[str] = None
    pending_question: Optional[str] = None
    last_action: Optional[ActionKind] = None

    def __post_init__(self) -> None:
        _set(self, "recent_turns", _tuple_of("recent_turns", self.recent_turns, Turn))
        _optional_text("current_topic", self.current_topic)
        _optional_text("pending_question", self.pending_question)
        if self.last_action is not None:
            _enum("last_action", self.last_action, ActionKind)


# --------------------------------------------------------------------------
# Identity (ownership.py) and authentication (security.py) are separate
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Identity:
    """A person known to Eris. `claimed` is what they said about themselves.

    Having an Identity, even with role OWNER, grants no privileges.
    """
    person_id: str
    display_name: str
    role: Role
    claimed: Optional[str] = None

    def __post_init__(self) -> None:
        _text("person_id", self.person_id)
        _text("display_name", self.display_name)
        _enum("role", self.role, Role)
        _optional_text("claimed", self.claimed)


@dataclass(frozen=True)
class SessionIdentity:
    """Who Eris believes she is talking to in this session.

    `established` means the identity is settled for conversation purposes. It
    is not authentication. Only OWNER or USER identities can be the speaker;
    a merely mentioned person never becomes a participant by being mentioned.
    """
    identity: Optional[Identity] = None
    established: bool = False

    def __post_init__(self) -> None:
        _bool("established", self.established)
        if self.identity is not None:
            _enum("identity", self.identity, Identity)
            if self.identity.role not in (Role.OWNER, Role.USER):
                raise ValueError("a session identity must have role OWNER or USER")
        if self.established and self.identity is None:
            raise ValueError("an established session needs an identity")


@dataclass(frozen=True)
class AuthState:
    """Outcome of authentication. The default is NOT authenticated.

    Only security.py should create an authenticated state. This class only
    keeps the fields consistent; it cannot stop other code from constructing
    one, so Core must always take it from the security subsystem.
    """
    authenticated: bool = False
    level: AuthLevel = AuthLevel.NONE
    method: AuthMethod = AuthMethod.NONE

    def __post_init__(self) -> None:
        _bool("authenticated", self.authenticated)
        _enum("level", self.level, AuthLevel)
        _enum("method", self.method, AuthMethod)
        granted = self.level is not AuthLevel.NONE and self.method is not AuthMethod.NONE
        if self.authenticated != granted:
            raise ValueError("authenticated, level and method are inconsistent")


# --------------------------------------------------------------------------
# Memory boundary (memory_search.py / memory_manager.py)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryRequest:
    """What Brain asks memory for. category None means search all categories."""
    query: str
    category: Optional[MemoryCategory] = None
    limit: int = 5

    def __post_init__(self) -> None:
        _text("query", self.query)
        if self.category is not None:
            _enum("category", self.category, MemoryCategory)
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("limit must be an integer")
        if self.limit < 1:
            raise ValueError("limit must be at least 1")


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved memory.

    is_verbatim must be stated explicitly: True only if `text` is an exact
    stored passage, False for a summary or reconstruction. Receiving a hit does
    not authorize revealing it; access control belongs to memory and security.
    """
    record_id: str
    category: MemoryCategory
    text: str
    score: float
    is_verbatim: bool

    def __post_init__(self) -> None:
        _text("record_id", self.record_id)
        _enum("category", self.category, MemoryCategory)
        _text("text", self.text)
        _set(self, "score", _unit("score", self.score))
        _bool("is_verbatim", self.is_verbatim)


# --------------------------------------------------------------------------
# Decisions and replies (brain.py, personality.py)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """What Brain decided to do.

    content is structured meaning, not final wording: for example keys such as
    "definition", "example" or "uncertainty", so Personality can express each
    part. It is made read-only on creation. needs_auth marks that the action
    requires authorization; it never grants it.
    """
    kind: ActionKind
    content: Mapping = field(default_factory=dict)
    needs_memory: Optional[MemoryRequest] = None
    needs_auth: bool = False
    style_hints: tuple = ()

    def __post_init__(self) -> None:
        _enum("kind", self.kind, ActionKind)
        if not isinstance(self.content, Mapping):
            raise TypeError("content must be a mapping")
        _set(self, "content", _freeze(self.content))
        if self.needs_memory is not None:
            _enum("needs_memory", self.needs_memory, MemoryRequest)
        _bool("needs_auth", self.needs_auth)
        _set(self, "style_hints", _tuple_of("style_hints", self.style_hints, StyleHint))


@dataclass(frozen=True)
class Reply:
    """Final text plus the decision that produced it."""
    text: str
    action: Action

    def __post_init__(self) -> None:
        _text("text", self.text)
        _enum("action", self.action, Action)
