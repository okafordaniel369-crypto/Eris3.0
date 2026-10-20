"""ownership.py - who is who: the assistant, the canonical owner, the session
user, and other people mentioned.

Ownership answers identity questions only. It does not authenticate or
authorize anyone, and it does not depend on security.py. A name or alias is a
claim, never proof: claiming an owner alias yields the owner identity marked
`established=False`, and Core decides when a session is established through
the verification process. Profiles and protected data belong to memory.
"""
from __future__ import annotations

import re
import uuid
from typing import Callable, Iterable, Mapping

from conversation.models import Identity, Role, SessionIdentity

# Aliases are configuration, supplied to the constructor. These defaults are
# for convenience; Core can later pass values loaded from the owner profile.
DEFAULT_OWNER_NAME = "Master Shadow"
DEFAULT_OWNER_ALIASES = ("Daniel", "Master Shadow", "Shadow")
MAX_NAME_LENGTH = 60


def _key(name: str) -> str:
    """Comparison form of a name: case, spacing and stray punctuation ignored."""
    if not isinstance(name, str):
        raise TypeError("a name must be a string")
    cleaned = re.sub(r"[^\w' -]", "", name).casefold()
    return " ".join(cleaned.split())


def _clean_display(name: str) -> str:
    key = _key(name)
    if not key:
        raise ValueError("a name must not be empty")
    if len(key) > MAX_NAME_LENGTH:
        raise ValueError("that name is too long")
    return " ".join(re.sub(r"[^\w' -]", "", name).split())


def _new_id(name: str) -> str:
    return "user-" + uuid.uuid4().hex[:12]


class Ownership:
    """Resolves names to identities. Pure logic: no files, no authentication."""

    def __init__(self, canonical_name: str = DEFAULT_OWNER_NAME,
                 aliases: Iterable[str] = DEFAULT_OWNER_ALIASES,
                 assistant_name: str = "Eris", owner_id: str = "owner",
                 id_factory: Callable[[str], str] = _new_id) -> None:
        self._owner_name = _clean_display(canonical_name)
        self._assistant_name = _clean_display(assistant_name)
        self._owner_id = owner_id
        self._id_factory = id_factory
        self._owner_keys = {_key(self._owner_name)} | {_key(a) for a in aliases if _key(a)}
        if _key(self._assistant_name) in self._owner_keys:
            raise ValueError("an owner alias cannot be the assistant's own name")

    @classmethod
    def from_profile(cls, profile: Mapping, **kwargs) -> "Ownership":
        """Build from owner-profile data, e.g. {"name": ..., "aliases": [...]}.

        Lets Core load aliases from a profile or configuration source later
        without changing this module.
        """
        return cls(canonical_name=profile["name"], aliases=profile.get("aliases", ()), **kwargs)

    # -- the fixed identities ------------------------------------------------

    def assistant_identity(self) -> Identity:
        return Identity("assistant", self._assistant_name, Role.ASSISTANT)

    def owner_identity(self) -> Identity:
        """The one canonical owner, whichever alias is used for them."""
        return Identity(self._owner_id, self._owner_name, Role.OWNER)

    def is_owner(self, identity: Identity) -> bool:
        """True if this identity IS the canonical owner (an identity comparison).

        This is not authentication or authorization and grants nothing: a
        claimed, unverified owner identity is also "the owner" by this test.
        """
        if not isinstance(identity, Identity):
            raise TypeError("is_owner needs an Identity")
        return identity.person_id == self._owner_id

    # -- names that appear in conversation -----------------------------------

    def mentioned(self, name: str) -> Identity:
        """Who a name in someone's message refers to (not who is speaking).

        Owner aliases resolve to the canonical owner; the assistant's name to
        the assistant; anyone else is a MENTIONED person, never a participant.
        A name alone cannot tell two people with the same alias apart.
        """
        display = _clean_display(name)
        if _key(display) in self._owner_keys:
            return self.owner_identity()
        if _key(display) == _key(self._assistant_name):
            return self.assistant_identity()
        return Identity(self._id_factory(display), display, Role.MENTIONED, claimed=display)

    def claim(self, name: str) -> SessionIdentity:
        """The session identity after someone says "I am <name>".

        An owner alias gives the owner identity but NOT established: it is only
        a claim. Any other name is a normal user, established for conversation
        only. Claiming to be the assistant yields no identity.
        """
        display = _clean_display(name)
        if _key(display) in self._owner_keys:
            owner = self.owner_identity()
            claimed = Identity(owner.person_id, owner.display_name, Role.OWNER, claimed=display)
            return SessionIdentity(claimed, established=False)
        if _key(display) == _key(self._assistant_name):
            return SessionIdentity()
        return SessionIdentity(Identity(self._id_factory(display), display, Role.USER, claimed=display),
                               established=True)
