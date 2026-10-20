"""Tests for ownership.py (no files or real data are touched).

Run from the project folder:  python -m unittest discover -s tests -v
or press Run on this file in VS Code.
"""
import ast
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from conversation.models import AuthState, Identity, Role, SessionIdentity  # noqa: E402
from ownership import Ownership  # noqa: E402


def make(**kwargs):
    counter = iter(range(1, 100))
    return Ownership(id_factory=lambda name: f"id-{next(counter)}", **kwargs)


class FixedIdentityTests(unittest.TestCase):
    def test_assistant_and_owner_are_distinct(self):
        o = make()
        self.assertIs(o.assistant_identity().role, Role.ASSISTANT)
        self.assertEqual(o.assistant_identity().display_name, "Eris")
        self.assertIs(o.owner_identity().role, Role.OWNER)
        self.assertNotEqual(o.assistant_identity().person_id, o.owner_identity().person_id)

    def test_daniel_master_shadow_and_shadow_are_one_owner(self):
        o = make()
        names = ["Daniel", "daniel", "Master Shadow", "shadow", "  SHADOW! "]
        self.assertEqual({o.mentioned(n).person_id for n in names}, {o.owner_identity().person_id})
        self.assertEqual({o.claim(n).identity.person_id for n in names}, {o.owner_identity().person_id})
        self.assertEqual(o.mentioned("daniel").display_name, "Master Shadow")

    def test_alias_may_not_be_the_assistants_name(self):
        with self.assertRaises(ValueError):
            Ownership(aliases=("Eris",))

    def test_aliases_are_constructor_input_and_can_load_from_a_profile(self):
        o = Ownership.from_profile({"name": "Boss One", "aliases": ["B1", "Chief"]})
        self.assertTrue(o.is_owner(o.claim("chief").identity))
        self.assertFalse(o.is_owner(o.claim("Daniel").identity))  # not an alias in this profile


class IsOwnerTests(unittest.TestCase):
    def test_is_owner_is_an_identity_comparison_only(self):
        o = make()
        self.assertTrue(o.is_owner(o.owner_identity()))
        self.assertTrue(o.is_owner(o.claim("Shadow").identity))   # a claim is still "the owner identity"
        self.assertFalse(o.is_owner(o.claim("Ada").identity))
        self.assertFalse(o.is_owner(o.mentioned("Ada")))
        self.assertFalse(o.is_owner(o.assistant_identity()))

    def test_is_owner_takes_no_authentication_and_changes_nothing(self):
        o = make()
        claimed = o.claim("Master Shadow")
        with self.assertRaises(TypeError):
            o.is_owner(claimed.identity, AuthState())
        self.assertTrue(o.is_owner(claimed.identity))
        self.assertFalse(claimed.established)  # asking never establishes a session

    def test_is_owner_rejects_non_identities(self):
        with self.assertRaises(TypeError):
            make().is_owner("owner")


class ClaimTests(unittest.TestCase):
    def test_claiming_an_owner_alias_is_only_a_claim(self):
        o = make()
        for name in ["Master Shadow", "Shadow", "Daniel"]:
            with self.subTest(name=name):
                session = o.claim(name)
                self.assertIs(session.identity.role, Role.OWNER)
                self.assertFalse(session.established)
                self.assertEqual(session.identity.claimed, name)

    def test_other_names_are_normal_users_with_separate_ids(self):
        o = make()
        ada, ada2 = o.claim("Ada"), o.claim("Ada")
        self.assertIs(ada.identity.role, Role.USER)
        self.assertTrue(ada.established)
        self.assertNotEqual(ada.identity.person_id, o.owner_identity().person_id)
        self.assertNotEqual(ada.identity.person_id, ada2.identity.person_id)

    def test_claiming_to_be_the_assistant_gives_no_identity(self):
        session = make().claim("Eris")
        self.assertIsNone(session.identity)
        self.assertFalse(session.established)

    def test_bad_names_are_rejected(self):
        o = make()
        for bad in ["", "   ", "!!!", "x" * 200]:
            with self.subTest(name=bad[:10]):
                with self.assertRaises(ValueError):
                    o.claim(bad)
        with self.assertRaises(TypeError):
            o.claim(None)


class MentionedTests(unittest.TestCase):
    def test_other_people_are_mentioned_not_participants(self):
        ada = make().mentioned("Ada")
        self.assertIs(ada.role, Role.MENTIONED)
        with self.assertRaises(ValueError):
            SessionIdentity(identity=ada, established=True)

    def test_mentioning_the_assistant_resolves_to_the_assistant(self):
        self.assertIs(make().mentioned("eris").role, Role.ASSISTANT)


class IndependenceTests(unittest.TestCase):
    def test_ownership_does_not_import_security_or_auth_types(self):
        with open(os.path.join(ROOT, "ownership.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
        self.assertNotIn("security", imported)
        for name in ("AuthState", "AuthLevel", "AuthMethod"):
            self.assertNotIn(name, imported)

    def test_ownership_has_no_authentication_methods(self):
        for name in ("with_authentication", "is_authenticated_owner", "authenticate", "authorize"):
            self.assertFalse(hasattr(Ownership, name), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
