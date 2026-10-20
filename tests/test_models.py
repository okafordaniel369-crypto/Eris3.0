"""Tests for conversation/models.py.

Run from the project folder, either of:
    python -m unittest discover -s tests -v
    python tests/test_models.py          (or press Run on this file in VS Code)
"""
import ast
import dataclasses
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from conversation import models as m  # noqa: E402


def make_analysis(**overrides):
    values = dict(raw_text="hello there", normalized_text="hello there")
    values.update(overrides)
    return m.Analysis(**values)


class FrozenTests(unittest.TestCase):
    def test_reassigning_a_field_fails(self):
        cases = [
            (m.Entity("Daniel", m.EntityKind.PERSON), "text", "Other"),
            (make_analysis(), "intent", m.Intent.GREETING),
            (m.Identity("u1", "Daniel", m.Role.USER), "display_name", "X"),
            (m.AuthState(), "authenticated", True),
            (m.Action(m.ActionKind.ANSWER), "needs_auth", True),
            (m.Reply("hi", m.Action(m.ActionKind.GREET)), "text", "bye"),
        ]
        for obj, field_name, value in cases:
            with self.subTest(type=type(obj).__name__):
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(obj, field_name, value)


class NestedImmutabilityTests(unittest.TestCase):
    def test_lists_become_tuples_and_cannot_be_modified(self):
        cues = ["worried"]
        analysis = make_analysis(emotion_cues=cues, ambiguities=["who?"], references=["it"])
        cues.append("sad")  # changing the original must not affect the object
        self.assertEqual(analysis.emotion_cues, ("worried",))
        for name in ("entities", "emotion_cues", "ambiguities", "references"):
            self.assertIsInstance(getattr(analysis, name), tuple, name)
        with self.assertRaises(AttributeError):
            analysis.emotion_cues.append("x")

    def test_context_and_style_hints_are_tuples(self):
        context = m.Context(recent_turns=[m.Turn(m.Speaker.USER, "hi", 1.0)])
        self.assertIsInstance(context.recent_turns, tuple)
        action = m.Action(m.ActionKind.ANSWER, style_hints=[m.StyleHint.CONCISE])
        self.assertIsInstance(action.style_hints, tuple)

    def test_action_content_is_read_only_including_nested_values(self):
        source = {"definition": "x", "examples": ["a", "b"], "detail": {"k": ["v"]}}
        action = m.Action(m.ActionKind.EXPLAIN, content=source)
        source["examples"].append("c")
        source["new"] = "y"
        self.assertEqual(action.content["examples"], ("a", "b"))
        self.assertNotIn("new", action.content)
        with self.assertRaises(TypeError):
            action.content["definition"] = "changed"
        self.assertIsInstance(action.content["detail"]["k"], tuple)

    def test_action_content_rejects_unsupported_values(self):
        with self.assertRaises(TypeError):
            m.Action(m.ActionKind.ANSWER, content={"bad": object()})
        with self.assertRaises(TypeError):
            m.Action(m.ActionKind.ANSWER, content={1: "non-string key"})


class AuthenticationTests(unittest.TestCase):
    def test_default_state_grants_no_owner_access(self):
        state = m.AuthState()
        self.assertFalse(state.authenticated)
        self.assertIs(state.level, m.AuthLevel.NONE)
        self.assertIs(state.method, m.AuthMethod.NONE)

    def test_inconsistent_states_are_rejected(self):
        with self.assertRaises(ValueError):
            m.AuthState(authenticated=True)
        with self.assertRaises(ValueError):
            m.AuthState(authenticated=False, level=m.AuthLevel.OWNER, method=m.AuthMethod.PASSCODE)
        with self.assertRaises(ValueError):
            m.AuthState(authenticated=True, level=m.AuthLevel.OWNER)

    def test_consistent_authenticated_state_is_accepted(self):
        state = m.AuthState(True, m.AuthLevel.OWNER, m.AuthMethod.PASSCODE)
        self.assertTrue(state.authenticated)

    def test_auth_values_must_be_enums_not_strings(self):
        with self.assertRaises(TypeError):
            m.AuthState(authenticated=True, level="owner", method="passcode")

    def test_needs_auth_is_only_a_flag(self):
        action = m.Action(m.ActionKind.REQUEST_AUTHENTICATION, needs_auth=True)
        self.assertTrue(action.needs_auth)
        self.assertFalse(m.AuthState().authenticated)


class IdentityVersusAuthenticationTests(unittest.TestCase):
    def test_establishing_an_identity_does_not_authenticate_it(self):
        owner = m.Identity("owner-1", "Master Shadow", m.Role.OWNER, claimed="Master Shadow")
        session = m.SessionIdentity(identity=owner, established=True)
        self.assertTrue(session.established)
        self.assertFalse(hasattr(session, "authenticated"))
        self.assertFalse(hasattr(owner, "authenticated"))
        self.assertFalse(m.AuthState().authenticated)

    def test_a_new_user_named_daniel_is_not_the_owner(self):
        daniel = m.Identity("user-7", "Daniel", m.Role.USER)
        self.assertIsNot(daniel.role, m.Role.OWNER)

    def test_established_session_requires_an_identity(self):
        with self.assertRaises(ValueError):
            m.SessionIdentity(identity=None, established=True)

    def test_mentioned_or_assistant_roles_cannot_be_the_session_user(self):
        for role in (m.Role.MENTIONED, m.Role.ASSISTANT):
            with self.subTest(role=role):
                with self.assertRaises(ValueError):
                    m.SessionIdentity(identity=m.Identity("p", "Someone", role), established=True)

    def test_default_session_is_unknown(self):
        session = m.SessionIdentity()
        self.assertIsNone(session.identity)
        self.assertFalse(session.established)


class AnalysisTests(unittest.TestCase):
    def test_confidence_below_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            make_analysis(confidence=-0.01)

    def test_confidence_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            make_analysis(confidence=1.01)

    def test_valid_confidence_values_are_accepted(self):
        for value in (0, 0.0, 0.5, 1, 1.0):
            with self.subTest(value=value):
                self.assertEqual(make_analysis(confidence=value).confidence, float(value))

    def test_non_numeric_nan_and_bool_confidence_are_rejected(self):
        with self.assertRaises(TypeError):
            make_analysis(confidence="high")
        with self.assertRaises(TypeError):
            make_analysis(confidence=True)
        with self.assertRaises(ValueError):
            make_analysis(confidence=float("nan"))

    def test_unknown_intent_is_represented_by_intent_unknown(self):
        analysis = make_analysis()
        self.assertIs(analysis.intent, m.Intent.UNKNOWN)
        self.assertIs(analysis.dialogue_act, m.DialogueAct.UNKNOWN)
        self.assertIs(analysis.understanding, m.Understanding.NOT_UNDERSTOOD)
        self.assertEqual(analysis.confidence, 0.0)

    def test_unknown_intent_cannot_be_marked_understood(self):
        with self.assertRaises(ValueError):
            make_analysis(understanding=m.Understanding.UNDERSTOOD)

    def test_guessing_is_distinct_from_understanding(self):
        guess = make_analysis(intent=m.Intent.GENERAL_QUESTION, understanding=m.Understanding.GUESSED)
        self.assertIsNot(guess.understanding, m.Understanding.UNDERSTOOD)

    def test_blank_text_and_wrong_enum_types_are_rejected(self):
        with self.assertRaises(ValueError):
            make_analysis(raw_text="   ")
        with self.assertRaises(TypeError):
            make_analysis(intent="greeting")

    def test_entity_outside_the_message_is_rejected(self):
        entity = m.Entity("hello", m.EntityKind.OTHER, span=(0, 50))
        with self.assertRaises(ValueError):
            make_analysis(entities=[entity])


class EntitySpanTests(unittest.TestCase):
    def test_invalid_character_positions_are_rejected(self):
        for span in [(-1, 3), (3, 3), (5, 2)]:
            with self.subTest(span=span):
                with self.assertRaises(ValueError):
                    m.Entity("x", m.EntityKind.OTHER, span=span)

    def test_malformed_spans_are_rejected(self):
        for span in [(0,), (0, 1, 2), ("a", 2), (True, 2), (0.5, 2)]:
            with self.subTest(span=span):
                with self.assertRaises(TypeError):
                    m.Entity("x", m.EntityKind.OTHER, span=span)

    def test_valid_span_and_no_span_are_accepted(self):
        self.assertEqual(m.Entity("x", m.EntityKind.OTHER, span=[0, 3]).span, (0, 3))
        self.assertIsNone(m.Entity("x", m.EntityKind.OTHER).span)


class MemoryTests(unittest.TestCase):
    def hit(self, **overrides):
        values = dict(record_id="r1", category=m.MemoryCategory.CONVERSATION_HISTORY,
                      text="stored text", score=0.8, is_verbatim=True)
        values.update(overrides)
        return m.MemoryHit(**values)

    def test_invalid_memory_score_is_rejected(self):
        for score in (-0.1, 1.1, float("inf")):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    self.hit(score=score)
        with self.assertRaises(TypeError):
            self.hit(score="0.5")

    def test_valid_scores_are_accepted(self):
        for score in (0, 0.5, 1):
            self.assertEqual(self.hit(score=score).score, float(score))

    def test_invalid_memory_limit_is_rejected(self):
        for limit in (0, -3):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    m.MemoryRequest("query", limit=limit)
        for limit in (True, 2.5, "5"):
            with self.subTest(limit=limit):
                with self.assertRaises(TypeError):
                    m.MemoryRequest("query", limit=limit)

    def test_valid_request_defaults(self):
        request = m.MemoryRequest("what did we discuss")
        self.assertEqual(request.limit, 5)
        self.assertIsNone(request.category)

    def test_is_verbatim_must_be_stated_explicitly(self):
        with self.assertRaises(TypeError):
            m.MemoryHit(record_id="r1", category=m.MemoryCategory.LEARNING, text="t", score=0.5)
        with self.assertRaises(TypeError):
            self.hit(is_verbatim="yes")
        self.assertFalse(self.hit(is_verbatim=False).is_verbatim)

    def test_all_five_categories_exist(self):
        names = {c.name for c in m.MemoryCategory}
        self.assertEqual(names, {"OWNER_PROFILE", "CONVERSATION_HISTORY", "OTHER_PEOPLE",
                                 "PROJECTS_KNOWLEDGE", "LEARNING"})


class SessionStateTests(unittest.TestCase):
    def test_turn_validation(self):
        with self.assertRaises(ValueError):
            m.Turn(m.Speaker.USER, "hi", -1.0)
        with self.assertRaises(ValueError):
            m.Turn(m.Speaker.USER, "hi", float("nan"))
        with self.assertRaises(ValueError):
            m.Turn(m.Speaker.USER, "", 1.0)
        with self.assertRaises(TypeError):
            m.Turn("user", "hi", 1.0)

    def test_turn_can_carry_an_analysis(self):
        turn = m.Turn(m.Speaker.USER, "hello there", 12, analysis=make_analysis())
        self.assertEqual(turn.timestamp, 12.0)
        self.assertIsInstance(turn.analysis, m.Analysis)

    def test_context_defaults_and_validation(self):
        context = m.Context()
        self.assertEqual(context.recent_turns, ())
        self.assertIsNone(context.pending_question)
        with self.assertRaises(ValueError):
            m.Context(pending_question="  ")
        with self.assertRaises(TypeError):
            m.Context(recent_turns=["not a turn"])
        with self.assertRaises(TypeError):
            m.Context(last_action="answer")


class ActionAndReplyTests(unittest.TestCase):
    def test_controlled_values_must_be_enums(self):
        with self.assertRaises(TypeError):
            m.Action("answer")
        with self.assertRaises(TypeError):
            m.Action(m.ActionKind.ANSWER, style_hints=["concise"])

    def test_action_can_request_memory(self):
        request = m.MemoryRequest("earlier topic", m.MemoryCategory.CONVERSATION_HISTORY)
        action = m.Action(m.ActionKind.RECALL, needs_memory=request)
        self.assertIs(action.needs_memory, request)
        with self.assertRaises(TypeError):
            m.Action(m.ActionKind.RECALL, needs_memory="earlier topic")

    def test_reply_keeps_text_and_action(self):
        action = m.Action(m.ActionKind.GREET, content={"text": "hi"})
        reply = m.Reply("Hello!", action)
        self.assertEqual(reply.text, "Hello!")
        self.assertIs(reply.action, action)
        with self.assertRaises(ValueError):
            m.Reply("", action)
        with self.assertRaises(TypeError):
            m.Reply("Hello!", "not an action")


class EnumAdditionTests(unittest.TestCase):
    def test_identity_question_and_self_reference_exist(self):
        self.assertIn("IDENTITY_QUESTION", m.Intent.__members__)
        self.assertIn("SELF_REFERENCE", m.EntityKind.__members__)
        entity = m.Entity("creator", m.EntityKind.SELF_REFERENCE)
        self.assertIsNone(entity.span)


class IndependenceTests(unittest.TestCase):
    def test_models_imports_only_the_standard_library(self):
        path = os.path.join(ROOT, "conversation", "models.py")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        allowed = {"__future__", "math", "dataclasses", "enum", "types", "typing"}
        self.assertTrue(imported <= allowed, f"unexpected imports: {imported - allowed}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
