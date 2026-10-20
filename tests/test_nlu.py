"""Practical checks for nlu.py (rule-based "rules-v1").

Run from the project folder, either of:
    python -m unittest discover -s tests -v
    python tests/test_nlu.py          (or press Run on this file in VS Code)

These check what the rules DO cover. They do not show general understanding.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from conversation.models import (  # noqa: E402
    Context, DialogueAct, EntityKind, Intent, Sentiment, Speaker, Turn, Understanding,
)
from nlu import NLU  # noqa: E402

nlu = NLU()
EARLIER = Context(recent_turns=(Turn(Speaker.ERIS, "Water boils at 100 degrees.", 1.0),))


def subjects(analysis):
    return [e.text for e in analysis.entities if e.kind is EntityKind.SELF_REFERENCE]


class IdentityTests(unittest.TestCase):
    def test_assistant_identity_phrasings_share_one_intent(self):
        for text in ["Who are you?", "Tell me about yourself.", "What's your name?",
                     "Remind me who I'm talking to.", "Introduce yourself.", "who r u",
                     "wats ur name", "whats your name"]:
            with self.subTest(text=text):
                a = nlu.analyze(text)
                self.assertIs(a.intent, Intent.IDENTITY_QUESTION)
                self.assertIs(a.understanding, Understanding.UNDERSTOOD)
                self.assertEqual(subjects(a), ["assistant"])

    def test_speaker_and_creator_are_told_apart(self):
        for text in ["Who am I?", "what's my name?"]:
            with self.subTest(text=text):
                self.assertEqual(subjects(nlu.analyze(text)), ["speaker"])
        for text in ["Who created you?", "who made u", "Who is your owner?"]:
            with self.subTest(text=text):
                self.assertEqual(subjects(nlu.analyze(text)), ["creator"])

    def test_who_am_i_talking_to_is_not_who_am_i(self):
        self.assertEqual(subjects(nlu.analyze("Who am I talking to?")), ["assistant"])

    def test_nlu_reports_a_claimed_name_but_grants_nothing(self):
        a = nlu.analyze("I am Master Shadow")
        self.assertIs(a.intent, Intent.PERSONAL_INFORMATION)
        self.assertEqual([e.text for e in a.entities if e.kind is EntityKind.PERSON], ["Master Shadow"])
        a = nlu.analyze("my name is Daniel")
        self.assertEqual([e.text for e in a.entities if e.kind is EntityKind.PERSON], ["Daniel"])


class SocialTests(unittest.TestCase):
    def test_greetings_farewells_thanks_apologies(self):
        cases = [("hi", Intent.GREETING), ("Hello Eris!", Intent.GREETING),
                 ("good morning", Intent.GREETING), ("Bye", Intent.FAREWELL),
                 ("talk to you later", Intent.FAREWELL), ("thx a lot", Intent.THANKS),
                 ("Thank you", Intent.THANKS), ("sorry o", Intent.APOLOGY)]
        for text, intent in cases:
            with self.subTest(text=text):
                self.assertIs(nlu.analyze(text).intent, intent)

    def test_a_greeting_word_inside_other_text_is_not_a_greeting(self):
        self.assertIsNot(nlu.analyze("he said hi to me").intent, Intent.GREETING)
        a = nlu.analyze("hello world program in python")
        self.assertIsNot(a.understanding, Understanding.UNDERSTOOD)

    def test_thanks_for_nothing_is_flagged_as_possibly_sarcastic(self):
        a = nlu.analyze("Thanks for nothing")
        self.assertIn("possible_sarcasm", a.emotion_cues)
        self.assertIsNot(a.understanding, Understanding.UNDERSTOOD)

    def test_typo_in_a_known_word_is_repaired_with_lower_confidence(self):
        clean, typo = nlu.analyze("hello"), nlu.analyze("helo")
        self.assertIs(typo.intent, Intent.GREETING)
        self.assertLess(typo.confidence, clean.confidence)

    def test_names_are_not_autocorrected(self):
        a = nlu.analyze("my name is Hallo")
        self.assertIn("Hallo", [e.text for e in a.entities])


class NigerianEnglishTests(unittest.TestCase):
    def test_abeg_help_me_is_a_request(self):
        a = nlu.analyze("Abeg, help me")
        self.assertIs(a.intent, Intent.REQUEST)
        self.assertIs(a.dialogue_act, DialogueAct.REQUEST)

    def test_wetin_happen_is_a_question_about_what_happened(self):
        a = nlu.analyze("Wetin happen?")
        self.assertIs(a.intent, Intent.FACTUAL_QUESTION)
        self.assertIs(a.dialogue_act, DialogueAct.QUESTION)

    def test_how_far_is_a_greeting_only_when_context_supports_it(self):
        start = nlu.analyze("How far?")
        mid = nlu.analyze("How far?", EARLIER)
        self.assertIs(start.intent, Intent.GREETING)
        self.assertIs(start.understanding, Understanding.UNDERSTOOD)
        self.assertIs(mid.understanding, Understanding.GUESSED)
        self.assertIs(nlu.analyze("how far is the airport").intent, Intent.FACTUAL_QUESTION)

    def test_negative_emotion_is_preserved_in_pidgin(self):
        a = nlu.analyze("I no dey happy")
        self.assertIs(a.intent, Intent.EMOTIONAL_DISCLOSURE)
        self.assertIs(a.sentiment, Sentiment.NEGATIVE)
        self.assertIn("negated:joy", a.emotion_cues)
        self.assertEqual(a.normalized_text, "i no dey happy")  # not rewritten
        self.assertEqual(a.raw_text, "I no dey happy")

    def test_wanting_to_rest_is_recognised_but_only_as_a_guess(self):
        a = nlu.analyze("I wan rest")
        self.assertIs(a.intent, Intent.EMOTIONAL_DISCLOSURE)
        self.assertIn("wants_rest", a.emotion_cues)
        self.assertIs(a.understanding, Understanding.GUESSED)


class ContextTests(unittest.TestCase):
    def test_short_answer_to_a_pending_question_is_a_clarification_response(self):
        ctx = Context(pending_question="Would you prefer Chemistry or Biology?")
        a = nlu.analyze("Chemistry.", ctx)
        self.assertIs(a.intent, Intent.CLARIFICATION_RESPONSE)
        self.assertIs(a.dialogue_act, DialogueAct.CLARIFICATION)

    def test_same_fragment_without_context_is_not_understood(self):
        a = nlu.analyze("Chemistry.")
        self.assertIs(a.intent, Intent.UNKNOWN)
        self.assertIs(a.understanding, Understanding.NOT_UNDERSTOOD)
        self.assertTrue(a.ambiguities)

    def test_yes_and_no_need_something_to_answer(self):
        ctx = Context(pending_question="Shall I continue?")
        self.assertIs(nlu.analyze("yes", ctx).dialogue_act, DialogueAct.CONFIRMATION)
        self.assertIs(nlu.analyze("nope", ctx).dialogue_act, DialogueAct.DENIAL)
        alone = nlu.analyze("yes")
        self.assertIs(alone.understanding, Understanding.NOT_UNDERSTOOD)
        self.assertIn("nothing earlier to confirm or deny", alone.ambiguities)

    def test_reference_with_a_plausible_target_is_clean(self):
        with_ctx = nlu.analyze("Could you explain this?", EARLIER)
        without = nlu.analyze("Could you explain this?")
        self.assertEqual(with_ctx.references, ("this",))
        self.assertEqual(with_ctx.ambiguities, ())
        self.assertIs(with_ctx.understanding, Understanding.UNDERSTOOD)
        self.assertTrue(without.ambiguities)
        self.assertIs(without.understanding, Understanding.GUESSED)

    def test_correction_needs_something_to_correct(self):
        with_ctx = nlu.analyze("No, I meant Chemistry", EARLIER)
        without = nlu.analyze("No, I meant Chemistry")
        self.assertIs(with_ctx.intent, Intent.CORRECTION)
        self.assertIs(with_ctx.dialogue_act, DialogueAct.CORRECTION)
        self.assertIs(with_ctx.understanding, Understanding.UNDERSTOOD)
        self.assertIs(without.understanding, Understanding.GUESSED)
        self.assertGreater(with_ctx.confidence, without.confidence)

    def test_memory_question_is_recognised_and_topic_is_extracted(self):
        a = nlu.analyze("Do you remember what we said about photosynthesis?")
        self.assertIs(a.intent, Intent.MEMORY_RECALL)
        self.assertIn("photosynthesis", [e.text for e in a.entities if e.kind is EntityKind.TOPIC])

    def test_context_is_not_modified(self):
        ctx = Context(recent_turns=EARLIER.recent_turns, pending_question="Which one?")
        before = (ctx.recent_turns, ctx.pending_question, ctx.current_topic, ctx.last_action)
        nlu.analyze("the first one", ctx)
        self.assertEqual(before, (ctx.recent_turns, ctx.pending_question, ctx.current_topic, ctx.last_action))


class EmotionTests(unittest.TestCase):
    def test_clear_emotional_cues(self):
        cases = [("I'm happy.", Sentiment.POSITIVE), ("I'm not happy.", Sentiment.NEGATIVE),
                 ("I'm so happy I could scream.", Sentiment.POSITIVE), ("I'm tired", Sentiment.NEGATIVE)]
        for text, sentiment in cases:
            with self.subTest(text=text):
                a = nlu.analyze(text)
                self.assertIs(a.sentiment, sentiment)
                self.assertIs(a.intent, Intent.EMOTIONAL_DISCLOSURE)

    def test_possible_sarcasm_is_flagged_not_resolved(self):
        a = nlu.analyze("Yeah, wonderful. Just what I needed.")
        self.assertIn("possible_sarcasm", a.emotion_cues)
        self.assertIs(a.sentiment, Sentiment.MIXED)
        self.assertIn("tone may be sarcastic", a.ambiguities)
        self.assertIsNot(a.understanding, Understanding.UNDERSTOOD)

    def test_possible_crisis_language_is_flagged_for_other_components(self):
        a = nlu.analyze("I want to die")
        self.assertIn("possible_crisis", a.emotion_cues)
        self.assertIs(a.intent, Intent.EMOTIONAL_DISCLOSURE)


class HonestyTests(unittest.TestCase):
    def test_every_analysis_names_its_method(self):
        self.assertEqual(nlu.analyze("hi").analyzer, "rules-v1")

    def test_a_statement_it_cannot_interpret_is_not_understood(self):
        a = nlu.analyze("The purple elephant is negotiating with the moon.")
        self.assertIs(a.intent, Intent.UNKNOWN)
        self.assertIs(a.understanding, Understanding.NOT_UNDERSTOOD)

    def test_a_question_full_of_unknown_words_is_at_most_a_guess(self):
        a = nlu.analyze("Who is the purple elephant that is negotiating with the moon?")
        self.assertIsNot(a.understanding, Understanding.UNDERSTOOD)
        self.assertLess(a.confidence, 0.7)
        self.assertTrue(a.ambiguities)

    def test_a_simple_factual_question_keeps_its_topic(self):
        a = nlu.analyze("What is photosynthesis?")
        self.assertIs(a.intent, Intent.FACTUAL_QUESTION)
        self.assertIn("photosynthesis", [e.text for e in a.entities])

    def test_gibberish_is_unknown_and_explained(self):
        for text in ["asdfgh qwerty", "zxcv lkjh poiu mnbv"]:
            with self.subTest(text=text):
                a = nlu.analyze(text)
                self.assertIs(a.intent, Intent.UNKNOWN)
                self.assertIsNot(a.understanding, Understanding.UNDERSTOOD)
                self.assertTrue(a.ambiguities)

    def test_close_competing_readings_are_reported(self):
        a = nlu.analyze("How far?", Context(recent_turns=EARLIER.recent_turns))
        self.assertIsNot(a.understanding, Understanding.UNDERSTOOD)

    def test_blank_input_is_rejected(self):
        for text in ["", "   "]:
            with self.assertRaises(ValueError):
                nlu.analyze(text)

    def test_entity_spans_point_into_the_raw_text(self):
        text = "Remind me at 5pm tomorrow, my sister Ada said 42."
        a = nlu.analyze(text)
        for entity in a.entities:
            if entity.span:
                start, end = entity.span
                self.assertEqual(text[start:end].lower(), entity.text.lower())
        kinds = {e.kind for e in a.entities}
        self.assertTrue({EntityKind.TIME, EntityKind.PERSON, EntityKind.NUMBER} <= kinds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
