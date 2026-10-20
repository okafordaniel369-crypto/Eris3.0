"""nlu.py - rule-based interpretation of one user message (analyzer "rules-v1").

NLU turns text plus session context into a models.Analysis. It never answers,
never decides what Eris does (Brain does), and never touches memory.

Honest limits: this is pattern evidence, not comprehension. It only recognises
what the lexicon below covers, and it says so through `understanding`:
UNDERSTOOD, GUESSED or NOT_UNDERSTOOD (then the intent is Intent.UNKNOWN).
Confidence comes from evidence quality (strength of cues, competing readings,
unresolved references, words it cannot interpret), not from word matches alone.
Pidgin and Nigerian English are matched as they are written; they are not
translated into other words, so their meaning is not lost in normalisation.
"""
from __future__ import annotations

import difflib
import re
from typing import Optional

from conversation.models import (
    ActionKind, Analysis, Context, DialogueAct, Entity, EntityKind, Intent,
    Sentiment, Understanding,
)

ANALYZER = "rules-v1"
UNDERSTOOD_AT = 0.70   # at or above: UNDERSTOOD
GUESSED_AT = 0.40      # at or above: GUESSED; below: NOT_UNDERSTOOD / UNKNOWN
GUESS_CAP = 0.69       # too many uninterpreted words: never better than a guess

# ==========================================================================
# LEXICON (word lists, cue weights, patterns). Patterns run on the normalised
# text padded with one space each side, so " word " matches whole words.
# ==========================================================================

# Unambiguous shorthand only. Pidgin words are deliberately NOT listed here.
ABBREVIATIONS = {
    "pls": "please", "plz": "please", "thx": "thanks", "tnx": "thanks",
    "thanx": "thanks", "ty": "thanks", "u": "you", "ur": "your", "r": "are",
    "im": "i'm", "ive": "i've", "dont": "don't", "cant": "can't",
    "doesnt": "doesn't", "didnt": "didn't", "isnt": "isn't", "wont": "won't",
    "whats": "what's", "wats": "what's", "whos": "who's", "thats": "that's",
    "hows": "how's", "wat": "what", "wht": "what", "wen": "when",
    "abt": "about", "bcos": "because", "cos": "because", "dis": "this",
    "dat": "that", "idk": "i don't know", "btw": "by the way",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
}

STOPWORDS = set("""
i me my mine myself you your yours we us our he him his she her it its they them
their the a an and or but if so is am are was were be been being do does did to of
in on at for with about as by from that this these those there here can could will
would should have has had not no yes just very too really also then than what who
whom which when where why how please ok okay oh o na dey don wey abi sha oga wan
""".split())

# word -> (cue, polarity): +1 positive, -1 negative, 0 neutral. Cues are evidence
# for other components to interpret; sarcasm and mixed feelings are NOT resolved.
EMOTION_WORDS = {
    "happy": ("joy", 1), "glad": ("joy", 1), "thrilled": ("joy", 1),
    "excited": ("excitement", 1), "great": ("joy", 1), "wonderful": ("joy", 1),
    "amazing": ("joy", 1), "awesome": ("joy", 1), "proud": ("pride", 1),
    "grateful": ("gratitude", 1), "relieved": ("relief", 1), "hopeful": ("hope", 1),
    "sad": ("sadness", -1), "unhappy": ("sadness", -1), "depressed": ("sadness", -1),
    "upset": ("sadness", -1), "heartbroken": ("sadness", -1), "lonely": ("loneliness", -1),
    "hurt": ("hurt", -1), "tired": ("fatigue", -1), "tire": ("fatigue", -1),
    "exhausted": ("fatigue", -1), "stressed": ("stress", -1),
    "overwhelmed": ("stress", -1), "anxious": ("anxiety", -1), "worried": ("anxiety", -1),
    "scared": ("fear", -1), "afraid": ("fear", -1), "angry": ("anger", -1),
    "annoyed": ("anger", -1), "vex": ("anger", -1), "vexed": ("anger", -1),
    "frustrated": ("frustration", -1), "disappointed": ("disappointment", -1),
    "bored": ("boredom", -1), "terrible": ("distress", -1), "awful": ("distress", -1),
    "fine": ("okay", 0), "okay": ("okay", 0), "ok": ("okay", 0),
}
NEGATORS = {"not", "no", "never", "don't", "doesn't", "didn't", "isn't", "aren't",
            "can't", "won't", "ain't", "hardly"}
INTENSIFIERS = {"so", "really", "very", "too", "extremely", "super", "soo"}
SARCASM_RE = re.compile(r" (?:just what i needed|oh great|yeah right|yeah wonderful|great just|thanks for nothing) ")

# Possible crisis language: NLU only flags it (cue "possible_crisis"); Brain and
# Personality decide how to respond. Weighted high so it is never overlooked.
CRISIS_PATTERNS = (
    r" (?:want|wanna|wan|going|plan|planning) to (?:die|kill myself|end it all|end my life|disappear) ",
    r" kill myself ", r" end my life ", r" no reason to live ", r" suicid",
    r" don't want to (?:be here|live|exist) ", r" i wish i (?:was|were) dead ",
)
CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS))

IMPERATIVE_VERBS = (
    "tell", "show", "give", "write", "list", "explain", "describe", "define", "find",
    "search", "open", "set", "stop", "start", "remember", "forget", "call", "remind",
    "translate", "summarize", "summarise", "calculate", "play", "make", "create",
    "repeat", "say", "introduce",
)
EMOTION_ALT = "|".join(sorted(EMOTION_WORDS, key=len, reverse=True))

INTENT_RULES = {
    Intent.GREETING: (
        (r"^ (?:(?:oh|well|so|and|ok|okay) )?(?:hi|hello|hey|hiya|heya|yo|howdy|sup|wassup) ", 0.75),
        (r" good (?:morning|afternoon|evening|day) ", 0.85),
        (r" (?:how are you|how are you doing|how you dey|how's it going|how is it going|what's up) ", 0.75),
        (r" how body ", 0.6),
        (r" nice to meet you ", 0.7),
    ),
    Intent.FAREWELL: (
        (r" (?:bye|goodbye|good night|goodnight|see you|see ya|cya|take care|ttyl|"
         r"talk to you later|talk later|catch you later|i dey go|i'm leaving|i'm off) ", 0.8),
        (r" (?:i have to go|i got to go|got to go|i must go) ", 0.65),
    ),
    Intent.THANKS: (
        (r" (?:thanks|thank you|thankyou|i appreciate|much appreciated) ", 0.85),
        (r" (?:well done|nice one) ", 0.55),
    ),
    Intent.APOLOGY: (
        (r" (?:sorry|apologies|apologise|apologize|my bad|forgive me|pardon me) ", 0.8),
    ),
    Intent.CORRECTION: (
        (r" i meant ", 0.85),
        (r" (?:no|nope) i (?:said|meant|mean) ", 0.85),
        (r" not what i (?:said|meant|asked) ", 0.85),
        (r" (?:that's|that is|it's|it is) (?:not|wrong|incorrect) ", 0.8),
        (r" (?:you're|you are) (?:wrong|mistaken|not right) ", 0.8),
        (r" i (?:didn't|did not|never) (?:say|mean|ask) ", 0.8),
        (r" (?:you misunderstood|you got it wrong|you got that wrong|you misheard) ", 0.85),
        (r" correction ", 0.7), (r" (?:wrong|incorrect) ", 0.4), (r" actually ", 0.35),
    ),
    Intent.MEMORY_RECALL: (
        (r" do you remember ", 0.85), (r" remember when ", 0.8),
        (r" what did (?:i|we) (?:say|tell|talk|discuss|mention|speak) ", 0.85),
        (r" what was i (?:saying|talking) ", 0.85),
        (r" we (?:talked|spoke|discussed) about ", 0.8),
        (r" did i (?:tell|mention|say) ", 0.8), (r" last time ", 0.5),
        (r" as i said ", 0.5), (r" you said ", 0.5), (r" earlier ", 0.4),
    ),
    Intent.CAPABILITY_QUESTION: (
        (r" what can you do ", 0.85), (r" what are you able to ", 0.85),
        (r" what are your (?:features|abilities|skills|limits) ", 0.85),
        (r" are you able to ", 0.7), (r" how do you work ", 0.75),
        (r" can you (?:remember|learn|search|read|see|hear|speak) ", 0.7),
        (r" what do you know ", 0.6),
    ),
    Intent.OPINION: (
        (r" what do you think ", 0.8), (r" your (?:opinion|view|thoughts|take) ", 0.75),
        (r" do you (?:like|love|hate|prefer|believe) ", 0.6),
        (r" which (?:is|do you) (?:better|best|prefer) ", 0.7),
        (r" in my opinion ", 0.7), (r" i think ", 0.5), (r" i believe ", 0.5),
    ),
    Intent.PERSONAL_INFORMATION: (
        (r" my name is ", 0.9), (r" call me ", 0.85), (r" (?:i'm|i am) called ", 0.85),
        (r" people call me ", 0.85), (r" (?:i'm|i am) from ", 0.8), (r" i live in ", 0.8),
        (r" i (?:work|study|teach) ", 0.7), (r" (?:i'm|i am) (?:a|an) ", 0.55),
        (r" my (?:birthday|age|job|favorite|favourite) ", 0.6),
    ),
    Intent.REQUEST: (
        (r" (?:can|could|would|will) you (?:please )?(?:help|explain|tell|show|give|write|"
         r"make|find|check|teach|remind|describe|list|send) ", 0.85),
        (r" (?:can|could|would) you ", 0.6), (r" help me ", 0.8), (r" assist me ", 0.8),
        (r" i want you to ", 0.8), (r" i'd like (?:you )?to ", 0.7),
        (r" do me a favou?r ", 0.8), (r" abeg ", 0.6), (r" please ", 0.55),
        (r" kindly ", 0.6), (r" i need ", 0.55),
    ),
    Intent.INSTRUCTION: (
        (r"^ (?:%s) " % "|".join(IMPERATIVE_VERBS), 0.8),
    ),
    Intent.FACTUAL_QUESTION: (
        (r"^ (?:who|what|when|where|which|how many|how much)(?:'s)? "
         r"(?!should |would |about |do you |if )", 0.8),
        (r" what happened ", 0.8), (r" wetin happen ", 0.75), (r"^ wetin ", 0.6),
        (r" how far is ", 0.8), (r" what does .* mean ", 0.8),
    ),
    Intent.GENERAL_QUESTION: (
        (r"^ why ", 0.75), (r"^ how (?:do|does|can|could|should|would|to|did|come) ", 0.75),
        (r" should i ", 0.65), (r"^ what (?:should|would|about) ", 0.7),
        (r"^ (?:what|how) if ", 0.65),
    ),
    Intent.EMOTIONAL_DISCLOSURE: (
        (r" i(?:'m| am| feel| felt| dey| no| don't| do not)(?: [a-z']+){0,3} (?:%s) " % EMOTION_ALT, 0.8),
        (r" i (?:wan|want to|need to) (?:rest|sleep|cry|be alone) ", 0.55),
        (r" my heart (?:is )?(?:heavy|broken|aching) ", 0.8),
        (r" i can't (?:cope|take it|do this) anymore ", 0.85),
    ) + tuple((p, 0.95) for p in CRISIS_PATTERNS),
}

# (pattern, weight, whose identity is being asked about)
IDENTITY_RULES = (
    (r" who (?:are|r) you ", 0.9, "assistant"),
    (r" (?:what's|what is) your name ", 0.9, "assistant"),
    (r" what are you called ", 0.9, "assistant"),
    (r" tell me about yourself ", 0.9, "assistant"),
    (r" introduce yourself ", 0.9, "assistant"),
    (r" who (?:i'm|i am|am i) talking to ", 0.9, "assistant"),
    (r" who be you ", 0.85, "assistant"),
    (r" wetin be your name ", 0.9, "assistant"),
    (r" your name ", 0.7, "assistant"),
    (r" who am i(?! talking) ", 0.9, "speaker"),
    (r" (?:what's|what is) my name ", 0.9, "speaker"),
    (r" do you know (?:who i am|me) ", 0.85, "speaker"),
    (r" who be me ", 0.85, "speaker"),
    (r" who (?:created|made|built|designed|programmed|developed|coded|wrote|owns|trained) you ", 0.9, "creator"),
    (r" who(?:'s| is) your (?:creator|maker|owner|master|developer|boss|builder|programmer) ", 0.9, "creator"),
    (r" who(?:'s| is) behind you ", 0.85, "creator"),
    (r" your (?:creator|maker|developer) ", 0.7, "creator"),
)

# A strong reading of one kind hides the broader, generic kinds it contains.
SUPERSEDES = {
    Intent.IDENTITY_QUESTION: (Intent.FACTUAL_QUESTION, Intent.GENERAL_QUESTION,
                               Intent.INSTRUCTION, Intent.REQUEST, Intent.CAPABILITY_QUESTION),
    Intent.MEMORY_RECALL: (Intent.FACTUAL_QUESTION, Intent.GENERAL_QUESTION, Intent.INSTRUCTION),
    Intent.CAPABILITY_QUESTION: (Intent.FACTUAL_QUESTION, Intent.GENERAL_QUESTION, Intent.REQUEST),
    Intent.OPINION: (Intent.FACTUAL_QUESTION, Intent.GENERAL_QUESTION),
    Intent.REQUEST: (Intent.INSTRUCTION, Intent.GENERAL_QUESTION),
}

# (pattern, label, only meaningful inside the current session)
EXPLICIT_REFERENCES = (
    (r" what i (?:said|told you|mentioned) ", "what I said", False),
    (r" (?:earlier|before|just now|a moment ago) ", "earlier", False),
    (r" the (?:previous|last|first|other) (?:one|thing|topic|question|answer|message) ", "previous one", True),
    (r" the same (?:thing|one) ", "the same one", True),
)
PRONOUN_RE = re.compile(r"(?<= )(that|this|it|those|them|these)(?=(?:'s)? )")
PRONOUN_INTENTS = {Intent.REQUEST, Intent.INSTRUCTION, Intent.CORRECTION, Intent.FACTUAL_QUESTION,
                   Intent.GENERAL_QUESTION, Intent.OPINION}
SOCIAL_INTENTS = {Intent.GREETING, Intent.FAREWELL, Intent.THANKS, Intent.APOLOGY}
TOPIC_INTENTS = {Intent.FACTUAL_QUESTION, Intent.GENERAL_QUESTION, Intent.REQUEST, Intent.INSTRUCTION,
                 Intent.MEMORY_RECALL, Intent.OPINION, Intent.CAPABILITY_QUESTION}

YES_WORDS = {"yes", "yeah", "yep", "yup", "yea", "sure", "correct", "exactly", "right", "definitely",
             "absolutely", "ok", "okay", "alright"}
NO_WORDS = {"no", "nope", "nah"}
WH_WORDS = {"who", "what", "when", "where", "why", "how", "which", "wetin"}
AUX_WORDS = {"is", "are", "am", "do", "does", "did", "can", "could", "will", "would", "should", "have", "has", "was", "were"}
SUBJECT_WORDS = {"i", "you", "he", "she", "it", "we", "they", "there", "this", "that", "the", "a", "my", "your"}
# ==========================================================================
# END OF LEXICON
# ==========================================================================


def _build_vocabulary() -> set:
    words = set()
    patterns = [p for rules in INTENT_RULES.values() for p, _ in rules]
    patterns += [p for p, _, _ in IDENTITY_RULES]
    for pattern in patterns:
        words |= set(re.findall(r"[a-z']{2,}", pattern))
    for value in ABBREVIATIONS.values():
        words |= set(value.split())
    return words | set(EMOTION_WORDS) | set(IMPERATIVE_VERBS)


KNOWN_WORDS = _build_vocabulary()
_BY_INITIAL: dict = {}
for _word in sorted(KNOWN_WORDS):
    if len(_word) >= 3 and "'" not in _word:
        _BY_INITIAL.setdefault(_word[0], []).append(_word)


# --------------------------------------------------------------------------
# Stage 1: normalise (clauses of tokens, abbreviations, careful typo repair)
# --------------------------------------------------------------------------

def _tokenize(raw: str) -> tuple:
    """Return (clauses, typo_fixes). Names (capitalised words) are never 'fixed'."""
    clauses, fixes, seen = [], 0, 0
    for part in re.split(r"[.,;:!?\n]+|\s[-\u2014]\s", raw.replace("\u2019", "'")):
        clause = []
        for word in re.findall(r"[A-Za-z0-9']+", part):
            low = word.lower().strip("'")
            is_name = word[0].isupper() and seen > 0
            seen += 1
            if not low:
                continue
            for piece in ABBREVIATIONS.get(low, low).split():
                if (piece == low and not is_name and piece not in KNOWN_WORDS
                        and piece not in STOPWORDS and len(piece) >= 4
                        and piece.isalpha()):
                    match = difflib.get_close_matches(piece, _BY_INITIAL.get(piece[0], ()), n=1, cutoff=0.85)
                    if match:
                        piece, fixes = match[0], fixes + 1
                clause.append(piece)
        if clause:
            clauses.append(clause)
    return clauses, fixes


# --------------------------------------------------------------------------
# Stage 2: message form
# --------------------------------------------------------------------------

def _form(raw: str, tokens: list) -> tuple:
    """Return (is_question, is_imperative) from the shape of the message."""
    first = tokens[0] if tokens else ""
    question = (raw.rstrip().endswith("?") or first.split("'")[0] in WH_WORDS
                or (first in AUX_WORDS and len(tokens) > 1 and tokens[1] in SUBJECT_WORDS))
    return question, (first in IMPERATIVE_VERBS and not question)


def _yes_no(tokens: list) -> Optional[str]:
    if 0 < len(tokens) <= 3:
        if tokens[0] in YES_WORDS:
            return "confirmation"
        if tokens[0] in NO_WORDS or " ".join(tokens[:2]) in ("not really", "never mind"):
            return "denial"
    return None


# --------------------------------------------------------------------------
# Stage 3: intent evidence
# --------------------------------------------------------------------------

def _find_cues(padded: str) -> tuple:
    """Return ({intent: [(weight, span)]}, identity subjects)."""
    found, subjects = {}, []
    for intent, rules in INTENT_RULES.items():
        for pattern, weight in rules:
            match = re.search(pattern, padded)
            if match:
                found.setdefault(intent, []).append((weight, match.span()))
    for pattern, weight, subject in IDENTITY_RULES:
        match = re.search(pattern, padded)
        if match:
            found.setdefault(Intent.IDENTITY_QUESTION, []).append((weight, match.span()))
            if subject not in subjects:
                subjects.append(subject)
    return found, subjects


def _scores(found: dict) -> dict:
    scores = {i: min(0.95, max(w for w, _ in hits) + 0.05 * (len(hits) - 1)) for i, hits in found.items()}
    for strong, hidden in SUPERSEDES.items():
        if scores.get(strong, 0) >= 0.6:
            for intent in hidden:
                scores.pop(intent, None)
    return scores


def _context_scores(scores: dict, padded: str, tokens: list, question: bool,
                    personal: float, context: Context, yes_no: Optional[str]) -> None:
    """Evidence that comes from the conversation or message shape, not keywords."""
    if (padded.startswith(" how far ") and not padded.startswith(" how far is ")
            and len(tokens) <= 6 and Intent.GREETING not in scores):
        supported = not context.recent_turns or context.last_action is ActionKind.GREET
        scores[Intent.GREETING] = 0.75 if supported else 0.5
    if personal:
        scores[Intent.PERSONAL_INFORMATION] = max(scores.get(Intent.PERSONAL_INFORMATION, 0), personal)
    top = max(scores.values(), default=0.0)
    if context.pending_question and 1 <= len(tokens) <= 8 and top < 0.7:
        scores[Intent.CLARIFICATION_RESPONSE] = 0.8
    question_kinds = {Intent.FACTUAL_QUESTION, Intent.GENERAL_QUESTION, Intent.IDENTITY_QUESTION,
                      Intent.CAPABILITY_QUESTION, Intent.MEMORY_RECALL, Intent.OPINION, Intent.REQUEST}
    if question and len(tokens) >= 2 and not question_kinds & set(scores) and top < 0.5:
        scores[Intent.GENERAL_QUESTION] = 0.5


# --------------------------------------------------------------------------
# Stage 4: entities
# --------------------------------------------------------------------------

_NAME = r"([A-Za-z][\w'-]*(?:\s+[A-Z][\w'-]*)*)"
_CAP_NAME = r"([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)"
_INTRO_RE = re.compile(r"(?i:\b(?:my name is|call me|i am called|i'm called|people call me|named))\s+" + _NAME)
_SELF_RE = re.compile(r"(?i:\bi(?:'m| am))\s+" + _CAP_NAME)
_KIN_RE = re.compile(r"(?i:\bmy (?:brother|sister|friend|mum|mom|mother|dad|father|boss|wife|husband|"
                     r"cousin|uncle|aunt|son|daughter|colleague|teacher))\s+" + _CAP_NAME)
_TIME_RE = re.compile(r"(?i)\b(?:today|tomorrow|yesterday|tonight|this (?:morning|afternoon|evening|week|weekend)"
                      r"|next (?:week|month|year)|\d{1,2}(?::\d{2})?\s?(?:am|pm))\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def _entities(raw: str) -> tuple:
    """Return (entities with spans in raw, evidence that this is personal info)."""
    entities, personal, taken = [], 0.0, []

    def add(match, group, kind, confidence):
        entities.append(Entity(match.group(group), kind, match.span(group), confidence))
        taken.append(match.span())

    for match in _INTRO_RE.finditer(raw):
        add(match, 1, EntityKind.PERSON, 0.9)
        personal = max(personal, 0.9)
    for match in _SELF_RE.finditer(raw):
        name = match.group(1)
        if name.lower() not in EMOTION_WORDS and name.lower() not in STOPWORDS and not any(
                e.span == match.span(1) for e in entities):
            add(match, 1, EntityKind.PERSON, 0.8)
            personal = max(personal, 0.7)
    for match in _KIN_RE.finditer(raw):
        add(match, 1, EntityKind.PERSON, 0.8)
        personal = max(personal, 0.55)
    times = []
    for match in _TIME_RE.finditer(raw):
        entities.append(Entity(match.group(0), EntityKind.TIME, match.span(), 0.9))
        times.append(match.span())
    for match in _NUMBER_RE.finditer(raw):
        if not any(a <= match.start() < b for a, b in times):
            entities.append(Entity(match.group(0), EntityKind.NUMBER, match.span(), 1.0))
    return entities, personal


# --------------------------------------------------------------------------
# Stage 5: emotion cues (evidence only; sarcasm and mixed feelings not resolved)
# --------------------------------------------------------------------------

def _emotion(clauses: list, padded: str) -> tuple:
    """Return (cues, sentiment, ambiguities)."""
    cues, positive, negative, neutral = [], 0, 0, False
    for clause in clauses:
        for i, word in enumerate(clause):
            if word not in EMOTION_WORDS:
                continue
            cue, base = EMOTION_WORDS[word]
            negated = any(w in NEGATORS for w in clause[max(0, i - 3):i])
            if negated:
                cues.append("negated:" + cue)
                negative += 1 if base >= 0 else 0  # "not happy" and "not okay" are negative
            else:
                cues.append(cue)
                positive += base > 0
                negative += base < 0
                neutral = neutral or base == 0
            if any(w in INTENSIFIERS for w in clause[max(0, i - 2):i]):
                cues.append("intense")
    ambiguities = []
    if SARCASM_RE.search(padded):
        cues.append("possible_sarcasm")
        ambiguities.append("tone may be sarcastic")
        negative += 1
    if CRISIS_RE.search(padded):
        cues.append("possible_crisis")
        negative += 1
    if re.search(r" i (?:wan|want to|need to) (?:rest|sleep) ", padded):
        cues.append("wants_rest")
    if positive and negative:
        sentiment = Sentiment.MIXED
    elif positive:
        sentiment = Sentiment.POSITIVE
    elif negative:
        sentiment = Sentiment.NEGATIVE
    else:
        sentiment = Sentiment.NEUTRAL if neutral else Sentiment.UNKNOWN
    return tuple(dict.fromkeys(cues)), sentiment, ambiguities


# --------------------------------------------------------------------------
# Stage 6: references and unexplained words
# --------------------------------------------------------------------------

def _references(padded: str, intent: Intent, tokens: list, has_context: bool) -> tuple:
    """Return (references, ambiguities). Reports whether context offers a target."""
    refs, problems = [], []
    for pattern, label, session_only in EXPLICIT_REFERENCES:
        if re.search(pattern, padded):
            refs.append(label)
            if session_only and not has_context:
                problems.append(f"'{label}' has nothing earlier in this conversation to refer to")
    if intent in PRONOUN_INTENTS:
        for pronoun in dict.fromkeys(PRONOUN_RE.findall(padded)):
            refs.append(pronoun)
            if not has_context and len(tokens) <= 6:
                problems.append(f"'{pronoun}' has nothing earlier to refer to")
    return tuple(refs), problems


def _unexplained(padded: str, tokens: list, found: dict, entities: list) -> list:
    """Content words that no cue, entity or known word accounts for."""
    spans = [span for hits in found.values() for _, span in hits]
    entity_words = {w.lower() for e in entities for w in re.findall(r"[\w']+", e.text)}
    result, offset = [], 1
    for token in tokens:
        start, end = offset, offset + len(token)
        offset = end + 1
        covered = any(start < b and end > a for a, b in spans)
        if not (covered or token in STOPWORDS or token in KNOWN_WORDS
                or token in entity_words or token.isdigit()):
            result.append((token, start))
    return result


def _topics(raw: str, unexplained: list) -> list:
    """Unexplained neighbouring words become TOPIC entities (up to three)."""
    runs = []
    for word, start in unexplained:
        if runs and start == runs[-1][-1][1] + len(runs[-1][-1][0]) + 1:
            runs[-1].append((word, start))
        else:
            runs.append([(word, start)])
    topics = []
    for run in runs[:3]:
        text = " ".join(word for word, _ in run)
        at = raw.lower().find(text)
        topics.append(Entity(text, EntityKind.TOPIC, (at, at + len(text)) if at >= 0 else None, 0.5))
    return topics


# --------------------------------------------------------------------------
# Stage 7: dialogue act
# --------------------------------------------------------------------------

_ACT_BY_INTENT = {
    Intent.GREETING: DialogueAct.GREETING, Intent.FAREWELL: DialogueAct.FAREWELL,
    Intent.THANKS: DialogueAct.THANKING, Intent.APOLOGY: DialogueAct.APOLOGY,
    Intent.CORRECTION: DialogueAct.CORRECTION, Intent.REQUEST: DialogueAct.REQUEST,
    Intent.INSTRUCTION: DialogueAct.COMMAND, Intent.EMOTIONAL_DISCLOSURE: DialogueAct.EXPRESSION_OF_EMOTION,
    Intent.PERSONAL_INFORMATION: DialogueAct.STATEMENT,
}
_QUESTION_INTENTS = {Intent.IDENTITY_QUESTION, Intent.CAPABILITY_QUESTION, Intent.MEMORY_RECALL,
                     Intent.FACTUAL_QUESTION, Intent.GENERAL_QUESTION}


def _dialogue_act(intent: Intent, question: bool, imperative: bool, yes_no: Optional[str], tokens: list):
    if intent is Intent.CLARIFICATION_RESPONSE:
        return {"confirmation": DialogueAct.CONFIRMATION, "denial": DialogueAct.DENIAL}.get(
            yes_no, DialogueAct.CLARIFICATION)
    if intent in _ACT_BY_INTENT:
        return _ACT_BY_INTENT[intent]
    if intent in _QUESTION_INTENTS:
        return DialogueAct.REQUEST if imperative else DialogueAct.QUESTION
    if intent is Intent.OPINION:
        return DialogueAct.QUESTION if question else DialogueAct.STATEMENT
    if yes_no:
        return DialogueAct.CONFIRMATION if yes_no == "confirmation" else DialogueAct.DENIAL
    if question:
        return DialogueAct.QUESTION
    if imperative:
        return DialogueAct.COMMAND
    return DialogueAct.STATEMENT if len(tokens) >= 3 else DialogueAct.UNKNOWN


# --------------------------------------------------------------------------
# The analyzer
# --------------------------------------------------------------------------

class NLU:
    """Rule-based analyzer. Swap in a stronger one by providing the same analyze()."""

    analyzer = ANALYZER

    def analyze(self, text: str, context: Optional[Context] = None) -> Analysis:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("NLU needs a non-empty message")
        context = context if context is not None else Context()
        clauses, fixes = _tokenize(text)
        tokens = [t for clause in clauses for t in clause]
        normalized = " ".join(tokens) or text.strip().lower()
        padded = f" {normalized} "
        has_context = bool(context.recent_turns or context.current_topic)
        question, imperative = _form(text, tokens)
        yes_no = _yes_no(tokens)

        found, subjects = _find_cues(padded)
        entities, personal = _entities(text)
        scores = _scores(found)
        _context_scores(scores, padded, tokens, question, personal, context, yes_no)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], list(Intent).index(kv[0])))
        intent, top = ranked[0] if ranked else (Intent.UNKNOWN, 0.0)

        ambiguities = []
        confidence = top - 0.05 * min(fixes, 3)
        if len(ranked) > 1 and ranked[1][1] > top - 0.14:
            ambiguities.append(f"could also be {ranked[1][0].name.lower()}")
            confidence -= 0.15
        refs, problems = _references(padded, intent, tokens, has_context)
        if problems:
            ambiguities += problems
            confidence -= 0.2
        if intent is Intent.CORRECTION and not has_context:
            ambiguities.append("nothing earlier in this conversation to correct")
            confidence -= 0.2
        cues, sentiment, tone = _emotion(clauses, padded)
        ambiguities += tone
        if "possible_sarcasm" in cues:
            confidence -= 0.1

        unexplained = _unexplained(padded, tokens, found, entities)
        social = intent in SOCIAL_INTENTS
        if len(unexplained) > (2 if social else 3):
            ambiguities.append("many words in this message are outside what this analyzer interprets")
        if ambiguities:  # an unresolved ambiguity means at most a guess
            confidence = min(confidence, GUESS_CAP)
        confidence = round(max(0.0, min(confidence, 0.95)), 3)

        if confidence >= UNDERSTOOD_AT:
            understanding = Understanding.UNDERSTOOD
        elif confidence >= GUESSED_AT:
            understanding = Understanding.GUESSED
        else:
            understanding, intent = Understanding.NOT_UNDERSTOOD, Intent.UNKNOWN
        if intent is Intent.UNKNOWN:
            if yes_no:
                if not context.pending_question:
                    ambiguities.append("nothing earlier to confirm or deny")
            elif len(tokens) <= 2:
                ambiguities.append("too short to tell what is meant without earlier context")
            if not ambiguities:
                ambiguities.append("no pattern recognised")

        if intent is Intent.IDENTITY_QUESTION:
            entities += [Entity(s, EntityKind.SELF_REFERENCE, None, 1.0) for s in subjects]
        if intent in TOPIC_INTENTS:
            entities += _topics(text, unexplained)

        return Analysis(
            raw_text=text, normalized_text=normalized, intent=intent,
            dialogue_act=_dialogue_act(intent, question, imperative, yes_no, tokens),
            entities=entities, sentiment=sentiment, emotion_cues=cues, confidence=confidence,
            understanding=understanding, ambiguities=ambiguities, references=refs, analyzer=ANALYZER,
        )
