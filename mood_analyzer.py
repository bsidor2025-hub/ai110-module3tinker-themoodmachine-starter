# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
import string
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

NEGATORS = {"not", "no", "never"}
NEGATION_WINDOW = 2  # how many tokens after a negator it can still flip a word

# Irregular contractions that the generic "n't" rule would get wrong.
CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "i'm": "i am",
}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          - Strip whitespace and lowercase
          - Expand contractions ("don't" -> "do not")
          - Normalize repeated characters ("soooo" -> "soo")
          - Tokenize into words, dropping ASCII punctuation
          - Keep emojis as separate tokens ("lol😂" -> "lol", "😂")
        """
        cleaned = text.strip().lower()

        # Normalize curly apostrophes, then expand contractions so negation
        # ("don't" -> "do not") is visible as its own token.
        cleaned = cleaned.replace("’", "'")
        for contraction, expanded in CONTRACTIONS.items():
            cleaned = cleaned.replace(contraction, expanded)
        cleaned = re.sub(r"n't\b", " not", cleaned)

        # Squash 3+ repeated characters down to 2 ("soooo" -> "soo").
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)

        # Words (letters, digits, inner apostrophes) stay together; every other
        # non-space character becomes its own token. ASCII punctuation is then
        # dropped, which leaves emojis and emoticon characters as tokens.
        raw_tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]", cleaned)
        tokens = [t for t in raw_tokens if not (len(t) == 1 and t in string.punctuation)]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        score, _, _ = self._analyze(text)
        return score

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Shared scoring logic used by score_text, predict_label and explain.

        Returns (score, positive_hits, negative_hits). A negator ("not", "no",
        "never") flips the polarity of the next sentiment word if it appears
        within NEGATION_WINDOW tokens, e.g. "not happy" counts as negative.
        Negated hits are recorded with their negator, e.g. "not happy".
        """
        tokens = self.preprocess(text)
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0
        negator: Optional[str] = None
        since_negator = 0

        for token in tokens:
            if token in NEGATORS:
                negator = token
                since_negator = 0
                continue

            polarity = 0
            if token in self.positive_words:
                polarity = 1
            elif token in self.negative_words:
                polarity = -1

            if polarity != 0:
                label = token
                if negator is not None and since_negator < NEGATION_WINDOW:
                    polarity = -polarity
                    label = f"{negator} {token}"
                    negator = None
                score += polarity
                (positive_hits if polarity > 0 else negative_hits).append(label)
            elif negator is not None:
                since_negator += 1
                if since_negator >= NEGATION_WINDOW:
                    negator = None

        return score, positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score, positive_hits, negative_hits = self._analyze(text)

        # Both kinds of signal present -> mixed, even if they cancel out.
        if positive_hits and negative_hits:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        score, positive_hits, negative_hits = self._analyze(text)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
