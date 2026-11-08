"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # core
    "glad",
    "proud",
    "grateful",
    "hopeful",
    # strong praise
    "wonderful",
    "fantastic",
    "perfect",
    # slang
    "hyped",
    "stoked",
    "dope",
    # emoji
    "😂",
    "🔥",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # core
    "annoyed",
    "frustrated",
    "anxious",
    # strong dislike
    "worst",
    "horrible",
    "sucks",
    # slang
    "meh",
    "cringe",
    "drained",
    # emoji
    "😭",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "Lowkey stressed but kind of proud of myself",
    "I absolutely love getting stuck in traffic 🙃",
    "Great, another meeting that could have been an email 🙄",
    "Oh perfect, my phone died right before the exam 💀",
    "bro might be cooked 💀",
    "she is so bad 🔥",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "mixed",  # "Lowkey stressed but kind of proud of myself"
    "negative",  # "I absolutely love getting stuck in traffic 🙃" (sarcasm)
    "negative",  # "Great, another meeting that could have been an email 🙄" (sarcasm)
    "negative",  # "Oh perfect, my phone died right before the exam 💀" (sarcasm)
    "neutral",  # "bro might be cooked 💀"
    "positive",  # "she is so bad 🔥" (slang: "bad" meaning impressive)
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
print(len(SAMPLE_POSTS) == len(TRUE_LABELS))
