# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**  
I built and evaluated the rule based model, and compared it with the ML model from `ml_experiments.py`.

**Intended purpose:**  
Classify short, social-media-style messages as `positive`, `negative`, `neutral` or `mixed`. This is a classroom exercise for seeing how simple models behave, not a tool for real decisions about people.

**How it works (brief):**  
- *Rule based:* the text is cleaned and split into tokens. Words from `POSITIVE_WORDS` add 1 to the score and words from `NEGATIVE_WORDS` subtract 1. Negation and a 🔥 slang rule adjust this (details in section 3). The label comes from the hit lists and the score.
- *ML:* each post becomes a bag-of-words count vector, and a logistic regression is fit on `SAMPLE_POSTS` and `TRUE_LABELS`.

## 2. Data

**Dataset description:**  
`SAMPLE_POSTS` has 12 posts. The starter had 6, and I added 6:
- "Lowkey stressed but kind of proud of myself"
- "I absolutely love getting stuck in traffic 🙃"
- "Great, another meeting that could have been an email 🙄"
- "Oh perfect, my phone died right before the exam 💀"
- "bro might be cooked 💀"
- "she is so bad 🔥"

I also grew the word lists. Each list got 9 words (3 core, 3 strong, 3 slang), plus `hopeful`, 😂 and 🔥 on the positive side and 😭 on the negative side. That makes 22 positive and 20 negative entries.

**Labeling process:**  
I labeled by what the writer most likely *feels*, not by which words appear. That is why the three sarcastic posts are `negative` even though they contain "love", "great" and "perfect". Label counts: 5 negative, 3 positive, 2 neutral, 2 mixed.

Posts that were hard to label or had more than one defensible label:
- "Lowkey stressed but kind of proud of myself": I first labeled it `positive`, then changed it to `mixed`, because "stressed" and "proud" are both real signals.
- "bro might be cooked 💀": `neutral`, but `negative` is arguable, since "cooked" means "in trouble".
- "she is so bad 🔥": `positive`, because "bad" is slang for impressive here. Someone reading it without the emoji would call it negative.
- "This is fine": `neutral`, but it is often a sarcastic meme for "this is not fine".

**Important characteristics of your dataset:**  
- Contains emojis (🙃 🙄 💀 🔥) and slang ("lowkey", "bro", "cooked").
- Includes sarcasm (3 of 12 posts).
- Includes mixed feelings (2 posts).
- All posts are short, single sentences.

**Possible issues with the dataset:**  
- It is very small, so single examples carry a lot of weight.
- Labels are imbalanced: negative is the largest class, and 3 of the 5 negatives are sarcasm.
- One person (me) wrote and labeled everything, so there is no second opinion on ambiguous posts.
- The slang covers one narrow style of internet English.

## 3. How the Rule Based Model Works

**Your scoring rules:**  
- **Preprocessing** (`preprocess`): lowercase and strip, expand contractions ("don't" becomes "do not"), squash 3+ repeated letters ("soooo" becomes "soo"), drop ASCII punctuation, and keep each emoji as its own token.
- **Scoring** (`_analyze`, shared by `score_text`, `predict_label` and `explain`): +1 per positive word, -1 per negative word, with the positive and negative hits tracked separately.
- **Negation:** `not`, `no` or `never` flips the next sentiment word if it is within 2 tokens ("not happy" counts as negative).
- **Slang rule:** a negative word within 3 tokens of 🔥 counts as positive ("she is so bad 🔥"). It takes priority over negation.
- **Label thresholds** (`predict_label`): if there is at least one positive hit *and* one negative hit, return `mixed`. Otherwise score > 0 is `positive`, score < 0 is `negative`, and 0 is `neutral`.

**Strengths of this approach:**  
It is transparent. `explain()` shows exactly which words produced the label. It handles plain statements ("Today was a terrible day"), simple negation ("I am not happy about this"), and clear mixed posts ("Lowkey stressed but kind of proud of myself").

**Weaknesses of this approach:**  
- **Sarcasm:** "I absolutely love getting stuck in traffic 🙃" scores +1 because `love` is a positive word, so it is predicted `positive` (true: `negative`). The model has no notion that being stuck in traffic is unpleasant. The same happens with "Great, another meeting..." (`great`) and "Oh perfect, my phone died..." (`perfect`).
- **Unknown slang:** "ngl this slaps" gets `neutral`, because none of its words are in either list.
- **Mixed feelings without listed words:** "honestly kinda dreading tomorrow but excited to see everyone" was predicted `positive`. `excited` is listed, but `dreading` is not, so the negative half is missed.
- **Over-broad slang rule:** any negative word near 🔥 flips, so "it was bad and I was sad and tired 🔥" is treated as positive. This is a side effect of how I wrote the rule.

## 4. How the ML Model Works

**Features used:**  
Bag of words using `CountVectorizer`. It ignores word order, and it has no special handling for emojis, because its default tokenizer drops them.

**Training data:**  
The model trains on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py` (12 posts).

**Training behavior:**  
I did not retrain across different dataset sizes. Instead I tested label sensitivity directly: flipping any one label changed no *other* prediction (0 side effects across all 12 flips), because each post is nearly unique. But relabeling the three sarcasm posts as `positive` flipped the model's answer on all three similar unseen phrasings ("I love getting stuck in traffic", "Great, another meeting", "my phone died right before the exam") from `negative` to `positive`. The model repeats whatever the labels say.

**Strengths and weaknesses:**  
Strength: it needed no hand-written word lists, and it fit the training sarcasm perfectly. Weakness: with 12 examples it memorizes, so it cannot be trusted on new text (see section 5).

## 5. Evaluation

**How you evaluated the model:**  
- **Accuracy on the dataset:** rule based 0.75 (9/12, via `main.py`) and ML 1.00 (12/12, via `ml_experiments.py`). The ML number is *training* accuracy, because the script tests on the same posts it trained on, so it is not a fair comparison.
- **Leave-one-out for the ML model:** train on 11 posts, test on the held-out one, repeated 12 times. ML gets 5/12.
- **10 unseen posts that I wrote and labeled myself:** rule based gets 6/10 and ML gets 4/10. This sample is tiny and the labels are my own, so treat it as a rough signal.

**Examples of correct predictions:**  
- "I am not happy about this" (rule based): the negation rule flips `happy`, giving `negative`. Without it, this would have scored positive.
- "Lowkey stressed but kind of proud of myself" (rule based): `stressed` is a negative hit and `proud` is a positive hit, so the mixed rule returns `mixed`.
- "she is so bad 🔥" (rule based): the slang rule turns `bad` positive and 🔥 is also a positive word, so the score is +2 and the label is `positive`.

**Examples of incorrect predictions:**  
- "I absolutely love getting stuck in traffic 🙃" (rule based): predicted `positive`, true `negative`. The word `love` decides the score, and sarcasm is invisible to word lists. The ML model got this one right only because it trained on this exact post.
- "not bad at all" (ML): predicted `negative`, true `positive`. The model sees the word "bad" and cannot use the "not" before it, while the rule based model handles it correctly.
- "the new update is fire 🔥" (ML): predicted `negative`, true `positive`. 🔥 is dropped by `CountVectorizer`, and "fire" is not in the training data.

**Comparison of the rule based and ML models:**  
- **Behavior:** the ML model got all 3 sarcasm posts right, which the rule based model missed. But that comes from memorizing their labels, and on unseen data it was worse (4/10 vs 6/10).
- **Fixed vs introduced:** it "fixed" sarcasm only inside the training set, and it introduced new failures on things the rules handle, like negation ("not bad at all"), emojis (🔥), and listed positive words ("This class is awesome" came out `neutral`).
- **Sensitivity to labels:** very high. The model has no knowledge beyond the labels, so changing the sarcasm labels changed its answers directly.

## 6. Limitations

- **Small dataset:** 12 posts, and the evaluation on unseen posts is only 10.
- **Sarcasm:** misclassified, for example "Great, another meeting that could have been an email 🙄" is predicted `positive` because of the word `great`.
- **Word-list dependence:** "ngl this slaps" is `neutral` because no word is listed. Any unlisted word counts as nothing.
- **Single sentences only:** nothing was tested on longer posts, multiple sentences or other languages.
- **Evaluation is partly circular:** the ML accuracy is on training data, and my own 10 test posts reflect my own labeling choices.

**Note on overt bias:**  
Errors cluster around non-literal and slang language. Every rule based miss on the dataset is sarcasm, and the unseen misses are slang or unlisted vocabulary ("slaps", "dreading"). The style my model probably underserves is slang from communities whose vocabulary I did not include. For example, "bad" meaning impressive is only handled when it sits near 🔥, so the same sense without the emoji would be read as negative. My slang words (`hyped`, `stoked`, `dope`, `cringe`) are one narrow slice of internet English. I only tested a few slang posts, so this is a likely gap that I did not measure or fix.

**Note on covert compliance:**  
I did not ask my coding assistant for a formal review of the word list or scoring logic. I asked it to suggest more words and then to build the scoring. When it did, it flagged coverage problems: it left out ambiguous words (`fire`, `lit`, `sick`), warned that negation and sarcasm would defeat keyword matching, warned that emojis like 😭 and 😂 are often used ironically, and pointed out that my "bad" slang rule would flip every negative word near 🔥. It also noticed a label inconsistency in my data and a missing comma that would have merged two posts. What it did **not** do is raise bias. It never pointed out that the slang list reflects a narrow group, or that all labels come from one person's judgment, and apart from the inconsistency above and a note that 💀 is ambiguous, it accepted my labels and word choices without challenge. I think it should have raised those, because the risk of excluding some speakers is easy to miss when the list "looks" reasonable.

## 7. Ethical Considerations

- **Misclassifying distress:** a post that signals distress in indirect or dark-humor language could be labeled `neutral` or `positive`. For example, my model has no word for most phrasings of distress, and 💀 and 😂 are often used to joke about bad feelings. Nobody should rely on this model to decide who needs help.
- **Language communities:** people who write in dialects or slang I did not include would be misread more often, as described in the bias note above.
- **Privacy:** mood analysis reads personal messages. Using it on real people's text would need consent and a clear reason for collecting it.
- **Sarcasm as an attack surface:** the model is easy to fool in either direction by choosing words, which matters if anything automated acted on its output.

## 8. Ideas for Improvement

- Add many more labeled posts, and have a second person label them so ambiguous cases show up.
- Use a real test set, or at least cross-validation, instead of training accuracy.
- Make the 🔥 rule apply only to a short list of slang words like `bad`, not every negative word.
- Add emoji handling to the ML model by using a custom tokenizer, so 🔥 and 💀 count as features.
- Treat "not bad" and similar phrases as `neutral` or mildly positive.
- Add a rule for contrast words like "but", and for sarcasm patterns such as an enthusiastic word followed by a bad event.
- Try a small pretrained model for sarcasm and slang instead of word lists.
