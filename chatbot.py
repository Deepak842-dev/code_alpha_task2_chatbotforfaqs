"""
FAQ Chatbot core engine.

Pipeline:
  1. Load FAQs (question/answer pairs) from a JSON file.
  2. Preprocess text (lowercase, tokenize, remove stopwords/punctuation, stem).
  3. Vectorize all FAQ questions with TF-IDF.
  4. For a new user query, vectorize it the same way and compute cosine
     similarity against every FAQ question.
  5. Return the best-matching answer (with a confidence score), or a
     fallback message if nothing matches well enough.
"""

import json
import re
import string
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Lightweight NLP preprocessing (no external downloads required).
# Functionally equivalent to what you'd get from NLTK (stopword removal +
# stemming) but implemented with a built-in list so it runs fully offline.
# --------------------------------------------------------------------------

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "s", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
}

# A tiny Porter-style suffix stripper. Not a full stemmer, but normalizes
# common variants (e.g. "shipping"/"shipped"/"ships" -> "ship") well enough
# for FAQ-length text.
_SUFFIXES = ("ing", "edly", "ed", "ly", "es", "s")


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if len(word) - len(suf) >= 3 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation, tokenize, drop stopwords, stem."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    tokens = [_stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# FAQ matching engine
# --------------------------------------------------------------------------

class FAQChatbot:
    def __init__(self, faq_path: str, threshold: float = 0.5, ambiguity_gap: float = 1.1):
        """
        faq_path: path to a JSON file: [{"question": "...", "answer": "..."}, ...]
        threshold: minimum cosine similarity to accept a match (0-1).
                   Below this, the bot admits it doesn't know.
        ambiguity_gap: the top match's score must be at least this many times
                   the runner-up's score to be trusted; otherwise the query
                   is treated as too ambiguous between two FAQs to answer.
        """
        self.threshold = threshold
        self.ambiguity_gap = ambiguity_gap
        self.faqs = self._load_faqs(faq_path)
        self.questions = [f["question"] for f in self.faqs]
        self._processed_questions = [preprocess(q) for q in self.questions]

        # unigrams + bigrams and sublinear TF scaling noticeably improve
        # short-question matching over a plain TF-IDF bag-of-words setup.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.question_vectors = self.vectorizer.fit_transform(self._processed_questions)

    @staticmethod
    def _load_faqs(path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not data:
            raise ValueError("FAQ file is empty.")
        return data

    def get_response(self, user_query: str) -> dict:
        """Return {'answer', 'matched_question', 'score'} for the best match."""
        if not user_query or not user_query.strip():
            return {
                "answer": "Could you type a question? I'm listening.",
                "matched_question": None,
                "score": 0.0,
            }

        processed = preprocess(user_query)
        query_vec = self.vectorizer.transform([processed])
        similarities = cosine_similarity(query_vec, self.question_vectors)[0]

        ranked = similarities.argsort()[::-1]
        best_idx, runner_up_idx = ranked[0], ranked[1]
        best_score = float(similarities[best_idx])
        runner_up_score = float(similarities[runner_up_idx])

        too_ambiguous = (
            runner_up_score > 0
            and best_score < runner_up_score * self.ambiguity_gap
        )

        if best_score < self.threshold or too_ambiguous:
            return {
                "answer": (
                    "I'm not confident I have an answer for that yet. "
                    "Could you rephrase, or ask about something else related "
                    "to our FAQs?"
                ),
                "matched_question": None,
                "score": best_score,
            }

        return {
            "answer": self.faqs[best_idx]["answer"],
            "matched_question": self.faqs[best_idx]["question"],
            "score": best_score,
        }

    def top_matches(self, user_query: str, k: int = 3):
        """Return the top-k candidate FAQ matches with scores (for debugging/UI)."""
        processed = preprocess(user_query)
        query_vec = self.vectorizer.transform([processed])
        similarities = cosine_similarity(query_vec, self.question_vectors)[0]
        ranked = similarities.argsort()[::-1][:k]
        return [
            {
                "question": self.faqs[i]["question"],
                "answer": self.faqs[i]["answer"],
                "score": float(similarities[i]),
            }
            for i in ranked
        ]


if __name__ == "__main__":
    # Simple command-line demo
    bot = FAQChatbot(str(Path(__file__).parent / "faq_data.json"))
    print("FAQ Chatbot (type 'quit' to exit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        result = bot.get_response(q)
        print(f"Bot: {result['answer']}  (score={result['score']:.2f})\n")
