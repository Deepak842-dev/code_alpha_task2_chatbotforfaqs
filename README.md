# FAQ Chatbot

A simple FAQ chatbot that matches a user's question against a stored set of
FAQs using **NLP preprocessing + TF-IDF vectorization + cosine similarity**,
served through a small **Flask** API with a ready-made **chat UI**.

## How it works

1. **FAQ data** — `faq_data.json` holds `{"question": ..., "answer": ...}`
   pairs. Replace this with your own product/topic FAQs.
2. **Preprocessing** (`chatbot.py`) — lowercases text, strips punctuation,
   removes stopwords, and lightly stems words (e.g. "shipping" → "shipp"),
   so "How do I track my package?" and "package tracking" line up.
   *(Implemented with a small built-in stopword list + suffix stripper
   instead of NLTK, so it needs no external data downloads — functionally
   the same preprocessing NLTK's `stopwords` + `PorterStemmer` would give.
   Swap in NLTK/spaCy directly in `preprocess()` if you prefer.)*
3. **Matching** — all FAQ questions are vectorized with `TfidfVectorizer`
   (unigrams + bigrams). A user's message is vectorized the same way and
   compared to every FAQ question with cosine similarity. The closest
   match is returned if its score clears a confidence threshold **and**
   isn't too close to the second-best match (which would mean the
   question is genuinely ambiguous between two FAQs).
4. **Fallback** — if nothing matches well, the bot admits it doesn't know
   instead of guessing.

## Run it

```bash
pip install flask scikit-learn
cd faq_chatbot
python3 app.py
```

Then open **http://localhost:5000** for the chat UI.

### Command-line version (no Flask)

```bash
python3 chatbot.py
```

## Customizing

- **Add your own FAQs**: edit `faq_data.json`. No code changes needed.
- **Tune matching strictness**: `FAQChatbot(faq_path, threshold=0.5)` in
  `app.py` — raise it for fewer, more confident matches; lower it for more
  lenient (but riskier) matches.
- **Swap in real NLTK/spaCy**: replace the `preprocess()` function in
  `chatbot.py` with your NLTK/spaCy pipeline (tokenize, remove stopwords,
  lemmatize) — the rest of the matching pipeline doesn't need to change.

## Known limitation

TF-IDF/cosine similarity matches on **shared words**, not meaning — it
doesn't know "return" and "money back" or "open" and "business hours" are
related unless those words literally co-occur in your FAQ questions. Two
ways to improve this if you outgrow it:

- Add more paraphrased questions per FAQ entry (e.g. list a few ways users
  phrase the same question, and treat a match to any of them as a hit).
- Move to embeddings (e.g. `sentence-transformers`) for meaning-based
  ("semantic") matching instead of exact word overlap — same overall
  pipeline, just replace the vectorizer + similarity step.

## Files

```
faq_chatbot/
├── app.py            # Flask server (API + serves the UI)
├── chatbot.py         # Preprocessing + TF-IDF matching engine (+ CLI demo)
├── faq_data.json       # Your FAQ question/answer pairs
├── templates/
│   └── index.html      # Chat UI markup
└── static/
    ├── style.css        # Chat UI styling
    └── script.js        # Chat UI client logic (calls /api/chat)
```
