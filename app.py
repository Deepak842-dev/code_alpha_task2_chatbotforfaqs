"""
Flask server for the FAQ chatbot.

Routes:
  GET  /            -> chat UI
  POST /api/chat    -> {"message": "..."}  =>  {"answer", "matched_question", "score"}
  GET  /api/faqs    -> list of all FAQs (used to render quick-suggestion chips)

Run:
  python3 app.py
Then open http://localhost:5000
"""

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from chatbot import FAQChatbot

app = Flask(__name__)

FAQ_PATH = Path(__file__).parent / "faq_data.json"
bot = FAQChatbot(str(FAQ_PATH), threshold=0.5)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    result = bot.get_response(message)
    return jsonify(result)


@app.route("/api/faqs")
def faqs():
    return jsonify([f["question"] for f in bot.faqs])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
