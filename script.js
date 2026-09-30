const messagesEl = document.getElementById("chat-messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("user-input");
const suggestionsEl = document.getElementById("suggestions");

function addMessage(text, sender, meta) {
  const wrap = document.createElement("div");
  wrap.className = `message ${sender}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (meta) {
    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }

  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage(text) {
  addMessage(text, "user");
  inputEl.value = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    const confidencePct = Math.round((data.score || 0) * 100);
    addMessage(data.answer, "bot", `Match confidence: ${confidencePct}%`);
  } catch (err) {
    addMessage("Sorry, something went wrong reaching the server.", "bot");
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (text) sendMessage(text);
});

// Load a few FAQ questions as quick-tap suggestion chips
async function loadSuggestions() {
  try {
    const res = await fetch("/api/faqs");
    const questions = await res.json();
    questions.slice(0, 4).forEach((q) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = q;
      chip.addEventListener("click", () => sendMessage(q));
      suggestionsEl.appendChild(chip);
    });
  } catch (err) {
    // Non-critical if suggestions fail to load
  }
}

loadSuggestions();
