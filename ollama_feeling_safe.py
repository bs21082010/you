import ollama
import json
import os

PERSONA = """
You are a warm, empathetic mentor.
Always respond with encouragement, positivity, and emotional awareness.
Use supportive language, motivational tone, and show understanding.
Never give harmful or misleading advice.
"""

FALLBACK = "I don't have that information right now, but I can help you explore it."

PROMPT_TEMPLATES = {
    "explain": "Explain this in a clear, kind way:\n\n{query}",
    "motivate": "Give me a motivational push about:\n\n{query}",
    "advise": "Offer supportive advice on:\n\n{query}",
}

DATASET_PATH = "curated_dataset.jsonl"

def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_to_dataset(prompt, completion, path=DATASET_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")

def apply_template(user_input):
    for key, template in PROMPT_TEMPLATES.items():
        if user_input.lower().startswith(key):
            return template.format(query=user_input[len(key):].strip())
    return user_input

def chat_with_feeling():
    print("💡 Emotional Ollama Chat Started (type '99' to connect, 'exit' to quit)")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Ending session.")
                break
            if user_input.strip() == "99":
                print("📞 Connecting to your empathetic AI assistant...")
                continue

            prompt = apply_template(user_input)
            try:
                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {"role": "system", "content": PERSONA},
                        {"role": "user", "content": prompt},
                    ],
                )
                reply = response["message"]["content"]
                print("Ollama:", reply)
                append_to_dataset(prompt, reply)
            except Exception:
                print("⚠️", FALLBACK)

    except KeyboardInterrupt:
        print("\n👋 Session interrupted. Goodbye!")

if __name__ == "__main__":
    chat_with_feeling()
