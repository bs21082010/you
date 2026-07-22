import ollama
import json
import os
import sys
from collections import Counter

MODEL_VERSION = "llama3.2"
DATASET_PATH = "curated_dataset.jsonl"
LAST_REPLY = None
LAST_PROMPT = None

PERSONA = f"""
You are a warm, empathetic mentor.
Always respond with encouragement, positivity, and emotional awareness.
Use supportive language, motivational tone, and show understanding.
Never give harmful or misleading advice.
If you are uncertain about something, say: "I don't have that information right now, but I can help you explore it."
You are running on {MODEL_VERSION}.
"""

FALLBACK = "I don't have that information right now, but I can help you explore it."

PROMPT_TEMPLATES = {
    "explain": "Explain this in a clear, kind way:\n\n{{query}}",
    "motivate": "Give me a motivational push about:\n\n{{query}}",
    "advise": "Offer supportive advice on:\n\n{{query}}",
}


def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_to_dataset(prompt, completion, path=DATASET_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")


def dataset_stats(path=DATASET_PATH):
    data = load_dataset(path)
    if not data:
        print("📭 Dataset is empty.")
        return
    prompts = [d["prompt"] for d in data]
    topics = Counter()
    for p in prompts:
        for key in PROMPT_TEMPLATES:
            if p.lower().startswith(key):
                topics[key] += 1
                break
        else:
            topics["general"] += 1
    print(f"📊 Dataset: {len(data)} entries | Model: {MODEL_VERSION}")
    for topic, count in topics.most_common():
        print(f"   {topic}: {count}")


def validate_dataset(path=DATASET_PATH):
    data = load_dataset(path)
    if not data:
        print("✅ No entries to validate.")
        return
    issues = 0
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "prompt" not in entry or "completion" not in entry:
            print(f"⚠️  Entry {i}: malformed (missing keys)")
            issues += 1
            continue
        if len(entry["completion"]) < 5:
            print(f"⚠️  Entry {i}: completion too short")
            issues += 1
        if "i don't know" in entry["completion"].lower() and len(entry["completion"]) < 20:
            print(f"⚠️  Entry {i}: uncertain completion without exploration offer")
            issues += 1
    if issues == 0:
        print("✅ All entries valid.")
    else:
        print(f"⚠️  {issues} issue(s) found. Consider reviewing.")


def apply_template(user_input):
    for key, template in PROMPT_TEMPLATES.items():
        if user_input.lower().startswith(key):
            return template.format(query=user_input[len(key):].strip())
    return user_input


def chat_with_feeling():
    global LAST_REPLY, LAST_PROMPT

    print("💡 Emotional Ollama Chat Started")
    print(f"   Model: {MODEL_VERSION}  |  Dataset: {DATASET_PATH}")
    print("   Commands:  exit/quit  |  99 (connect)  |  correct <new text>  |  stats  |  validate")
    try:
        while True:
            user_input = input("You: ")
            cmd = user_input.strip().lower()

            if cmd in ["exit", "quit"]:
                print("👋 Ending session.")
                break

            if cmd == "99":
                print("📞 Connecting to your empathetic AI assistant...")
                continue

            if cmd == "stats":
                dataset_stats()
                continue

            if cmd == "validate":
                validate_dataset()
                continue

            if cmd.startswith("correct") and LAST_REPLY is not None:
                corrected = user_input[len("correct"):].strip()
                if corrected:
                    append_to_dataset(LAST_PROMPT, corrected)
                    print("✅ Correction saved to dataset.")
                    LAST_REPLY = corrected
                continue

            prompt = apply_template(user_input)
            try:
                response = ollama.chat(
                    model=MODEL_VERSION,
                    messages=[
                        {"role": "system", "content": PERSONA},
                        {"role": "user", "content": prompt},
                    ],
                )
                reply = response["message"]["content"]
                print("Ollama:", reply)

                LAST_PROMPT = prompt
                LAST_REPLY = reply
                append_to_dataset(prompt, reply)
            except Exception:
                print("⚠️", FALLBACK)

    except KeyboardInterrupt:
        print("\n👋 Session interrupted. Goodbye!")


if __name__ == "__main__":
    if "--stats" in sys.argv:
        dataset_stats()
    elif "--validate" in sys.argv:
        validate_dataset()
    else:
        chat_with_feeling()
