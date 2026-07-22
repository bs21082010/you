import ollama
import json
import os
import sys
from collections import Counter

MODEL_VERSION = "llama3.2"
DATASET_PATH = "curated_dataset.jsonl"
WEAK_REPLIES_PATH = "weak_replies.jsonl"
SCORE_THRESHOLD = 70
LAST_REPLY = None
LAST_PROMPT = None
LAST_SCORE = None

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


def append_to_weak(prompt, completion, score, path=WEAK_REPLIES_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "completion": completion, "score": score}) + "\n")


def generate_variations(prompt, corrected):
    variations = [
        corrected,
        corrected.replace("I think", "I believe"),
        corrected.replace("maybe", "").replace("perhaps", ""),
    ]
    variations = [v for v in variations if len(v) > 20]
    variations = list(set(variations))
    for v in variations:
        append_to_dataset(prompt, v)


def score_reply(reply):
    score = 100
    words = reply.split()
    if len(words) < 20:
        score -= 20
    if len(words) < 10:
        score -= 15
    lower = reply.lower()
    if "i don't know" in lower or "unsure" in lower:
        score -= 30
    if "i don't have that information" in lower:
        score -= 10
    if any(w in lower for w in ["maybe", "sort of", "perhaps", "kind of"]):
        score -= 10
    if reply.endswith("?") or reply.endswith("..."):
        score -= 5
    return max(score, 0)


def topic_balance(data):
    prompts = [d["prompt"] for d in data]
    topics = Counter()
    for p in prompts:
        for key in PROMPT_TEMPLATES:
            if p.lower().startswith(key):
                topics[key] += 1
                break
        else:
            topics["general"] += 1
    if not topics:
        return topics
    total = sum(topics.values())
    ideal = total / len(topics)
    print("   ⚖️  Balance:")
    for topic, count in topics.most_common():
        sign = "+" if count >= ideal else "−"
        print(f"      {topic}: {count} ({sign}{abs(count - ideal):.0f} vs ideal)")
    return topics


def dataset_stats(path=DATASET_PATH):
    data = load_dataset(path)
    if not data:
        print("📭 Dataset is empty.")
        return
    topics = topic_balance(data)
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


def train_from_dataset(base_model=MODEL_VERSION, path=DATASET_PATH, new_model="empathetic-mentor"):
    data = load_dataset(path)
    if len(data) < 5:
        print("⚠️  Need at least 5 dataset entries to train.")
        return
    print(f"🧠 Training '{new_model}' from {base_model} using {len(data)} examples...")
    few_shot = ""
    for i, entry in enumerate(data[:20]):
        few_shot += f"User: {entry['prompt']}\nAssistant: {entry['completion']}\n\n"
    modelfile = f"""FROM {base_model}
SYSTEM \"\"\"{PERSONA.strip()}\"\"\"
TEMPLATE \"\"\"System: {PERSONA.strip()}

{few_shot}
User: {{.Prompt}}
Assistant: \"\"\"
"""
    modelfile_path = "Modelfile.tmp"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile)
    try:
        ollama.create(model=new_model, modelfile=modelfile_path)
        print(f"✅ Model '{new_model}' created. Run it with: ollama run {new_model}")
    except Exception as e:
        print(f"⚠️ Training failed: {e}")
    finally:
        if os.path.exists(modelfile_path):
            os.remove(modelfile_path)


def apply_template(user_input):
    for key, template in PROMPT_TEMPLATES.items():
        if user_input.lower().startswith(key):
            return template.format(query=user_input[len(key):].strip())
    return user_input


def chat_with_feeling():
    global LAST_REPLY, LAST_PROMPT, LAST_SCORE

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
                    generate_variations(LAST_PROMPT, corrected)
                    print("   Variations generated.")
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
                score = score_reply(reply)
                print(f"Ollama: {reply}  [confidence: {score}/100]")

                LAST_PROMPT = prompt
                LAST_REPLY = reply
                LAST_SCORE = score

                if score < 50:
                    append_to_weak(prompt, reply, score)
                    print(f"⚠️  Score {score}/100 — auto-rejected. Using fallback.")
                    print("Ollama:", FALLBACK)
                    append_to_dataset(prompt, FALLBACK)
                elif score < SCORE_THRESHOLD:
                    append_to_weak(prompt, reply, score)
                    print(f"⚠️  Reply scored {score}/100 — below threshold.")
                    fix = input("   Correct it? (leave blank to skip, or type correction): ").strip()
                    if fix:
                        append_to_dataset(prompt, fix)
                        print("✅ Correction saved.")
                        generate_variations(prompt, fix)
                        print("   Variations generated.")
                else:
                    if score > 90:
                        print("   ✅ Auto-approved (score > 90).")
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
    elif "--train" in sys.argv:
        train_from_dataset()
    else:
        chat_with_feeling()
