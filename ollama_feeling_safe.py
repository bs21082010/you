import ollama
import json
import os
import sys
import shutil
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from collections import Counter

MODEL_VERSION = "llama3.2"
DATASET_PATH = "curated_dataset.jsonl"
WEAK_REPLIES_PATH = "weak_replies.jsonl"
CACHE_PATH = "knowledge_cache.jsonl"
CONFLICT_PATH = "conflict_queue.jsonl"
GOLD_DIR = "gold"
SNAPSHOT_DIR = "snapshots"
SYNC_DIR = None
SCORE_THRESHOLD = 70
CACHE_MAX_DAYS = 7
LOW_CONF_ESCALATION_COUNT = 3
LAST_REPLY = None
LAST_PROMPT = None
LAST_SCORE = None
RECENT_SCORES = []
CONSECUTIVE_LOW = 0

EXTERNAL_ENABLED = False
EXTERNAL_SOURCES = {
    "duckduckgo": "https://api.duckduckgo.com/?q={query}&format=json&no_html=1",
    "wikipedia": "https://en.wikipedia.org/api/rest_v1/page/summary/{query}",
}

PERSONAS = {
    "mentor": """You are a warm, empathetic mentor.
Always respond with encouragement, positivity, and emotional awareness.
Use supportive language, motivational tone, and show understanding.
Never give harmful or misleading advice.
If uncertain, say you don't know but offer to help explore it.""",

    "coach": """You are a direct, results-oriented coach.
Always respond with clarity, actionable steps, and accountability.
Push the user toward growth with firm but supportive language.
Never give harmful or misleading advice.
If uncertain, say you don't know but offer to help explore it.""",

    "teacher": """You are a patient, knowledgeable teacher.
Always respond with structured explanations, examples, and clarity.
Break complex topics into digestible steps.
Never give harmful or misleading advice.
If uncertain, say you don't know but offer to help explore it.""",
}

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


def append_conflict(prompt, completion_a, score_a, completion_b, score_b):
    entry = {
        "prompt": prompt,
        "completions": [
            {"completion": completion_a, "score": score_a},
            {"completion": completion_b, "score": score_b},
        ],
        "flagged_at": datetime.now().isoformat(),
    }
    with open(CONFLICT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print("   \u2697\ufe0f Conflict flagged — both completions saved to conflict_queue.jsonl")


def query_hash(query):
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def cache_lookup(query):
    if not os.path.exists(CACHE_PATH):
        return None
    qh = query_hash(query)
    now = datetime.now()
    fresh = []
    stale_entries = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cached_at = entry.get("cached_at", "")
            try:
                age = now - datetime.fromisoformat(cached_at)
                if age > timedelta(days=CACHE_MAX_DAYS):
                    stale_entries.append(entry)
                    continue
            except Exception:
                pass
            fresh.append(entry)
    if stale_entries:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            for entry in fresh:
                f.write(json.dumps(entry) + "\n")
    for entry in fresh:
        if entry.get("query_hash") == qh:
            return entry["results"]
    return None


def cache_store(query, results):
    entry = {
        "query_hash": query_hash(query), "query": query,
        "results": results, "cached_at": datetime.now().isoformat(),
    }
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


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


def score_completion(completion):
    return score_reply(completion)


def topic_distribution(data):
    prompts = [d["prompt"] for d in data]
    topics = Counter()
    for p in prompts:
        for key in PROMPT_TEMPLATES:
            if p.lower().startswith(key):
                topics[key] += 1
                break
        else:
            topics["general"] += 1
    return topics


def orchestrate_persona(spec):
    parts = spec.split("+")
    segments = []
    for part in parts:
        part = part.strip()
        if ":" in part:
            name, weight_str = part.rsplit(":", 1)
            try:
                weight = int(weight_str)
            except ValueError:
                weight = 50
        else:
            name = part
            weight = 50
        base = PERSONAS.get(name.strip())
        if base:
            segments.append((base.strip(), weight))
    if not segments:
        return PERSONAS["mentor"] + f"\n\nYou are running on {MODEL_VERSION}."

    total_weight = sum(w for _, w in segments)
    if total_weight == 0:
        total_weight = 1

    blended_parts = []
    for text, weight in segments:
        pct = weight / total_weight
        emphasis = "strongly" if pct > 0.5 else "moderately" if pct > 0.2 else "slightly"
        blended_parts.append(f"[{emphasis} weighted at {weight}%]\n{text}")
    combined = "\n\n".join(blended_parts)
    combined += f"\n\nYou are running on {MODEL_VERSION}."
    return combined


def build_persona(data=None, persona_name="mentor"):
    if "+" in persona_name and ":" in persona_name:
        persona = orchestrate_persona(persona_name)
    elif "+" in persona_name:
        persona = "\n\n".join(PERSONAS.get(p.strip(), "").strip() for p in persona_name.split("+") if p.strip() in PERSONAS)
        if not persona:
            persona = PERSONAS["mentor"]
        persona += f"\n\nYou are running on {MODEL_VERSION}."
    else:
        persona = PERSONAS.get(persona_name, PERSONAS["mentor"])

    avg_conf = (sum(RECENT_SCORES) / len(RECENT_SCORES)) if RECENT_SCORES else 100
    if avg_conf < 50:
        persona += "\nBe cautious and acknowledge uncertainty clearly."
    elif avg_conf < 70:
        persona += "\nBalance confidence with openness to correction."
    else:
        persona += "\nRespond with strong confidence and clarity."

    if data and len(data) >= 3:
        topics = topic_distribution(data)
        total = sum(topics.values()) or 1
        if topics.get("motivate", 0) / total < 0.2:
            persona += "\nEmphasize motivation and encouragement in your responses."
        if topics.get("advise", 0) / total < 0.2:
            persona += "\nActively offer actionable advice and guidance."
        if topics.get("explain", 0) / total > 0.5:
            persona += "\nPrioritize clarity and simplicity in explanations."
    return persona


def topic_balance(data):
    topics = topic_distribution(data)
    if not topics:
        return topics
    total = sum(topics.values())
    ideal = total / len(topics)
    print("   \u2696\ufe0f  Balance:")
    for topic, count in topics.most_common():
        sign = "+" if count >= ideal else "\u2212"
        print(f"      {topic}: {count} ({sign}{abs(count - ideal):.0f} vs ideal)")
    return topics


def dataset_stats(path=DATASET_PATH):
    data = load_dataset(path)
    if not data:
        print("\U0001f4ad Dataset is empty.")
        return
    topics = topic_balance(data)
    print(f"\U0001f4ca Dataset: {len(data)} entries | Model: {MODEL_VERSION}")
    for topic, count in topics.most_common():
        print(f"   {topic}: {count}")
    avg_conf = (sum(RECENT_SCORES) / len(RECENT_SCORES)) if RECENT_SCORES else None
    if avg_conf is not None:
        print(f"   Recent avg confidence: {avg_conf:.0f}/100")
        print(f"   Consecutive low: {CONSECUTIVE_LOW}/{LOW_CONF_ESCALATION_COUNT}")


def validate_dataset(path=DATASET_PATH):
    data = load_dataset(path)
    if not data:
        print("\u2705 No entries to validate.")
        return
    issues = 0
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "prompt" not in entry or "completion" not in entry:
            print(f"\u26a0\ufe0f  Entry {i}: malformed (missing keys)")
            issues += 1
            continue
        if len(entry["completion"]) < 5:
            print(f"\u26a0\ufe0f  Entry {i}: completion too short")
            issues += 1
        if "i don't know" in entry["completion"].lower() and len(entry["completion"]) < 20:
            print(f"\u26a0\ufe0f  Entry {i}: uncertain completion without exploration offer")
            issues += 1
    if issues == 0:
        print("\u2705 All entries valid.")
    else:
        print(f"\u26a0\ufe0f  {issues} issue(s) found. Consider reviewing.")


def export_snapshot(path=DATASET_PATH, label=""):
    if not os.path.exists(path):
        print("\u26a0\ufe0f  No dataset to snapshot.")
        return
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"curated_{timestamp}{suffix}.jsonl")
    shutil.copy2(path, snapshot_path)
    print(f"\U0001f4f8 Snapshot saved: {snapshot_path}")
    return snapshot_path


def export_gold_dataset(path=DATASET_PATH, min_score=90):
    data = load_dataset(path)
    if not data:
        print("\U0001f4ad No entries to export.")
        return
    os.makedirs(GOLD_DIR, exist_ok=True)
    gold_path = os.path.join(GOLD_DIR, "gold_dataset.jsonl")
    count = 0
    with open(gold_path, "w", encoding="utf-8") as out:
        for entry in data:
            s = score_completion(entry["completion"])
            if s >= min_score:
                out.write(json.dumps({"prompt": entry["prompt"], "completion": entry["completion"], "_score": s}) + "\n")
                count += 1
    print(f"\U0001f947 Gold dataset exported: {gold_path} ({count} entries, score >= {min_score})")
    return gold_path


def sync_dataset(target_dir=None):
    dest = target_dir or SYNC_DIR
    if not dest:
        print("\u26a0\ufe0f  No sync target directory set. Configure SYNC_DIR or pass --sync-dir.")
        return
    os.makedirs(dest, exist_ok=True)
    files_to_sync = []
    if os.path.exists(DATASET_PATH):
        files_to_sync.append(DATASET_PATH)
    if os.path.exists(WEAK_REPLIES_PATH):
        files_to_sync.append(WEAK_REPLIES_PATH)
    if os.path.exists(CONFLICT_PATH):
        files_to_sync.append(CONFLICT_PATH)
    snapshot = export_snapshot()
    if snapshot:
        files_to_sync.append(snapshot)
    gold = export_gold_dataset()
    if gold:
        files_to_sync.append(gold)
    for f in files_to_sync:
        shutil.copy2(f, os.path.join(dest, os.path.basename(f)))
        print(f"   Synced: {os.path.basename(f)}")
    print(f"\U0001f4e4 Sync complete \u2192 {dest}")


def federated_sync(remote_dir):
    if not os.path.isdir(remote_dir):
        print(f"\u26a0\ufe0f  Remote dir not found: {remote_dir}")
        return
    local_data = load_dataset()
    remote_path = os.path.join(remote_dir, DATASET_PATH)
    remote_data = load_dataset(remote_path) if os.path.exists(remote_path) else []
    seen = {}
    for entry in local_data + remote_data:
        key = entry.get("prompt", "")
        existing = seen.get(key)
        if existing:
            existing_score = score_completion(existing.get("completion", ""))
            current_score = score_completion(entry.get("completion", ""))
            if current_score > existing_score:
                seen[key] = entry
            elif current_score == existing_score:
                append_conflict(key, existing["completion"], existing_score, entry["completion"], current_score)
        else:
            seen[key] = entry
    merged = list(seen.values())
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for entry in merged:
            f.write(json.dumps({"prompt": entry["prompt"], "completion": entry["completion"]}) + "\n")
    print(f"\U0001f91d Federated merge complete: {len(merged)} entries ({len(local_data)} local + {len(remote_data)} remote)")


def fetch_source(source_name, query):
    url_template = EXTERNAL_SOURCES.get(source_name)
    if not url_template:
        return None
    try:
        encoded = urllib.parse.quote(query)
        if source_name == "wikipedia":
            clean = query.strip().replace(" ", "_")
            url = url_template.format(query=clean)
        else:
            url = url_template.format(query=encoded)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if source_name == "duckduckgo":
                abstract = data.get("AbstractText", "")
                if abstract:
                    return ("DuckDuckGo", abstract)
                results = data.get("RelatedTopics", [])
                if results:
                    text = results[0].get("Text", "")
                    if text:
                        return ("DuckDuckGo", text)
            elif source_name == "wikipedia":
                extract = data.get("extract", "")
                if extract:
                    return ("Wikipedia", extract[:500])
    except Exception:
        pass
    return None


def layered_lookup(query):
    cached = cache_lookup(query)
    if cached:
        print("   \U0001f4e1 Cache hit for query")
        return cached
    results = []
    for source in EXTERNAL_SOURCES:
        result = fetch_source(source, query)
        if result:
            results.append(result)
    if results:
        cache_store(query, results)
    return results


def apply_template(user_input):
    for key, template in PROMPT_TEMPLATES.items():
        if user_input.lower().startswith(key):
            return template.format(query=user_input[len(key):].strip())
    return user_input


def train_from_dataset(base_model=MODEL_VERSION, path=DATASET_PATH, new_model="empathetic-mentor", min_score=80):
    data = load_dataset(path)
    if len(data) < 5:
        print("\u26a0\ufe0f  Need at least 5 dataset entries to train.")
        return

    scored = []
    for entry in data:
        s = score_completion(entry["completion"])
        entry["_score"] = s
        scored.append(entry)
    scored.sort(key=lambda x: x["_score"], reverse=True)
    high_quality = [e for e in scored if e["_score"] >= min_score]
    if not high_quality:
        print(f"\u26a0\ufe0f  No entries scored >= {min_score}. Training with top entries anyway.")
        high_quality = scored[:10]

    export_snapshot(path)
    print(f"\U0001f9e0 Training '{new_model}' from {base_model} using {len(high_quality)} high-quality examples (score >= {min_score})...")

    few_shot = ""
    for entry in high_quality[:20]:
        few_shot += f"User: {entry['prompt']}\nAssistant: {entry['completion']}\n\n"

    data = load_dataset(path)
    persona = build_persona(data)
    modelfile = f"""FROM {base_model}
SYSTEM \"\"\"{persona.strip()}\"\"\"
TEMPLATE \"\"\"System: {persona.strip()}

{few_shot}
User: {{.Prompt}}
Assistant: \"\"\"
"""
    modelfile_path = "Modelfile.tmp"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile)
    try:
        ollama.create(model=new_model, modelfile=modelfile_path)
        print(f"\u2705 Model '{new_model}' created. Run it with: ollama run {new_model}")
    except Exception as e:
        print(f"\u26a0\ufe0f Training failed: {e}")
    finally:
        if os.path.exists(modelfile_path):
            os.remove(modelfile_path)


def chat_with_feeling(persona_name="mentor"):
    global LAST_REPLY, LAST_PROMPT, LAST_SCORE, RECENT_SCORES, CONSECUTIVE_LOW

    data = load_dataset()
    active_persona = build_persona(data, persona_name)

    print(f"\U0001f4a1 Emotional Ollama Chat Started  [persona: {persona_name}]")
    print(f"   Model: {MODEL_VERSION}  |  Dataset: {DATASET_PATH}")
    print("   Commands:  exit/quit  |  99 (connect)  |  correct <new text>  |  stats  |  validate | gold")
    if EXTERNAL_ENABLED:
        print(f"   \U0001f310 External knowledge: {', '.join(EXTERNAL_SOURCES)}")
        print(f"   Cache TTL: {CACHE_MAX_DAYS}d  |  Escalation after {LOW_CONF_ESCALATION_COUNT} consecutive low")
    try:
        while True:
            user_input = input("You: ")
            cmd = user_input.strip().lower()

            if cmd in ["exit", "quit"]:
                print("\U0001f44b Ending session.")
                break

            if cmd == "99":
                print("\U0001f4de Connecting to your empathetic AI assistant...")
                data = load_dataset()
                active_persona = build_persona(data, persona_name)
                print(f"   Persona adapted ({avg_conf_str(RECENT_SCORES)}).")
                continue

            if cmd == "stats":
                dataset_stats()
                continue

            if cmd == "validate":
                validate_dataset()
                continue

            if cmd == "gold":
                export_gold_dataset()
                continue

            if cmd.startswith("correct") and LAST_REPLY is not None:
                corrected = user_input[len("correct"):].strip()
                if corrected:
                    append_to_dataset(LAST_PROMPT, corrected)
                    print("\u2705 Correction saved to dataset.")
                    generate_variations(LAST_PROMPT, corrected)
                    print("   Variations generated.")
                    LAST_REPLY = corrected
                continue

            prompt = apply_template(user_input)
            try:
                response = ollama.chat(
                    model=MODEL_VERSION,
                    messages=[
                        {"role": "system", "content": active_persona},
                        {"role": "user", "content": prompt},
                    ],
                )
                reply = response["message"]["content"]
                score = score_reply(reply)
                RECENT_SCORES.append(score)
                if len(RECENT_SCORES) > 20:
                    RECENT_SCORES.pop(0)

                if score < 50:
                    CONSECUTIVE_LOW += 1
                else:
                    CONSECUTIVE_LOW = 0

                print(f"Ollama: {reply}  [confidence: {score}/100]")

                LAST_PROMPT = prompt
                LAST_REPLY = reply
                LAST_SCORE = score

                escalated = EXTERNAL_ENABLED and CONSECUTIVE_LOW >= LOW_CONF_ESCALATION_COUNT

                if score < 50:
                    append_to_weak(prompt, reply, score)
                    external_results = layered_lookup(user_input) if EXTERNAL_ENABLED else []
                    if escalated:
                        print(f"   \U0001f6c8 Escalating — {CONSECUTIVE_LOW} consecutive low-confidence replies.")
                    if external_results:
                        parts = [f"[{source}] {text}" for source, text in external_results]
                        fallback_reply = f"I checked multiple sources:\n" + "\n".join(parts)
                        print(f"   \U0001f310 Layered results:\n" + "\n".join(parts))
                        append_to_dataset(prompt, fallback_reply)
                    else:
                        print(f"\u26a0\ufe0f  Score {score}/100 — auto-rejected. Using fallback.")
                        print("Ollama:", FALLBACK)
                        append_to_dataset(prompt, FALLBACK)
                elif score < SCORE_THRESHOLD:
                    append_to_weak(prompt, reply, score)
                    print(f"\u26a0\ufe0f  Reply scored {score}/100 — below threshold.")
                    fix = input("   Correct it? (leave blank to skip, or type correction): ").strip()
                    if fix:
                        append_to_dataset(prompt, fix)
                        print("\u2705 Correction saved.")
                        generate_variations(prompt, fix)
                        print("   Variations generated.")
                else:
                    if score > 90:
                        print("   \u2705 Auto-approved (score > 90).")
                    append_to_dataset(prompt, reply)

                if escalated:
                    print(f"   \U0001f6c8 Low-confidence streak reset.")
                    CONSECUTIVE_LOW = 0

            except Exception:
                print("\u26a0\ufe0f", FALLBACK)

    except KeyboardInterrupt:
        print("\n\U0001f44b Session interrupted. Goodbye!")


def avg_conf_str(scores):
    if not scores:
        return "no data"
    avg = sum(scores) / len(scores)
    return f"avg {avg:.0f}/100"


if __name__ == "__main__":
    persona_name = "mentor"
    sync_target = None
    rest = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--persona" and i + 1 < len(sys.argv):
            persona_name = sys.argv[i + 1]
            i += 2
        elif arg == "--sync" and i + 1 < len(sys.argv):
            sync_target = sys.argv[i + 1]
            i += 2
        elif arg == "--sync-dir" and i + 1 < len(sys.argv):
            sync_target = sys.argv[i + 1]
            i += 2
        elif arg == "--federated-sync" and i + 1 < len(sys.argv):
            federated_sync(sys.argv[i + 1])
            sys.exit(0)
        else:
            rest.append(arg)
            i += 1
    sys.argv = [sys.argv[0]] + rest

    if "--stats" in sys.argv:
        dataset_stats()
    elif "--validate" in sys.argv:
        validate_dataset()
    elif "--gold" in sys.argv:
        export_gold_dataset()
    elif "--train" in sys.argv:
        train_from_dataset()
    elif "--sync" in sys.argv or sync_target:
        sync_dataset(sync_target)
    else:
        chat_with_feeling(persona_name=persona_name)
