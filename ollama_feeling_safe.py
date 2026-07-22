import ollama
import json
import os
import sys
import shutil
import hashlib
import urllib.request
import urllib.parse
import re
from datetime import datetime, timedelta
from collections import Counter

MODEL_VERSION = "llama3.2"
DATASET_PATH = "curated_dataset.jsonl"
WEAK_REPLIES_PATH = "weak_replies.jsonl"
CACHE_PATH = "knowledge_cache.jsonl"
CONFLICT_PATH = "conflict_queue.jsonl"
ESCALATION_LOG_PATH = "escalation_log.jsonl"
GOLD_DIR = "gold"
SNAPSHOT_DIR = "snapshots"
SYNC_DIR = None
SCORE_THRESHOLD = 70
WEIGHT_BOOST_FACTOR = 12
WEIGHT_DECAY_PER_TURN = 3
LAST_REPLY = None
LAST_PROMPT = None
LAST_SCORE = None
RECENT_SCORES = []
CONSECUTIVE_LOW = 0
DYNAMIC_WEIGHTS = {}
PERSONA_SPEC = "mentor"

EXTERNAL_ENABLED = False
EXTERNAL_SOURCES = {
    "duckduckgo": "https://api.duckduckgo.com/?q={query}&format=json&no_html=1",
    "wikipedia": "https://en.wikipedia.org/api/rest_v1/page/summary/{query}",
}
FRESHNESS_DECAY = {"duckduckgo": 7, "wikipedia": 3}
FRESHNESS_STALE_AT = 30

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

ESCALATION_EVENTS = []


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
    topic_priority = topic_priority_for(prompt)
    entry = {
        "prompt": prompt,
        "completions": [
            {"completion": completion_a, "score": score_a},
            {"completion": completion_b, "score": score_b},
        ],
        "priority": topic_priority,
        "flagged_at": datetime.now().isoformat(),
    }
    with open(CONFLICT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"   \u2697\ufe0f Conflict flagged (priority: {topic_priority}) \u2014 saved to conflict_queue.jsonl")


def log_escalation(tier, count, user_input):
    entry = {
        "tier": tier,
        "consecutive_low": count,
        "user_input": user_input[:100],
        "timestamp": datetime.now().isoformat(),
    }
    ESCALATION_EVENTS.append(entry)
    with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def show_escalation_stats():
    if not os.path.exists(ESCALATION_LOG_PATH):
        print("   No escalation events logged.")
        return
    tiers = Counter()
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            tiers[entry.get("tier", "?")] += 1
    print("   \U0001f4ca Escalation stats:")
    for tier, count in tiers.most_common():
        print(f"      Tier {tier}: {count} event(s)")


def query_hash(query):
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def freshness_score(source, cached_at_str):
    try:
        age_days = (datetime.now() - datetime.fromisoformat(cached_at_str)).days
    except Exception:
        return 0
    decay_rate = FRESHNESS_DECAY.get(source, 5)
    return max(0, 100 - age_days * decay_rate)


def cache_lookup(query):
    if not os.path.exists(CACHE_PATH):
        return None
    qh = query_hash(query)
    fresh = []
    stale = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            source = entry.get("source", "duckduckgo")
            fs = freshness_score(source, entry.get("cached_at", ""))
            entry["_freshness"] = fs
            if fs < FRESHNESS_STALE_AT:
                stale.append(entry)
            else:
                fresh.append(entry)
    if stale:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            for entry in fresh:
                e = {k: v for k, v in entry.items() if not k.startswith("_")}
                f.write(json.dumps(e) + "\n")
    for entry in fresh:
        if entry.get("query_hash") == qh:
            return entry["results"]
    return None


def cache_store(query, results):
    for source, _ in results:
        entry = {
            "query_hash": query_hash(query), "query": query,
            "source": source, "results": results,
            "cached_at": datetime.now().isoformat(),
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


def topic_priority_for(text):
    lower = text.lower()
    for word in ["motivate", "inspire", "encourage", "push"]:
        if word in lower:
            return 90
    for word in ["advise", "recommend", "suggest", "should"]:
        if word in lower:
            return 70
    for word in ["explain", "what", "how", "define"]:
        if word in lower:
            return 50
    return 30


def detect_intent(user_input):
    lower = user_input.lower()
    if any(w in lower for w in ["motivate", "inspire", "encourage", "push", "hype"]):
        return "motivate"
    if any(w in lower for w in ["explain", "what is", "how does", "define", "tell me about"]):
        return "explain"
    if any(w in lower for w in ["advise", "should i", "what should", "recommend", "suggest"]):
        return "advise"
    return None


def normalize_weights(weights):
    if not weights:
        return
    total = sum(weights.values())
    if total == 0:
        return
    factor = 100 / total
    for k in weights:
        weights[k] = round(weights[k] * factor)


def decay_weights(weights):
    for k in weights:
        if weights[k] > 50:
            weights[k] = max(50, weights[k] - WEIGHT_DECAY_PER_TURN)


def orchestrate_persona(spec, dynamic_overrides=None):
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
        if dynamic_overrides and name.strip() in dynamic_overrides:
            weight = dynamic_overrides[name.strip()]
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


def build_persona(data=None, persona_name="mentor", dynamic_overrides=None):
    if "+" in persona_name:
        persona = orchestrate_persona(persona_name, dynamic_overrides)
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
        print(f"   Consecutive low: {CONSECUTIVE_LOW}")
    show_escalation_stats()


def show_conflicts():
    if not os.path.exists(CONFLICT_PATH):
        print("   \u2705 No conflicts in queue.")
        return
    entries = []
    with open(CONFLICT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    entries.sort(key=lambda e: e.get("priority", 30), reverse=True)
    print(f"   \u2697\ufe0f Conflict queue ({len(entries)} total, sorted by priority):")
    for e in entries[:5]:
        prompt_preview = e.get("prompt", "")[:60]
        priority = e.get("priority", 30)
        scores = [c.get("score", 0) for c in e.get("completions", [])]
        print(f"      [{priority}] {prompt_preview}... scores={scores}")


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
    if os.path.exists(ESCALATION_LOG_PATH):
        files_to_sync.append(ESCALATION_LOG_PATH)
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


def federated_sync(remote_dir, merge_policy="highest"):
    if not os.path.isdir(remote_dir):
        print(f"\u26a0\ufe0f  Remote dir not found: {remote_dir}")
        return
    local_data = load_dataset()
    remote_path = os.path.join(remote_dir, DATASET_PATH)
    remote_data = load_dataset(remote_path) if os.path.exists(remote_path) else []
    seen = {}
    conflicts = []
    for entry in local_data + remote_data:
        key = entry.get("prompt", "")
        existing = seen.get(key)
        if existing:
            existing_score = score_completion(existing.get("completion", ""))
            current_score = score_completion(entry.get("completion", ""))
            if merge_policy == "keep-both":
                conflicts.append((key, existing["completion"], existing_score, entry["completion"], current_score))
                continue
            elif merge_policy == "manual":
                conflicts.append((key, existing["completion"], existing_score, entry["completion"], current_score))
                continue
            else:
                if current_score > existing_score:
                    seen[key] = entry
                elif current_score == existing_score:
                    conflicts.append((key, existing["completion"], existing_score, entry["completion"], current_score))
        else:
            seen[key] = entry
    for key, ca, sa, cb, sb in conflicts:
        if merge_policy == "keep-both":
            seen[key + "_v1"] = {"prompt": key + " [v1]", "completion": ca}
            seen[key + "_v2"] = {"prompt": key + " [v2]", "completion": cb}
        elif merge_policy == "manual":
            append_conflict(key, ca, sa, cb, sb)
    merged = list(seen.values())
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for entry in merged:
            f.write(json.dumps({"prompt": entry["prompt"], "completion": entry["completion"]}) + "\n")
    print(f"\U0001f91d Federated merge [{merge_policy}]: {len(merged)} entries ({len(local_data)} local + {len(remote_data)} remote)")


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
                    return (source_name, abstract)
                results = data.get("RelatedTopics", [])
                if results:
                    text = results[0].get("Text", "")
                    if text:
                        return (source_name, text)
            elif source_name == "wikipedia":
                extract = data.get("extract", "")
                if extract:
                    return (source_name, extract[:500])
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


def apply_template(user_input, dynamic_weights=None):
    intent = detect_intent(user_input) if dynamic_weights is not None else None
    if intent and dynamic_weights is not None:
        decay_weights(dynamic_weights)
        if intent == "motivate":
            for name in ["coach", "mentor"]:
                if name in dynamic_weights:
                    dynamic_weights[name] = min(dynamic_weights.get(name, 50) + WEIGHT_BOOST_FACTOR, 100)
        elif intent == "explain" and "teacher" in dynamic_weights:
            dynamic_weights["teacher"] = min(dynamic_weights.get("teacher", 50) + WEIGHT_BOOST_FACTOR, 100)
        elif intent == "advise":
            for name in ["mentor", "coach"]:
                if name in dynamic_weights:
                    dynamic_weights[name] = min(dynamic_weights.get(name, 50) + int(WEIGHT_BOOST_FACTOR * 0.8), 100)
        normalize_weights(dynamic_weights)
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


def handle_escalation_tier(count, user_input, prompt):
    if count >= 7:
        print(f"   \U0001f6d1 Tier 3 ({count} consecutive low) \u2014 hard fallback activated.")
        log_escalation(3, count, user_input)
        return "I want to be honest with you \u2014 I\u2019m not confident I can give you a reliable answer right now. Let\u2019s try a different approach or topic."
    if count >= 5:
        print(f"   \U0001f6c8 Tier 2 ({count} consecutive low) \u2014 shifting persona to cautious mode.")
        log_escalation(2, count, user_input)
        return None
    if count >= 3:
        print(f"   \U0001f310 Tier 1 ({count} consecutive low) \u2014 querying external sources.")
        log_escalation(1, count, user_input)
        external_results = layered_lookup(user_input) if EXTERNAL_ENABLED else []
        if external_results:
            parts = [f"[{source}] {text}" for source, text in external_results]
            return "I checked multiple sources:\n" + "\n".join(parts)
        return None
    return None


def chat_with_feeling(persona_name="mentor"):
    global LAST_REPLY, LAST_PROMPT, LAST_SCORE, RECENT_SCORES, CONSECUTIVE_LOW, DYNAMIC_WEIGHTS, PERSONA_SPEC
    PERSONA_SPEC = persona_name

    parts = persona_name.split("+")
    DYNAMIC_WEIGHTS = {}
    for part in parts:
        p = part.strip()
        if ":" in p:
            name = p.rsplit(":", 1)[0].strip()
        else:
            name = p
        if name in PERSONAS:
            DYNAMIC_WEIGHTS[name] = 50

    data = load_dataset()
    active_persona = build_persona(data, persona_name)

    print(f"\U0001f4a1 Emotional Ollama Chat Started  [persona: {persona_name}]")
    print(f"   Model: {MODEL_VERSION}  |  Dataset: {DATASET_PATH}")
    print("   Commands:  exit/quit  |  99 (connect)  |  correct <new text>  |  stats  |  validate | gold | conflicts")
    if EXTERNAL_ENABLED:
        print(f"   \U0001f310 External: {', '.join(EXTERNAL_SOURCES)} | Cache freshness score < {FRESHNESS_STALE_AT} = stale")
        print(f"   Escalation tiers: 3\u2192external  5\u2192persona shift  7\u2192hard fallback")
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
                active_persona = build_persona(data, persona_name, DYNAMIC_WEIGHTS)
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

            if cmd == "conflicts":
                show_conflicts()
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

            prompt = apply_template(user_input, DYNAMIC_WEIGHTS)
            if DYNAMIC_WEIGHTS and "+" in persona_name:
                active_persona = build_persona(data, persona_name, DYNAMIC_WEIGHTS)
                print(f"   \U0001f504 Weights: {DYNAMIC_WEIGHTS}")

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

                if score < 50:
                    append_to_weak(prompt, reply, score)
                    tier_reply = handle_escalation_tier(CONSECUTIVE_LOW, user_input, prompt)
                    if tier_reply:
                        print(f"   \U0001f310 Escalation result:", tier_reply[:100])
                        append_to_dataset(prompt, tier_reply)
                    else:
                        print(f"\u26a0\ufe0f  Score {score}/100 \u2014 auto-rejected. Using fallback.")
                        print("Ollama:", FALLBACK)
                        append_to_dataset(prompt, FALLBACK)
                elif score < SCORE_THRESHOLD:
                    append_to_weak(prompt, reply, score)
                    print(f"\u26a0\ufe0f  Reply scored {score}/100 \u2014 below threshold.")
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
    merge_policy = "highest"
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
        elif arg == "--merge-policy" and i + 1 < len(sys.argv):
            merge_policy = sys.argv[i + 1]
            i += 2
        elif arg == "--federated-sync" and i + 1 < len(sys.argv):
            federated_sync(sys.argv[i + 1], merge_policy)
            sys.exit(0)
        elif arg == "--conflicts":
            show_conflicts()
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
