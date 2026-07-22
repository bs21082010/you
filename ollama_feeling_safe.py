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
from collections import Counter, defaultdict

MODEL_VERSION = "llama3.2"
DATASET_PATH = "curated_dataset.jsonl"
WEAK_REPLIES_PATH = "weak_replies.jsonl"
CACHE_PATH = "knowledge_cache.jsonl"
CONFLICT_PATH = "conflict_queue.jsonl"
ESCALATION_LOG_PATH = "escalation_log.jsonl"
WEIGHT_LOG_PATH = "weight_history.jsonl"
GOLD_DIR = "gold"
SNAPSHOT_DIR = "snapshots"
SYNC_DIR = None
SCORE_THRESHOLD = 70
WEIGHT_BOOST_FACTOR = 12
WEIGHT_DECAY_PER_TURN = 3
DRIFT_THRESHOLD = 15
PREDICTIVE_BOOST = 8
CONFLICT_AUTO_POLICY = "low"
STALE_WARNING_DAYS = 1
LAST_REPLY = None
LAST_PROMPT = None
LAST_SCORE = None
RECENT_SCORES = []
CONSECUTIVE_LOW = 0
DYNAMIC_WEIGHTS = {}
PERSONA_SPEC = "mentor"
CACHE_HITS = 0
CACHE_MISSES = 0

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
BASELINE_WEIGHTS = {}


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
    pri = topic_priority_for(prompt)
    entry = {"prompt": prompt, "completions": [{"completion": completion_a, "score": score_a}, {"completion": completion_b, "score": score_b}], "priority": pri, "flagged_at": datetime.now().isoformat()}
    with open(CONFLICT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"   Conflict flagged (priority: {pri})")


def log_escalation(tier, count, user_input):
    entry = {"tier": tier, "consecutive_low": count, "user_input": user_input[:100], "timestamp": datetime.now().isoformat()}
    ESCALATION_EVENTS.append(entry)
    with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_weights(weights):
    entry = {"weights": dict(weights), "timestamp": datetime.now().isoformat()}
    with open(WEIGHT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def query_hash(query):
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def freshness_score(source, cached_at_str):
    try:
        age_days = (datetime.now() - datetime.fromisoformat(cached_at_str)).days
    except Exception:
        return 0
    return max(0, 100 - age_days * FRESHNESS_DECAY.get(source, 5))


def time_until_stale(source, cached_at_str):
    try:
        age_days = (datetime.now() - datetime.fromisoformat(cached_at_str)).days
    except Exception:
        return 0
    decay = FRESHNESS_DECAY.get(source, 5)
    current = max(0, 100 - age_days * decay)
    if current <= FRESHNESS_STALE_AT:
        return 0
    return (current - FRESHNESS_STALE_AT) / decay


def cache_lookup(query):
    global CACHE_HITS, CACHE_MISSES
    if not os.path.exists(CACHE_PATH):
        CACHE_MISSES += 1; return None
    qh = query_hash(query)
    fresh, stale = [], []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            entry = json.loads(line)
            fs = freshness_score(entry.get("source", "duckduckgo"), entry.get("cached_at", ""))
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
            CACHE_HITS += 1
            return entry["results"]
    CACHE_MISSES += 1
    return None


def cache_store(query, results):
    for source, _ in results:
        entry = {"query_hash": query_hash(query), "query": query, "source": source, "results": results, "cached_at": datetime.now().isoformat()}
        with open(CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def forecast_cache_staleness():
    if not os.path.exists(CACHE_PATH):
        return
    with open(CACHE_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries:
        return
    warning = []
    for e in entries:
        src = e.get("source", "duckduckgo")
        days_left = time_until_stale(src, e.get("cached_at", ""))
        if 0 < days_left <= STALE_WARNING_DAYS:
            warning.append((src, e.get("query", "")[:60], days_left))
    if warning:
        print(f"   Cache forecast ({len(warning)} entries nearing staleness):")
        for src, q, d in warning:
            print(f"      {src}: \"{q}\"... stale in {d:.1f}d")
        print("   Run a relevant query to auto-refresh these entries.")


def generate_variations(prompt, corrected):
    vs = [corrected, corrected.replace("I think", "I believe"), corrected.replace("maybe", "").replace("perhaps", "")]
    for v in set(v for v in vs if len(v) > 20):
        append_to_dataset(prompt, v)


def score_reply(reply):
    score = 100
    words = reply.split()
    if len(words) < 20: score -= 20
    if len(words) < 10: score -= 15
    lower = reply.lower()
    if "i don't know" in lower or "unsure" in lower: score -= 30
    if "i don't have that information" in lower: score -= 10
    if any(w in lower for w in ["maybe", "sort of", "perhaps", "kind of"]): score -= 10
    if reply.endswith("?") or reply.endswith("..."): score -= 5
    return max(score, 0)


def score_completion(c):
    return score_reply(c)


def topic_distribution(data):
    topics = Counter()
    for p in [d["prompt"] for d in data]:
        matched = False
        for key in PROMPT_TEMPLATES:
            if p.lower().startswith(key):
                topics[key] += 1; matched = True; break
        if not matched: topics["general"] += 1
    return topics


def topic_priority_for(text):
    lower = text.lower()
    for words, score in [(["motivate", "inspire", "encourage", "push"], 90),
                          (["advise", "recommend", "suggest", "should"], 70),
                          (["explain", "what", "how", "define"], 50)]:
        if any(w in lower for w in words): return score
    return 30


def detect_intent(text):
    lower = text.lower()
    if any(w in lower for w in ["motivate", "inspire", "encourage", "push", "hype"]): return "motivate"
    if any(w in lower for w in ["explain", "what is", "how does", "define", "tell me about"]): return "explain"
    if any(w in lower for w in ["advise", "should i", "what should", "recommend", "suggest"]): return "advise"
    return None


def predictive_intent_boost(intent):
    if not os.path.exists(ESCALATION_LOG_PATH):
        return {}
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events:
        return {}
    by_intent = Counter()
    for e in events:
        text = e.get("user_input", "")
        i = detect_intent(text) or "general"
        by_intent[i] += 1
    total = sum(by_intent.values())
    if total == 0:
        return {}
    rate = by_intent.get(intent, 0) / total
    boost_map = {}
    if rate > 0.3 and intent == "motivate":
        boost_map["coach"] = PREDICTIVE_BOOST
        boost_map["mentor"] = PREDICTIVE_BOOST
    elif rate > 0.3 and intent == "explain":
        boost_map["teacher"] = PREDICTIVE_BOOST
    elif rate > 0.3 and intent == "advise":
        boost_map["mentor"] = PREDICTIVE_BOOST
        boost_map["coach"] = int(PREDICTIVE_BOOST * 0.7)
    return boost_map


def normalize_weights(w):
    if not w: return
    total = sum(w.values())
    if total == 0: return
    factor = 100 / total
    for k in w: w[k] = round(w[k] * factor)


def decay_weights(w):
    for k in w:
        if w[k] > 50: w[k] = max(50, w[k] - WEIGHT_DECAY_PER_TURN)


def orchestrate_persona(spec, overrides=None):
    parts = spec.split("+")
    segs = []
    for part in parts:
        part = part.strip()
        if ":" in part:
            name, ws = part.rsplit(":", 1)
            weight = int(ws) if ws.isdigit() else 50
        else:
            name = part; weight = 50
        base = PERSONAS.get(name.strip())
        if overrides and name.strip() in overrides: weight = overrides[name.strip()]
        if base: segs.append((base.strip(), weight))
    if not segs: return PERSONAS["mentor"] + f"\n\nRunning on {MODEL_VERSION}."
    tw = sum(w for _, w in segs) or 1
    parts_out = []
    for t, w in segs:
        pct = w / tw
        emp = "strongly" if pct > 0.5 else "moderately" if pct > 0.2 else "slightly"
        parts_out.append(f"[{emp} weighted at {w}%]\n{t}")
    return "\n\n".join(parts_out) + f"\n\nRunning on {MODEL_VERSION}."


def build_persona(data=None, persona_name="mentor", overrides=None):
    if "+" in persona_name:
        persona = orchestrate_persona(persona_name, overrides)
    else:
        persona = PERSONAS.get(persona_name, PERSONAS["mentor"])
    avg_c = (sum(RECENT_SCORES) / len(RECENT_SCORES)) if RECENT_SCORES else 100
    if avg_c < 50: persona += "\nBe cautious and acknowledge uncertainty clearly."
    elif avg_c < 70: persona += "\nBalance confidence with openness to correction."
    else: persona += "\nRespond with strong confidence and clarity."
    if data and len(data) >= 3:
        t = topic_distribution(data); tot = sum(t.values()) or 1
        if t.get("motivate", 0) / tot < 0.2: persona += "\nEmphasize motivation and encouragement."
        if t.get("advise", 0) / tot < 0.2: persona += "\nActively offer actionable advice."
        if t.get("explain", 0) / tot > 0.5: persona += "\nPrioritize clarity and simplicity."
    return persona


def topic_balance(data):
    topics = topic_distribution(data)
    if not topics: return topics
    total = sum(topics.values()); ideal = total / len(topics)
    print("   Balance:")
    for topic, count in topics.most_common():
        sign = "+" if count >= ideal else "-"
        print(f"      {topic}: {count} ({sign}{abs(count - ideal):.0f} vs ideal)")
    return topics


def show_cache_stats():
    if not os.path.exists(CACHE_PATH):
        print("   Cache: empty"); return
    with open(CACHE_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries: print("   Cache: empty"); return
    fs_by_src = defaultdict(list)
    for e in entries:
        fs_by_src[e.get("source", "unknown")].append(freshness_score(e.get("source", ""), e.get("cached_at", "")))
    tq = CACHE_HITS + CACHE_MISSES; hr = (CACHE_HITS / tq * 100) if tq else 0
    print(f"   Cache: {len(entries)} entries | hits: {CACHE_HITS} | misses: {CACHE_MISSES} | hit rate: {hr:.0f}%")
    for src, scores in fs_by_src.items():
        avg = sum(scores) / len(scores)
        bar = chr(9608) * int(avg / 10) + chr(9617) * (10 - int(avg / 10))
        print(f"      {src}: {avg:.0f}/100 {bar}")
    forecast_cache_staleness()


def show_weight_trends():
    if not os.path.exists(WEIGHT_LOG_PATH): print("   Weight history: no data"); return
    with open(WEIGHT_LOG_PATH, encoding="utf-8") as f:
        history = [json.loads(line) for line in f if line.strip()]
    if not history: return
    names = list(history[0].get("weights", {}).keys())
    print(f"   Weight trends ({len(history)} snapshots):")
    for name in names:
        vals = [h["weights"].get(name, 0) for h in history]
        recent5 = vals[-5:]
        d = chr(8599) if len(vals) > 1 and vals[-1] > vals[0] else chr(8600) if len(vals) > 1 and vals[-1] < vals[0] else chr(8594)
        print(f"      {name}: {d} avg {sum(recent5)/len(recent5):.0f} last5: {recent5}")
    return history


def show_escalation_dashboard():
    if not os.path.exists(ESCALATION_LOG_PATH): print("   No escalations."); return
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events: print("   Escalations: none"); return
    tiers = Counter(e.get("tier") for e in events)
    triggers = Counter(e.get("user_input", "")[:30] for e in events)
    total = len(events)
    labels = {1: "external lookup", 2: "persona shift", 3: "hard fallback"}
    print(f"   Escalations ({total} total):")
    for t in sorted(tiers):
        c = tiers[t]; p = c / total * 100
        bar = chr(9608) * int(p / 5) + chr(9617) * (20 - int(p / 5))
        print(f"      Tier {t} ({labels.get(t)}): {c} ({p:.0f}%) {bar}")
    print("      Top triggers:")
    for text, count in triggers.most_common(3):
        print(f"         \"{text}...\" x{count}")


def show_escalation_heatmap():
    if not os.path.exists(ESCALATION_LOG_PATH): print("   No escalation data."); return
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events: return
    heat = defaultdict(lambda: Counter())
    for e in events:
        text = e.get("user_input", "")
        intent = detect_intent(text) or "general"
        tier = e.get("tier", "?")
        heat[intent][tier] += 1
    print("   Escalation heatmap (intent vs tier):")
    tiers = sorted({e.get("tier") for e in events})
    header = "   " + "".join(f"  T{t}" for t in tiers)
    print(header)
    for intent in sorted(heat):
        row = f"   {intent:10s}"
        for t in tiers:
            c = heat[intent].get(t, 0)
            marker = chr(9608) if c >= 3 else chr(9618) if c >= 1 else " "
            row += f"   {marker} {c}"
        max_c = max(heat[intent].values()) if heat[intent] else 0
        bar = chr(9608) * min(max_c, 10)
        print(f"{row}  {bar}")
    risk_intents = sorted(heat, key=lambda i: sum(heat[i].values()), reverse=True)
    if risk_intents:
        print(f"   Hotspot: \"{risk_intents[0]}\" ({sum(heat[risk_intents[0]].values())} events)")


def show_predictive_analytics():
    if not os.path.exists(ESCALATION_LOG_PATH): print("   No escalation data."); return
    with open(ESCALATION_LOG_PATH, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events: return
    weak_topics = Counter()
    for e in events:
        text = e.get("user_input", ""); intent = detect_intent(text) or "general"
        weak_topics[intent] += 1
    print("   Predictive analytics (high-risk topics):")
    for topic, count in weak_topics.most_common():
        bar = chr(9608) * count + chr(9617) * (10 - count)
        print(f"      {topic}: {count} escalation(s) {bar}")
    riskiest = weak_topics.most_common(1)
    if riskiest: print(f"   ! Recommend pre-training on \"{riskiest[0][0]}\" examples.")


def detect_drift():
    if not os.path.exists(WEIGHT_LOG_PATH): return
    with open(WEIGHT_LOG_PATH, encoding="utf-8") as f:
        history = [json.loads(line) for line in f if line.strip()]
    if len(history) < 3: return
    latest = history[-1].get("weights", {})
    print("   Drift detection:")
    for name in latest:
        vals = [h["weights"].get(name, 50) for h in history]
        bl, cur = vals[0], vals[-1]; drift = abs(cur - bl)
        if drift > DRIFT_THRESHOLD:
            d = "above" if cur > bl else "below"
            print(f"      !! {name}: drifted {drift:.0f} pts {d} baseline ({bl} -> {cur})")
        else:
            print(f"      OK {name}: stable ({bl} -> {cur}, drift {drift:.0f})")


def auto_resolve_conflicts():
    cutoff = {"low": 50, "medium": 70, "all": 100}.get(CONFLICT_AUTO_POLICY, 50)
    if not os.path.exists(CONFLICT_PATH): return 0
    with open(CONFLICT_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries: return 0
    kept, resolved = [], 0
    for e in entries:
        pri = e.get("priority", 30)
        if pri < cutoff:
            ca, cb = e["completions"][0], e["completions"][1]
            sa, sb = ca.get("score", 0), cb.get("score", 0)
            if abs(sa - sb) >= 20:
                best = ca if sa > sb else cb
                append_to_dataset(e["prompt"], best["completion"])
                resolved += 1
            elif sa >= 70 and sb >= 70:
                combined = ca["completion"] + "\n\n" + cb["completion"]
                append_to_dataset(e["prompt"], combined)
                resolved += 1
            else:
                best = ca if sa >= sb else cb
                append_to_dataset(e["prompt"], best["completion"])
                resolved += 1
        else:
            kept.append(e)
    if kept:
        with open(CONFLICT_PATH, "w", encoding="utf-8") as f:
            for e in kept: f.write(json.dumps(e) + "\n")
    elif os.path.exists(CONFLICT_PATH):
        os.remove(CONFLICT_PATH)
    return resolved


def show_dashboard():
    print("=" * 58)
    print("  EMOTIONAL OLLAMA DASHBOARD")
    print("=" * 58)
    data = load_dataset()
    if data:
        print(f"\nDataset: {len(data)} entries"); topic_balance(data)
    else: print("\nDataset: empty")
    print(); show_cache_stats(); print()
    hist = show_weight_trends(); print()
    if hist and len(hist) >= 3: detect_drift(); print()
    show_escalation_dashboard(); print()
    show_escalation_heatmap(); print()
    show_predictive_analytics(); print()
    resolved = auto_resolve_conflicts()
    if resolved: print(f"   Auto-resolved {resolved} conflict(s) (policy: {CONFLICT_AUTO_POLICY}).")
    show_conflicts()
    print("=" * 58)


def dataset_stats(path=DATASET_PATH):
    data = load_dataset(path)
    if not data: print("Dataset is empty."); return
    topic_balance(data)
    print(f"Dataset: {len(data)} entries | Model: {MODEL_VERSION}")
    avg_c = (sum(RECENT_SCORES) / len(RECENT_SCORES)) if RECENT_SCORES else None
    if avg_c is not None: print(f"   Recent avg confidence: {avg_c:.0f}/100 | Consecutive low: {CONSECUTIVE_LOW}")
    show_cache_stats(); show_weight_trends(); show_escalation_dashboard(); show_escalation_heatmap(); show_predictive_analytics(); detect_drift()


def show_conflicts():
    if not os.path.exists(CONFLICT_PATH): print("   No conflicts."); return
    with open(CONFLICT_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    entries.sort(key=lambda e: e.get("priority", 30), reverse=True)
    print(f"   Conflicts ({len(entries)}):")
    for i, e in enumerate(entries):
        pp = e.get("prompt", "")[:60]
        sc = [c.get("score", 0) for c in e.get("completions", [])]
        print(f"   [{i}] [pri {e.get('priority', 30)}] {pp}... scores={sc}")
    print("   resolve <i> <text> | resolve-skip <i>")


def resolve_conflict(idx, correction, path=CONFLICT_PATH):
    if not os.path.exists(path): print("No conflict queue."); return
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if idx < 0 or idx >= len(entries): print(f"Invalid index {idx}."); return
    prompt = entries[idx]["prompt"]; append_to_dataset(prompt, correction)
    print(f"Conflict [{idx}] resolved."); entries.pop(idx)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")
    if correction: generate_variations(prompt, correction)


def skip_conflict(idx, path=CONFLICT_PATH):
    if not os.path.exists(path): print("No conflict queue."); return
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if idx < 0 or idx >= len(entries): print(f"Invalid index {idx}."); return
    entries.pop(idx)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")
    print(f"Skipped conflict [{idx}].")


def validate_dataset(path=DATASET_PATH):
    data = load_dataset(path)
    if not data: print("No entries to validate."); return
    issues = 0
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "prompt" not in entry or "completion" not in entry:
            print(f"Entry {i}: malformed"); issues += 1; continue
        if len(entry["completion"]) < 5: print(f"Entry {i}: too short"); issues += 1
        if "i don't know" in entry["completion"].lower() and len(entry["completion"]) < 20:
            print(f"Entry {i}: weak uncertain reply"); issues += 1
    print(f"{'All valid.' if issues == 0 else f'{issues} issue(s).'}")


def export_snapshot(path=DATASET_PATH, label=""):
    if not os.path.exists(path): print("No dataset to snapshot."); return
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = f"_{label}" if label else ""
    sp = os.path.join(SNAPSHOT_DIR, f"curated_{ts}{suf}.jsonl")
    shutil.copy2(path, sp); print(f"Snapshot: {sp}"); return sp


def export_gold_dataset(path=DATASET_PATH, min_score=90):
    data = load_dataset(path)
    if not data: print("No entries."); return
    os.makedirs(GOLD_DIR, exist_ok=True)
    gp = os.path.join(GOLD_DIR, "gold_dataset.jsonl"); count = 0
    with open(gp, "w", encoding="utf-8") as out:
        for entry in data:
            if score_completion(entry["completion"]) >= min_score:
                out.write(json.dumps({"prompt": entry["prompt"], "completion": entry["completion"], "_score": score_completion(entry["completion"])}) + "\n"); count += 1
    print(f"Gold: {gp} ({count} entries)"); return gp


def sync_dataset(target_dir=None):
    dest = target_dir or SYNC_DIR
    if not dest: print("No sync target."); return
    os.makedirs(dest, exist_ok=True)
    files = [f for f in [DATASET_PATH, WEAK_REPLIES_PATH, CONFLICT_PATH, ESCALATION_LOG_PATH, WEIGHT_LOG_PATH] if os.path.exists(f)]
    s = export_snapshot(); g = export_gold_dataset()
    if s: files.append(s)
    if g: files.append(g)
    for f in files: shutil.copy2(f, os.path.join(dest, os.path.basename(f))); print(f"   Synced: {os.path.basename(f)}")
    print(f"Sync -> {dest}")


def federated_sync(remote_dir, merge_policy="highest"):
    if not os.path.isdir(remote_dir): print(f"Remote not found: {remote_dir}"); return
    local = load_dataset()
    rp = os.path.join(remote_dir, DATASET_PATH)
    remote = load_dataset(rp) if os.path.exists(rp) else []
    seen, conflicts = {}, []
    for entry in local + remote:
        key = entry.get("prompt", "")
        if key in seen:
            es = score_completion(seen[key].get("completion", ""))
            cs = score_completion(entry.get("completion", ""))
            if merge_policy in ("keep-both", "manual"):
                conflicts.append((key, seen[key]["completion"], es, entry["completion"], cs)); continue
            else:
                if cs > es: seen[key] = entry
                elif cs == es: conflicts.append((key, seen[key]["completion"], es, entry["completion"], cs))
        else: seen[key] = entry
    for key, ca, sa, cb, sb in conflicts:
        if merge_policy == "keep-both":
            seen[key + "_v1"] = {"prompt": key + " [v1]", "completion": ca}
            seen[key + "_v2"] = {"prompt": key + " [v2]", "completion": cb}
        elif merge_policy == "manual": append_conflict(key, ca, sa, cb, sb)
    merged = list(seen.values())
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for entry in merged: f.write(json.dumps({"prompt": entry["prompt"], "completion": entry["completion"]}) + "\n")
    print(f"Merge [{merge_policy}]: {len(merged)} entries ({len(local)} local + {len(remote)} remote)")


def fetch_source(source_name, query):
    url = EXTERNAL_SOURCES.get(source_name)
    if not url: return None
    try:
        if source_name == "wikipedia":
            full = url.format(query=query.strip().replace(" ", "_"))
        else: full = url.format(query=urllib.parse.quote(query))
        req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if source_name == "duckduckgo":
                ab = data.get("AbstractText", "")
                if ab: return (source_name, ab)
                rs = data.get("RelatedTopics", [])
                if rs and rs[0].get("Text"): return (source_name, rs[0]["Text"])
            elif source_name == "wikipedia":
                ext = data.get("extract", "")
                if ext: return (source_name, ext[:500])
    except Exception: pass
    return None


def layered_lookup(query):
    cached = cache_lookup(query)
    if cached: return cached
    results = [fetch_source(src, query) for src in EXTERNAL_SOURCES]
    results = [r for r in results if r]
    if results: cache_store(query, results)
    return results


def apply_template(user_input, dynamic_weights=None):
    intent = detect_intent(user_input) if dynamic_weights is not None else None
    if intent and dynamic_weights is not None:
        decay_weights(dynamic_weights)
        boost = predictive_intent_boost(intent)
        if intent == "motivate":
            for n in ["coach", "mentor"]:
                if n in dynamic_weights: dynamic_weights[n] = min(dynamic_weights.get(n, 50) + WEIGHT_BOOST_FACTOR + boost.get(n, 0), 100)
        elif intent == "explain" and "teacher" in dynamic_weights:
            dynamic_weights["teacher"] = min(dynamic_weights["teacher"] + WEIGHT_BOOST_FACTOR + boost.get("teacher", 0), 100)
        elif intent == "advise":
            for n in ["mentor", "coach"]:
                if n in dynamic_weights: dynamic_weights[n] = min(dynamic_weights.get(n, 50) + int(WEIGHT_BOOST_FACTOR * 0.8) + boost.get(n, 0), 100)
        normalize_weights(dynamic_weights)
    for key, template in PROMPT_TEMPLATES.items():
        if user_input.lower().startswith(key): return template.format(query=user_input[len(key):].strip())
    return user_input


def train_from_dataset(base_model=MODEL_VERSION, path=DATASET_PATH, new_model="empathetic-mentor", min_score=80):
    data = load_dataset(path)
    if len(data) < 5: print("Need >= 5 entries."); return
    scored = sorted([(score_completion(e["completion"]), e) for e in data], key=lambda x: x[0], reverse=True)
    hq = [e for s, e in scored if s >= min_score] or [e for _, e in scored[:10]]
    export_snapshot(path)
    print(f"Training '{new_model}' from {base_model} ({len(hq)} examples)...")
    few_shot = "\n\n".join(f"User: {e['prompt']}\nAssistant: {e['completion']}" for e in hq[:20])
    persona = build_persona(data)
    mf = f"""FROM {base_model}
SYSTEM \"\"\"{persona.strip()}\"\"\"
TEMPLATE \"\"\"System: {persona.strip()}

{few_shot}
User: {{.Prompt}}
Assistant: \"\"\"
"""
    mp = "Modelfile.tmp"
    with open(mp, "w", encoding="utf-8") as f: f.write(mf)
    try:
        ollama.create(model=new_model, modelfile=mp)
        print(f"Model '{new_model}' created.")
    except Exception as e: print(f"Training failed: {e}")
    finally:
        if os.path.exists(mp): os.remove(mp)


def handle_escalation_tier(count, user_input, prompt):
    if count >= 7:
        print(f"   Tier 3 ({count}) - hard fallback"); log_escalation(3, count, user_input)
        return "I'm not confident I can give a reliable answer right now. Let's try a different approach."
    if count >= 5:
        print(f"   Tier 2 ({count}) - persona shift"); log_escalation(2, count, user_input); return None
    if count >= 3:
        print(f"   Tier 1 ({count}) - external lookup"); log_escalation(1, count, user_input)
        results = layered_lookup(user_input) if EXTERNAL_ENABLED else []
        if results: return "I checked multiple sources:\n" + "\n".join(f"[{s}] {t}" for s, t in results)
        return None
    return None


def chat_with_feeling(persona_name="mentor"):
    global LAST_REPLY, LAST_PROMPT, LAST_SCORE, RECENT_SCORES, CONSECUTIVE_LOW, DYNAMIC_WEIGHTS, PERSONA_SPEC
    PERSONA_SPEC = persona_name
    DYNAMIC_WEIGHTS = {}
    for part in persona_name.split("+"):
        p = part.strip()
        name = p.rsplit(":", 1)[0].strip() if ":" in p else p
        if name in PERSONAS: DYNAMIC_WEIGHTS[name] = 50
    data = load_dataset()
    active_persona = build_persona(data, persona_name)
    print(f"Chat started [persona: {persona_name}]")
    print("Commands: exit | 99 | correct | stats | validate | gold | conflicts | resolve <i> <t> | resolve-skip <i> | dashboard | heatmap")
    if EXTERNAL_ENABLED: print(f"External: {', '.join(EXTERNAL_SOURCES)}")
    try:
        while True:
            ui = input("You: "); cmd = ui.strip().lower()
            if cmd in ("exit", "quit"): print("Bye."); break
            if cmd == "99":
                data = load_dataset(); active_persona = build_persona(data, persona_name, DYNAMIC_WEIGHTS)
                print(f"Reconnected ({avg_conf_str(RECENT_SCORES)})."); continue
            if cmd == "stats": dataset_stats(); continue
            if cmd == "validate": validate_dataset(); continue
            if cmd == "gold": export_gold_dataset(); continue
            if cmd == "conflicts": show_conflicts(); continue
            if cmd == "dashboard": show_dashboard(); continue
            if cmd == "heatmap": show_escalation_heatmap(); continue
            if cmd.startswith("resolve ") and len(cmd.split()) >= 2:
                parts = ui.split(maxsplit=2)
                try: idx = int(parts[1]); resolve_conflict(idx, parts[2] if len(parts) > 2 else "")
                except (ValueError, IndexError): print("Usage: resolve <i> <text>")
                continue
            if cmd.startswith("resolve-skip") and len(cmd.split()) >= 2:
                try: skip_conflict(int(cmd.split()[1]))
                except (ValueError, IndexError): print("Usage: resolve-skip <i>")
                continue
            if cmd.startswith("correct") and LAST_REPLY is not None:
                fix = ui[len("correct"):].strip()
                if fix: append_to_dataset(LAST_PROMPT, fix); generate_variations(LAST_PROMPT, fix); LAST_REPLY = fix; print("Corrected.")
                continue
            prompt = apply_template(ui, DYNAMIC_WEIGHTS)
            if DYNAMIC_WEIGHTS and "+" in persona_name:
                active_persona = build_persona(data, persona_name, DYNAMIC_WEIGHTS)
                print(f"   Weights: {DYNAMIC_WEIGHTS}")
                log_weights(DYNAMIC_WEIGHTS)
            try:
                resp = ollama.chat(model=MODEL_VERSION, messages=[
                    {"role": "system", "content": active_persona},
                    {"role": "user", "content": prompt},
                ])
                reply = resp["message"]["content"]; score = score_reply(reply)
                RECENT_SCORES.append(score)
                if len(RECENT_SCORES) > 20: RECENT_SCORES.pop(0)
                CONSECUTIVE_LOW = CONSECUTIVE_LOW + 1 if score < 50 else 0
                print(f"Ollama: {reply}  [{score}/100]")
                LAST_PROMPT, LAST_REPLY, LAST_SCORE = prompt, reply, score
                if score < 50:
                    append_to_weak(prompt, reply, score)
                    tr = handle_escalation_tier(CONSECUTIVE_LOW, ui, prompt)
                    if tr: print("Escalation:", tr[:100]); append_to_dataset(prompt, tr)
                    else: print("Fallback:", FALLBACK); append_to_dataset(prompt, FALLBACK)
                elif score < SCORE_THRESHOLD:
                    append_to_weak(prompt, reply, score)
                    fix = input("Correct? (blank to skip): ").strip()
                    if fix: append_to_dataset(prompt, fix); generate_variations(prompt, fix)
                else:
                    if score > 90: print("Auto-approved.")
                    append_to_dataset(prompt, reply)
            except Exception: print("Fallback:", FALLBACK)
    except KeyboardInterrupt: print("\nInterrupted.")


def avg_conf_str(s):
    return f"avg {sum(s)/len(s):.0f}/100" if s else "no data"


if __name__ == "__main__":
    persona_name, sync_target, merge_policy = "mentor", None, "highest"
    rest = []
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--persona" and i + 1 < len(sys.argv): persona_name = sys.argv[i + 1]; i += 2
        elif a in ("--sync", "--sync-dir") and i + 1 < len(sys.argv): sync_target = sys.argv[i + 1]; i += 2
        elif a == "--merge-policy" and i + 1 < len(sys.argv): merge_policy = sys.argv[i + 1]; i += 2
        elif a == "--conflict-policy" and i + 1 < len(sys.argv):
            v = sys.argv[i + 1].lower()
            if v in ("low", "medium", "all"): CONFLICT_AUTO_POLICY = v
            i += 2
        elif a == "--federated-sync" and i + 1 < len(sys.argv): federated_sync(sys.argv[i + 1], merge_policy); sys.exit(0)
        elif a == "--conflicts": show_conflicts(); sys.exit(0)
        elif a == "--dashboard": show_dashboard(); sys.exit(0)
        elif a == "--heatmap": show_escalation_heatmap(); sys.exit(0)
        else: rest.append(a); i += 1
    sys.argv = [sys.argv[0]] + rest
    if "--stats" in sys.argv: dataset_stats()
    elif "--validate" in sys.argv: validate_dataset()
    elif "--gold" in sys.argv: export_gold_dataset()
    elif "--train" in sys.argv: train_from_dataset()
    elif "--sync" in sys.argv or sync_target: sync_dataset(sync_target)
    else: chat_with_feeling(persona_name=persona_name)
