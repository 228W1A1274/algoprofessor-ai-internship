"""
llm_comparator.py
=================
MeetScribe — LLM Summarisation Module
AlgoProfessor AI Internship | Milestone 4 | Day 25

Responsibilities:
    1. Build a structured meeting summarisation prompt
    2. Call Groq (FREE Llama 3.3 70B), Claude 3.5, and GPT-4o in parallel
    3. Return results with latency, cost, and token usage for comparison
    4. Save comparison JSON to outputs/
"""

import os
import time
import json
import concurrent.futures
from typing import Optional

# ── LLM Clients ──────────────────────────────────────────────
# Each is imported lazily so missing packages don't break the module
try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ─────────────────────────────────────────────────────────
# 1. CLIENT INITIALISATION
# ─────────────────────────────────────────────────────────

def init_clients(
    groq_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    openai_key: Optional[str] = None
) -> dict:
    """
    Initialise LLM API clients from provided keys.
    Returns a dict of {name: client} for available clients only.

    Args:
        groq_key      : Groq API key (free at console.groq.com)
        anthropic_key : Anthropic API key (optional)
        openai_key    : OpenAI API key (optional)

    Returns:
        {'groq': client, 'anthropic': client, 'openai': client}
        (only includes clients with valid keys)
    """
    clients = {}

    if groq_key and _GROQ_AVAILABLE:
        clients["groq"] = Groq(api_key=groq_key)
        print("  ✅ Groq (Llama 3.3 70B FREE)   — Ready")
    else:
        print("  ❌ Groq                         — No key (get free key at console.groq.com)")

    if anthropic_key and _ANTHROPIC_AVAILABLE:
        clients["anthropic"] = anthropic.Anthropic(api_key=anthropic_key)
        print("  ✅ Anthropic (Claude 3.5 Sonnet) — Ready")
    else:
        print("  ⚠️  Anthropic                    — Skipped (optional)")

    if openai_key and _OPENAI_AVAILABLE:
        clients["openai"] = openai.OpenAI(api_key=openai_key)
        print("  ✅ OpenAI (GPT-4o)              — Ready")
    else:
        print("  ⚠️  OpenAI                       — Skipped (optional / no free tier)")

    return clients


# ─────────────────────────────────────────────────────────
# 2. PROMPT BUILDER
# ─────────────────────────────────────────────────────────

def build_meeting_prompt(
    diarised_transcript: list,
    meeting_context: str = "Business meeting"
) -> str:
    """
    Build a structured prompt for meeting summarisation.

    The prompt instructs the LLM to extract:
    - Executive summary
    - Key decisions
    - Action items (with speaker attribution)
    - Topics discussed
    - Meeting sentiment

    Args:
        diarised_transcript : List of {'speaker', 'start', 'end', 'text'} dicts
        meeting_context     : Brief description of what the meeting was about

    Returns:
        Formatted prompt string ready to send to any LLM
    """
    # Format transcript as "[SPEAKER_XX] (Xs): text" lines
    transcript_lines = "\n".join([
        f"[{seg['speaker']}] ({seg['start']:.0f}s): {seg['text']}"
        for seg in diarised_transcript
    ])

    prompt = f"""You are an expert meeting analyst and minute-taker. Carefully read the transcript below and produce a professional meeting report.

## MEETING CONTEXT
{meeting_context}

---

## YOUR TASK
Produce the following sections. Use the exact headings shown.

### 1. EXECUTIVE SUMMARY
Write 3-4 sentences covering what the meeting was about, who attended, and the overall outcome.

### 2. KEY DECISIONS MADE
Bullet list of concrete decisions agreed upon.
If none detected: write "No formal decisions recorded."

### 3. ACTION ITEMS
List every task assigned, in this exact format:
- [SPEAKER_XX] must [specific action] — [deadline if mentioned, else "ASAP"]
If none: write "No explicit action items identified."

### 4. TOPICS DISCUSSED
Bullet list of all topics covered in the meeting.

### 5. MEETING SENTIMENT
State: Positive / Neutral / Negative / Mixed
Then write one sentence explaining why.

---

## TRANSCRIPT
{transcript_lines}

---

Respond using exactly the headings above. Be concise, professional, and precise."""

    return prompt


# ─────────────────────────────────────────────────────────
# 3. INDIVIDUAL LLM CALLERS
# ─────────────────────────────────────────────────────────

def call_groq(client, prompt: str, model: str = "llama-3.3-70b-versatile") -> dict:
    """
    Call Groq API (FREE Llama 3.3 70B at 400+ tokens/sec).

    Why Groq?
    - Completely free (generous rate limits)
    - Llama 3.3 70B is comparable to GPT-4o on most tasks
    - 400+ tokens/second — fastest available
    - No credit card required

    Args:
        client : Groq client instance
        prompt : Formatted meeting prompt
        model  : Groq model ID

    Returns:
        Result dict with summary, latency, cost, tokens, status
    """
    MODEL_LABEL = f"Llama 3.3-70B (Groq FREE)"
    t0 = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3,   # Low temp = consistent, professional output
        )
        return {
            "model":       MODEL_LABEL,
            "provider":    "Groq",
            "summary":     response.choices[0].message.content,
            "tokens":      response.usage.total_tokens,
            "cost_usd":    0.0,           # Completely free
            "latency_sec": round(time.time() - t0, 2),
            "status":      "success",
        }
    except Exception as e:
        return {
            "model":    MODEL_LABEL,
            "provider": "Groq",
            "summary":  "",
            "status":   f"error: {str(e)}",
        }


def call_claude(client, prompt: str, model: str = "claude-3-5-sonnet-20241022") -> dict:
    """
    Call Anthropic Claude 3.5 Sonnet.

    Pricing (as of 2025):
        Input:  $3.00 per million tokens
        Output: $15.00 per million tokens
        New accounts get $5 free credit.

    Args:
        client : Anthropic client instance
        prompt : Formatted meeting prompt
        model  : Claude model ID

    Returns:
        Result dict with summary, latency, cost, tokens, status
    """
    MODEL_LABEL = "Claude 3.5 Sonnet"
    t0 = time.time()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        in_tok  = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        cost    = round(in_tok * 0.000003 + out_tok * 0.000015, 6)

        return {
            "model":       MODEL_LABEL,
            "provider":    "Anthropic",
            "summary":     response.content[0].text,
            "tokens":      in_tok + out_tok,
            "cost_usd":    cost,
            "latency_sec": round(time.time() - t0, 2),
            "status":      "success",
        }
    except Exception as e:
        return {
            "model":    MODEL_LABEL,
            "provider": "Anthropic",
            "summary":  "",
            "status":   f"error: {str(e)}",
        }


def call_gpt4o(client, prompt: str, model: str = "gpt-4o") -> dict:
    """
    Call OpenAI GPT-4o.

    Note: OpenAI no longer provides free credits to new accounts.
    The function is fully coded — it just needs a paid API key.
    Architecture is identical to call_groq() and call_claude().

    Pricing (as of 2025):
        Input:  $5.00 per million tokens
        Output: $15.00 per million tokens

    Args:
        client : OpenAI client instance
        prompt : Formatted meeting prompt
        model  : OpenAI model ID

    Returns:
        Result dict with summary, latency, cost, tokens, status
    """
    MODEL_LABEL = "GPT-4o"
    t0 = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3,
        )
        tok  = response.usage.total_tokens
        cost = round(tok * 0.000005, 6)

        return {
            "model":       MODEL_LABEL,
            "provider":    "OpenAI",
            "summary":     response.choices[0].message.content,
            "tokens":      tok,
            "cost_usd":    cost,
            "latency_sec": round(time.time() - t0, 2),
            "status":      "success",
        }
    except Exception as e:
        return {
            "model":    MODEL_LABEL,
            "provider": "OpenAI",
            "summary":  "",
            "status":   f"error: {str(e)}",
        }


# ─────────────────────────────────────────────────────────
# 4. PARALLEL 3-WAY COMPARISON
# ─────────────────────────────────────────────────────────

def run_llm_comparison(
    diarised_transcript: list,
    clients: dict,
    meeting_context: str = "Business meeting"
) -> dict:
    """
    Run all available LLMs in parallel and return comparison results.

    Uses ThreadPoolExecutor so all 3 LLMs are called simultaneously.
    Total time = slowest individual call (not sum of all calls).

    Args:
        diarised_transcript : Merged speaker+text segments
        clients             : Dict from init_clients()
        meeting_context     : Meeting description for the prompt

    Returns:
        Dict of {model_label: result_dict}
    """
    prompt = build_meeting_prompt(diarised_transcript, meeting_context)

    print(f"\n🤖 Running 3-way LLM comparison (parallel)...")
    print(f"   Prompt length: {len(prompt):,} characters")

    tasks = []
    if "groq" in clients:
        tasks.append(("groq",      lambda: call_groq(clients["groq"], prompt)))
    if "anthropic" in clients:
        tasks.append(("anthropic", lambda: call_claude(clients["anthropic"], prompt)))
    if "openai" in clients:
        tasks.append(("openai",    lambda: call_gpt4o(clients["openai"], prompt)))

    if not tasks:
        print("❌ No LLM clients available. Add at least GROQ_API_KEY.")
        return {}

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_map = {executor.submit(fn): name for name, fn in tasks}
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results[result["model"]] = result
            status   = "✅" if result["status"] == "success" else "❌"
            cost_str = "FREE" if result.get("cost_usd", 0) == 0 else f"${result.get('cost_usd', 0):.5f}"
            lat      = result.get("latency_sec", "—")
            print(f"   {status} {result['model']:35s} | {lat}s | {cost_str}")

    return results


def print_summaries(results: dict):
    """Pretty-print all LLM summaries to console."""
    print("\n" + "=" * 60)
    print("LLM SUMMARY COMPARISON")
    print("=" * 60)

    for model, result in results.items():
        if result.get("status") == "success":
            cost_str = "FREE" if result["cost_usd"] == 0 else f"${result['cost_usd']:.5f}"
            print(f"\n{'─'*60}")
            print(f"📌 {result['model']}")
            print(f"   Latency: {result['latency_sec']}s  |  "
                  f"Tokens: {result['tokens']}  |  Cost: {cost_str}")
            print(f"{'─'*60}")
            print(result["summary"])
        else:
            print(f"\n⚠️  {model}: {result['status']}")


def save_results(results: dict, output_path: str = "outputs/llm_comparison.json"):
    """Save LLM comparison results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 LLM results saved → {output_path}")


# ─────────────────────────────────────────────────────────
# 5. QUICK USAGE EXAMPLE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Load diarised transcript from JSON
    transcript_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/diarised_transcript.json"
    with open(transcript_path, "r") as f:
        diarised = json.load(f)

    # Init clients from environment variables
    print("\n🔑 Initialising LLM clients...")
    clients = init_clients(
        groq_key      = os.getenv("GROQ_API_KEY"),
        anthropic_key = os.getenv("ANTHROPIC_API_KEY"),
        openai_key    = os.getenv("OPENAI_API_KEY"),
    )

    # Run comparison
    results = run_llm_comparison(
        diarised,
        clients,
        meeting_context="Business team meeting"
    )

    # Print + save
    print_summaries(results)
    save_results(results)
