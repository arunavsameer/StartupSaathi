"""
models.py — Model Loaders & Inference Wrappers
StartupSaathi | Bharat Bricks Hacks 2026

CPU-only. No device='cuda' anywhere.

LLM strategy — API-based (no local model download, zero RAM cost):
  Priority 1: Sarvam AI API  (set SARVAM_API_KEY in App env vars)
              → sarvam-30b  (recommended, Indian-origin, multilingual)
              → sarvam-m    (legacy fallback within Sarvam)
  Priority 2: Databricks Foundation Models API (free, no key needed)
  Priority 3: Extractive fallback (returns best-matching chunk, no LLM)

  Get a Sarvam API key at: https://dashboard.sarvam.ai

Embedding (unchanged — small model, works fine on free tier):
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (~120 MB)
  Loaded once at startup, cached in st.session_state["embed_model"].

NEW in this version:
  get_system_prompt(query_type) → type-specific system prompt that tells the LLM
    what answer structure to produce, reducing generic/disconnected answers.
  _call_databricks() now accepts system_prompt for consistency with _call_sarvam().
  DEFAULT_SYSTEM_PROMPT significantly strengthened with explicit grounding rules.
"""

import os
import logging
import streamlit as st

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Sarvam AI — OpenAI-compatible endpoint
SARVAM_BASE_URL   = "https://api.sarvam.ai/v1"
SARVAM_MODELS     = ["sarvam-30b", "sarvam-m"]   # tried in order

# Databricks Foundation Models — free serverless fallback
DATABRICKS_MODELS = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-dbrx-instruct",
]


# ─────────────────────────────────────────────────────────────
# Embedding Model (local, ~120 MB — fine on free tier)
# ─────────────────────────────────────────────────────────────

def load_embedding_model():
    """
    Load the multilingual MiniLM embedding model on CPU.
    Called once at app startup, cached in st.session_state.
    Returns a SentenceTransformer instance.
    """
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    logger.info("Embedding model loaded.")
    return model


def embed_query(query: str, embed_model) -> "np.ndarray":
    """Embed a query → (1, 384) float32 numpy array."""
    import numpy as np
    emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb.astype("float32")


def embed_texts_batch(texts: list, embed_model, batch_size: int = 64) -> "np.ndarray":
    """
    Embed a list of texts in one batched forward pass.

    Returns an (N, 384) float32 array — one row per input text.
    Using a single encode() call instead of N individual calls eliminates
    Python loop overhead and lets the model process texts in parallel,
    which is the main source of speed-up on CPU.

    batch_size=64 is a good default for CPU; tune down if you hit OOM.
    """
    import numpy as np
    emb = embed_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb.astype("float32")


# ─────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────

# Base identity and hard rules shared by all system prompts.
_BASE_IDENTITY = (
    "You are StartupSaathi, a precise and practical AI legal assistant for Indian startups. "
    "You help founders understand compliance, registrations, and regulations in India. "
)

_BASE_RULES = (
    "STRICT RULES — follow these exactly:\n"
    "1. Answer ONLY from the legal excerpts provided in the user message.\n"
    "2. Do NOT invent, infer, or add information not present in the excerpts.\n"
    "3. For every specific fact (fee amount, form number, deadline, section of law), "
    "cite the excerpt number, e.g. [EXCERPT 1].\n"
    "4. If the excerpts do not cover the question, say: "
    "'The available documents do not cover this in sufficient detail.' "
    "Do NOT guess or use general knowledge.\n"
    "5. Do NOT show your reasoning or thinking. "
    "Wrap your final answer in <final_answer>...</final_answer> tags.\n"
    "6. Reply in the same language as the question (English or Hindi). "
    "For Hindi questions, keep legal terms (GST, MCA, ROC, etc.) in English.\n"
    "7. If the user sends a greeting (hi, hello, namaste), respond warmly and briefly — "
    "explain what you can help with. Do not reference excerpts for greetings.\n"
)

# The default system prompt — used when no specific query type is identified.
DEFAULT_SYSTEM_PROMPT = (
    "You are StartupSaathi, a knowledgeable AI assistant helping Indian startups with legal compliance. "
    "If the user sends a greeting or small talk (like 'hello', 'hi', 'hey'), respond warmly and briefly — "
    "tell them what you can help with and invite their question. Do NOT reference excerpts for greetings. "
    "For legal questions, answer ONLY from the provided legal excerpts. "
    "IMPORTANT: When citing rules or requirements, quote the EXACT words from the excerpt in quotation marks, "
    "then cite as [EXCERPT N]. Do NOT write [EXCERPT N] as a placeholder — always include the actual quoted text before it. "
    "Example of correct citation: The regulations state \"floors and walls must be maintained in sound condition\" [EXCERPT 1]. "
    "Example of WRONG citation: Location requirements [EXCERPT 1] — this is wrong, never do this. "
    "Do NOT show your reasoning or thinking process. "
    "Write your final answer inside <final_answer> tags. "
    "Use clear headings (##) and numbered points where helpful. "
    "Be thorough and detailed — cover every relevant point from the excerpts. "
    "If the answer is not in the excerpts, say so honestly. "
    "Reply in the same language as the question (English or Hindi)."
)
# Per-query-type system prompts.
# Each builds on the base identity + rules but adds type-specific output guidance.
_SYSTEM_PROMPTS_BY_TYPE = {

    "procedural": _BASE_IDENTITY + (
        "The user is asking about a PROCESS or PROCEDURE. "
        "Structure your answer as:\n"
        "  • One-line overview of what this process achieves.\n"
        "  • Numbered step-by-step instructions (be specific about forms, portals, fees).\n"
        "  • Estimated timeline if mentioned in excerpts.\n"
        "  • What to do after the process is complete.\n"
        + _BASE_RULES
    ),

    "eligibility": _BASE_IDENTITY + (
        "The user is asking about WHO QUALIFIES or WHAT CRITERIA apply. "
        "Structure your answer as:\n"
        "  • Direct yes/no or general eligibility statement first.\n"
        "  • Bullet-pointed list of ALL eligibility criteria from the excerpts.\n"
        "  • Any exclusions or disqualifying factors.\n"
        "  • Next step if eligible.\n"
        + _BASE_RULES
    ),

    "definition": _BASE_IDENTITY + (
        "The user wants to UNDERSTAND or DEFINE a concept. "
        "Structure your answer as:\n"
        "  • A clear, plain-language definition (1-2 sentences).\n"
        "  • Why it matters for Indian startups.\n"
        "  • Any key thresholds, limits, or variations mentioned in the excerpts.\n"
        "  • A practical example if the excerpts support it.\n"
        + _BASE_RULES
    ),

    "penalty": _BASE_IDENTITY + (
        "The user is asking about PENALTIES, FINES, or CONSEQUENCES of non-compliance. "
        "Structure your answer as:\n"
        "  • The specific penalty/fine amount or range (exact figures from excerpts).\n"
        "  • The governing law section or authority.\n"
        "  • Whether penalties are per day, per instance, or one-time.\n"
        "  • How to avoid or remedy the violation.\n"
        "Be specific. Never say 'heavy penalty' without quoting the actual amount from excerpts.\n"
        + _BASE_RULES
    ),

    "deadline": _BASE_IDENTITY + (
        "The user is asking about TIMING, DEADLINES, or DUE DATES. "
        "Structure your answer as:\n"
        "  • The specific deadline or frequency (e.g., within 30 days, annually by 30 Sept).\n"
        "  • What triggers the deadline (incorporation date, financial year, etc.).\n"
        "  • Consequence of missing the deadline.\n"
        "  • Any extensions or grace periods mentioned.\n"
        + _BASE_RULES
    ),

    "document": _BASE_IDENTITY + (
        "The user wants a DOCUMENTS CHECKLIST. "
        "Structure your answer as:\n"
        "  • A numbered checklist of required documents.\n"
        "  • Group by category where helpful (Identity / Address / Business docs).\n"
        "  • Note the required format, attestation, or self-certification if mentioned.\n"
        "  • Flag any documents that differ by entity type or state.\n"
        + _BASE_RULES
    ),

    "cost": _BASE_IDENTITY + (
        "The user is asking about FEES or COSTS. "
        "Structure your answer as:\n"
        "  • Government fee (exact amount from excerpts).\n"
        "  • Professional or filing fee if mentioned.\n"
        "  • Recurring vs one-time cost distinction.\n"
        "  • Concessions for startups, MSMEs, or women-led businesses if available.\n"
        "Never quote a fee range you have not seen in the excerpts.\n"
        + _BASE_RULES
    ),

    "comparison": _BASE_IDENTITY + (
        "The user is COMPARING two or more options. "
        "Structure your answer as a comparison table or parallel bullet points covering:\n"
        "  Registration | Liability | Compliance burden | Tax treatment | Best suited for.\n"
        "End with a clear recommendation based on the scenario in the excerpts.\n"
        + _BASE_RULES
    ),

    "general": DEFAULT_SYSTEM_PROMPT,
}


def get_system_prompt(query_type: str = "general") -> str:
    """
    Return the system prompt tailored to the detected query type.

    Different query types need different answer structures:
      - A 'procedural' query needs numbered steps.
      - An 'eligibility' query needs criteria bullet points.
      - A 'penalty' query needs exact figures and governing law.
    Feeding the LLM a type-specific system prompt dramatically reduces
    generic, disconnected answers — without any extra API calls.

    Args:
        query_type: One of the types returned by rag.detect_query_type().

    Returns:
        System prompt string.
    """
    return _SYSTEM_PROMPTS_BY_TYPE.get(query_type, DEFAULT_SYSTEM_PROMPT)


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _strip_reasoning(text: str) -> str:
    """
    Sarvam-30B and sarvam-m are reasoning models that output <think>...</think>
    blocks before the actual answer. Strip those first, then look for
    <final_answer> tags, then fall back to heuristics.
    """
    import re

    # Step 1: strip <think>...</think> blocks (sarvam-30b reasoning traces)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Step 2: extract content inside <final_answer>...</final_answer>
    match = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Step 3: strip up to last "Answer:" marker
    for marker in ["Answer:", "answer:", "ANSWER:"]:
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()

    # Step 4: drop lines that look like internal reasoning
    reasoning_patterns = re.compile(
        r"^(Okay|OK|Alright|Let me|First,|Looking at|I need|I should|"
        r"Now,|So,|Next,|Then,|Also,|Additionally|Putting it|"
        r"The user|Based on|According to EXCERPT|EXCERPT \d)",
        re.IGNORECASE,
    )
    import re as _re
    lines = text.split("\n")
    clean_lines = []
    found_answer = False
    for line in lines:
        if not found_answer and reasoning_patterns.match(line.strip()):
            continue
        found_answer = True
        clean_lines.append(line)
    result = "\n".join(clean_lines).strip()
    return result if result else text.strip()


def _call_sarvam(
    model_name: str,
    prompt: str,
    max_tokens: int,
    system_prompt: str = None,
) -> str:
    """
    Call the Sarvam AI chat completions API.
    Endpoint is OpenAI-compatible — uses the openai package with a custom base_url.
    Requires SARVAM_API_KEY in environment.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["SARVAM_API_KEY"],
        base_url=SARVAM_BASE_URL,
    )
    sys_msg = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    # Reasoning models pay more attention to the user turn than the system prompt.
    # Prepend a compact instruction header so the model actually follows formatting.
    user_msg = (
        "[INSTRUCTIONS: Do NOT show your thinking. "
        "Reply ONLY inside <final_answer>...</final_answer> tags. "
        "Answer only from the excerpts below. "
        "When citing, ALWAYS quote the exact text first then write [EXCERPT N]. "
        "Never write a bare [EXCERPT N] without the quoted text before it.]\\n\\n"
        + prompt
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.15,     # slightly lower than before for more factual answers
    )
    content = response.choices[0].message.content
    raw = (content or "").strip()
    return _strip_reasoning(raw)


def _call_databricks(
    endpoint_name: str,
    prompt: str,
    max_tokens: int,
    system_prompt: str = None,
) -> str:
    """
    Call a Databricks Foundation Models serving endpoint.
    Now accepts system_prompt for consistency with _call_sarvam().
    """
    from databricks.sdk import WorkspaceClient
    sys_msg = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    w = WorkspaceClient()
    resp = w.serving_endpoints.query(
        name=endpoint_name,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content.strip()
    return _strip_reasoning(raw)


def _probe_sarvam(model_name: str) -> bool:
    """Quick probe to verify Sarvam API key + model are reachable."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["SARVAM_API_KEY"],
            base_url=SARVAM_BASE_URL,
        )
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            temperature=0.2,
        )
        logger.info(f"Sarvam probe succeeded for {model_name}")
        return True
    except Exception as e:
        logger.warning(f"Sarvam probe FAILED for {model_name}: {type(e).__name__}: {e}")
        return False


def _probe_databricks(endpoint_name: str) -> bool:
    """Quick probe to verify a Databricks endpoint is reachable."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        resp = w.serving_endpoints.query(
            name=endpoint_name,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return resp is not None
    except Exception as e:
        logger.warning(f"Databricks probe failed for {endpoint_name}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def load_llm() -> tuple:
    """
    Determine which LLM API to use and return a sentinel model name.

    Returns:
        (None, None, model_name_string)
        model/tokenizer are always None — inference happens via API.
        'extractive' is returned only if no API is reachable.
    """
    sarvam_key = os.environ.get("SARVAM_API_KEY", "").strip()

    # ── Priority 1: Sarvam AI API ────────────────────────────
    if sarvam_key:
        for model_name in SARVAM_MODELS:
            logger.info(f"Probing Sarvam model: {model_name}")
            if _probe_sarvam(model_name):
                logger.info(f"Using Sarvam model: {model_name}")
                st.success(f"✅ Connected to Sarvam AI ({model_name})", icon="🇮🇳")
                return None, None, f"sarvam:{model_name}"
        logger.warning("SARVAM_API_KEY set but all Sarvam models failed.")
        st.warning("⚠️ SARVAM_API_KEY found but API unreachable. Trying Databricks fallback.")
    else:
        logger.info("SARVAM_API_KEY not set — skipping Sarvam.")

    # ── Priority 2: Databricks Foundation Models (free) ──────
    for endpoint in DATABRICKS_MODELS:
        logger.info(f"Probing Databricks endpoint: {endpoint}")
        if _probe_databricks(endpoint):
            logger.info(f"Using Databricks endpoint: {endpoint}")
            return None, None, f"databricks:{endpoint}"

    # ── Priority 3: Extractive fallback ──────────────────────
    logger.warning("No LLM API reachable. Falling back to extractive mode.")
    st.warning(
        "⚠️ No LLM API could be reached. Showing retrieved passages instead.\n\n"
        "👉 Add `SARVAM_API_KEY` in the Databricks Apps environment variables "
        "to get full AI-generated answers using Sarvam's Indian language models.",
        icon="⚠️",
    )
    return None, None, "extractive"


def generate_answer(
    prompt: str,
    model,                      # always None; kept for app.py signature compatibility
    tokenizer,                  # always None; kept for app.py signature compatibility
    model_name: str = "extractive",
    max_new_tokens: int = 512,
    system_prompt: str = None,  # override DEFAULT_SYSTEM_PROMPT; use get_system_prompt()
) -> str:
    """
    Generate an answer via the appropriate API.

    model_name format:
      'sarvam:<model>'        → Sarvam AI API
      'databricks:<endpoint>' → Databricks Foundation Models
      'extractive'            → extractive fallback (no LLM)

    system_prompt:
      Use get_system_prompt(query_type) to get a type-specific prompt.
      Falls back to DEFAULT_SYSTEM_PROMPT if None.
    """
    if model_name == "extractive":
        return _extractive_fallback(prompt)

    try:
        if model_name.startswith("sarvam:"):
            sarvam_model = model_name.split(":", 1)[1]
            return _call_sarvam(sarvam_model, prompt, max_new_tokens, system_prompt=system_prompt)

        elif model_name.startswith("databricks:"):
            db_endpoint = model_name.split(":", 1)[1]
            return _call_databricks(db_endpoint, prompt, max_new_tokens, system_prompt=system_prompt)

        else:
            logger.error(f"Unknown model_name format: {model_name}")
            return _extractive_fallback(prompt)

    except Exception as e:
        logger.error(f"generate_answer failed ({model_name}): {e}")
        st.warning(f"⚠️ API call failed: {e}. Showing retrieved passages instead.")
        return _extractive_fallback(prompt)


def _extractive_fallback(prompt: str) -> str:
    """
    When no LLM is available, extract and display the top-scoring excerpt.
    Improved to pick the most relevant excerpt using simple keyword density
    rather than always returning the first one.
    """
    import re

    try:
        # Extract all EXCERPT blocks
        excerpt_blocks = re.findall(
            r"\[EXCERPT \d+.*?\]\n(.*?)(?=\[EXCERPT |\Z)",
            prompt,
            re.DOTALL,
        )

        if not excerpt_blocks:
            # Fallback for old prompt format
            lines = prompt.split("\n")
            for line in lines:
                if line.strip() and not line.startswith("---") and not line.startswith("RULES"):
                    return (
                        "**[Extractive Mode — LLM unavailable]**\n\n"
                        f"{line.strip()}\n\n"
                        + _extractive_setup_hint()
                    )
            return "Unable to extract a passage. Please add a SARVAM_API_KEY.\n" + _extractive_setup_hint()

        # Extract user question from the prompt
        q_match = re.search(r"User question:\s*(.+)", prompt)
        user_question = q_match.group(1).strip() if q_match else ""

        # Score excerpts by keyword overlap with question
        best_excerpt = excerpt_blocks[0]
        if user_question:
            q_tokens = set(re.split(r"\W+", user_question.lower()))
            best_score = -1
            for block in excerpt_blocks:
                block_tokens = set(re.split(r"\W+", block.lower()))
                score = len(q_tokens & block_tokens)
                if score > best_score:
                    best_score = score
                    best_excerpt = block

        return (
            "**[Extractive Mode — LLM unavailable]**\n\n"
            "Most relevant legal passage found:\n\n"
            f"{best_excerpt.strip()}\n\n"
            "---\n"
            + _extractive_setup_hint()
        )

    except Exception:
        return "Unable to generate an answer. Please review the source excerpts above.\n" + _extractive_setup_hint()


def _extractive_setup_hint() -> str:
    return (
        "💡 *To enable AI-synthesised answers:*\n"
        "1. Get a free API key at [dashboard.sarvam.ai](https://dashboard.sarvam.ai)\n"
        "2. In Databricks Apps UI → your app → **Environment variables**\n"
        "3. Add `SARVAM_API_KEY = <your key>`\n"
        "4. Restart the app"
    )