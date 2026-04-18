"""
rag.py — FAISS Loading, Retrieval, Query Intelligence, and Prompt Construction
StartupSaathi | Bharat Bricks Hacks 2026

Reads the FAISS index and metadata pickle that were built by Notebook 04.
Files live on a Unity Catalog Volume.

Access strategy:
  1. Try /Volumes/... directly (works when running on a Databricks cluster/notebook).
  2. Fall back to Databricks SDK Files API → download to /tmp
     (required in Databricks Apps runtime, which cannot resolve /Volumes/ via os.path).

Query Intelligence Pipeline (NEW — zero LLM cost, pure Python):
  expand_query()          → Domain-aware keyword enrichment for short/vague queries
  detect_query_type()     → Classify query intent (procedural/eligibility/definition/...)
  is_ambiguous_query()    → Detect queries too vague for reliable retrieval
  generate_query_variants() → 2-3 semantically related reformulations
  rerank_chunks()         → BM25-lite reranking of FAISS results by keyword overlap
  multi_query_retrieve()  → Retrieve for multiple variants, merge & deduplicate
  build_rag_prompt()      → Query-type-aware grounded prompt with anti-hallucination rules
  rewrite_query_with_history() → Conversational query rewriting using last N turns
"""

import os
import re
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ─── Unity Catalog Volume paths ──────────────────────────────
ARTIFACT_VOLUME   = "/Volumes/startup_hackathon/legal_data/model_artifacts"
FAISS_INDEX_PATH  = f"{ARTIFACT_VOLUME}/faiss_index.bin"
METADATA_PKL_PATH = f"{ARTIFACT_VOLUME}/faiss_chunk_metadata.pkl"

# ─── Local /tmp cache (used by Databricks Apps runtime) ──────
TMP_DIR           = "/tmp/startup_saathi_artifacts"
TMP_INDEX_PATH    = f"{TMP_DIR}/faiss_index.bin"
TMP_METADATA_PATH = f"{TMP_DIR}/faiss_chunk_metadata.pkl"

EMBED_DIM = 384
TOP_K = 5


# ═══════════════════════════════════════════════════════════════
# QUERY INTELLIGENCE — Constants
# ═══════════════════════════════════════════════════════════════

# Stopwords to ignore during keyword matching and ambiguity detection
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "up", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "each", "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "this", "that", "these", "those", "and", "or", "but", "if", "because",
    "as", "until", "while", "so", "than", "too", "very", "just", "also",
    "what", "how", "when", "where", "who", "which", "tell", "please",
    "ke", "liye", "kya", "hai", "mein", "se", "ka", "ki", "ko", "ne",
}

# Maps detected domain keywords → expansion phrases to append.
# Goal: turn "GST?" into something FAISS can retrieve well.
# Keys are lowercase substrings to match in the query.
_DOMAIN_EXPANSIONS = {
    "gst":           "GST GSTIN goods services tax registration threshold turnover India",
    "fssai":         "FSSAI food safety standards authority license registration food business operator FBO",
    "dpiit":         "DPIIT startup India recognition certificate tax exemption 80-IAC benefits",
    "trademark":     "trademark intellectual property IP registration brand TM protection",
    "copyright":     "copyright creative works protection registration infringement",
    "patent":        "patent invention intellectual property protection filing",
    "msme":          "MSME udyam registration small medium enterprise benefits subsidies",
    "udyam":         "MSME udyam registration small medium enterprise benefits",
    "provident fund":"provident fund EPF EPFO employee employer contribution PF deduction",
    "epf":           "provident fund EPF EPFO employee employer contribution deduction",
    "esi":           "ESI ESIC employee state insurance health maternity benefit",
    "shop":          "shop establishment act state registration license premises",
    "labour":        "labour law employment contract minimum wages act India",
    "income tax":    "income tax PAN TAN TDS filing return assessment India",
    "tds":           "TDS tax deducted source income tax return TAN",
    "pan":           "PAN permanent account number income tax individual business",
    "tan":           "TAN tax deduction account number TDS employer",
    "roc":           "ROC registrar of companies MCA incorporation filing",
    "mca":           "MCA ministry corporate affairs ROC company registration",
    "llp":           "LLP limited liability partnership registration incorporation deed",
    "pvt":           "private limited company incorporation MCA ROC shareholders directors",
    "startup india": "startup India DPIIT recognition certificate tax benefits fund",
    "foreign":       "foreign investment FDI FEMA RBI compliance reporting approval",
    "fdi":           "FDI foreign direct investment FEMA RBI approval sector limits",
    "fema":          "FEMA foreign exchange management act RBI compliance reporting",
    "pollution":     "pollution control board NOC environmental clearance consent",
    "fire":          "fire NOC safety clearance fire department permission",
    "professional tax":"professional tax PT employer employee state deduction",
    "import export": "import export IEC code DGFT registration licence",
    "iec":           "import export code IEC DGFT registration international trade",
    "sebi":          "SEBI securities exchange board India regulatory compliance",
    "company law":   "company law Companies Act 2013 MCA compliance director",
    "incorporation": "company incorporation private limited MCA ROC memorandum articles",
    "director":      "director DIN digital signature DSC appointment resignation",
    "din":           "director identification number DIN MCA registration",
    "equity":        "equity share capital shareholders agreement term sheet valuation",
    "funding":       "startup funding angel venture capital seed series investor",
    "contract":      "contract agreement legal binding terms conditions",
    "compliance":    "compliance regulatory legal requirement India startup annual filing",
    "annual":        "annual compliance filing return ROC MCA income tax yearly",
    "digital signature": "digital signature certificate DSC MCA e-filing",
    "registered office": "registered office address company MCA verification",
    "posh":          "POSH sexual harassment workplace policy committee Internal Complaints",
    "maternity":     "maternity leave benefit Maternity Benefit Act employee women",
    "gratuity":      "gratuity Payment of Gratuity Act employee benefit 5 years",
    "bonus":         "bonus Payment of Bonus Act employee profit minimum",
}

# Query intent patterns → query type label
_QUERY_TYPE_PATTERNS = {
    "procedural": [
        r"\bhow (to|do|can|should)\b", r"\bsteps?\b", r"\bprocess\b",
        r"\bprocedure\b", r"\bregister\b", r"\bapply\b", r"\bfil(e|ing)\b",
        r"\bobtain\b", r"\bopen\b", r"\bget\b.*\b(license|permit|certificate|registration)\b",
        r"\bsetup\b", r"\bset up\b", r"\bstart\b.*\b(company|business|startup)\b",
        r"\bkaise\b", r"\bkaro\b",
    ],
    "eligibility": [
        r"\beligib\w*\b", r"\bqualif\w*\b", r"\bwho can\b", r"\bwho is\b",
        r"\bcriteria\b", r"\bcan (i|a|we|my)\b", r"\bam i\b",
        r"\ballow\w*\b", r"\bentitl\w*\b", r"\bkyaa main\b",
    ],
    "definition": [
        r"\bwhat (is|are|does)\b", r"\bdefin\w*\b", r"\bmeaning\b",
        r"\bexplain\b", r"\bdescribe\b", r"\bkya hai\b", r"\bkya hota\b",
        r"\bwhat does .+ mean\b",
    ],
    "penalty": [
        r"\bpenalt\w*\b", r"\bfine\b", r"\bpunish\w*\b", r"\bconsequen\w*\b",
        r"\bviolat\w*\b", r"\bnon.?compli\w*\b", r"\bdefault\b", r"\blate fee\b",
        r"\bpenalized\b", r"\bjail\b", r"\bimprison\w*\b",
    ],
    "deadline": [
        r"\bdeadline\b", r"\bdue date\b", r"\bwhen\b.*(file|submit|pay|renew)",
        r"\btimeline\b", r"\bhow long\b", r"\blast date\b", r"\bvalidity\b",
        r"\bexpiry\b", r"\brenew\w*\b", r"\bwithin\b.*\bdays?\b",
    ],
    "document": [
        r"\bdocument\w*\b", r"\bcertificate\b", r"\bproof\b", r"\brequired\b",
        r"\bpaperwork\b", r"\bform\b.*\b(need|requir)\b", r"\bklist\b",
        r"\bsubmit\b", r"\battach\w*\b", r"\bupload\b",
    ],
    "cost": [
        r"\bcost\b", r"\bfee\b", r"\bcharge\b", r"\bpric\w*\b",
        r"\bhow much\b", r"\bkitna\b", r"\bexpens\w*\b", r"\bpay\b",
    ],
    "comparison": [
        r"\bvs\.?\b", r"\bversus\b", r"\bdifferen\w*\b", r"\bcompar\w*\b",
        r"\bbetter\b", r"\bor\b.*(pvt|llp|opc|partnership)",
        r"\bwhich (is|one)\b",
    ],
}


# ═══════════════════════════════════════════════════════════════
# QUERY INTELLIGENCE — Public Functions
# ═══════════════════════════════════════════════════════════════

def detect_query_type(query: str) -> str:
    """
    Classify the query into one of 8 intent types using regex pattern matching.
    Zero LLM cost — pure Python.

    Types: procedural | eligibility | definition | penalty |
           deadline   | document    | cost       | comparison | general

    Used by build_rag_prompt() to inject type-specific instructions into the
    prompt, which guides the LLM to structure the answer appropriately
    (e.g., numbered steps for procedural, criteria list for eligibility).
    """
    q = query.lower()
    for qtype, patterns in _QUERY_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                return qtype
    return "general"


def expand_query(query: str) -> str:
    """
    Enrich a query with domain-relevant terms using a keyword → expansion dict.
    Zero LLM cost — pure dictionary lookup.

    Why this helps:
      - Short queries like "GST?" have poor FAISS recall because the query
        vector sits far from verbose legal document chunks.
      - Adding "GST GSTIN goods services tax registration threshold turnover"
        pulls the query vector closer to relevant chunks in embedding space.
      - Only expands when a domain keyword is actually found — no noise added
        for unrecognized queries.

    Returns the original query + appended domain terms (or original if no match).
    """
    q_lower = query.lower()
    expansions_to_add = []

    for keyword, expansion in _DOMAIN_EXPANSIONS.items():
        if keyword in q_lower:
            # Don't add the expansion if most of its terms are already in the query
            new_terms = [
                t for t in expansion.split()
                if t.lower() not in q_lower
            ]
            if new_terms:
                expansions_to_add.append(" ".join(new_terms))

    if not expansions_to_add:
        return query

    expanded = query.rstrip("?. ") + " " + " ".join(expansions_to_add)
    logger.debug(f"Query expanded: '{query}' → '{expanded[:120]}...'")
    return expanded


def is_ambiguous_query(query: str) -> bool:
    """
    Detect queries that are too short or vague for reliable retrieval.
    Heuristic — zero LLM cost.

    A query is considered ambiguous when:
      1. Fewer than 3 meaningful (non-stopword) tokens remain, OR
      2. No domain keyword from the expansion dict is found AND
         the query is under 5 words total

    Ambiguous flag is passed into build_rag_prompt() which adds an
    instruction for the LLM to ask a clarifying follow-up.

    Examples of ambiguous: "compliance?", "register", "help me"
    Examples of non-ambiguous: "how to get GST registration?", "FSSAI license documents"
    """
    tokens = [
        t for t in re.split(r"\W+", query.lower())
        if t and t not in _STOPWORDS and len(t) > 2
    ]

    if len(tokens) < 3:
        return True

    # Short query with no recognized domain keyword
    total_words = len(query.split())
    has_domain_keyword = any(kw in query.lower() for kw in _DOMAIN_EXPANSIONS)
    if total_words < 5 and not has_domain_keyword:
        return True

    return False


def generate_query_variants(query: str) -> list:
    """
    Generate 2-3 semantically distinct query variants for multi-query retrieval.
    Zero LLM cost — pure heuristics.

    Why this helps:
      FAISS retrieves the single nearest-neighbour region of the query vector.
      A slightly different phrasing of the same question sits in a different
      region and retrieves different (often complementary) chunks.
      Merging results from multiple variants dramatically improves recall.

    Variants produced:
      1. Original query (always included)
      2. Domain-expanded version (adds domain keywords → better FAISS recall)
      3. Reformulated question (structure-transformed version of query)

    Returns a deduplicated list of 1-3 query strings.
    """
    variants = [query]

    # Variant 2: domain-expanded
    expanded = expand_query(query)
    if expanded != query and expanded not in variants:
        variants.append(expanded)

    # Variant 3: structural reformulation based on query type
    q = query.strip().lower().rstrip("?. ")
    reformulated = None

    if re.search(r"^what (is|are)\b", q):
        topic = re.sub(r"^what (is|are)\s+", "", q).strip()
        reformulated = f"explain {topic} requirements procedure India startup"

    elif re.search(r"^how (to|do|can|should)\b", q):
        topic = re.sub(r"^how (to|do|can|should)\s+", "", q).strip()
        reformulated = f"steps procedure {topic} India registration"

    elif re.search(r"^(can i|can a|am i|is it|do i)\b", q):
        topic = re.sub(r"^(can i|can a|am i|is it|do i)\s+", "", q).strip()
        reformulated = f"eligibility criteria {topic} India startup rules"

    elif re.search(r"^(what|which) documents?\b", q):
        topic = re.sub(r"^(what|which) documents?\s*(are|do)?\s*(required|needed)?\s*(for)?\s*", "", q).strip()
        reformulated = f"{topic} required documents checklist India"

    elif len(query.split()) <= 4:
        # Very short query — create a "tell me about X" expansion
        reformulated = f"complete guide {query} startup India legal compliance requirements"

    if reformulated and reformulated not in variants:
        variants.append(reformulated)

    return variants


def rerank_chunks(chunks: list, query: str, top_k: int = TOP_K) -> list:
    """
    Rerank FAISS-retrieved chunks using a BM25-inspired hybrid score.

    Why pure FAISS distance isn't always enough:
      FAISS L2 distance captures semantic similarity well on average, but
      can retrieve topically adjacent chunks that miss exact query terms.
      Keyword overlap provides a complementary signal that FAISS misses.

    Scoring formula (per chunk):
      faiss_score   = 1 - normalised_L2_distance   (higher = closer match)
      overlap_score = |query_tokens ∩ chunk_tokens| / |query_tokens|
      phrase_bonus  = +0.1 per 2-word query phrase found in chunk (capped 0.3)

      final_score = 0.50 * faiss_score
                  + 0.35 * overlap_score
                  + 0.15 * phrase_bonus

    The blend weights are tuned for legal document retrieval where exact
    terminology matters (GST vs GSTIN vs "goods and services tax" are all
    semantically close but FAISS may rank them differently from keyword match).

    Args:
        chunks:  List of chunk dicts with '_distance' field from FAISS.
        query:   Raw user query (or rewritten query) for token extraction.
        top_k:   Return at most this many chunks.

    Returns:
        Reranked list (top_k), each chunk has '_rerank_score' added.
    """
    if not chunks:
        return chunks

    # Tokenize query — remove stopwords, keep meaningful terms
    query_tokens = set(
        t for t in re.split(r"\W+", query.lower())
        if t and t not in _STOPWORDS and len(t) > 2
    )
    if not query_tokens:
        return chunks[:top_k]

    query_words = [
        t for t in re.split(r"\W+", query.lower())
        if t and len(t) > 2
    ]

    # Normalise FAISS distances across this result set
    distances = [c.get("_distance", 0.0) for c in chunks]
    max_d = max(distances) if distances else 1.0
    min_d = min(distances) if distances else 0.0
    d_range = (max_d - min_d) if (max_d - min_d) > 1e-6 else 1.0

    scored = []
    for chunk in chunks:
        dist = chunk.get("_distance", 0.0)
        faiss_score = 1.0 - (dist - min_d) / d_range

        chunk_text = chunk.get("chunk_text", "").lower()
        chunk_tokens = set(re.split(r"\W+", chunk_text))

        # Keyword overlap
        overlap = len(query_tokens & chunk_tokens) / len(query_tokens)

        # Phrase bonus: bigrams
        phrase_bonus = 0.0
        for i in range(len(query_words) - 1):
            bigram = f"{query_words[i]} {query_words[i+1]}"
            if bigram in chunk_text:
                phrase_bonus = min(phrase_bonus + 0.1, 0.3)

        final_score = 0.50 * faiss_score + 0.35 * overlap + 0.15 * phrase_bonus
        scored.append({**chunk, "_rerank_score": round(final_score, 4)})

    scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
    return scored[:top_k]


def multi_query_retrieve(
    query: str,
    embed_fn,
    faiss_index,
    faiss_metadata: list,
    sector: str = "all",
    top_k: int = TOP_K,
) -> list:
    """
    Retrieve chunks for multiple query variants and merge results.

    Why this helps more than a single FAISS query:
      Each query variant embeds to a slightly different point in vector space,
      retrieving a different neighbourhood of chunks.  Merging all neighbourhoods
      dramatically improves recall — especially for short/vague queries where a
      single embedding sits in a low-density region.

    Process:
      1. Generate 2-3 variants of the input query (zero LLM cost).
      2. Embed each variant independently.
      3. Retrieve top-(top_k * 2) for each variant (over-fetch to allow merging).
      4. Deduplicate by chunk_id — keep the copy with the lowest distance score.
      5. BM25-lite rerank the merged pool against the original query.
      6. Return top_k after reranking.

    Args:
        query:          Raw or rewritten user query.
        embed_fn:       Callable — takes a string, returns (1, 384) float32 array.
                        Typically: lambda q: models.embed_query(q, embed_model)
        faiss_index:    Loaded FAISS index.
        faiss_metadata: Metadata list aligned to the index.
        sector:         Sector filter tag passed to retrieve_chunks().
        top_k:          Final number of chunks to return.

    Returns:
        Reranked list of up to top_k chunk dicts.
    """
    variants = generate_query_variants(query)
    logger.info(f"multi_query_retrieve: {len(variants)} variants for query '{query[:60]}'")

    seen_chunk_ids: dict = {}   # chunk_id → chunk dict (keep lowest distance)
    fetch_k = top_k * 2        # over-fetch per variant to have room to merge

    for variant in variants:
        emb = embed_fn(variant)
        chunks = retrieve_chunks(
            query_embedding=emb,
            faiss_index=faiss_index,
            faiss_metadata=faiss_metadata,
            sector=sector,
            top_k=fetch_k,
        )
        for chunk in chunks:
            cid = chunk.get("chunk_id", chunk.get("chunk_text", "")[:40])
            if cid not in seen_chunk_ids:
                seen_chunk_ids[cid] = chunk
            else:
                # Keep the retrieval with the lower distance (higher relevance)
                if chunk.get("_distance", 999) < seen_chunk_ids[cid].get("_distance", 999):
                    seen_chunk_ids[cid] = chunk

    merged = list(seen_chunk_ids.values())
    logger.info(f"multi_query_retrieve: {len(merged)} unique chunks before rerank")

    return rerank_chunks(merged, query, top_k=top_k)


def rewrite_query_with_history(
    query: str,
    history: list,
    model,
    tokenizer,
    model_name: str,
) -> str:
    """
    Rewrite the latest user query into a self-contained standalone question
    using the last 1-2 conversation turns as context.

    Only called when history is non-empty (no cost on first question).
    Falls back to the original query on any error.

    Args:
        query:      Raw new question from the user.
        history:    List of {"role": "user"/"assistant", "content": str} dicts.
        model:      LLM model (None for API-based models).
        tokenizer:  LLM tokenizer (None for API-based models).
        model_name: Model name string for generate_answer routing.

    Returns:
        A rewritten standalone question string.
    """
    try:
        recent = history[-2:] if len(history) >= 2 else history[-1:]

        convo_lines = []
        for turn in recent:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            # Truncate long assistant answers — keep rewrite prompt compact
            if turn.get("role") == "assistant" and len(content) > 300:
                content = content[:300] + "..."
            convo_lines.append(f"{role}: {content}")

        convo_block = "\n".join(convo_lines)

        rewrite_prompt = (
            f"Conversation so far:\n{convo_block}\n\n"
            f"New question: {query}\n\n"
            "If the new question is vague (like 'more', 'that', 'it', 'those'), "
            "you MUST include the topic from previous conversation.\n\n"
            "Rewrite it into a FULLY SELF-CONTAINED question.\n\n"
            "Examples:\n"
            "User: rules for food startups\n"
            "New: give more rules\n"
            "→ give more rules for food startup compliance in India\n\n"
            "Output only the rewritten question."
        )

        from src import models as _models
        rewritten = _models.generate_answer(
            prompt=rewrite_prompt,
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            max_new_tokens=80,
            system_prompt=(
                "You are a query rewriting assistant for a legal chatbot. "
                "Output ONLY the rewritten question as a single sentence. "
                "No preamble, no explanation, no tags."
            ),
        )

        rewritten = rewritten.strip()
        if not rewritten or len(rewritten) > 300:
            return query

        logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.warning(f"Query rewriting failed ({e}), using original.")
        return query


# ═══════════════════════════════════════════════════════════════
# FAISS ARTIFACT MANAGEMENT (unchanged from original)
# ═══════════════════════════════════════════════════════════════

def _volumes_directly_accessible() -> bool:
    try:
        return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PKL_PATH)
    except Exception:
        return False


def _tmp_cache_valid() -> bool:
    return os.path.exists(TMP_INDEX_PATH) and os.path.exists(TMP_METADATA_PATH)


def _download_via_sdk() -> None:
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as e:
        raise ImportError(
            "databricks-sdk is not installed. "
            "Add 'databricks-sdk' to requirements.txt."
        ) from e

    os.makedirs(TMP_DIR, exist_ok=True)
    w = WorkspaceClient()

    logger.info(f"Downloading FAISS index via SDK -> {TMP_INDEX_PATH}")
    response = w.files.download(FAISS_INDEX_PATH)
    with open(TMP_INDEX_PATH, "wb") as f:
        f.write(response.contents.read())
    logger.info(f"  faiss_index.bin saved ({os.path.getsize(TMP_INDEX_PATH)/1024/1024:.2f} MB)")

    logger.info(f"Downloading metadata pickle via SDK -> {TMP_METADATA_PATH}")
    response = w.files.download(METADATA_PKL_PATH)
    with open(TMP_METADATA_PATH, "wb") as f:
        f.write(response.contents.read())
    logger.info(f"  faiss_chunk_metadata.pkl saved ({os.path.getsize(TMP_METADATA_PATH)/1024:.1f} KB)")


def _resolve_local_paths() -> tuple:
    if _volumes_directly_accessible():
        logger.info("FAISS artifacts accessible via /Volumes/ path directly.")
        return FAISS_INDEX_PATH, METADATA_PKL_PATH

    if _tmp_cache_valid():
        logger.info("FAISS artifacts found in /tmp cache.")
        return TMP_INDEX_PATH, TMP_METADATA_PATH

    logger.info(
        "/Volumes/ path not directly accessible (Databricks Apps runtime). "
        "Downloading artifacts via SDK ..."
    )
    try:
        _download_via_sdk()
    except Exception as e:
        raise FileNotFoundError(
            f"Could not resolve FAISS artifacts from '{ARTIFACT_VOLUME}'. "
            f"SDK download also failed: {e}\n"
            "Make sure Notebook 04 (04_build_faiss_index) has been run "
            "and the app has permission to access the Volume."
        ) from e

    if not _tmp_cache_valid():
        raise FileNotFoundError(
            f"SDK download appeared to succeed but files are missing from {TMP_DIR}."
        )

    return TMP_INDEX_PATH, TMP_METADATA_PATH


def faiss_artifacts_exist() -> bool:
    if _volumes_directly_accessible():
        return True
    if _tmp_cache_valid():
        return True
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        files = list(w.files.list(ARTIFACT_VOLUME))
        names = {f.name for f in files}
        return "faiss_index.bin" in names and "faiss_chunk_metadata.pkl" in names
    except Exception as e:
        logger.warning(f"SDK existence check failed: {e}")
        return False


def load_faiss_artifacts() -> tuple:
    """
    Load FAISS index and metadata pickle.
    Returns (faiss_index, faiss_metadata).
    """
    index_path, metadata_path = _resolve_local_paths()

    import faiss

    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    logger.info(f"FAISS index loaded: {index.ntotal} vectors")

    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    logger.info(f"Metadata loaded: {len(metadata)} entries")

    assert index.ntotal == len(metadata), (
        f"Index/metadata misalignment: {index.ntotal} vectors vs {len(metadata)} metadata entries. "
        "Re-run Notebook 04."
    )

    return index, metadata


# ═══════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════

def retrieve_chunks(
    query_embedding: np.ndarray,
    faiss_index,
    faiss_metadata: list,
    sector: str = "all",
    top_k: int = TOP_K,
) -> list:
    """
    Search the FAISS index and return the top_k most relevant chunks.
    Low-level retrieval used by multi_query_retrieve().
    For the full pipeline (multi-query + reranking) call multi_query_retrieve().
    """
    search_k = top_k * 6 if sector != "all" else top_k * 2

    q = query_embedding.astype(np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    distances, indices = faiss_index.search(q, min(search_k, faiss_index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        meta = faiss_metadata[idx].copy()
        meta["_distance"] = float(dist)
        results.append(meta)

    if sector != "all":
        sector_matched = [
            r for r in results
            if sector in (r.get("sector_tag") or "")
            or "all" in (r.get("sector_tag") or "all")
        ]
        if len(sector_matched) >= top_k:
            results = sector_matched

    return results[:top_k]


# ═══════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

# Per-query-type output format instructions injected into the prompt.
# These tell the LLM what structure to use for the answer.
_TYPE_INSTRUCTIONS = {
    "procedural": (
        "The user is asking HOW to do something. "
        "Structure your answer as a clear numbered step-by-step process. "
        "Include specific forms, portals, fees, and timelines where mentioned in the excerpts. "
        "Start with an overview sentence, then list the steps."
    ),
    "eligibility": (
        "The user is asking WHO qualifies or WHAT the requirements are. "
        "List the eligibility criteria clearly as bullet points. "
        "Mention any exceptions or exclusions if present in the excerpts. "
        "End with a brief note on what to do if requirements are met."
    ),
    "definition": (
        "The user wants to understand WHAT something is. "
        "Give a clear plain-English definition first. "
        "Then explain its relevance to Indian startups. "
        "Use examples from the excerpts if available."
    ),
    "penalty": (
        "The user is asking about CONSEQUENCES or FINES. "
        "Be specific — state the exact penalty amounts or ranges if in the excerpts. "
        "Mention the governing section/act. "
        "Briefly note how to avoid the penalty."
    ),
    "deadline": (
        "The user is asking about TIMING or DUE DATES. "
        "State the specific deadline, frequency, or validity period first. "
        "Then mention the consequence of missing it. "
        "If deadlines vary by category/size, note the variations."
    ),
    "document": (
        "The user wants to know WHAT DOCUMENTS are needed. "
        "Present the required documents as a numbered checklist. "
        "Group by category if there are many (e.g., identity proof, address proof, business docs). "
        "Note the format or attesting authority if mentioned."
    ),
    "cost": (
        "The user is asking about FEES or COSTS. "
        "List the specific fee amounts from the excerpts. "
        "Distinguish between government fees, professional fees, and recurring costs. "
        "Mention any concessions available to startups or MSMEs."
    ),
    "comparison": (
        "The user is COMPARING options. "
        "Structure your answer as a side-by-side comparison covering: "
        "registration, liability, compliance burden, tax treatment, and suitability. "
        "Recommend based on the startup scenario if the excerpts support it."
    ),
    "general": (
        "Answer the question clearly and completely from the provided excerpts. "
        "Use headings and bullet points where they improve readability. "
        "Cite the relevant law, section, or authority for key claims."
    ),
}


def build_rag_prompt(
    query: str,
    chunks: list,
    history: list = None,
    query_type: str = "general",
    is_ambiguous: bool = False,
) -> str:
    """
    Construct a grounded, query-type-aware RAG prompt for the LLM.

    Improvements over the original:
      1. Query type instruction — tells the LLM what answer structure to use.
      2. Conversation history injection — last 2 turns for follow-up coherence.
      3. Explicit anti-hallucination rules — numbered, hard to ignore.
      4. Ambiguity flag — instructs LLM to ask a clarifying question when needed.
      5. Source citation requirement — forces attribution per claim.

    Args:
        query:        The (possibly rewritten/expanded) user question.
        chunks:       Reranked retrieved chunks.
        history:      Optional last 2 conversation turns as {"role", "content"} dicts.
        query_type:   Output of detect_query_type() — controls answer structure.
        is_ambiguous: Output of is_ambiguous_query() — adds clarification instruction.

    Returns:
        Formatted prompt string ready for generate_answer().
    """
    # ── Format reference excerpts ──────────────────────────────
    excerpts = []
    for i, chunk in enumerate(chunks, 1):
        source  = chunk.get("source_file", "unknown")
        page    = chunk.get("page_number", "?")
        text    = chunk.get("chunk_text", "").strip()
        score   = chunk.get("_rerank_score", "")
        score_tag = f" | relevance {score:.2f}" if score else ""
        excerpts.append(f"[EXCERPT {i} — {source}, p.{page}{score_tag}]\n{text}")

    excerpts_block = "\n\n".join(excerpts)

    # ── Format conversation history block ──────────────────────
    history_block = ""
    if history:
        recent = history[-2:] if len(history) >= 2 else history
        lines = []
        for turn in recent:
            role    = "User" if turn.get("role") == "user" else "Assistant"
            content = turn.get("content", "")
            if role == "Assistant" and len(content) > 400:
                content = content[:400] + "..."
            lines.append(f"{role}: {content}")
        history_block = (
            "\n--- PRIOR CONVERSATION ---\n"
            + "\n".join(lines)
            + "\n--- END PRIOR CONVERSATION ---\n"
        )

    # ── Query type instruction ─────────────────────────────────
    type_instruction = _TYPE_INSTRUCTIONS.get(query_type, _TYPE_INSTRUCTIONS["general"])

    # ── Ambiguity instruction ──────────────────────────────────
    ambiguity_note = ""
    if is_ambiguous:
        ambiguity_note = (
            "\nNOTE: The question appears vague or incomplete. "
            "After providing whatever answer you can from the excerpts, "
            "ask the user ONE specific clarifying question to help give a better answer.\n"
        )

    # ── Assemble prompt ────────────────────────────────────────
    prompt = f"""--- LEGAL REFERENCE EXCERPTS ---
{excerpts_block}
--- END EXCERPTS ---
{history_block}
ANSWER FORMAT GUIDANCE: {type_instruction}
{ambiguity_note}
RULES YOU MUST FOLLOW:
1. Answer ONLY using information from the excerpts above. Do not use outside knowledge.
2. For every specific claim (fee, deadline, form number, section), cite its EXCERPT number, e.g. [EXCERPT 2].
3. If the excerpts do not contain enough information to answer fully, say exactly: "The available documents do not cover this in detail" — do not guess.
4. Keep the answer focused and practical. Avoid repeating the question back.
5. If the question involves a number (fee, days, employees), quote it exactly as it appears in the excerpt.

User question: {query}"""

    return prompt


def format_sources(chunks: list) -> list:
    """Return a deduplicated list of human-readable source citations."""
    seen = set()
    sources = []
    for chunk in chunks:
        source   = chunk.get("source_file", "unknown")
        page     = chunk.get("page_number", "?")
        citation = f"{source} — p.{page}"
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)
    return sources