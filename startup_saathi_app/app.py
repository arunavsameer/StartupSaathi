"""
StartupSaathi · app.py
Legal Navigator for Indian Startups — Multilingual Edition
Bharat Bricks Hacks 2026 | IIT Indore | Track: Nyaya-Sahayak

Multilingual support powered by Sarvam AI (Mayura translation model).
Supports: English, Hindi, Marathi, Gujarati, Tamil, Telugu, Kannada,
          Malayalam, Punjabi, Bengali, Odia
"""

import uuid
import time
import logging
import re
import numpy as np
import pydeck as pdk
import streamlit as st
import sys
import os

# ── Ensure src/ is on the path ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src import db, rag, models
from src.translate import (
    maybe_translate_query,
    translate_response,
    ts,               # pre-baked dict + live API fallback
    SARVAM_LANGUAGES, # code -> display name dict
    detect_language,
    is_non_english,
)
from src import nsws_rag
from src.constants import (
    SIZE_OPTIONS, INDIAN_STATES, FALLBACK_SCHEMES, CITY_COORDS, STATE_COORDS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StartupSaathi — Legal Navigator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SECTOR_OPTIONS: dict[str, str] = {s: s for s in nsws_rag.KNOWN_DPIIT_SECTORS}
STATE_OPTIONS: dict[str, str]  = nsws_rag.SUPPORTED_STATES
STATE_DISPLAY_LIST              = list(STATE_OPTIONS.values())
STATE_KEY_BY_DISPLAY            = {v: k for k, v in STATE_OPTIONS.items()}

# Language selector options
LANG_OPTIONS = list(SARVAM_LANGUAGES.keys())     # ["en-IN", "hi-IN", ...]
LANG_LABELS  = list(SARVAM_LANGUAGES.values())   # ["English", "हिंदी…", ...]

# ─────────────────────────────────────────────────────────────────────────────
# Helper: get current UI language from session state
# ─────────────────────────────────────────────────────────────────────────────
def _lang() -> str:
    """Return the currently selected UI language code."""
    return st.session_state.get("ui_lang", "en-IN")

# ─────────────────────────────────────────────────────────────────────────────
# Pure Logic Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_session_defaults(default_sector: str, default_state_key: str = "maharashtra") -> dict:
    return {
        "session_id":       str(uuid.uuid4()),
        "embed_model":      None,
        "faiss_index":      None,
        "faiss_metadata":   None,
        "faiss_loaded":     False,
        "faiss_error":      None,
        "llm_model":        None,
        "llm_tokenizer":    None,
        "llm_name":         None,
        "llm_loaded":       False,
        "nsws_checklist":   [],
        "nsws_sector":      default_sector,
        "nsws_state_key":   default_state_key,
        "completed_tasks":  set(),
        "checklist_ready":  False,
        "sector":           "all",
        "size":             "all",
        "location":         "Maharashtra",
        "startup_desc":     "",
        "chat_history":     [],
        "ui_lang":          "en-IN",   # NEW: selected UI language
    }

def get_coords(city: str, state: str) -> tuple[float, float] | None:
    if city and city.lower().strip() in CITY_COORDS:
        return CITY_COORDS[city.lower().strip()]
    if state and state.lower().strip() in STATE_COORDS:
        return STATE_COORDS[state.lower().strip()]
    return None

def is_placeholder_data(rows: list) -> bool:
    if not rows: return True
    sample = (rows[0].get("chunk_text") or "").strip()
    return bool(re.match(r"^Scheme:\s+Scheme\s+\d+", sample))

def filter_schemes_by_profile(schemes: list[dict], sector: str, size: str) -> list[dict]:
    result = []
    for scheme in schemes:
        s_match = "all" in scheme.get("sectors", ["all"]) or sector in scheme.get("sectors", ["all"])
        z_match = "all" in scheme.get("sizes", ["all"]) or size in scheme.get("sizes", ["all"])
        if s_match and z_match: result.append((2, scheme))
        elif s_match or z_match: result.append((1, scheme))
    result.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in result]

# ─────────────────────────────────────────────────────────────────────────────
# Cached Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embed_model():
    return nsws_rag.load_embed_model()

@st.cache_resource(show_spinner="Loading NSWS licence indices…")
def _load_nsws_indices():
    return nsws_rag.load_nsws_indices()

def _init_session_state() -> None:
    defaults = get_session_defaults(
        default_sector=nsws_rag.KNOWN_DPIIT_SECTORS[0],
        default_state_key="maharashtra",
    )
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def _ensure_faiss() -> bool:
    if st.session_state["faiss_loaded"]: return st.session_state["faiss_error"] is None
    try:
        idx, meta = rag.load_faiss_artifacts()
        st.session_state.update(faiss_index=idx, faiss_metadata=meta, faiss_loaded=True, faiss_error=None)
        return True
    except Exception as exc:
        st.session_state.update(faiss_loaded=True, faiss_error=str(exc))
        return False

def _ensure_llm() -> None:
    if not st.session_state["llm_loaded"]:
        m, tok, name = models.load_llm()
        st.session_state.update(llm_model=m, llm_tokenizer=tok, llm_name=name, llm_loaded=True)

# ─────────────────────────────────────────────────────────────────────────────
# UI — Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    lang = _lang()

    with st.sidebar:
        st.title("StartupSaathi")
        st.caption("Legal Navigator · IIT Indore 2026")
        st.divider()

        # ── Language selector (topmost) ──────────────────────────────────────
        st.subheader("🌐 " + ts("Language", lang))
        current_lang_idx = LANG_OPTIONS.index(lang) if lang in LANG_OPTIONS else 0
        selected_lang_label = st.selectbox(
            ts("Language", lang),
            options=LANG_LABELS,
            index=current_lang_idx,
            label_visibility="collapsed",
            key="sidebar_lang_selector",
        )
        selected_lang_code = LANG_OPTIONS[LANG_LABELS.index(selected_lang_label)]
        if selected_lang_code != lang:
            st.session_state["ui_lang"] = selected_lang_code
            st.rerun()

        st.divider()

        # ── Startup Profile ──────────────────────────────────────────────────
        st.subheader("🏢 " + ts("Startup Profile", lang))
        sector_label = st.selectbox(
            ts("Sector", lang),
            options=list(SECTOR_OPTIONS.keys()),
            index=list(SECTOR_OPTIONS.keys()).index(
                st.session_state.get("nsws_sector", nsws_rag.KNOWN_DPIIT_SECTORS[0])
            ) if st.session_state.get("nsws_sector") in SECTOR_OPTIONS else 0,
            key="sidebar_sector",
        )

        current_state_display = STATE_OPTIONS.get(
            st.session_state.get("nsws_state_key", "maharashtra"), "Maharashtra"
        )
        state_display = st.selectbox(
            ts("State of Operation", lang),
            options=STATE_DISPLAY_LIST,
            index=STATE_DISPLAY_LIST.index(current_state_display)
            if current_state_display in STATE_DISPLAY_LIST else 0,
            key="sidebar_state",
        )
        state_key = STATE_KEY_BY_DISPLAY[state_display]

        size_label = st.selectbox(
            ts("Company Size", lang),
            options=list(SIZE_OPTIONS.keys()),
            index=0,
            key="sidebar_size",
        )

        st.subheader("✍️ " + ts("Describe Your Startup", lang))

        # ── Language hint banner ─────────────────────────────────────────────
        if lang != "en-IN":
            lang_name = SARVAM_LANGUAGES.get(lang, lang)
            st.info(
                f"💬 You can describe your startup in **{lang_name}** — "
                f"it will be automatically translated for best results.",
                icon="🌐",
            )

        startup_desc = st.text_area(
            ts("What does your startup do?", lang),
            value=st.session_state.get("startup_desc", ""),
            placeholder=_desc_placeholder(lang),
            height=120,
        )

        if st.button(
            ts("🚀 Generate My Checklist", lang),
            use_container_width=True,
            type="primary",
        ):
            if not startup_desc.strip():
                st.warning(_no_desc_warning(lang))
                return

            # Translate description to English for RAG if non-English
            english_desc, detected_lang = maybe_translate_query(
                startup_desc.strip(), ui_lang=lang
            )

            st.session_state.update(
                nsws_sector=sector_label, nsws_state_key=state_key,
                sector=sector_label, size=SIZE_OPTIONS[size_label],
                location=state_display, startup_desc=startup_desc.strip(),
                completed_tasks=set(),
            )

            try:
                embed_model = _load_embed_model()
                dpiit_indices, state_idxs = _load_nsws_indices()
            except Exception as exc:
                st.error(f"❌ Could not load NSWS indices: {exc}")
                return

            with st.spinner(f"Searching {sector_label} licences in {state_display}…"):
                try:
                    raw_results = nsws_rag.search_nsws_all(
                        query=english_desc,
                        embed_model=embed_model, sector=sector_label,
                        state_key=state_key, dpiit_indices=dpiit_indices,
                        state_indices=state_idxs, top_k=12,
                        include_general=nsws_rag.INCLUDE_GENERAL_IN_SECTOR_SEARCH,
                    )
                except Exception as exc:
                    st.error(f"Search failed: {exc}")
                    return

            checklist = nsws_rag.format_nsws_checklist(raw_results)
            if checklist:
                st.session_state.update(nsws_checklist=checklist, checklist_ready=True)
                st.success(
                    f"✅ Found {len(raw_results['dpiit'])} central + "
                    f"{len(raw_results['state'])} {state_display} licences!"
                )
            else:
                st.warning("No matching licences found. Try a more detailed description.")


def _desc_placeholder(lang: str) -> str:
    placeholders = {
        "en-IN": "e.g. We're building a B2B SaaS platform for restaurant supply-chain management...",
        "hi-IN": "जैसे: हम रेस्टोरेंट सप्लाई चेन के लिए B2B SaaS बना रहे हैं...",
        "mr-IN": "उदा: आम्ही रेस्टॉरंट सप्लाय चेनसाठी B2B SaaS बनवत आहोत...",
        "gu-IN": "દા.ત. અમે રેસ્ટોરન્ટ સપ્લાય ચેઇન માટે B2B SaaS બનાવી રહ્યા છીએ...",
        "ta-IN": "எ.கா. உணவக விநியோக சங்கிலிக்கான B2B SaaS உருவாக்குகிறோம்...",
        "te-IN": "ఉదా: రెస్టారెంట్ సప్లై చెయిన్ కోసం B2B SaaS నిర్మిస్తున్నాం...",
        "kn-IN": "ಉದಾ: ರೆಸ್ಟೋರೆಂಟ್ ಪೂರೈಕೆ ಸರಪಳಿಗಾಗಿ B2B SaaS ನಿರ್ಮಿಸುತ್ತಿದ್ದೇವೆ...",
        "ml-IN": "ഉദാ: റസ്റ്റോറന്റ് സപ്ലൈ ചെയ്നിനായി B2B SaaS നിർമ്മിക്കുന്നു...",
        "pa-IN": "ਜਿਵੇਂ: ਅਸੀਂ ਰੈਸਟੋਰੈਂਟ ਸਪਲਾਈ ਚੇਨ ਲਈ B2B SaaS ਬਣਾ ਰਹੇ ਹਾਂ...",
        "bn-IN": "যেমন: রেস্টুরেন্ট সাপ্লাই চেইনের জন্য B2B SaaS তৈরি করছি...",
        "od-IN": "ଯଥା: ଆମେ ରେଷ୍ଟୁରେଣ୍ଟ ସପ୍ଲାଇ ଚେନ ପାଇଁ B2B SaaS ତିଆରି କରୁଛୁ...",
    }
    return placeholders.get(lang, placeholders["en-IN"])


def _no_desc_warning(lang: str) -> str:
    warnings = {
        "en-IN": "Please describe your startup before generating the checklist.",
        "hi-IN": "कृपया चेकलिस्ट बनाने से पहले अपना स्टार्टअप बताएं।",
        "mr-IN": "कृपया चेकलिस्ट तयार करण्यापूर्वी तुमचा स्टार्टअप सांगा।",
        "gu-IN": "કૃપા કરીને ચેકલિસ્ટ બનાવવા પહેલાં તમારો સ્ટાર્ટઅપ વર્ણવો.",
        "ta-IN": "சரிபார்ப்பு பட்டியல் உருவாக்குவதற்கு முன் உங்கள் நிறுவனம் பற்றி கூறுங்கள்.",
        "te-IN": "చెక్‌లిస్ట్ రూపొందించే ముందు మీ స్టార్టప్ వివరించండి.",
        "kn-IN": "ಚೆಕ್‌ಲಿಸ್ಟ್ ರಚಿಸುವ ಮೊದಲು ನಿಮ್ಮ ಸ್ಟಾರ್ಟಪ್ ವಿವರಿಸಿ.",
        "ml-IN": "ചെക്ക്‌ലിസ്റ്റ് ഉണ്ടാക്കുന്നതിന് മുമ്പ് നിങ്ങളുടെ സ്റ്റാർട്ടപ്പ് വിവരിക്കൂ.",
        "pa-IN": "ਚੈੱਕਲਿਸਟ ਬਣਾਉਣ ਤੋਂ ਪਹਿਲਾਂ ਆਪਣਾ ਸਟਾਰਟਅੱਪ ਦੱਸੋ।",
        "bn-IN": "চেকলিস্ট তৈরির আগে আপনার স্টার্টআপ বর্ণনা করুন।",
        "od-IN": "ଚେକଲିଷ୍ଟ ତିଆରି ପୂର୍ବରୁ ଆପଣଙ୍କ ଷ୍ଟାର୍ଟଅପ ବର୍ଣ୍ଣନା କରନ୍ତୁ।",
    }
    return warnings.get(lang, warnings["en-IN"])


# ─────────────────────────────────────────────────────────────────────────────
# UI — Checklist Tab
# ─────────────────────────────────────────────────────────────────────────────
def _render_checklist_item(item: dict, completed: set) -> bool:
    tid = item["task_id"]
    is_done = tid in completed
    changed = False
    lang = _lang()

    with st.container(border=True):
        col1, col2 = st.columns([0.05, 0.95])
        with col1:
            checked = st.checkbox("", value=is_done, key=f"chk_{tid}")
        with col2:
            name = item["task_name"]
            if is_non_english(lang):
                name = translate_response(name, lang)
            title_text = f"~~{name}~~" if is_done else f"**{name}**"
            st.markdown(title_text)

            if item.get("description"):
                desc = item["description"]
                if is_non_english(lang):
                    desc = translate_response(desc, lang)
                st.caption(desc)

            fee = item.get("fee_display", "Free")
            fee_str = f"🟢 **Free**" if fee.lower() in ("free", "nil", "₹0", "0") else f"🟡 **₹ {fee}**"
            score = int(item.get("score", 0.0) * 100)
            approval = item.get("approval_type", "")
            appr_str = f" | 📝 {approval}" if approval else ""
            match_label = ts("Match", lang) if lang != "en-IN" else "Match"
            st.markdown(f"{fee_str} | 🎯 {match_label}: {score}% {appr_str}")

            if item.get("portal_url"):
                st.markdown(f"[🔗 View on NSWS Portal]({item['portal_url']})")

    if checked and tid not in completed:
        completed.add(tid)
        changed = True
    elif not checked and tid in completed:
        completed.discard(tid)
        changed = True

    return changed


def render_checklist_tab() -> None:
    lang = _lang()

    if not st.session_state["checklist_ready"]:
        st.info("📋 Fill in your startup profile in the sidebar and click **"
                + ts("🚀 Generate My Checklist", lang) + "** to get started.")
        return

    checklist = st.session_state["nsws_checklist"]
    completed = st.session_state["completed_tasks"]
    total = len(checklist)
    done = len(completed)

    if total == 0:
        st.warning("No checklist items found.")
        return

    pct = int(done / total * 100) if total > 0 else 0
    progress_label = f"{done}/{total} Tasks Completed ({pct}%)"
    if lang != "en-IN":
        progress_label = translate_response(progress_label, lang)
    st.progress(pct / 100.0, text=f"Overall Progress: {progress_label}")
    st.divider()

    dpiit_items = [i for i in checklist if i["source"] == "dpiit"]
    state_items = [i for i in checklist if i["source"] == "state"]
    any_change = False

    central_label = ts("Central Licences", lang) if lang != "en-IN" else "Central Licences"
    state_label   = ts("State Licences", lang) if lang != "en-IN" else "State Licences"

    st.subheader(f"🇮🇳 {central_label} ({len(dpiit_items)})")
    for item in dpiit_items:
        if _render_checklist_item(item, completed): any_change = True
    if not dpiit_items:
        st.caption("No central licences matched your description.")

    st.subheader(f"🗺️ {state_label} ({len(state_items)})")
    for item in state_items:
        if _render_checklist_item(item, completed): any_change = True
    if not state_items:
        st.caption("No state-level licences matched your description.")

    if any_change:
        st.session_state["completed_tasks"] = completed
        db.upsert_user_profile(
            session_id=st.session_state["session_id"],
            sector=st.session_state.get("nsws_sector", ""),
            size=st.session_state.get("size", "all"),
            location=st.session_state.get("location", ""),
            completed_tasks=list(completed),
        )


# ─────────────────────────────────────────────────────────────────────────────
# UI — Legal Q&A Tab
# ─────────────────────────────────────────────────────────────────────────────
def render_qa_tab() -> None:
    lang = _lang()

    st.subheader("💬 " + ts("Ask a Legal Question", lang))

    lang_name = SARVAM_LANGUAGES.get(lang, "English")
    if lang != "en-IN":
        st.info(
            f"🌐 Ask your question in **{lang_name}**. "
            f"It will be translated to English for retrieval, then the answer will be translated back to {lang_name}.",
            icon="🌐",
        )
    else:
        st.caption(
            "Ask anything about Indian startup compliance. "
            "Follow-up questions use previous context automatically."
        )

    if not _ensure_faiss():
        st.error("❌ FAISS index not found. Ensure backend notebooks have been run.")
        return

    question_label = ts("Your question", lang)
    query = st.text_area(question_label, placeholder=_qa_placeholder(lang))

    col1, col2 = st.columns([4, 1])
    with col1:
        ask_clicked = st.button(ts("🔍 Get Answer", lang), type="primary", use_container_width=True)
    with col2:
        if st.button(ts("🔄 Clear Chat", lang), use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    if ask_clicked and query.strip():
        # Step 1: translate query to English if needed
        spinner_label = ts("Analysing and retrieving...", lang)
        with st.spinner(spinner_label):
            history = st.session_state.get("chat_history", [])
            _ensure_llm()

            english_query, query_lang = maybe_translate_query(query.strip(), ui_lang=lang)

            embed_model = st.session_state.get("embed_model")
            if embed_model is None:
                embed_model = models.load_embedding_model()
                st.session_state["embed_model"] = embed_model

            processed_query = (
                rag.rewrite_query_with_history(
                    english_query, history,
                    st.session_state["llm_model"],
                    st.session_state["llm_tokenizer"],
                    st.session_state.get("llm_name", "unknown"),
                )
                if history else english_query
            )

            chunks = rag.multi_query_retrieve(
                query=processed_query,
                embed_fn=lambda q: models.embed_query(q, embed_model),
                faiss_index=st.session_state["faiss_index"],
                faiss_metadata=st.session_state["faiss_metadata"],
                sector=st.session_state.get("sector", "all"),
                top_k=5,
            )

            if not chunks:
                st.warning("No relevant documents found.")
                return

            prompt = rag.build_rag_prompt(
                processed_query, chunks, history[-2:] if history else None, "general", False
            )
            english_answer = models.generate_answer(
                prompt,
                st.session_state["llm_model"],
                st.session_state["llm_tokenizer"],
                st.session_state.get("llm_name", "unknown"),
                max_new_tokens=512,
                system_prompt=models.get_system_prompt("general"),
            )
            sources = rag.format_sources(chunks)

        # Step 2: translate answer back to user's language
        if is_non_english(lang):
            translate_spinner = ts("Translating response...", lang)
            with st.spinner(translate_spinner):
                final_answer = translate_response(english_answer, lang)
        else:
            final_answer = english_answer

        st.session_state["chat_history"].append({"role": "user", "content": query})
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": final_answer,
            "content_en": english_answer,   # keep English copy for re-translation
            "sources": sources,
        })

    history = st.session_state.get("chat_history", [])
    if history:
        st.divider()
        st.subheader("📜 " + ts("Conversation History", lang))
        for msg in reversed(history):
            if msg["role"] == "user":
                st.info(f"**Q:** {msg['content']}")
            else:
                with st.container(border=True):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        st.caption("📄 **Sources:** " + ", ".join(msg["sources"]))


def _qa_placeholder(lang: str) -> str:
    placeholders = {
        "en-IN": "e.g. What are the steps to register under GST as a startup?",
        "hi-IN": "जैसे: GST के तहत स्टार्टअप के रूप में पंजीकरण कैसे करें?",
        "mr-IN": "उदा: GST अंतर्गत स्टार्टअप म्हणून नोंदणी कशी करावी?",
        "gu-IN": "દા.ત. GST હેઠળ સ્ટાર્ટઅપ તરીકે નોંધણી કેવી રીતે કરવી?",
        "ta-IN": "எ.கா. GST-யில் ஒரு தொடக்க நிறுவனமாக பதிவு செய்வது எப்படி?",
        "te-IN": "ఉదా: GST కింద స్టార్టప్‌గా నమోదు చేసుకోవడం ఎలా?",
        "kn-IN": "ಉದಾ: GST ಅಡಿಯಲ್ಲಿ ಸ್ಟಾರ್ಟಪ್ ಆಗಿ ನೋಂದಾಯಿಸುವುದು ಹೇಗೆ?",
        "ml-IN": "ഉദാ: GST-ൽ സ്റ്റാർട്ടപ്പ് ആയി രജിസ്ട്രേഷൻ ചെയ്യുന്നത് എങ്ങനെ?",
        "pa-IN": "ਜਿਵੇਂ: GST ਤਹਿਤ ਸਟਾਰਟਅੱਪ ਵਜੋਂ ਰਜਿਸਟ੍ਰੇਸ਼ਨ ਕਿਵੇਂ ਕਰੀਏ?",
        "bn-IN": "যেমন: GST-এর অধীনে স্টার্টআপ হিসেবে নিবন্ধন কীভাবে করবেন?",
        "od-IN": "ଯଥା: GST ଅଧୀନ ଷ୍ଟାର୍ଟଅପ ଭାବରେ ପଞ୍ଜୀକରଣ କିପରି କରିବେ?",
    }
    return placeholders.get(lang, placeholders["en-IN"])


# ─────────────────────────────────────────────────────────────────────────────
# UI — Government Schemes Tab
# ─────────────────────────────────────────────────────────────────────────────
def render_schemes_tab() -> None:
    lang = _lang()

    st.subheader("🏛️ " + ts("Government Schemes for Startups", lang))
    if lang != "en-IN":
        st.caption(
            translate_response(
                "Filtered to your sector and company size.",
                lang,
            )
        )
    else:
        st.caption("Filtered to your sector and company size.")

    sector = st.session_state.get("nsws_sector", "General")
    size   = st.session_state.get("size", "all")

    with st.spinner("Loading schemes…"):
        rows = db.get_schemes(limit=15)

    if is_placeholder_data(rows):
        filtered = filter_schemes_by_profile(FALLBACK_SCHEMES, sector, size) or FALLBACK_SCHEMES
        st.success(f"Showing {len(filtered)} curated schemes matching your profile.")

        for scheme in filtered:
            with st.container(border=True):
                name = scheme["name"]
                desc = scheme["description"]
                elig = scheme["eligibility"]
                bene = scheme["benefits"]

                if is_non_english(lang):
                    name = translate_response(name, lang)
                    desc = translate_response(desc, lang)
                    elig = translate_response(elig, lang)
                    bene = translate_response(bene, lang)

                st.markdown(f"#### {name}")
                st.markdown(f"**{ts('Description', lang) if lang != 'en-IN' else 'Description'}:** {desc}")
                st.markdown(f"**{ts('Eligibility', lang) if lang != 'en-IN' else 'Eligibility'}:** {elig}")
                st.markdown(f"**{ts('Benefits', lang) if lang != 'en-IN' else 'Benefits'}:** {bene}")
                if scheme.get("link"):
                    st.link_button("View Official Portal", scheme["link"])
    else:
        for row in rows:
            text = row.get("chunk_text") or ""
            parts = {
                p.split(":", 1)[0].strip(): p.split(":", 1)[1].strip()
                for p in text.split(" | ") if ":" in p
            }
            title = parts.get("Scheme") or text[:80]
            with st.container(border=True):
                st.markdown(f"#### {title}")
                for k, v in parts.items():
                    if k != "Scheme":
                        st.markdown(f"**{k}:** {v}")


# ─────────────────────────────────────────────────────────────────────────────
# UI — Opportunities Tab
# ─────────────────────────────────────────────────────────────────────────────
def render_opportunities_tab() -> None:
    lang = _lang()

    st.subheader("🚀 " + ts("Incubation Opportunities", lang))
    if lang != "en-IN":
        st.caption(translate_response("Incubators from the Startup India Seed Fund portal.", lang))
    else:
        st.caption("Incubators from the Startup India Seed Fund portal.")

    sector   = st.session_state.get("nsws_sector", "General")
    location = st.session_state.get("location", "")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        chosen_sector_label = st.selectbox(
            ts("Sector", lang),
            options=list(SECTOR_OPTIONS.keys()),
            index=list(SECTOR_OPTIONS.keys()).index(sector) if sector in SECTOR_OPTIONS else 0,
            key="opp_sector_filter",
        )
    with col2:
        state_options = ["All States"] + INDIAN_STATES
        chosen_state_label = st.selectbox(
            ts("State of Operation", lang),
            options=state_options,
            index=state_options.index(location) if location in state_options else 0,
            key="opp_state_filter",
        )
        chosen_state = "" if chosen_state_label == "All States" else chosen_state_label
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key="opp_refresh"):
            st.rerun()

    with st.spinner("Loading opportunities…"):
        opportunities = db.get_opportunities(limit=50, sector=chosen_sector_label, state=chosen_state)

    if not opportunities:
        st.warning("No matching incubators found. Try broadening your filters.")
        return

    map_data = []
    for opp in opportunities:
        coords = get_coords(opp.get("city", ""), opp.get("state", ""))
        if coords:
            map_data.append({
                "name": opp.get("name", ""),
                "city": opp.get("city", ""),
                "state": opp.get("state", ""),
                "sector": (opp.get("sector") or "")[:60],
                "lat": coords[0],
                "lon": coords[1],
            })

    if map_data:
        st.success(f"📍 {len(map_data)} of {len(opportunities)} incubators mapped.")
        layer = pdk.Layer(
            "ScatterplotLayer", data=map_data,
            get_position=["lon", "lat"],
            get_color=[5, 150, 105, 210],
            get_radius=40000, pickable=True, auto_highlight=True,
        )
        view    = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=4, pitch=0)
        tooltip = {"html": "<b>{name}</b><br/>📍 {city}, {state}<br/>🏷️ {sector}"}
        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style="light"),
            use_container_width=True,
        )

    st.divider()

    for opp in opportunities:
        with st.container(border=True):
            name = opp.get("name", "Unnamed Incubator")
            desc = opp.get("description", "No description available.")
            if is_non_english(lang):
                desc = translate_response(desc, lang)

            st.markdown(f"#### {name}")
            loc_str = ", ".join(filter(None, [opp.get("city", ""), opp.get("state", "")])) or "Location unknown"
            st.caption(f"📍 {loc_str} | 🏷️ {opp.get('sector', 'General')}")
            st.markdown(desc)
            if opp.get("link"):
                st.markdown(f"[🔗 View on Startup India]({opp['link']})")


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _init_session_state()
    lang = _lang()

    # Hero Section
    st.title("StartupSaathi")
    subtitle = ts("AI-powered compliance & legal navigator for Indian startups", lang)
    st.subheader(subtitle)
    st.caption("Powered by Sarvam AI · MiniLM · NSWS FAISS · Delta Lake")
    st.divider()

    render_sidebar()

    # Tab labels translated
    tab_labels = [
        ts("📋 Compliance Checklist", lang),
        ts("💬 Legal Q&A", lang),
        ts("🏛️ Government Schemes", lang),
        ts("🚀 Opportunities", lang),
    ]

    tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

    with tab1: render_checklist_tab()
    with tab2: render_qa_tab()
    with tab3: render_schemes_tab()
    with tab4: render_opportunities_tab()


if __name__ == "__main__":
    main()
