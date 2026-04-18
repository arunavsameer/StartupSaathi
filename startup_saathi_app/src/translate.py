"""
translate.py — Sarvam AI Powered Translation & Language Detection
StartupSaathi | Bharat Bricks Hacks 2026

Uses Sarvam AI's /v1/translate endpoint (Mayura model) for high-quality
Indian language <-> English translation.

Features:
  - Language detection (langdetect)
  - Query translation: Indian language -> English (for RAG retrieval)
  - Response translation: English -> user's chosen language
  - UI string translation with in-memory caching (avoids duplicate API calls)
  - Graceful fallback to original text if API is unreachable
"""

from __future__ import annotations

import os
import logging
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language Registry
# ---------------------------------------------------------------------------

SARVAM_LANGUAGES: dict[str, str] = {
    "en-IN": "English",
    "hi-IN": "हिंदी (Hindi)",
    "mr-IN": "मराठी (Marathi)",
    "gu-IN": "ગુજરાતી (Gujarati)",
    "ta-IN": "தமிழ் (Tamil)",
    "te-IN": "తెలుగు (Telugu)",
    "kn-IN": "ಕನ್ನಡ (Kannada)",
    "ml-IN": "മലയാളം (Malayalam)",
    "pa-IN": "ਪੰਜਾਬੀ (Punjabi)",
    "bn-IN": "বাংলা (Bengali)",
    "od-IN": "ଓଡ଼ିଆ (Odia)",
}

# Map langdetect ISO-639-1 codes -> Sarvam codes
_LANGDETECT_TO_SARVAM: dict[str, str] = {
    "hi": "hi-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "bn": "bn-IN",
    "or": "od-IN",
    "en": "en-IN",
}

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/v1/translate"

# ---------------------------------------------------------------------------
# Core Translation via Sarvam AI
# ---------------------------------------------------------------------------

def _sarvam_translate(
    text: str,
    source_lang: str,
    target_lang: str,
    mode: str = "formal",
) -> str:
    """Call Sarvam AI /v1/translate. Returns original text on any failure."""
    if source_lang == target_lang or not text.strip():
        return text

    api_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not api_key:
        logger.warning("SARVAM_API_KEY not set — skipping translation.")
        return text

    try:
        payload = {
            "input": text,
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "speaker_gender": "Male",
            "mode": mode,
            "model": "mayura:v1",
            "enable_preprocessing": False,
        }
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }
        response = requests.post(
            SARVAM_TRANSLATE_URL,
            json=payload,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        translated = data.get("translated_text", "").strip()
        if translated:
            return translated
        logger.warning("Sarvam translate returned empty text.")
        return text
    except requests.exceptions.Timeout:
        logger.warning("Sarvam translate timed out.")
        return text
    except Exception as exc:
        logger.warning("Sarvam translate failed: %s", exc)
        return text


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Detect language, returns Sarvam code (e.g. 'hi-IN'). Defaults to 'en-IN'."""
    if len(text.strip()) < 15:
        return "en-IN"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        iso = detect(text)
        return _LANGDETECT_TO_SARVAM.get(iso, "en-IN")
    except Exception as exc:
        logger.warning("Language detection failed: %s", exc)
        return "en-IN"


def is_non_english(lang_code: str) -> bool:
    """Return True if the language is a non-English Indian language."""
    return lang_code != "en-IN" and lang_code in SARVAM_LANGUAGES


# ---------------------------------------------------------------------------
# Query Translation (user input -> English for RAG)
# ---------------------------------------------------------------------------

def maybe_translate_query(query: str, ui_lang: str = "en-IN") -> tuple[str, str]:
    """
    Prepare a user query for RAG retrieval.

    1. Uses ui_lang if non-English (user explicitly chose a language).
    2. Otherwise auto-detects from the query text.
    3. If non-English, translates to English for embedding / retrieval.

    Returns:
        (english_query, original_lang_code)
    """
    lang = ui_lang if (ui_lang and ui_lang != "en-IN") else detect_language(query)

    if is_non_english(lang):
        english_query = _sarvam_translate(query, lang, "en-IN", mode="formal")
        logger.info("Query translated [%s -> en-IN]: %.80s", lang, query)
        return english_query, lang

    return query, "en-IN"


# ---------------------------------------------------------------------------
# Response Translation (English answer -> user's language)
# ---------------------------------------------------------------------------

def translate_response(text: str, target_lang: str) -> str:
    """
    Translate an English AI response to the user's chosen language.
    Uses formal mode — Sarvam's Mayura model preserves legal abbreviations.
    """
    if not target_lang or target_lang == "en-IN":
        return text
    return _sarvam_translate(text, "en-IN", target_lang, mode="formal")


# ---------------------------------------------------------------------------
# UI String Translation with Caching
# ---------------------------------------------------------------------------

_ui_translation_cache: dict[tuple[str, str], str] = {}


def translate_ui(text: str, target_lang: str) -> str:
    """Translate a short UI label with in-memory caching."""
    if not target_lang or target_lang == "en-IN":
        return text
    cache_key = (text, target_lang)
    if cache_key in _ui_translation_cache:
        return _ui_translation_cache[cache_key]
    translated = _sarvam_translate(text, "en-IN", target_lang, mode="modern-colloquial")
    _ui_translation_cache[cache_key] = translated
    return translated


def t(text: str, target_lang: str) -> str:
    """Short alias for translate_ui."""
    return translate_ui(text, target_lang)


# ---------------------------------------------------------------------------
# Pre-baked UI Strings (instant load, no API call)
# ---------------------------------------------------------------------------

UI_STRINGS: dict[str, dict[str, str]] = {
    "Startup Profile": {
        "hi-IN": "स्टार्टअप प्रोफाइल", "mr-IN": "स्टार्टअप प्रोफाइल",
        "gu-IN": "સ્ટાર્ટઅપ પ્રોફાઇલ", "ta-IN": "தொடக்க நிறுவன சுயவிவரம்",
        "te-IN": "స్టార్టప్ ప్రొఫైల్", "kn-IN": "ಸ್ಟಾರ್ಟಪ್ ಪ್ರೊಫೈಲ್",
        "ml-IN": "സ്റ്റാർട്ടപ്പ് പ്രൊഫൈൽ", "pa-IN": "ਸਟਾਰਟਅੱਪ ਪ੍ਰੋਫਾਈਲ",
        "bn-IN": "স্টার্টআপ প্রোফাইল", "od-IN": "ଷ୍ଟାର୍ଟଅପ ପ୍ରୋଫାଇଲ",
    },
    "Sector": {
        "hi-IN": "क्षेत्र", "mr-IN": "क्षेत्र", "gu-IN": "ક્ષેત્ર",
        "ta-IN": "துறை", "te-IN": "రంగం", "kn-IN": "ವಲಯ",
        "ml-IN": "മേഖല", "pa-IN": "ਸੈਕਟਰ", "bn-IN": "খাত", "od-IN": "କ୍ଷେତ୍ର",
    },
    "State of Operation": {
        "hi-IN": "संचालन का राज्य", "mr-IN": "कार्यक्षेत्र राज्य",
        "gu-IN": "કાર્યક્ષેત્ર રાજ્ય", "ta-IN": "செயல்பாட்டு மாநிலம்",
        "te-IN": "కార్యాచరణ రాష్ట్రం", "kn-IN": "ಕಾರ್ಯಾಚರಣೆ ರಾಜ್ಯ",
        "ml-IN": "പ്രവർത്തന സംസ്ഥാനം", "pa-IN": "ਕਾਰਜ ਰਾਜ",
        "bn-IN": "কার্যক্ষেত্র রাজ্য", "od-IN": "କାର୍ଯ୍ୟ ରାଜ୍ୟ",
    },
    "Company Size": {
        "hi-IN": "कंपनी का आकार", "mr-IN": "कंपनीचा आकार", "gu-IN": "કંપની નું કદ",
        "ta-IN": "நிறுவன அளவு", "te-IN": "కంపెనీ పరిమాణం", "kn-IN": "ಕಂಪನಿ ಗಾತ್ರ",
        "ml-IN": "കമ്പനി വലിപ്പം", "pa-IN": "ਕੰਪਨੀ ਦਾ ਆਕਾਰ",
        "bn-IN": "কোম্পানির আকার", "od-IN": "କମ୍ପାନି ଆକାର",
    },
    "Describe Your Startup": {
        "hi-IN": "अपना स्टार्टअप बताएं", "mr-IN": "तुमचा स्टार्टअप सांगा",
        "gu-IN": "તમારો સ્ટાર્ટઅપ વર્ણવો", "ta-IN": "உங்கள் தொடக்க நிறுவனம் பற்றி கூறுங்கள்",
        "te-IN": "మీ స్టార్టప్ వివరించండి", "kn-IN": "ನಿಮ್ಮ ಸ್ಟಾರ್ಟಪ್ ವಿವರಿಸಿ",
        "ml-IN": "നിങ്ങളുടെ സ്റ്റാർട്ടപ്പ് വിവരിക്കൂ", "pa-IN": "ਆਪਣਾ ਸਟਾਰਟਅੱਪ ਦੱਸੋ",
        "bn-IN": "আপনার স্টার্টআপ বর্ণনা করুন", "od-IN": "ଆପଣଙ୍କ ଷ୍ଟାର୍ଟଅପ ବର୍ଣ୍ଣନା କରନ୍ତୁ",
    },
    "What does your startup do?": {
        "hi-IN": "आपका स्टार्टअप क्या करता है?", "mr-IN": "तुमचा स्टार्टअप काय करतो?",
        "gu-IN": "તમારો સ્ટાર્ટઅપ શું કરે છે?", "ta-IN": "உங்கள் நிறுவனம் என்ன செய்கிறது?",
        "te-IN": "మీ స్టార్టప్ ఏం చేస్తుంది?", "kn-IN": "ನಿಮ್ಮ ಸ್ಟಾರ್ಟಪ್ ಏನು ಮಾಡುತ್ತದೆ?",
        "ml-IN": "നിങ്ങളുടെ സ്റ്റാർട്ടപ്പ് എന്ത് ചെയ്യുന്നു?", "pa-IN": "ਤੁਹਾਡਾ ਸਟਾਰਟਅੱਪ ਕੀ ਕਰਦਾ ਹੈ?",
        "bn-IN": "আপনার স্টার্টআপ কী করে?", "od-IN": "ଆପଣଙ୍କ ଷ୍ଟାର୍ଟଅପ କ'ଣ କରେ?",
    },
    "🚀 Generate My Checklist": {
        "hi-IN": "🚀 मेरी चेकलिस्ट बनाएं", "mr-IN": "🚀 माझी चेकलिस्ट तयार करा",
        "gu-IN": "🚀 મારી ચેકલિસ્ટ બનાવો", "ta-IN": "🚀 எனது சரிபார்ப்பு பட்டியல் உருவாக்கு",
        "te-IN": "🚀 నా చెక్‌లిస్ట్ రూపొందించు", "kn-IN": "🚀 ನನ್ನ ಚೆಕ್‌ಲಿಸ್ಟ್ ರಚಿಸು",
        "ml-IN": "🚀 എന്റെ ചെക്ക്‌ലിസ്റ്റ് ഉണ്ടാക്കൂ", "pa-IN": "🚀 ਮੇਰੀ ਚੈੱਕਲਿਸਟ ਬਣਾਓ",
        "bn-IN": "🚀 আমার চেকলিস্ট তৈরি করুন", "od-IN": "🚀 ମୋ ଚେକଲିଷ୍ଟ ତିଆରି କରନ୍ତୁ",
    },
    "Ask a Legal Question": {
        "hi-IN": "कानूनी सवाल पूछें", "mr-IN": "कायदेशीर प्रश्न विचारा",
        "gu-IN": "કાનૂની પ્રશ્ન પૂછો", "ta-IN": "சட்டக் கேள்வி கேளுங்கள்",
        "te-IN": "న్యాయపరమైన ప్రశ్న అడగండి", "kn-IN": "ಕಾನೂನು ಪ್ರಶ್ನೆ ಕೇಳಿ",
        "ml-IN": "നിയമ ചോദ്യം ചോദിക്കൂ", "pa-IN": "ਕਾਨੂੰਨੀ ਸਵਾਲ ਪੁੱਛੋ",
        "bn-IN": "আইনি প্রশ্ন জিজ্ঞাসা করুন", "od-IN": "ଆଇନଗତ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ",
    },
    "🔍 Get Answer": {
        "hi-IN": "🔍 उत्तर पाएं", "mr-IN": "🔍 उत्तर मिळवा", "gu-IN": "🔍 જવાબ મેળવો",
        "ta-IN": "🔍 பதில் பெறுங்கள்", "te-IN": "🔍 సమాధానం పొందండి", "kn-IN": "🔍 ಉತ್ತರ ಪಡೆಯಿರಿ",
        "ml-IN": "🔍 ഉത്തരം നേടൂ", "pa-IN": "🔍 ਜਵਾਬ ਪ੍ਰਾਪਤ ਕਰੋ",
        "bn-IN": "🔍 উত্তর পান", "od-IN": "🔍 ଉତ୍ତର ପ୍ରାପ୍ତ କରନ୍ତୁ",
    },
    "🔄 Clear Chat": {
        "hi-IN": "🔄 चैट साफ़ करें", "mr-IN": "🔄 चॅट साफ करा", "gu-IN": "🔄 ચેટ સાફ કરો",
        "ta-IN": "🔄 அரட்டை அழிக்கவும்", "te-IN": "🔄 చాట్ క్లియర్ చేయి", "kn-IN": "🔄 ಚಾಟ್ ತೆರವುಗೊಳಿಸಿ",
        "ml-IN": "🔄 ചാറ്റ് മായ്ക്കൂ", "pa-IN": "🔄 ਚੈਟ ਸਾਫ਼ ਕਰੋ",
        "bn-IN": "🔄 চ্যাট মুছুন", "od-IN": "🔄 ଚାଟ ସଫା କରନ୍ତୁ",
    },
    "Language": {
        "hi-IN": "भाषा", "mr-IN": "भाषा", "gu-IN": "ભાષા", "ta-IN": "மொழி",
        "te-IN": "భాష", "kn-IN": "ಭಾಷೆ", "ml-IN": "ഭാഷ", "pa-IN": "ਭਾਸ਼ਾ",
        "bn-IN": "ভাষা", "od-IN": "ଭାଷା",
    },
    "AI-powered compliance & legal navigator for Indian startups": {
        "hi-IN": "भारतीय स्टार्टअप के लिए AI-संचालित अनुपालन और कानूनी सहायक",
        "mr-IN": "भारतीय स्टार्टअपसाठी AI-आधारित अनुपालन व कायदेशीर सहाय्यक",
        "gu-IN": "ભારતીય સ્ટાર્ટઅપ માટે AI-સંચાલિત કાનૂની સહાયક",
        "ta-IN": "இந்திய தொடக்க நிறுவனங்களுக்கான AI சட்ட வழிகாட்டி",
        "te-IN": "భారతీయ స్టార్టప్‌లకు AI-ఆధారిత న్యాయ మార్గదర్శి",
        "kn-IN": "ಭಾರತೀಯ ಸ್ಟಾರ್ಟಪ್‌ಗಳಿಗಾಗಿ AI ಕಾನೂನು ಸಹಾಯಕ",
        "ml-IN": "ഇന്ത്യൻ സ്റ്റാർട്ടപ്പുകൾക്കുള്ള AI നിയമ സഹായി",
        "pa-IN": "ਭਾਰਤੀ ਸਟਾਰਟਅੱਪਾਂ ਲਈ AI ਕਾਨੂੰਨੀ ਸਹਾਇਕ",
        "bn-IN": "ভারতীয় স্টার্টআপগুলির জন্য AI আইনি সহায়ক",
        "od-IN": "ଭାରତୀୟ ଷ୍ଟାର୍ଟଅପ ପାଇଁ AI ଆଇନ ସହାୟକ",
    },
    "Your question": {
        "hi-IN": "आपका प्रश्न", "mr-IN": "तुमचा प्रश्न", "gu-IN": "તમારો પ્રશ્ન",
        "ta-IN": "உங்கள் கேள்வி", "te-IN": "మీ ప్రశ్న", "kn-IN": "ನಿಮ್ಮ ಪ್ರಶ್ನೆ",
        "ml-IN": "നിങ്ങളുടെ ചോദ്യം", "pa-IN": "ਤੁਹਾਡਾ ਸਵਾਲ",
        "bn-IN": "আপনার প্রশ্ন", "od-IN": "ଆପଣଙ୍କ ପ୍ରଶ୍ନ",
    },
    "Conversation History": {
        "hi-IN": "बातचीत का इतिहास", "mr-IN": "संभाषण इतिहास",
        "gu-IN": "વાર્તાલાપ ઇતિહાસ", "ta-IN": "உரையாடல் வரலாறு",
        "te-IN": "సంభాషణ చరిత్ర", "kn-IN": "ಸಂಭಾಷಣೆ ಇತಿಹಾಸ",
        "ml-IN": "സംഭാഷണ ചരിത്രം", "pa-IN": "ਗੱਲਬਾਤ ਦਾ ਇਤਿਹਾਸ",
        "bn-IN": "কথোপকথনের ইতিহাস", "od-IN": "କଥୋପକଥନ ଇତିହାସ",
    },
    "Government Schemes for Startups": {
        "hi-IN": "स्टार्टअप के लिए सरकारी योजनाएं", "mr-IN": "स्टार्टअपसाठी सरकारी योजना",
        "gu-IN": "સ્ટાર્ટઅપ માટે સરકારી યોજનાઓ", "ta-IN": "தொடக்க நிறுவனங்களுக்கான அரசு திட்டங்கள்",
        "te-IN": "స్టార్టప్‌లకు ప్రభుత్వ పథకాలు", "kn-IN": "ಸ್ಟಾರ್ಟಪ್‌ಗಳಿಗೆ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
        "ml-IN": "സ്റ്റാർട്ടപ്പുകൾക്കുള്ള സർക്കാർ പദ്ധതികൾ", "pa-IN": "ਸਟਾਰਟਅੱਪਾਂ ਲਈ ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ",
        "bn-IN": "স্টার্টআপের জন্য সরকারি প্রকল্প", "od-IN": "ଷ୍ଟାର୍ଟଅପ ପାଇଁ ସରକାରୀ ଯୋଜନା",
    },
    "Incubation Opportunities": {
        "hi-IN": "ऊष्मायन अवसर", "mr-IN": "इन्क्युबेशन संधी",
        "gu-IN": "ઇન્ક્યુબેશન તકો", "ta-IN": "ஊக்குவிப்பு வாய்ப்புகள்",
        "te-IN": "ఇంక్యుబేషన్ అవకాశాలు", "kn-IN": "ಇನ್ಕ್ಯುಬೇಶನ್ ಅವಕಾಶಗಳು",
        "ml-IN": "ഇൻക്യുബേഷൻ അവസരങ്ങൾ", "pa-IN": "ਇਨਕਿਊਬੇਸ਼ਨ ਮੌਕੇ",
        "bn-IN": "ইনকিউবেশন সুযোগ", "od-IN": "ଇନ୍‌କ୍ୟୁବେସନ ସୁଯୋଗ",
    },
    "📋 Compliance Checklist": {
        "hi-IN": "📋 अनुपालन चेकलिस्ट", "mr-IN": "📋 अनुपालन चेकलिस्ट",
        "gu-IN": "📋 અનુપાલન ચેકલિસ્ટ", "ta-IN": "📋 இணக்க சரிபார்ப்பு பட்டியல்",
        "te-IN": "📋 కమ్ప్లయన్స్ చెక్‌లిస్ట్", "kn-IN": "📋 ಅನುಪಾಲನೆ ಪರಿಶೀಲನಾ ಪಟ್ಟಿ",
        "ml-IN": "📋 കംപ്ലയൻസ് ചെക്ക്‌ലിസ്റ്റ്", "pa-IN": "📋 ਪਾਲਣਾ ਚੈੱਕਲਿਸਟ",
        "bn-IN": "📋 কমপ্লায়েন্স চেকলিস্ট", "od-IN": "📋 ଅନୁପାଳନ ଚେକଲିଷ୍ଟ",
    },
    "💬 Legal Q&A": {
        "hi-IN": "💬 कानूनी प्रश्नोत्तर", "mr-IN": "💬 कायदेशीर प्रश्नोत्तर",
        "gu-IN": "💬 કાનૂની પ્રશ્ન-ઉત્તર", "ta-IN": "💬 சட்ட கேள்வி-பதில்",
        "te-IN": "💬 న్యాయ ప్రశ్నోత్తరాలు", "kn-IN": "💬 ಕಾನೂನು ಪ್ರಶ್ನೋತ್ತರ",
        "ml-IN": "💬 നിയമ ചോദ്യോത്തരം", "pa-IN": "💬 ਕਾਨੂੰਨੀ ਸਵਾਲ-ਜਵਾਬ",
        "bn-IN": "💬 আইনি প্রশ্নোত্তর", "od-IN": "💬 ଆଇନ ପ୍ରଶ୍ନୋତ୍ତର",
    },
    "🏛️ Government Schemes": {
        "hi-IN": "🏛️ सरकारी योजनाएं", "mr-IN": "🏛️ सरकारी योजना",
        "gu-IN": "🏛️ સરકારી યોજનાઓ", "ta-IN": "🏛️ அரசு திட்டங்கள்",
        "te-IN": "🏛️ ప్రభుత్వ పథకాలు", "kn-IN": "🏛️ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
        "ml-IN": "🏛️ സർക്കാർ പദ്ധതികൾ", "pa-IN": "🏛️ ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ",
        "bn-IN": "🏛️ সরকারি প্রকল্প", "od-IN": "🏛️ ସରକାରୀ ଯୋଜନା",
    },
    "🚀 Opportunities": {
        "hi-IN": "🚀 अवसर", "mr-IN": "🚀 संधी", "gu-IN": "🚀 તકો",
        "ta-IN": "🚀 வாய்ப்புகள்", "te-IN": "🚀 అవకాశాలు", "kn-IN": "🚀 ಅವಕಾಶಗಳು",
        "ml-IN": "🚀 അവസരങ്ങൾ", "pa-IN": "🚀 ਮੌਕੇ",
        "bn-IN": "🚀 সুযোগ", "od-IN": "🚀 ସୁଯୋଗ",
    },
    "Analysing and retrieving...": {
        "hi-IN": "विश्लेषण और खोज हो रहा है...", "mr-IN": "विश्लेषण आणि शोध सुरू आहे...",
        "gu-IN": "વિશ્લેષણ અને શોધ ચાલી રહ્યો છે...", "ta-IN": "பகுப்பாய்வு மற்றும் தேடல்...",
        "te-IN": "విశ్లేషణ మరియు శోధన జరుగుతోంది...", "kn-IN": "ವಿಶ್ಲೇಷಣೆ ಮತ್ತು ಹುಡುಕಾಟ...",
        "ml-IN": "വിശകലനം ചെയ്യുന്നു...", "pa-IN": "ਵਿਸ਼ਲੇਸ਼ਣ ਅਤੇ ਖੋਜ...",
        "bn-IN": "বিশ্লেষণ ও অনুসন্ধান চলছে...", "od-IN": "ବିଶ୍ଳେଷଣ ଓ ଖୋଜ ଜାରି...",
    },
    "Translating response...": {
        "hi-IN": "उत्तर का अनुवाद हो रहा है...", "mr-IN": "उत्तराचे भाषांतर होत आहे...",
        "gu-IN": "જવાબ અનુવાદ થઈ રહ્યો છે...", "ta-IN": "பதில் மொழிபெயர்க்கப்படுகிறது...",
        "te-IN": "సమాధానం అనువదించబడుతోంది...", "kn-IN": "ಉತ್ತರ ಅನುವಾದಿಸಲಾಗುತ್ತಿದೆ...",
        "ml-IN": "ഉത്തരം വിവർത്തനം ചെയ്യുന്നു...", "pa-IN": "ਜਵਾਬ ਦਾ ਅਨੁਵਾਦ ਹੋ ਰਿਹਾ ਹੈ...",
        "bn-IN": "উত্তর অনুবাদ হচ্ছে...", "od-IN": "ଉତ୍ତର ଅନୁବାଦ ହେଉଛି...",
    },
}


def ts(key: str, target_lang: str) -> str:
    """
    Translate a UI string using the pre-baked dictionary first,
    falling back to Sarvam API if not found.
    """
    if not target_lang or target_lang == "en-IN":
        return key
    lang_dict = UI_STRINGS.get(key, {})
    if target_lang in lang_dict:
        return lang_dict[target_lang]
    return translate_ui(key, target_lang)
