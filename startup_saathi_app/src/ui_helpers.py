# """
# StartupSaathi · src/ui_helpers.py

# CSS design system, HTML badge generators, geographic coordinate lookups,
# and pure data-transformation helpers.

# No Streamlit imports — this module has zero side-effects and can be imported
# freely in notebooks, unit tests, or the UI layer.
# """
# from __future__ import annotations
# import re
# import uuid

# # ─────────────────────────────────────────────────────────────────────────────
# # Minimalist Professional CSS Design System
# # ─────────────────────────────────────────────────────────────────────────────
# PREMIUM_CSS = """
# <style>
# /* ── Google Fonts ──────────────────────────────────────────────────────────── */
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

# /* ── CSS Custom Properties (Clean SaaS Palette) ────────────────────────────── */
# :root {
#   --brand-primary:   #2563EB; /* Corporate Blue */
#   --brand-light:     #EFF6FF;
#   --brand-hover:     #1D4ED8;
  
#   --text-main:       #111827;
#   --text-secondary:  #4B5563;
#   --text-tertiary:   #9CA3AF;
  
#   --bg-app:          #F9FAFB;
#   --bg-card:         #FFFFFF;
#   --bg-sidebar:      #FFFFFF;
  
#   --border-subtle:   #F3F4F6;
#   --border-main:     #E5E7EB;
#   --border-strong:   #D1D5DB;
  
#   --success:         #059669;
#   --success-bg:      #ECFDF5;
#   --warning:         #D97706;
#   --warning-bg:      #FFFBEB;
  
#   --radius-sm:       4px;
#   --radius-md:       8px;
#   --radius-lg:       12px;
  
#   --shadow-sm:       0 1px 2px 0 rgba(0, 0, 0, 0.05);
#   --shadow-md:       0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
# }

# /* ── Global typography & layout ─────────────────────────────────────────────── */
# html, body, [class*="css"], .stApp {
#     font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
#     color: var(--text-main);
# }

# .stApp {
#     background: var(--bg-app) !important;
# }

# .main .block-container {
#     padding-top: 2rem !important;
#     padding-bottom: 3rem !important;
#     max-width: 1100px !important;
# }

# /* ── Hero Header (Clean Banner) ─────────────────────────────────────────────── */
# .ss-hero {
#     background: var(--bg-card);
#     border: 1px solid var(--border-main);
#     border-radius: var(--radius-lg);
#     padding: 2rem 2.5rem;
#     margin-bottom: 2rem;
#     box-shadow: var(--shadow-sm);
#     display: flex;
#     align-items: center;
#     gap: 1.5rem;
# }
# .ss-hero-icon {
#     font-size: 2.5rem;
#     background: var(--brand-light);
#     padding: 1rem;
#     border-radius: var(--radius-md);
# }
# .ss-hero-text h1 {
#     margin: 0 0 0.25rem;
#     font-size: 1.75rem;
#     font-weight: 700;
#     color: var(--text-main);
#     letter-spacing: -0.02em;
# }
# .ss-hero-text p {
#     margin: 0;
#     font-size: 0.95rem;
#     color: var(--text-secondary);
# }
# .ss-tech-pills {
#     display: flex;
#     gap: 0.5rem;
#     flex-wrap: wrap;
#     margin-top: 1rem;
# }
# .ss-tech-pill {
#     background: var(--bg-app);
#     border: 1px solid var(--border-main);
#     border-radius: var(--radius-sm);
#     padding: 2px 8px;
#     font-size: 0.75rem;
#     color: var(--text-secondary);
#     font-weight: 500;
# }

# /* ── Sidebar ─────────────────────────────────────────────────────────────────── */
# [data-testid="stSidebar"] {
#     background: var(--bg-sidebar) !important;
#     border-right: 1px solid var(--border-main) !important;
# }
# .ss-sidebar-brand {
#     display: flex;
#     align-items: center;
#     gap: 0.75rem;
#     padding-bottom: 1.5rem;
#     margin-bottom: 1.5rem;
#     border-bottom: 1px solid var(--border-subtle);
# }
# .ss-sidebar-brand-name {
#     font-size: 1.1rem;
#     font-weight: 700;
#     color: var(--text-main);
# }
# .ss-sidebar-brand-tagline {
#     font-size: 0.75rem;
#     color: var(--text-tertiary);
# }
# .ss-section-label {
#     font-size: 0.75rem;
#     font-weight: 600;
#     color: var(--text-secondary);
#     text-transform: uppercase;
#     letter-spacing: 0.05em;
#     margin: 1.5rem 0 0.75rem;
# }

# /* ── Native UI Overrides ─────────────────────────────────────────────────────── */
# .stButton > button[kind="primary"] {
#     background: var(--brand-primary) !important;
#     color: white !important;
#     border: none !important;
#     border-radius: var(--radius-md) !important;
#     font-weight: 500 !important;
#     box-shadow: var(--shadow-sm) !important;
# }
# .stButton > button[kind="primary"]:hover {
#     background: var(--brand-hover) !important;
# }
# .stButton > button[kind="secondary"],
# .stButton > button:not([kind]) {
#     background: var(--bg-card) !important;
#     border: 1px solid var(--border-strong) !important;
#     border-radius: var(--radius-md) !important;
#     color: var(--text-main) !important;
#     font-weight: 500 !important;
#     box-shadow: var(--shadow-sm) !important;
# }
# .stTextArea textarea,
# .stTextInput input,
# .stSelectbox [data-baseweb="select"] > div:first-child {
#     border: 1px solid var(--border-strong) !important;
#     border-radius: var(--radius-md) !important;
#     background: var(--bg-card) !important;
# }
# .stTextArea textarea:focus,
# .stTextInput input:focus {
#     border-color: var(--brand-primary) !important;
#     box-shadow: 0 0 0 1px var(--brand-primary) !important;
# }

# /* ── Progress Banner ─────────────────────────────────────────────────────────── */
# .ss-progress-banner {
#     background: var(--bg-card);
#     border: 1px solid var(--border-main);
#     border-radius: var(--radius-lg);
#     padding: 1.5rem;
#     margin-bottom: 1.5rem;
#     display: flex;
#     align-items: center;
#     gap: 2rem;
#     box-shadow: var(--shadow-sm);
# }
# .ss-progress-label {
#     font-size: 0.75rem;
#     font-weight: 600;
#     color: var(--text-tertiary);
#     text-transform: uppercase;
#     margin-bottom: 0.25rem;
# }
# .ss-progress-val {
#     font-size: 1.5rem;
#     font-weight: 700;
#     color: var(--text-main);
# }
# .ss-progress-track {
#     flex: 1;
#     height: 8px;
#     background: var(--border-subtle);
#     border-radius: var(--radius-sm);
#     overflow: hidden;
# }
# .ss-progress-fill {
#     height: 100%;
#     background: var(--brand-primary);
#     border-radius: var(--radius-sm);
#     transition: width 0.3s ease;
# }

# /* ── Section Headers ─────────────────────────────────────────────────────────── */
# .ss-section-hdr {
#     display: flex;
#     align-items: center;
#     gap: 0.75rem;
#     padding: 1rem 0;
#     border-bottom: 1px solid var(--border-main);
#     margin: 1.5rem 0 1rem;
# }
# .ss-section-hdr-title {
#     font-size: 1rem;
#     font-weight: 600;
#     color: var(--text-main);
#     flex: 1;
# }
# .ss-section-hdr-count {
#     font-size: 0.8rem;
#     background: var(--border-subtle);
#     color: var(--text-secondary);
#     padding: 2px 8px;
#     border-radius: var(--radius-sm);
#     font-weight: 500;
# }

# /* ── Checklist Items ─────────────────────────────────────────────────────────── */
# .ss-item-title {
#     font-size: 0.95rem;
#     font-weight: 500;
#     color: var(--text-main);
#     margin-bottom: 0.25rem;
# }
# .ss-item-title.done {
#     text-decoration: line-through;
#     color: var(--text-tertiary);
# }
# .ss-item-desc {
#     font-size: 0.85rem;
#     color: var(--text-secondary);
#     margin-bottom: 0.5rem;
# }
# .ss-item-footer {
#     display: flex;
#     align-items: center;
#     gap: 0.5rem;
#     font-size: 0.8rem;
#     color: var(--text-tertiary);
# }
# .ss-portal-link {
#     color: var(--brand-primary);
#     text-decoration: none;
#     font-weight: 500;
# }
# .ss-portal-link:hover {
#     text-decoration: underline;
# }

# /* ── Badges & Pills ──────────────────────────────────────────────────────────── */
# .ss-pill {
#     display: inline-flex;
#     align-items: center;
#     padding: 2px 8px;
#     border-radius: var(--radius-sm);
#     font-size: 0.75rem;
#     font-weight: 500;
# }
# .ss-pill-green { background: var(--success-bg); color: var(--success); }
# .ss-pill-amber { background: var(--warning-bg); color: var(--warning); }
# .ss-pill-slate { background: var(--border-subtle); color: var(--text-secondary); }
# .ss-pill-match-hi { background: var(--success-bg); color: var(--success); }
# .ss-pill-match-mid { background: var(--warning-bg); color: var(--warning); }
# .ss-pill-match-lo { background: var(--bg-app); color: var(--text-secondary); }
# .ss-query-type {
#     background: var(--brand-light);
#     color: var(--brand-primary);
#     padding: 2px 8px;
#     border-radius: var(--radius-sm);
#     font-size: 0.75rem;
#     font-weight: 500;
# }

# /* ── Q&A Answer Box ──────────────────────────────────────────────────────────── */
# .ss-answer {
#     background: var(--bg-card);
#     border: 1px solid var(--border-main);
#     border-left: 4px solid var(--brand-primary);
#     border-radius: var(--radius-md);
#     padding: 1.25rem;
#     margin: 1rem 0;
#     color: var(--text-main);
#     font-size: 0.95rem;
#     line-height: 1.6;
#     box-shadow: var(--shadow-sm);
# }
# .ss-source {
#     background: var(--border-subtle);
#     color: var(--text-secondary);
#     padding: 2px 8px;
#     border-radius: var(--radius-sm);
#     font-size: 0.75rem;
#     margin-right: 0.25rem;
# }

# /* ── Cards (Schemes & Opportunities) ─────────────────────────────────────────── */
# .ss-scheme, .ss-opp {
#     background: var(--bg-card);
#     border: 1px solid var(--border-main);
#     border-radius: var(--radius-md);
#     padding: 1.5rem;
#     margin-bottom: 1rem;
#     box-shadow: var(--shadow-sm);
#     transition: border-color 0.2s;
# }
# .ss-scheme:hover, .ss-opp:hover {
#     border-color: var(--border-strong);
# }
# .ss-scheme-name, .ss-opp-name {
#     font-size: 1rem;
#     font-weight: 600;
#     color: var(--text-main);
#     margin-bottom: 0.75rem;
# }
# .ss-scheme-row, .ss-opp-desc {
#     font-size: 0.85rem;
#     color: var(--text-secondary);
#     margin-bottom: 0.5rem;
#     line-height: 1.5;
# }
# .ss-scheme-row strong {
#     color: var(--text-main);
#     font-weight: 500;
# }
# .ss-scheme-link {
#     display: inline-block;
#     margin-top: 1rem;
#     font-size: 0.85rem;
#     color: var(--brand-primary);
#     text-decoration: none;
#     font-weight: 500;
# }
# .ss-scheme-link:hover {
#     text-decoration: underline;
# }
# .ss-opp-loc {
#     font-size: 0.8rem;
#     color: var(--text-tertiary);
#     margin-bottom: 0.75rem;
# }

# /* ── Empty State ─────────────────────────────────────────────────────────────── */
# .ss-empty {
#     text-align: center;
#     padding: 4rem 2rem;
#     background: var(--bg-card);
#     border: 1px dashed var(--border-strong);
#     border-radius: var(--radius-lg);
# }
# .ss-empty-icon {
#     font-size: 2.5rem;
#     color: var(--text-tertiary);
#     margin-bottom: 1rem;
#     display: block;
# }
# .ss-empty h3 {
#     font-size: 1.1rem;
#     font-weight: 600;
#     color: var(--text-main);
#     margin-bottom: 0.5rem;
# }
# .ss-empty p {
#     font-size: 0.9rem;
#     color: var(--text-secondary);
# }
# </style>
# """

# # ─────────────────────────────────────────────────────────────────────────────
# # Session state defaults
# # ─────────────────────────────────────────────────────────────────────────────
# def get_session_defaults(default_sector: str, default_state_key: str = "maharashtra") -> dict:
#     return {
#         "session_id":       str(uuid.uuid4()),
#         "embed_model":      None,
#         "faiss_index":      None,
#         "faiss_metadata":   None,
#         "faiss_loaded":     False,
#         "faiss_error":      None,
#         "llm_model":        None,
#         "llm_tokenizer":    None,
#         "llm_name":         None,
#         "llm_loaded":       False,
#         "nsws_checklist":   [],
#         "nsws_sector":      default_sector,
#         "nsws_state_key":   default_state_key,
#         "completed_tasks":  set(),
#         "checklist_ready":  False,
#         "sector":           "all",
#         "size":             "all",
#         "location":         "Maharashtra",
#         "startup_desc":     "",
#         "chat_history":     [],
#     }

# # ─────────────────────────────────────────────────────────────────────────────
# # Badge HTML generators
# # ─────────────────────────────────────────────────────────────────────────────
# def fee_badge_html(fee_display: str) -> str:
#     if fee_display.strip().lower() in ("free", "nil", "₹0", "0"):
#         return f'<span class="ss-pill ss-pill-green">Free</span>'
#     return f'<span class="ss-pill ss-pill-amber">₹ {fee_display}</span>'

# def score_badge_html(score: float) -> str:
#     pct = min(100, int(score * 100))
#     if pct >= 80:
#         return f'<span class="ss-pill ss-pill-match-hi">{pct}% match</span>'
#     elif pct >= 60:
#         return f'<span class="ss-pill ss-pill-match-mid">{pct}% match</span>'
#     return f'<span class="ss-pill ss-pill-match-lo">{pct}% match</span>'

# # ─────────────────────────────────────────────────────────────────────────────
# # Geographic coordinate helpers
# # ─────────────────────────────────────────────────────────────────────────────
# def get_coords(
#     city: str,
#     state: str,
#     city_coords: dict[str, tuple[float, float]],
#     state_coords: dict[str, tuple[float, float]],
# ) -> tuple[float, float] | None:
#     if city:
#         coords = city_coords.get(city.lower().strip())
#         if coords:
#             return coords
#     if state:
#         coords = state_coords.get(state.lower().strip())
#         if coords:
#             return coords
#     return None

# # ─────────────────────────────────────────────────────────────────────────────
# # Schemes data helpers
# # ─────────────────────────────────────────────────────────────────────────────
# def is_placeholder_data(rows: list) -> bool:
#     if not rows:
#         return True
#     sample = (rows[0].get("chunk_text") or "").strip()
#     return bool(re.match(r"^Scheme:\s+Scheme\s+\d+", sample))

# def filter_schemes_by_profile(
#     schemes: list[dict],
#     sector: str,
#     size: str,
# ) -> list[dict]:
#     result: list[tuple[int, dict]] = []
#     for scheme in schemes:
#         s_match = "all" in scheme.get("sectors", ["all"]) or sector in scheme.get("sectors", ["all"])
#         z_match = "all" in scheme.get("sizes", ["all"]) or size in scheme.get("sizes", ["all"])
#         if s_match and z_match:
#             result.append((2, scheme))
#         elif s_match or z_match:
#             result.append((1, scheme))
#     result.sort(key=lambda x: x[0], reverse=True)
#     return [s for _, s in result]