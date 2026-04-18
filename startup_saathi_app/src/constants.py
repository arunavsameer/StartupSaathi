"""
StartupSaathi · src/constants.py
All static data — no Streamlit, no business-logic imports.
Safe to import anywhere (UI, notebooks, tests).
"""
from __future__ import annotations

# ── Company size options ──────────────────────────────────────────────────────
SIZE_OPTIONS: dict[str, str] = {
    "Solo / Pre-incorporation": "all",
    "Micro (< 10 employees)":   "micro",
    "Small (10–50 employees)":  "small",
    "Medium (50–250 employees)":"medium",
}

# ── All Indian states + UTs (Opportunities tab state picker) ─────────────────
INDIAN_STATES: list[str] = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka",
    "Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram",
    "Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana",
    "Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
    "Andaman & Nicobar Islands","Chandigarh","Dadra & Nagar Haveli",
    "Daman & Diu","Delhi","Jammu and Kashmir","Ladakh","Lakshadweep",
    "Puducherry",
]

# ── Curated fallback schemes (displayed when Delta data is placeholder) ───────
FALLBACK_SCHEMES: list[dict] = [
    {
        "name": "Startup India Initiative (DPIIT Recognition)",
        "description": "DPIIT-recognised startups get tax exemptions (80-IAC), self-certification under labour & environment laws, fast-track patent examination at 80% fee rebate, and access to Fund of Funds.",
        "eligibility": "Indian private limited company, partnership, or LLP incorporated ≤10 years ago with annual turnover <₹100 crore, working on an innovative product/service.",
        "benefits": "Tax holiday for 3 years, ₹10 lakh patent support, relaxed public procurement norms, access to ₹10,000 crore Fund of Funds.",
        "link": "https://www.startupindia.gov.in/content/sih/en/recognition.html",
        "sectors": ["all"], "sizes": ["all"],
    },
    {
        "name": "PM Mudra Yojana (PMMY)",
        "description": "Provides collateral-free loans to micro and small enterprises including startups in manufacturing, trading, and services.",
        "eligibility": "Non-corporate, non-farm small/micro enterprises. No collateral required.",
        "benefits": "Shishu: up to ₹50,000 | Kishore: ₹50,001–₹5 lakh | Tarun: ₹5 lakh–₹10 lakh.",
        "link": "https://www.mudra.org.in/",
        "sectors": ["all", "FoodTech"], "sizes": ["all", "micro", "small"],
    },
    {
        "name": "Atal Innovation Mission (AIM) — Atal Incubation Centres",
        "description": "NITI Aayog's flagship programme to set up world-class incubators and support deep-tech startups across India.",
        "eligibility": "Startups in deep-tech, agri-tech, health-tech, fintech, and other innovation sectors.",
        "benefits": "Up to ₹10 crore grant for setting up Atal Incubation Centres.",
        "link": "https://aim.gov.in/",
        "sectors": ["all", "DeepTech", "AgriTech", "HealthTech", "FinTech"], "sizes": ["all"],
    },
    {
        "name": "SIDBI Fund of Funds for Startups (FFS)",
        "description": "₹10,000 crore corpus managed by SIDBI to invest in SEBI-registered AIFs which in turn invest in DPIIT-recognised startups.",
        "eligibility": "DPIIT-recognised startups. Investment is indirect — through AIFs selected by SIDBI.",
        "benefits": "Equity/quasi-equity funding of ₹25 lakh to ₹10 crore via AIFs.",
        "link": "https://www.sidbi.in/en/startups",
        "sectors": ["all"], "sizes": ["all"],
    },
    {
        "name": "Credit Guarantee Scheme for Startups (CGSS)",
        "description": "Provides credit guarantees to loans extended by scheduled commercial banks, NBFCs, and VCFs to DPIIT-recognised startups.",
        "eligibility": "DPIIT-recognised startups borrowing from eligible lenders.",
        "benefits": "Guarantee cover up to ₹10 crore per startup.",
        "link": "https://www.startupindia.gov.in/",
        "sectors": ["all"], "sizes": ["all"],
    },
    {
        "name": "FSSAI Food Business Operator (FBO) Support",
        "description": "Central licensing and registration support for food businesses including startups in food processing, packaged foods, and food tech.",
        "eligibility": "Any startup involved in manufacturing, processing, packaging, or distributing food products.",
        "benefits": "Simplified online registration, state-level compliance support, access to FSSAI labs.",
        "link": "https://foscos.fssai.gov.in/",
        "sectors": ["FoodTech"], "sizes": ["all"],
    },
    {
        "name": "Software Technology Parks of India (STPI)",
        "description": "Provides infrastructure, single-window clearance, and 100% export-oriented unit benefits for IT/software startups.",
        "eligibility": "IT/ITeS/software companies with export orientation.",
        "benefits": "Tax exemptions (10A/10B), import duty waiver on equipment, high-speed internet.",
        "link": "https://www.stpi.in/",
        "sectors": ["IT/SaaS", "DeepTech"], "sizes": ["all"],
    },
    {
        "name": "RBI Regulatory Sandbox for Fintech",
        "description": "Allows fintech startups to test innovative products under a relaxed regulatory environment with real customers.",
        "eligibility": "Fintech startups with innovative financial products/services. Min net worth ₹25 lakh.",
        "benefits": "Test live without full regulatory burden; direct RBI engagement.",
        "link": "https://rbi.org.in/Scripts/bs_viewcontent.aspx?Id=3747",
        "sectors": ["FinTech"], "sizes": ["all", "micro", "small"],
    },
    {
        "name": "GeM (Government e-Marketplace) for Startups",
        "description": "DPIIT-recognised startups can sell products and services directly to government buyers without tendering.",
        "eligibility": "DPIIT-recognised startups. Register on GeM portal as a seller.",
        "benefits": "Access to ₹2+ lakh crore government procurement market.",
        "link": "https://gem.gov.in/",
        "sectors": ["all"], "sizes": ["all"],
    },
    {
        "name": "Production Linked Incentive (PLI) Scheme",
        "description": "Incentivises domestic manufacturing in 14 key sectors including electronics, pharma, textiles, food processing.",
        "eligibility": "Startups and companies in eligible sectors meeting minimum investment thresholds.",
        "benefits": "4%–20% incentive on incremental sales over a 4–6 year period.",
        "link": "https://www.investindia.gov.in/pli",
        "sectors": ["Climate/Energy", "FoodTech", "HealthTech"], "sizes": ["small", "medium"],
    },
    {
        "name": "MSME Samadhaan — Delayed Payment Portal",
        "description": "Online portal for MSME startups to file cases against buyers who delay payment beyond the agreed period.",
        "eligibility": "Micro and small enterprises registered under MSME Act.",
        "benefits": "Statutory right to 45-day payment, interest at 3× bank rate on delayed payments.",
        "link": "https://samadhaan.msme.gov.in/",
        "sectors": ["all"], "sizes": ["all", "micro", "small"],
    },
    {
        "name": "National SC-ST Hub (NSSH)",
        "description": "Dedicated support for SC/ST entrepreneurs to participate in public procurement.",
        "eligibility": "Startups and MSMEs owned/managed by SC/ST entrepreneurs.",
        "benefits": "Mentorship, incubation support, preferential procurement from CPSEs.",
        "link": "https://www.scsthub.in/",
        "sectors": ["all"], "sizes": ["all", "micro", "small"],
    },
]

# ── City → (lat, lon) ─────────────────────────────────────────────────────────
CITY_COORDS: dict[str, tuple[float, float]] = {
    "hyderabad": (17.3850, 78.4867), "itanagar": (27.0844, 93.6053),
    "dispur": (26.1433, 91.7898), "patna": (25.5941, 85.1376),
    "raipur": (21.2514, 81.6296), "panaji": (15.4909, 73.8278),
    "gandhinagar": (23.2156, 72.6369), "chandigarh": (30.7333, 76.7794),
    "shimla": (31.1048, 77.1734), "ranchi": (23.3441, 85.3096),
    "bengaluru": (12.9716, 77.5946), "bangalore": (12.9716, 77.5946),
    "thiruvananthapuram": (8.5241, 76.9366), "bhopal": (23.2599, 77.4126),
    "mumbai": (19.0760, 72.8777), "imphal": (24.8170, 93.9368),
    "shillong": (25.5788, 91.8933), "aizawl": (23.7307, 92.7173),
    "kohima": (25.6751, 94.1086), "bhubaneswar": (20.2961, 85.8245),
    "amritsar": (31.6340, 74.8723), "jaipur": (26.9124, 75.7873),
    "gangtok": (27.3314, 88.6138), "chennai": (13.0827, 80.2707),
    "agartala": (23.8315, 91.2868), "lucknow": (26.8467, 80.9462),
    "dehradun": (30.3165, 78.0322), "kolkata": (22.5726, 88.3639),
    "new delhi": (28.6139, 77.2090), "delhi": (28.6139, 77.2090),
    "pune": (18.5204, 73.8567), "ahmedabad": (23.0225, 72.5714),
    "surat": (21.1702, 72.8311), "nagpur": (21.1458, 79.0882),
    "indore": (22.7196, 75.8577), "coimbatore": (11.0168, 76.9558),
    "kochi": (9.9312, 76.2673), "visakhapatnam": (17.6868, 83.2185),
    "gurgaon": (28.4595, 77.0266), "gurugram": (28.4595, 77.0266),
    "noida": (28.5355, 77.3910), "mysuru": (12.2958, 76.6394),
    "goa": (15.2993, 74.1240), "jammu": (32.7266, 74.8570),
    "varanasi": (25.3176, 82.9739), "agra": (27.1767, 78.0081),
    "kanpur": (26.4499, 80.3319), "guwahati": (26.1445, 91.7362),
    "thane": (19.2183, 72.9781), "aurangabad": (19.8762, 75.3433),
    "nashik": (19.9975, 73.7898), "vijayawada": (16.5062, 80.6480),
    "madurai": (9.9252, 78.1198), "puducherry": (11.9416, 79.8083),
}

# ── State → (lat, lon) ────────────────────────────────────────────────────────
STATE_COORDS: dict[str, tuple[float, float]] = {
    "andhra pradesh": (17.3850, 78.4867), "assam": (26.1433, 91.7898),
    "bihar": (25.5941, 85.1376), "chhattisgarh": (21.2514, 81.6296),
    "goa": (15.2993, 74.1240), "gujarat": (23.2156, 72.6369),
    "haryana": (30.7333, 76.7794), "himachal pradesh": (31.1048, 77.1734),
    "jharkhand": (23.3441, 85.3096), "karnataka": (12.9716, 77.5946),
    "kerala": (8.5241, 76.9366), "madhya pradesh": (23.2599, 77.4126),
    "maharashtra": (19.0760, 72.8777), "odisha": (20.2961, 85.8245),
    "punjab": (30.9010, 75.8573), "rajasthan": (26.9124, 75.7873),
    "tamil nadu": (13.0827, 80.2707), "telangana": (17.3850, 78.4867),
    "uttar pradesh": (26.8467, 80.9462), "uttarakhand": (30.3165, 78.0322),
    "west bengal": (22.5726, 88.3639), "delhi": (28.6139, 77.2090),
    "delhi": (28.6139, 77.2090), "jammu and kashmir": (33.7782, 76.5762),
    "chandigarh": (30.7333, 76.7794), "puducherry": (11.9416, 79.8083),
}
