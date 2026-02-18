"""
Stock Registry — NSE symbol lookup, fuzzy search, sector and index groupings.
No network calls. Pure data.
"""

REGISTRY = [
    # ── Nifty 50 ──────────────────────────────────────────────────────────
    {"symbol": "RELIANCE",    "name": "Reliance Industries",             "sector": "Energy",      "indices": ["Nifty 50"]},
    {"symbol": "TCS",         "name": "Tata Consultancy Services",       "sector": "IT",          "indices": ["Nifty 50", "Nifty IT"]},
    {"symbol": "HDFCBANK",    "name": "HDFC Bank",                       "sector": "Banking",     "indices": ["Nifty 50", "Nifty Bank"]},
    {"symbol": "INFY",        "name": "Infosys",                         "sector": "IT",          "indices": ["Nifty 50", "Nifty IT"]},
    {"symbol": "ICICIBANK",   "name": "ICICI Bank",                      "sector": "Banking",     "indices": ["Nifty 50", "Nifty Bank"]},
    {"symbol": "HINDUNILVR",  "name": "Hindustan Unilever",              "sector": "FMCG",        "indices": ["Nifty 50", "Nifty FMCG"]},
    {"symbol": "ITC",         "name": "ITC Limited",                     "sector": "FMCG",        "indices": ["Nifty 50", "Nifty FMCG"]},
    {"symbol": "SBIN",        "name": "State Bank of India",             "sector": "Banking",     "indices": ["Nifty 50", "Nifty Bank"]},
    {"symbol": "BHARTIARTL",  "name": "Bharti Airtel",                   "sector": "Telecom",     "indices": ["Nifty 50"]},
    {"symbol": "KOTAKBANK",   "name": "Kotak Mahindra Bank",             "sector": "Banking",     "indices": ["Nifty 50", "Nifty Bank"]},
    {"symbol": "LT",          "name": "Larsen & Toubro",                 "sector": "Infra",       "indices": ["Nifty 50"]},
    {"symbol": "AXISBANK",    "name": "Axis Bank",                       "sector": "Banking",     "indices": ["Nifty 50", "Nifty Bank"]},
    {"symbol": "ASIANPAINT",  "name": "Asian Paints",                    "sector": "Paints",      "indices": ["Nifty 50"]},
    {"symbol": "MARUTI",      "name": "Maruti Suzuki",                   "sector": "Auto",        "indices": ["Nifty 50", "Nifty Auto"]},
    {"symbol": "TITAN",       "name": "Titan Company",                   "sector": "Consumer",    "indices": ["Nifty 50"]},
    {"symbol": "SUNPHARMA",   "name": "Sun Pharmaceutical",              "sector": "Pharma",      "indices": ["Nifty 50", "Nifty Pharma"]},
    {"symbol": "WIPRO",       "name": "Wipro",                           "sector": "IT",          "indices": ["Nifty 50", "Nifty IT"]},
    {"symbol": "ULTRACEMCO",  "name": "UltraTech Cement",                "sector": "Cement",      "indices": ["Nifty 50"]},
    {"symbol": "NTPC",        "name": "NTPC Limited",                    "sector": "Power",       "indices": ["Nifty 50"]},
    {"symbol": "POWERGRID",   "name": "Power Grid Corporation",          "sector": "Power",       "indices": ["Nifty 50"]},
    {"symbol": "BAJFINANCE",  "name": "Bajaj Finance",                   "sector": "NBFC",        "indices": ["Nifty 50"]},
    {"symbol": "BAJAJFINSV",  "name": "Bajaj Finserv",                   "sector": "NBFC",        "indices": ["Nifty 50"]},
    {"symbol": "ONGC",        "name": "Oil and Natural Gas Corporation", "sector": "Energy",      "indices": ["Nifty 50"]},
    {"symbol": "TECHM",       "name": "Tech Mahindra",                   "sector": "IT",          "indices": ["Nifty 50", "Nifty IT"]},
    {"symbol": "HCLTECH",     "name": "HCL Technologies",                "sector": "IT",          "indices": ["Nifty 50", "Nifty IT"]},
    {"symbol": "TATASTEEL",   "name": "Tata Steel",                      "sector": "Metals",      "indices": ["Nifty 50"]},
    {"symbol": "ADANIENT",    "name": "Adani Enterprises",               "sector": "Conglomerate","indices": ["Nifty 50"]},
    {"symbol": "ADANIPORTS",  "name": "Adani Ports",                     "sector": "Infra",       "indices": ["Nifty 50"]},
    {"symbol": "JSWSTEEL",    "name": "JSW Steel",                       "sector": "Metals",      "indices": ["Nifty 50"]},
    {"symbol": "COALINDIA",   "name": "Coal India",                      "sector": "Mining",      "indices": ["Nifty 50"]},
    {"symbol": "NESTLEIND",   "name": "Nestle India",                    "sector": "FMCG",        "indices": ["Nifty 50", "Nifty FMCG"]},
    {"symbol": "BRITANNIA",   "name": "Britannia Industries",            "sector": "FMCG",        "indices": ["Nifty 50", "Nifty FMCG"]},
    {"symbol": "HEROMOTOCO",  "name": "Hero MotoCorp",                   "sector": "Auto",        "indices": ["Nifty 50", "Nifty Auto"]},
    {"symbol": "BAJAJ-AUTO",  "name": "Bajaj Auto",                      "sector": "Auto",        "indices": ["Nifty 50", "Nifty Auto"]},
    {"symbol": "EICHERMOT",   "name": "Eicher Motors",                   "sector": "Auto",        "indices": ["Nifty 50", "Nifty Auto"]},
    {"symbol": "CIPLA",       "name": "Cipla",                           "sector": "Pharma",      "indices": ["Nifty 50", "Nifty Pharma"]},
    {"symbol": "DRREDDY",     "name": "Dr Reddys Laboratories",          "sector": "Pharma",      "indices": ["Nifty 50", "Nifty Pharma"]},
    {"symbol": "DIVISLAB",    "name": "Divis Laboratories",              "sector": "Pharma",      "indices": ["Nifty 50", "Nifty Pharma"]},
    {"symbol": "APOLLOHOSP",  "name": "Apollo Hospitals",                "sector": "Healthcare",  "indices": ["Nifty 50"]},
    {"symbol": "HINDALCO",    "name": "Hindalco Industries",             "sector": "Metals",      "indices": ["Nifty 50"]},
    {"symbol": "VEDL",        "name": "Vedanta",                         "sector": "Metals",      "indices": ["Nifty 50"]},
    {"symbol": "TATACONSUM",  "name": "Tata Consumer Products",          "sector": "FMCG",        "indices": ["Nifty 50", "Nifty FMCG"]},
    {"symbol": "UPL",         "name": "UPL Limited",                     "sector": "Agro Chem",   "indices": ["Nifty 50"]},
    {"symbol": "GRASIM",      "name": "Grasim Industries",               "sector": "Cement",      "indices": ["Nifty 50"]},
    {"symbol": "M&M",         "name": "Mahindra & Mahindra",             "sector": "Auto",        "indices": ["Nifty 50", "Nifty Auto"]},
    {"symbol": "INDUSINDBK",  "name": "IndusInd Bank",                   "sector": "Banking",     "indices": ["Nifty 50", "Nifty Bank"]},
    {"symbol": "BPCL",        "name": "Bharat Petroleum",                "sector": "Energy",      "indices": ["Nifty 50"]},
    {"symbol": "IOC",         "name": "Indian Oil Corporation",          "sector": "Energy",      "indices": ["Nifty 50"]},
    {"symbol": "SHREECEM",    "name": "Shree Cement",                    "sector": "Cement",      "indices": ["Nifty 50"]},
    {"symbol": "LTIM",        "name": "LTIMindtree",                     "sector": "IT",          "indices": ["Nifty 50", "Nifty IT"]},

    # ── Nifty IT (extras) ─────────────────────────────────────────────────
    {"symbol": "MPHASIS",     "name": "Mphasis",                         "sector": "IT",          "indices": ["Nifty IT"]},
    {"symbol": "PERSISTENT",  "name": "Persistent Systems",              "sector": "IT",          "indices": ["Nifty IT"]},
    {"symbol": "COFORGE",     "name": "Coforge",                         "sector": "IT",          "indices": ["Nifty IT"]},
    {"symbol": "LTTS",        "name": "L&T Technology Services",         "sector": "IT",          "indices": ["Nifty IT"]},
    {"symbol": "OFSS",        "name": "Oracle Financial Services",       "sector": "IT",          "indices": ["Nifty IT"]},
    {"symbol": "KPITTECH",    "name": "KPIT Technologies",               "sector": "IT",          "indices": ["Nifty IT"]},
    {"symbol": "TATAELXSI",   "name": "Tata Elxsi",                      "sector": "IT",          "indices": ["Nifty IT"]},

    # ── Nifty Bank (extras) ───────────────────────────────────────────────
    {"symbol": "BANDHANBNK",  "name": "Bandhan Bank",                    "sector": "Banking",     "indices": ["Nifty Bank"]},
    {"symbol": "FEDERALBNK",  "name": "Federal Bank",                    "sector": "Banking",     "indices": ["Nifty Bank"]},
    {"symbol": "IDFCFIRSTB",  "name": "IDFC First Bank",                 "sector": "Banking",     "indices": ["Nifty Bank"]},
    {"symbol": "PNB",         "name": "Punjab National Bank",            "sector": "Banking",     "indices": ["Nifty Bank"]},
    {"symbol": "BANKBARODA",  "name": "Bank of Baroda",                  "sector": "Banking",     "indices": ["Nifty Bank"]},
    {"symbol": "CANBK",       "name": "Canara Bank",                     "sector": "Banking",     "indices": ["Nifty Bank"]},
    {"symbol": "RBLBANK",     "name": "RBL Bank",                        "sector": "Banking",     "indices": ["Nifty Bank"]},
    {"symbol": "AUBANK",      "name": "AU Small Finance Bank",           "sector": "Banking",     "indices": ["Nifty Bank"]},

    # ── Nifty Pharma (extras) ─────────────────────────────────────────────
    {"symbol": "AUROPHARMA",  "name": "Aurobindo Pharma",                "sector": "Pharma",      "indices": ["Nifty Pharma"]},
    {"symbol": "BIOCON",      "name": "Biocon",                          "sector": "Pharma",      "indices": ["Nifty Pharma"]},
    {"symbol": "LUPIN",       "name": "Lupin",                           "sector": "Pharma",      "indices": ["Nifty Pharma"]},
    {"symbol": "TORNTPHARM",  "name": "Torrent Pharmaceuticals",         "sector": "Pharma",      "indices": ["Nifty Pharma"]},
    {"symbol": "ALKEM",       "name": "Alkem Laboratories",              "sector": "Pharma",      "indices": ["Nifty Pharma"]},
    {"symbol": "ABBOTINDIA",  "name": "Abbott India",                    "sector": "Pharma",      "indices": ["Nifty Pharma"]},
    {"symbol": "IPCALAB",     "name": "IPCA Laboratories",               "sector": "Pharma",      "indices": ["Nifty Pharma"]},

    # ── Nifty Auto (extras) ───────────────────────────────────────────────
    {"symbol": "TVSMOTOR",    "name": "TVS Motor Company",               "sector": "Auto",        "indices": ["Nifty Auto"]},
    {"symbol": "ASHOKLEY",    "name": "Ashok Leyland",                   "sector": "Auto",        "indices": ["Nifty Auto"]},
    {"symbol": "MOTHERSON",   "name": "Samvardhana Motherson",           "sector": "Auto",        "indices": ["Nifty Auto"]},
    {"symbol": "BOSCHLTD",    "name": "Bosch",                           "sector": "Auto",        "indices": ["Nifty Auto"]},
    {"symbol": "BALKRISIND",  "name": "Balkrishna Industries",           "sector": "Auto",        "indices": ["Nifty Auto"]},
    {"symbol": "MRF",         "name": "MRF",                             "sector": "Auto",        "indices": ["Nifty Auto"]},
    {"symbol": "TATAMOTOR",   "name": "Tata Motors",                     "sector": "Auto",        "indices": ["Nifty Auto"]},

    # ── Nifty FMCG (extras) ───────────────────────────────────────────────
    {"symbol": "DABUR",       "name": "Dabur India",                     "sector": "FMCG",        "indices": ["Nifty FMCG"]},
    {"symbol": "MARICO",      "name": "Marico",                          "sector": "FMCG",        "indices": ["Nifty FMCG"]},
    {"symbol": "COLPAL",      "name": "Colgate Palmolive India",         "sector": "FMCG",        "indices": ["Nifty FMCG"]},
    {"symbol": "GODREJCP",    "name": "Godrej Consumer Products",        "sector": "FMCG",        "indices": ["Nifty FMCG"]},
    {"symbol": "EMAMILTD",    "name": "Emami",                           "sector": "FMCG",        "indices": ["Nifty FMCG"]},
    {"symbol": "VBL",         "name": "Varun Beverages",                 "sector": "FMCG",        "indices": ["Nifty FMCG"]},

    # ── Other popular stocks ──────────────────────────────────────────────
    {"symbol": "ZOMATO",      "name": "Zomato",                          "sector": "Consumer Tech","indices": []},
    {"symbol": "PAYTM",       "name": "Paytm (One97 Communications)",    "sector": "Fintech",     "indices": []},
    {"symbol": "NYKAA",       "name": "Nykaa (FSN E-Commerce)",          "sector": "Consumer Tech","indices": []},
    {"symbol": "POLICYBZR",   "name": "PB Fintech (PolicyBazaar)",       "sector": "Fintech",     "indices": []},
    {"symbol": "DELHIVERY",   "name": "Delhivery",                       "sector": "Logistics",   "indices": []},
    {"symbol": "IRCTC",       "name": "IRCTC",                           "sector": "Travel",      "indices": []},
    {"symbol": "HAL",         "name": "Hindustan Aeronautics",           "sector": "Defence",     "indices": []},
    {"symbol": "BEL",         "name": "Bharat Electronics",              "sector": "Defence",     "indices": []},
    {"symbol": "DIXON",       "name": "Dixon Technologies",              "sector": "Electronics", "indices": []},
    {"symbol": "ABFRL",       "name": "Aditya Birla Fashion",            "sector": "Retail",      "indices": []},
    {"symbol": "TRENT",       "name": "Trent (Westside)",                "sector": "Retail",      "indices": []},
    {"symbol": "DMART",       "name": "Avenue Supermarts (DMart)",       "sector": "Retail",      "indices": []},
    {"symbol": "PIDILITIND",  "name": "Pidilite Industries",             "sector": "Chemicals",   "indices": []},
    {"symbol": "SRF",         "name": "SRF Limited",                     "sector": "Chemicals",   "indices": []},
    {"symbol": "DEEPAKNTR",   "name": "Deepak Nitrite",                  "sector": "Chemicals",   "indices": []},
    {"symbol": "TATACHEM",    "name": "Tata Chemicals",                  "sector": "Chemicals",   "indices": []},
    {"symbol": "ADANIGREEN",  "name": "Adani Green Energy",              "sector": "Renewable",   "indices": []},
    {"symbol": "ADANITRANS",  "name": "Adani Transmission",              "sector": "Power",       "indices": []},
    {"symbol": "ADANIPOWER",  "name": "Adani Power",                     "sector": "Power",       "indices": []},
    {"symbol": "TATAPOWER",   "name": "Tata Power",                      "sector": "Power",       "indices": []},
    {"symbol": "CESC",        "name": "CESC Limited",                    "sector": "Power",       "indices": []},
    {"symbol": "NHPC",        "name": "NHPC Limited",                    "sector": "Power",       "indices": []},
    {"symbol": "RECLTD",      "name": "REC Limited",                     "sector": "NBFC",        "indices": []},
    {"symbol": "PFC",         "name": "Power Finance Corporation",       "sector": "NBFC",        "indices": []},
    {"symbol": "MUTHOOTFIN",  "name": "Muthoot Finance",                 "sector": "NBFC",        "indices": []},
    {"symbol": "CHOLAFIN",    "name": "Cholamandalam Investment",        "sector": "NBFC",        "indices": []},
    {"symbol": "HDFCAMC",     "name": "HDFC AMC",                        "sector": "AMC",         "indices": []},
    {"symbol": "NAUKRI",      "name": "Info Edge (Naukri)",              "sector": "Consumer Tech","indices": []},
    {"symbol": "JUSTDIAL",    "name": "Just Dial",                       "sector": "Consumer Tech","indices": []},
    {"symbol": "INDIGO",      "name": "IndiGo (InterGlobe Aviation)",    "sector": "Aviation",    "indices": []},
    {"symbol": "SPICEJET",    "name": "SpiceJet",                        "sector": "Aviation",    "indices": []},
]

# ── Aliases (common short names / abbreviations) ─────────────────────────
ALIASES = {
    "ril":         "RELIANCE",
    "reliance":    "RELIANCE",
    "tata motors": "TATAMOTOR",
    "tata steel":  "TATASTEEL",
    "tata motors": "TATAMOTOR",
    "tata power":  "TATAPOWER",
    "tata chem":   "TATACHEM",
    "tata consumer": "TATACONSUM",
    "hdfc":        "HDFCBANK",
    "hdfc bank":   "HDFCBANK",
    "sbi":         "SBIN",
    "infosys":     "INFY",
    "infy":        "INFY",
    "wipro":       "WIPRO",
    "hcl":         "HCLTECH",
    "tech mahindra": "TECHM",
    "bajaj finance": "BAJFINANCE",
    "bajaj finserv": "BAJAJFINSV",
    "kotak":       "KOTAKBANK",
    "kotak bank":  "KOTAKBANK",
    "icici":       "ICICIBANK",
    "icici bank":  "ICICIBANK",
    "axis":        "AXISBANK",
    "axis bank":   "AXISBANK",
    "sun pharma":  "SUNPHARMA",
    "dr reddy":    "DRREDDY",
    "dr reddys":   "DRREDDY",
    "divi":        "DIVISLAB",
    "divis":       "DIVISLAB",
    "asian paints":"ASIANPAINT",
    "l&t":         "LT",
    "larsen":      "LT",
    "airtel":      "BHARTIARTL",
    "bharti":      "BHARTIARTL",
    "ongc":        "ONGC",
    "coal india":  "COALINDIA",
    "ntpc":        "NTPC",
    "power grid":  "POWERGRID",
    "pgcil":       "POWERGRID",
    "ultratech":   "ULTRACEMCO",
    "shree cement":"SHREECEM",
    "grasim":      "GRASIM",
    "hindalco":    "HINDALCO",
    "vedanta":     "VEDL",
    "jswsteel":    "JSWSTEEL",
    "jsw":         "JSWSTEEL",
    "mahindra":    "M&M",
    "m&m":         "M&M",
    "maruti":      "MARUTI",
    "hero":        "HEROMOTOCO",
    "hero moto":   "HEROMOTOCO",
    "bajaj auto":  "BAJAJ-AUTO",
    "eicher":      "EICHERMOT",
    "royal enfield": "EICHERMOT",
    "dmart":       "DMART",
    "avenue supermarts": "DMART",
    "zomato":      "ZOMATO",
    "paytm":       "PAYTM",
    "nykaa":       "NYKAA",
    "policybazaar":"POLICYBZR",
    "irctc":       "IRCTC",
    "hal":         "HAL",
    "bel":         "BEL",
    "indigo":      "INDIGO",
    "spicejet":    "SPICEJET",
    "pidilite":    "PIDILITIND",
    "fevicol":     "PIDILITIND",
    "dabur":       "DABUR",
    "marico":      "MARICO",
    "colgate":     "COLPAL",
    "godrej":      "GODREJCP",
    "nestle":      "NESTLEIND",
    "britannia":   "BRITANNIA",
    "itc":         "ITC",
    "titan":       "TITAN",
    "mrf":         "MRF",
    "bosch":       "BOSCHLTD",
    "tvs":         "TVSMOTOR",
    "ashok leyland": "ASHOKLEY",
    "trent":       "TRENT",
    "westside":    "TRENT",
    "pfc":         "PFC",
    "rec":         "RECLTD",
    "muthoot":     "MUTHOOTFIN",
    "chola":       "CHOLAFIN",
    "ltimindtree": "LTIM",
    "mindtree":    "LTIM",
    "mphasis":     "MPHASIS",
    "persistent":  "PERSISTENT",
    "coforge":     "COFORGE",
    "oracle financial": "OFSS",
    "kpit":        "KPITTECH",
    "tata elxsi":  "TATAELXSI",
    "bandhan":     "BANDHANBNK",
    "federal bank":"FEDERALBNK",
    "idfc":        "IDFCFIRSTB",
    "pnb":         "PNB",
    "bob":         "BANKBARODA",
    "bank of baroda": "BANKBARODA",
    "canara":      "CANBK",
    "rbl":         "RBLBANK",
    "au bank":     "AUBANK",
    "au small":    "AUBANK",
    "aurobindo":   "AUROPHARMA",
    "biocon":      "BIOCON",
    "lupin":       "LUPIN",
    "torrent":     "TORNTPHARM",
    "alkem":       "ALKEM",
    "abbott":      "ABBOTINDIA",
    "ipca":        "IPCALAB",
    "adani":       "ADANIENT",
    "adani green": "ADANIGREEN",
    "adani ports": "ADANIPORTS",
    "adani power": "ADANIPOWER",
    "tata power":  "TATAPOWER",
    "naukri":      "NAUKRI",
    "info edge":   "NAUKRI",
    "delhivery":   "DELHIVERY",
    "dixon":       "DIXON",
    "varun beverages": "VBL",
    "emami":       "EMAMILTD",
    "apollo":      "APOLLOHOSP",
    "deepak nitrite": "DEEPAKNTR",
    "srf":         "SRF",
    "upl":         "UPL",
    "cipla":       "CIPLA",
    "hindustan unilever": "HINDUNILVR",
    "hul":         "HINDUNILVR",
    "unilever":    "HINDUNILVR",
    "hdfc amc":    "HDFCAMC",
    "justdial":    "JUSTDIAL",
}

# ── Sector keyword → sector name mapping ──────────────────────────────────
SECTOR_KEYWORDS = {
    "it":          "IT",
    "tech":        "IT",
    "technology":  "IT",
    "software":    "IT",
    "banking":     "Banking",
    "bank":        "Banking",
    "banks":       "Banking",
    "pharma":      "Pharma",
    "pharmaceutical": "Pharma",
    "healthcare":  "Healthcare",
    "health":      "Healthcare",
    "fmcg":        "FMCG",
    "consumer":    "FMCG",
    "auto":        "Auto",
    "automobile":  "Auto",
    "automotive":  "Auto",
    "energy":      "Energy",
    "oil":         "Energy",
    "gas":         "Energy",
    "power":       "Power",
    "electricity": "Power",
    "metals":      "Metals",
    "metal":       "Metals",
    "steel":       "Metals",
    "mining":      "Mining",
    "cement":      "Cement",
    "infra":       "Infra",
    "infrastructure": "Infra",
    "nbfc":        "NBFC",
    "finance":     "NBFC",
    "fintech":     "Fintech",
    "telecom":     "Telecom",
    "chemicals":   "Chemicals",
    "chemical":    "Chemicals",
    "retail":      "Retail",
    "aviation":    "Aviation",
    "airline":     "Aviation",
    "defence":     "Defence",
    "defense":     "Defence",
    "renewable":   "Renewable",
    "solar":       "Renewable",
    "logistics":   "Logistics",
    "paints":      "Paints",
}

# ── Index keyword → index name mapping ────────────────────────────────────
INDEX_KEYWORDS = {
    "nifty 50":    "Nifty 50",
    "nifty50":     "Nifty 50",
    "nifty it":    "Nifty IT",
    "niftyit":     "Nifty IT",
    "nifty bank":  "Nifty Bank",
    "niftybank":   "Nifty Bank",
    "bank nifty":  "Nifty Bank",
    "nifty pharma":"Nifty Pharma",
    "nifty auto":  "Nifty Auto",
    "nifty fmcg":  "Nifty FMCG",
}

# ── Pre-built lookup structures ────────────────────────────────────────────
_symbol_set = {r["symbol"] for r in REGISTRY}
_by_sector: dict[str, list[dict]] = {}
_by_index:  dict[str, list[dict]] = {}
_by_symbol: dict[str, dict] = {}

for _r in REGISTRY:
    _by_symbol[_r["symbol"]] = _r
    _by_sector.setdefault(_r["sector"], []).append(_r)
    for _idx in _r["indices"]:
        _by_index.setdefault(_idx, []).append(_r)


def resolve(query: str) -> dict:
    """
    Resolve a free-text query to stock(s).

    Returns:
        {
          "type": "single" | "list" | "none",
          "results": [ {"symbol", "name", "sector", "indices"}, ... ],
          "match_type": "exact" | "alias" | "index" | "sector" | "fuzzy"
        }
    """
    from rapidfuzz import process, fuzz

    q = query.strip().lower()
    qu = q.upper()

    # 1. Exact symbol match
    if qu in _symbol_set:
        return {"type": "single", "results": [_by_symbol[qu]], "match_type": "exact"}

    # 2. Alias match
    if q in ALIASES:
        sym = ALIASES[q]
        return {"type": "single", "results": [_by_symbol[sym]], "match_type": "alias"}

    # 3. Index keyword match  (e.g. "nifty bank", "bank nifty")
    for kw, idx_name in INDEX_KEYWORDS.items():
        if kw in q:
            stocks = _by_index.get(idx_name, [])
            return {"type": "list", "results": stocks, "match_type": "index", "group": idx_name}

    # 4. Sector keyword match  (e.g. "banking stocks", "pharma sector")
    for kw, sector_name in SECTOR_KEYWORDS.items():
        if kw in q:
            stocks = _by_sector.get(sector_name, [])
            if stocks:
                return {"type": "list", "results": stocks, "match_type": "sector", "group": sector_name}

    # 5. Fuzzy match on company names + symbols
    all_names = [(r["name"], r["symbol"]) for r in REGISTRY]
    all_symbols = [(r["symbol"], r["symbol"]) for r in REGISTRY]
    candidates = all_names + all_symbols

    matches = process.extract(query, [c[0] for c in candidates], scorer=fuzz.WRatio, limit=5, score_cutoff=55)
    if matches:
        seen = set()
        results = []
        for match_str, score, idx in matches:
            sym = candidates[idx][1]
            if sym not in seen:
                seen.add(sym)
                results.append(_by_symbol[sym])
        if len(results) == 1:
            return {"type": "single", "results": results, "match_type": "fuzzy"}
        return {"type": "list", "results": results, "match_type": "fuzzy"}

    return {"type": "none", "results": [], "match_type": "none"}


def get_all_symbols() -> list[str]:
    return [r["symbol"] for r in REGISTRY]


def get_sector_list() -> list[str]:
    return sorted(_by_sector.keys())


def get_index_list() -> list[str]:
    return sorted(_by_index.keys())
