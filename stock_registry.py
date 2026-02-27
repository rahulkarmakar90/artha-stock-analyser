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
    {"symbol": "UNIONBANK",   "name": "Union Bank of India",             "sector": "Banking",     "indices": ["Nifty Bank"]},
    {"symbol": "INDIANB",     "name": "Indian Bank",                     "sector": "Banking",     "indices": []},
    {"symbol": "IOB",         "name": "Indian Overseas Bank",            "sector": "Banking",     "indices": []},
    {"symbol": "CENTRALBK",   "name": "Central Bank of India",          "sector": "Banking",     "indices": []},
    {"symbol": "MAHABANK",    "name": "Bank of Maharashtra",             "sector": "Banking",     "indices": []},
    {"symbol": "UCOBANK",     "name": "UCO Bank",                        "sector": "Banking",     "indices": []},
    {"symbol": "BANKINDIA",   "name": "Bank of India",                   "sector": "Banking",     "indices": []},

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

    # ── Insurance ─────────────────────────────────────────────────────────
    {"symbol": "SBILIFE",     "name": "SBI Life Insurance",              "sector": "Insurance",   "indices": []},
    {"symbol": "HDFCLIFE",    "name": "HDFC Life Insurance",             "sector": "Insurance",   "indices": []},
    {"symbol": "ICICIPRULI",  "name": "ICICI Prudential Life Insurance", "sector": "Insurance",   "indices": []},
    {"symbol": "ICICIGI",     "name": "ICICI Lombard General Insurance", "sector": "Insurance",   "indices": []},
    {"symbol": "STARHEALTH",  "name": "Star Health and Allied Insurance","sector": "Insurance",   "indices": []},
    {"symbol": "NIACL",       "name": "New India Assurance",             "sector": "Insurance",   "indices": []},
    {"symbol": "GICRE",       "name": "GIC Re",                          "sector": "Insurance",   "indices": []},
    {"symbol": "LIC",         "name": "Life Insurance Corporation",      "sector": "Insurance",   "indices": []},
    {"symbol": "SBICARD",     "name": "SBI Cards and Payment Services",  "sector": "Fintech",     "indices": []},
    {"symbol": "JIOFIN",      "name": "Jio Financial Services",          "sector": "Fintech",     "indices": []},
    {"symbol": "BAJAJHFL",    "name": "Bajaj Housing Finance",           "sector": "NBFC",        "indices": []},
    {"symbol": "LTF",         "name": "L&T Finance",                     "sector": "NBFC",        "indices": []},

    # ── PSU / Govt ─────────────────────────────────────────────────────────
    {"symbol": "SAIL",        "name": "Steel Authority of India",        "sector": "Metals",      "indices": []},
    {"symbol": "NMDC",        "name": "NMDC",                            "sector": "Mining",      "indices": []},
    {"symbol": "GAIL",        "name": "GAIL India",                      "sector": "Energy",      "indices": []},
    {"symbol": "OIL",         "name": "Oil India",                       "sector": "Energy",      "indices": []},
    {"symbol": "CONCOR",      "name": "Container Corporation of India",  "sector": "Logistics",   "indices": []},
    {"symbol": "NBCC",        "name": "NBCC India",                      "sector": "Infra",       "indices": []},
    {"symbol": "RVNL",        "name": "Rail Vikas Nigam",                "sector": "Infra",       "indices": []},
    {"symbol": "IRFC",        "name": "Indian Railway Finance Corp",     "sector": "NBFC",        "indices": []},
    {"symbol": "IREDA",       "name": "Indian Renewable Energy Dev Agency","sector": "NBFC",      "indices": []},
    {"symbol": "HUDCO",       "name": "Housing and Urban Dev Corp",      "sector": "NBFC",        "indices": []},
    {"symbol": "SJVN",        "name": "SJVN",                            "sector": "Power",       "indices": []},
    {"symbol": "BHEL",        "name": "Bharat Heavy Electricals",        "sector": "Capital Goods","indices": []},
    {"symbol": "COCHINSHIP",  "name": "Cochin Shipyard",                 "sector": "Defence",     "indices": []},
    {"symbol": "GRSE",        "name": "Garden Reach Shipbuilders",       "sector": "Defence",     "indices": []},
    {"symbol": "MAZAGON",     "name": "Mazagon Dock Shipbuilders",       "sector": "Defence",     "indices": []},
    {"symbol": "NATIONALUM",  "name": "National Aluminium Company",      "sector": "Metals",      "indices": []},
    {"symbol": "HINDCOPPER",  "name": "Hindustan Copper",                "sector": "Metals",      "indices": []},
    {"symbol": "MOIL",        "name": "MOIL",                            "sector": "Mining",      "indices": []},

    # ── Real Estate ────────────────────────────────────────────────────────
    {"symbol": "DLF",         "name": "DLF",                             "sector": "Real Estate", "indices": []},
    {"symbol": "GODREJPROP",  "name": "Godrej Properties",               "sector": "Real Estate", "indices": []},
    {"symbol": "OBEROIRLTY",  "name": "Oberoi Realty",                   "sector": "Real Estate", "indices": []},
    {"symbol": "PRESTIGE",    "name": "Prestige Estates Projects",       "sector": "Real Estate", "indices": []},
    {"symbol": "PHOENIXLTD",  "name": "Phoenix Mills",                   "sector": "Real Estate", "indices": []},
    {"symbol": "BRIGADE",     "name": "Brigade Enterprises",             "sector": "Real Estate", "indices": []},
    {"symbol": "SOBHA",       "name": "Sobha",                           "sector": "Real Estate", "indices": []},

    # ── Capital Goods / Infra ──────────────────────────────────────────────
    {"symbol": "ABB",         "name": "ABB India",                       "sector": "Capital Goods","indices": []},
    {"symbol": "SIEMENS",     "name": "Siemens India",                   "sector": "Capital Goods","indices": []},
    {"symbol": "HAVELLS",     "name": "Havells India",                   "sector": "Capital Goods","indices": []},
    {"symbol": "VOLTAS",      "name": "Voltas",                          "sector": "Capital Goods","indices": []},
    {"symbol": "BLUESTAR",    "name": "Blue Star",                       "sector": "Capital Goods","indices": []},
    {"symbol": "POLYCAB",     "name": "Polycab India",                   "sector": "Capital Goods","indices": []},
    {"symbol": "AMBER",       "name": "Amber Enterprises",               "sector": "Electronics", "indices": []},
    {"symbol": "VGUARD",      "name": "V-Guard Industries",              "sector": "Capital Goods","indices": []},

    # ── Auto Ancillaries ───────────────────────────────────────────────────
    {"symbol": "BHARATFORG",  "name": "Bharat Forge",                    "sector": "Auto",        "indices": []},
    {"symbol": "CUMMINSIND",  "name": "Cummins India",                   "sector": "Auto",        "indices": []},
    {"symbol": "APOLLOTYRE",  "name": "Apollo Tyres",                    "sector": "Auto",        "indices": []},
    {"symbol": "EXIDEIND",    "name": "Exide Industries",                "sector": "Auto",        "indices": []},
    {"symbol": "SUNDRMFAST",  "name": "Sundram Fasteners",               "sector": "Auto",        "indices": []},
    {"symbol": "SCHAEFFLER",  "name": "Schaeffler India",                "sector": "Auto",        "indices": []},
    {"symbol": "AMARAJABAT",  "name": "Amara Raja Energy and Mobility",  "sector": "Auto",        "indices": []},

    # ── Pharma / Healthcare (extras) ──────────────────────────────────────
    {"symbol": "MANKIND",     "name": "Mankind Pharma",                  "sector": "Pharma",      "indices": []},
    {"symbol": "GLAND",       "name": "Gland Pharma",                    "sector": "Pharma",      "indices": []},
    {"symbol": "GRANULES",    "name": "Granules India",                  "sector": "Pharma",      "indices": []},
    {"symbol": "LAURUSLABS",  "name": "Laurus Labs",                     "sector": "Pharma",      "indices": []},
    {"symbol": "AARTI",       "name": "Aarti Industries",                "sector": "Chemicals",   "indices": []},
    {"symbol": "SYNGENE",     "name": "Syngene International",           "sector": "Pharma",      "indices": []},
    {"symbol": "MAXHEALTH",   "name": "Max Healthcare Institute",        "sector": "Healthcare",  "indices": []},
    {"symbol": "FORTIS",      "name": "Fortis Healthcare",               "sector": "Healthcare",  "indices": []},

    # ── Chemicals (extras) ─────────────────────────────────────────────────
    {"symbol": "PIIND",       "name": "PI Industries",                   "sector": "Chemicals",   "indices": []},
    {"symbol": "ASTRAL",      "name": "Astral",                          "sector": "Chemicals",   "indices": []},
    {"symbol": "SUPREMEIND",  "name": "Supreme Industries",              "sector": "Chemicals",   "indices": []},
    {"symbol": "NAVINFLUOR",  "name": "Navin Fluorine International",    "sector": "Chemicals",   "indices": []},
    {"symbol": "CLEAN",       "name": "Clean Science and Technology",    "sector": "Chemicals",   "indices": []},
    {"symbol": "ALKYLAMINE",  "name": "Alkyl Amines Chemicals",          "sector": "Chemicals",   "indices": []},

    # ── IT / Tech (extras) ─────────────────────────────────────────────────
    {"symbol": "CYIENT",      "name": "Cyient",                          "sector": "IT",          "indices": []},
    {"symbol": "ZENSAR",      "name": "Zensar Technologies",             "sector": "IT",          "indices": []},
    {"symbol": "HEXAWARE",    "name": "Hexaware Technologies",           "sector": "IT",          "indices": []},
    {"symbol": "HAPPSTMNDS",  "name": "Happiest Minds Technologies",     "sector": "IT",          "indices": []},
    {"symbol": "INTELLECT",   "name": "Intellect Design Arena",          "sector": "IT",          "indices": []},
    {"symbol": "MASTEK",      "name": "Mastek",                          "sector": "IT",          "indices": []},
    {"symbol": "TANLA",       "name": "Tanla Platforms",                 "sector": "IT",          "indices": []},
    {"symbol": "RATEGAIN",    "name": "RateGain Travel Technologies",    "sector": "IT",          "indices": []},
    {"symbol": "TEJASNET",    "name": "Tejas Networks",                  "sector": "IT",          "indices": []},

    # ── Technology Distribution ─────────────────────────────────────────────
    {"symbol": "REDINGTON",   "name": "Redington Limited",               "sector": "IT",          "indices": []},

    # ── Telecom / Media ────────────────────────────────────────────────────
    {"symbol": "IDEA",        "name": "Vodafone Idea",                   "sector": "Telecom",     "indices": []},
    {"symbol": "TATACOMM",    "name": "Tata Communications",             "sector": "Telecom",     "indices": []},
    {"symbol": "HFCL",        "name": "HFCL",                            "sector": "Telecom",     "indices": []},
    {"symbol": "ZEEL",        "name": "Zee Entertainment Enterprises",   "sector": "Media",       "indices": []},
    {"symbol": "SUNTV",       "name": "Sun TV Network",                  "sector": "Media",       "indices": []},
    {"symbol": "PVRINOX",     "name": "PVR Inox",                        "sector": "Media",       "indices": []},

    # ── Consumer ───────────────────────────────────────────────────────────
    {"symbol": "RAYMOND",     "name": "Raymond",                         "sector": "Consumer",    "indices": []},
    {"symbol": "VMART",       "name": "V-Mart Retail",                   "sector": "Retail",      "indices": []},

    # ── ETFs — Broad Market ────────────────────────────────────────────────
    {"symbol": "NIFTYBEES",   "name": "Nippon India Nifty 50 BeES ETF",  "sector": "ETF",         "indices": ["ETF", "ETF Nifty"]},
    {"symbol": "JUNIORBEES",  "name": "Nippon India Nifty Next 50 BeES", "sector": "ETF",         "indices": ["ETF", "ETF Nifty"]},
    {"symbol": "SETFNN50",    "name": "SBI Nifty Next 50 ETF",           "sector": "ETF",         "indices": ["ETF", "ETF Nifty"]},
    {"symbol": "MOM100",      "name": "Nippon India Nifty Momentum 100 ETF","sector": "ETF",      "indices": ["ETF", "ETF Nifty"]},
    {"symbol": "SMALLCAP",    "name": "Nippon India Nifty Smallcap 250 BeES","sector": "ETF",     "indices": ["ETF", "ETF Nifty"]},
    {"symbol": "MONIFTY500",  "name": "Motilal Oswal Nifty 500 ETF",     "sector": "ETF",         "indices": ["ETF", "ETF Nifty"]},

    # ── ETFs — Sectoral ────────────────────────────────────────────────────
    {"symbol": "BANKBEES",    "name": "Nippon India Bank BeES ETF",       "sector": "ETF",         "indices": ["ETF", "ETF Sectoral"]},
    {"symbol": "PSUBNKBEES",  "name": "Nippon India PSU Bank BeES ETF",   "sector": "ETF",         "indices": ["ETF", "ETF Sectoral"]},
    {"symbol": "ITBEES",      "name": "Nippon India IT BeES ETF",          "sector": "ETF",         "indices": ["ETF", "ETF Sectoral"]},
    {"symbol": "PHARMABEES",  "name": "Nippon India Pharma BeES ETF",     "sector": "ETF",         "indices": ["ETF", "ETF Sectoral"]},
    {"symbol": "INFRABEES",   "name": "Nippon India Infra BeES ETF",      "sector": "ETF",         "indices": ["ETF", "ETF Sectoral"]},
    {"symbol": "AUTOBEES",    "name": "Nippon India Auto BeES ETF",       "sector": "ETF",         "indices": ["ETF", "ETF Sectoral"]},
    {"symbol": "CONSUMBEES",  "name": "Nippon India Consumption BeES ETF","sector": "ETF",         "indices": ["ETF", "ETF Sectoral"]},
    {"symbol": "DIVOPPBEES",  "name": "Nippon India Dividend Opportunities BeES","sector": "ETF",  "indices": ["ETF", "ETF Sectoral"]},
    {"symbol": "ICICIB22",    "name": "ICICI Prudential BSE 500 ETF",     "sector": "ETF",         "indices": ["ETF", "ETF Sectoral"]},

    # ── ETFs — Gold ────────────────────────────────────────────────────────
    {"symbol": "GOLDBEES",    "name": "Nippon India Gold BeES ETF",       "sector": "ETF",         "indices": ["ETF", "ETF Gold"]},
    {"symbol": "SETFGOLD",    "name": "SBI Gold ETF",                     "sector": "ETF",         "indices": ["ETF", "ETF Gold"]},
    {"symbol": "HDFCGOLD",    "name": "HDFC Gold ETF",                    "sector": "ETF",         "indices": ["ETF", "ETF Gold"]},
    {"symbol": "AXISGOLD",    "name": "Axis Gold ETF",                    "sector": "ETF",         "indices": ["ETF", "ETF Gold"]},
    {"symbol": "BSLGOLDETF",  "name": "BSL Gold ETF",                     "sector": "ETF",         "indices": ["ETF", "ETF Gold"]},

    # ── ETFs — Silver ──────────────────────────────────────────────────────
    {"symbol": "SILVERBEES",  "name": "Nippon India Silver ETF",          "sector": "ETF",         "indices": ["ETF", "ETF Silver"]},

    # ── ETFs — Debt / Liquid ───────────────────────────────────────────────
    {"symbol": "LIQUIDBEES",  "name": "Nippon India Liquid BeES ETF",     "sector": "ETF",         "indices": ["ETF", "ETF Debt"]},
    {"symbol": "SETF10GILT",  "name": "SBI 10Y Gilt ETF",                 "sector": "ETF",         "indices": ["ETF", "ETF Debt"]},

    # ── ETFs — International ───────────────────────────────────────────────
    {"symbol": "MAFANG",      "name": "Mirae Asset NYSE FANG+ ETF",       "sector": "ETF",         "indices": ["ETF", "ETF International"]},
    {"symbol": "HNGSNGBEES",  "name": "Nippon India Hang Seng BeES ETF",  "sector": "ETF",         "indices": ["ETF", "ETF International"]},
    {"symbol": "MASPTOP50",   "name": "Mirae Asset S&P 500 Top 50 ETF",   "sector": "ETF",         "indices": ["ETF", "ETF International"]},
    {"symbol": "MON100",      "name": "Motilal Oswal Nasdaq 100 ETF",     "sector": "ETF",         "indices": ["ETF", "ETF International"]},
    {"symbol": "NIFTYIETF",   "name": "Nippon India Nifty India Innovation ETF","sector": "ETF",  "indices": ["ETF", "ETF International"]},

    # ── ETFs — Factor / Smart Beta ─────────────────────────────────────────
    {"symbol": "LOWVOLIETF",  "name": "ICICI Pru Nifty Low Volatility 30 ETF","sector": "ETF",    "indices": ["ETF", "ETF Factor"]},
    {"symbol": "ALPHA",       "name": "ICICI Pru Alpha Low Vol 30 ETF",   "sector": "ETF",         "indices": ["ETF", "ETF Factor"]},
    {"symbol": "QUAL30IETF",  "name": "ICICI Pru Nifty Quality Low Vol ETF","sector": "ETF",       "indices": ["ETF", "ETF Factor"]},
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
    "union bank":  "UNIONBANK",
    "union bank of india": "UNIONBANK",
    "ubi":         "UNIONBANK",
    "indian bank": "INDIANB",
    "iob":         "IOB",
    "indian overseas": "IOB",
    "central bank": "CENTRALBK",
    "bank of maharashtra": "MAHABANK",
    "uco":         "UCOBANK",
    "bank of india": "BANKINDIA",
    "boi":         "BANKINDIA",
    # Insurance
    "sbi life":    "SBILIFE",
    "hdfc life":   "HDFCLIFE",
    "icici pru":   "ICICIPRULI",
    "icici lombard": "ICICIGI",
    "star health": "STARHEALTH",
    "lic":         "LIC",
    "sbi card":    "SBICARD",
    "sbi cards":   "SBICARD",
    "jio financial": "JIOFIN",
    "jio fin":     "JIOFIN",
    "bajaj housing": "BAJAJHFL",
    "lt finance":  "LTF",
    "l&t finance": "LTF",
    # PSU / Govt
    "sail":        "SAIL",
    "steel authority": "SAIL",
    "nmdc":        "NMDC",
    "gail":        "GAIL",
    "oil india":   "OIL",
    "concor":      "CONCOR",
    "container corp": "CONCOR",
    "nbcc":        "NBCC",
    "rvnl":        "RVNL",
    "rail vikas":  "RVNL",
    "irfc":        "IRFC",
    "ireda":       "IREDA",
    "hudco":       "HUDCO",
    "sjvn":        "SJVN",
    "bhel":        "BHEL",
    "cochin ship": "COCHINSHIP",
    "mazagon":     "MAZAGON",
    "national aluminium": "NATIONALUM",
    "nalco":       "NATIONALUM",
    "hindustan copper": "HINDCOPPER",
    "moil":        "MOIL",
    # Real estate
    "dlf":         "DLF",
    "godrej prop": "GODREJPROP",
    "godrej properties": "GODREJPROP",
    "oberoi":      "OBEROIRLTY",
    "prestige":    "PRESTIGE",
    "phoenix mills": "PHOENIXLTD",
    "brigade":     "BRIGADE",
    "sobha":       "SOBHA",
    # Capital goods
    "abb":         "ABB",
    "siemens":     "SIEMENS",
    "havells":     "HAVELLS",
    "voltas":      "VOLTAS",
    "blue star":   "BLUESTAR",
    "polycab":     "POLYCAB",
    "amber":       "AMBER",
    "vguard":      "VGUARD",
    # Auto ancillaries
    "bharat forge": "BHARATFORG",
    "cummins":     "CUMMINSIND",
    "apollo tyre": "APOLLOTYRE",
    "exide":       "EXIDEIND",
    "schaeffler":  "SCHAEFFLER",
    "amara raja":  "AMARAJABAT",
    # Pharma / Healthcare
    "mankind":     "MANKIND",
    "gland":       "GLAND",
    "granules":    "GRANULES",
    "laurus":      "LAURUSLABS",
    "syngene":     "SYNGENE",
    "max health":  "MAXHEALTH",
    "max healthcare": "MAXHEALTH",
    "fortis":      "FORTIS",
    # Chemicals
    "pi industries": "PIIND",
    "astral":      "ASTRAL",
    "supreme ind": "SUPREMEIND",
    "navin fluorine": "NAVINFLUOR",
    "clean science": "CLEAN",
    "alkyl amines": "ALKYLAMINE",
    # IT extras
    "cyient":      "CYIENT",
    "zensar":      "ZENSAR",
    "hexaware":    "HEXAWARE",
    "happiest minds": "HAPPSTMNDS",
    "intellect":   "INTELLECT",
    "mastek":      "MASTEK",
    "tanla":       "TANLA",
    "rategain":    "RATEGAIN",
    "tejas":       "TEJASNET",
    "tejas networks": "TEJASNET",
    "tejasnet":    "TEJASNET",
    "redington":   "REDINGTON",
    "redington india": "REDINGTON",
    # Telecom / Media
    "vodafone idea": "IDEA",
    "vi":          "IDEA",
    "tata comm":   "TATACOMM",
    "tata communications": "TATACOMM",
    "hfcl":        "HFCL",
    "zee":         "ZEEL",
    "sun tv":      "SUNTV",
    "pvr":         "PVRINOX",
    "pvr inox":    "PVRINOX",
    "inox":        "PVRINOX",
    # Consumer
    "raymond":     "RAYMOND",
    "vmart":       "VMART",
    # ETFs — Broad Market
    "nifty bees":  "NIFTYBEES",
    "niftybees":   "NIFTYBEES",
    "junior bees": "JUNIORBEES",
    "juniorbees":  "JUNIORBEES",
    "nifty next 50 etf": "SETFNN50",
    "momentum etf": "MOM100",
    "mom100":      "MOM100",
    "smallcap etf": "SMALLCAP",
    "nifty 500 etf": "MONIFTY500",
    # ETFs — Sectoral
    "bank bees":   "BANKBEES",
    "bankbees":    "BANKBEES",
    "psu bank etf": "PSUBNKBEES",
    "it bees":     "ITBEES",
    "itbees":      "ITBEES",
    "pharma bees": "PHARMABEES",
    "infra bees":  "INFRABEES",
    "auto bees":   "AUTOBEES",
    "consumption etf": "CONSUMBEES",
    "dividend etf": "DIVOPPBEES",
    # ETFs — Gold
    "gold bees":   "GOLDBEES",
    "goldbees":    "GOLDBEES",
    "sbi gold etf": "SETFGOLD",
    "hdfc gold etf": "HDFCGOLD",
    "axis gold etf": "AXISGOLD",
    "bsl gold":    "BSLGOLDETF",
    # ETFs — Silver
    "silver bees": "SILVERBEES",
    "silverbees":  "SILVERBEES",
    # ETFs — Debt
    "liquid bees": "LIQUIDBEES",
    "liquidbees":  "LIQUIDBEES",
    "gilt etf":    "SETF10GILT",
    # ETFs — International
    "fang etf":    "MAFANG",
    "hang seng etf": "HNGSNGBEES",
    "s&p 500 etf": "MASPTOP50",
    "nasdaq etf":  "MON100",
    "nasdaq 100":  "MON100",
    # ETFs — Factor
    "low vol etf": "LOWVOLIETF",
    "alpha etf":   "ALPHA",
    "quality etf": "QUAL30IETF",
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
    # ETF index keywords
    "all etf":     "ETF",
    "all etfs":    "ETF",
    "etf list":    "ETF",
    "gold etf":    "ETF Gold",
    "silver etf":  "ETF Silver",
    "debt etf":    "ETF Debt",
    "liquid etf":  "ETF Debt",
    "international etf": "ETF International",
    "global etf":  "ETF International",
    "us etf":      "ETF International",
    "factor etf":  "ETF Factor",
    "smart beta":  "ETF Factor",
    "sectoral etf":"ETF Sectoral",
    "sector etf":  "ETF Sectoral",
    "nifty etf":   "ETF Nifty",
    "index etf":   "ETF Nifty",
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

    # 6. Raw symbol fallback — if query looks like a ticker, let the analyse
    #    endpoint try NSE then BSE directly (supports any BSE/NSE symbol not
    #    in the registry, including BSE numeric codes like "500325")
    import re
    clean = query.strip().upper().replace(" ", "")
    if re.match(r'^[A-Z0-9&.\-]{1,20}$', clean):
        return {
            "type": "single",
            "results": [{"symbol": clean, "name": clean, "sector": "", "indices": []}],
            "match_type": "direct",
        }

    return {"type": "none", "results": [], "match_type": "none"}


def get_all_symbols() -> list[str]:
    return [r["symbol"] for r in REGISTRY]


def get_sector_list() -> list[str]:
    return sorted(_by_sector.keys())


def get_index_list() -> list[str]:
    return sorted(_by_index.keys())
