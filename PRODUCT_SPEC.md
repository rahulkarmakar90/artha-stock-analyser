# Artha (अर्थ) — Product Specification
**Version:** 1.0 | **Date:** February 2026 | **Type:** Local-first web application

---

## What is Artha?

Artha is a personal Indian stock market analysis tool that runs entirely on your laptop. It pulls live stock data, computes technical indicators, reads news sentiment, and lets you chat with a local AI about any stock — all without sending your data to any cloud service or paying for an API.

The name *Artha* (अर्थ) means *wealth* and *meaning* in Sanskrit.

---

## How to Run It

```
cd backend/
uvicorn api:app --reload     # start the server
open http://localhost:8000   # open in browser
```

---

## Product Layers

### Layer 1 — Data (What the system knows)

**Source: Yahoo Finance via `yfinance`**
Every analysis starts here. When you search for a stock, the system fetches:
- 3 months of daily closing prices (used to calculate all indicators)
- Live quote: current price, day high/low, previous close, volume, market cap, 52-week range

**Source: Google News RSS**
The system searches Google News for the stock name and pulls up to 10 recent headlines. No API key required — it uses the free RSS feed.

**Stock Registry (`stock_registry.py`)**
A hardcoded lookup table of ~120 NSE-listed stocks with their full company name, sector, and index membership. This powers the search feature. No network calls — it's pure local data.

---

### Layer 2 — Analysis (What the system computes)

All calculations run locally in Python using `pandas`, `numpy`, and `scikit-learn`.

**Technical Indicators** (`technical_analysis.py`)

| Indicator | What it is | How it's used |
|---|---|---|
| RSI (14-day) | Relative Strength Index — measures if a stock is overbought or oversold on a 0–100 scale | Below 30 = oversold (green), above 70 = overbought (red) |
| MACD | Moving Average Convergence Divergence — measures momentum by comparing 12-day vs 26-day exponential averages | Positive = bullish momentum, negative = bearish |
| SMA 5 / SMA 20 | Simple Moving Averages over 5 and 20 days | SMA5 above SMA20 = short-term uptrend |
| Bollinger Bands | Price channel based on 20-day average ± 2 standard deviations | Used internally to measure volatility |

**Sentiment Score** (`news_fetcher.py`)
Each news headline is scored using TextBlob (a simple NLP library) on a scale of −1 (very negative) to +1 (very positive). The scores are averaged into a single number that feeds into the price prediction.

**Price Prediction** (`technical_analysis.py → predict_price_range`)
Combines three signals to estimate where the stock might close tomorrow:
- SMA trend weight: 40%
- RSI signal weight: 30%
- News sentiment weight: 30%

The output is a low–high range, not a single number, to reflect inherent uncertainty. This is a simplified statistical estimate, not a machine learning model.

**Outlook** (`technical_analysis.py → get_outlook`)
Scores three timeframes — short (1–3 days), medium (1–2 weeks), long (1 month) — and maps each to one of five labels: Strongly Bullish, Bullish, Neutral, Bearish, Strongly Bearish. A confidence percentage is also computed.

---

### Layer 3 — Search (How you find stocks)

The search bar accepts plain English. The system resolves your query in priority order:

1. **Exact symbol** — "RELIANCE" → direct match
2. **Alias** — "sbi", "ril", "hdfc bank", "infosys" → mapped to correct symbol
3. **Index** — "nifty bank", "nifty it" → returns all stocks in that index
4. **Sector** — "banking stocks", "pharma sector" → returns all stocks in that sector
5. **Fuzzy match** — typos or partial names → ranked by similarity using `rapidfuzz`

The registry covers Nifty 50, Nifty IT, Nifty Bank, Nifty Pharma, Nifty Auto, Nifty FMCG, and ~70 additional popular stocks (Zomato, IRCTC, HAL, Adani group, etc.).

---

### Layer 4 — API (How frontend talks to backend)

A FastAPI server (`api.py`) runs locally on port 8000 and exposes four endpoints:

| Endpoint | What it does |
|---|---|
| `GET /` | Serves the web UI |
| `GET /api/analyse/{symbol}` | Full analysis for one stock: quote, history, indicators, outlook, news |
| `GET /api/search?q=...` | Resolves any text query to a stock or list of stocks |
| `GET /api/scan` | Scans all 50 Nifty 50 stocks and ranks them by expected price movement |
| `POST /api/chat` | Sends a message to the local Ollama LLM with live stock data as context |

---

### Layer 5 — Chat (How you ask questions)

**Model:** `llama3.2` (3B parameters, ~2GB, runs locally via Ollama)

When you open the Chat tab after analysing a stock, the system automatically sends the following data as hidden context to the LLM before your first message:
- Current price, day high/low, previous close, 52-week range, volume
- RSI, MACD, SMA 5, SMA 20
- Expected close range (low–high)
- Short / medium / long-term outlook with confidence scores
- News sentiment label and up to 5 recent headlines

The LLM then answers your questions grounded in that live data. The last 10 turns of conversation are also included so it remembers the thread.

If you mention a stock in your chat message without setting a context (e.g., "What about TCS?"), the system automatically resolves it and fetches fresh data for that stock.

**Privacy:** All LLM inference runs on your Mac. Nothing leaves your machine.

---

### Layer 6 — UI (What you see)

A single HTML file (`static/index.html`) with no build step, no npm, no framework — just HTML, CSS, and vanilla JavaScript. Chart.js (loaded from CDN) draws the 30-day price sparkline.

**Three tabs:**

- **Analyse Stock** — Search bar with live autocomplete dropdown, then a dashboard showing the quote, price chart, indicators, outlook cards, expected range bar, and news headlines. A "Chat about [symbol] →" button sets the chat context and switches tabs.

- **Nifty 50 Scanner** — One button scans all 50 stocks (~60 seconds). Results table sorted by expected price move, with rows ≥3% highlighted in green. Clicking any symbol jumps to its full analysis.

- **Chat** — Conversational interface. Shows which stock is loaded as context. Suggested starter questions appear when the chat is empty. Typing indicator shown while Ollama is generating a response.

---

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| Backend server | FastAPI + Uvicorn | Fast, modern Python web framework |
| Stock data | yfinance (Yahoo Finance) | Free, no API key, reliable for NSE data |
| Technical analysis | pandas, numpy, scikit-learn | Standard scientific Python stack |
| News | Google News RSS + feedparser | Free, no API key required |
| Sentiment scoring | TextBlob | Lightweight NLP, no model download |
| Fuzzy search | rapidfuzz | Fast string matching for company name search |
| Local LLM | Ollama + llama3.2 | Fully offline, no API key, runs on Apple Silicon |
| Frontend | Vanilla HTML/CSS/JS | Zero build step, easy to modify |
| Charts | Chart.js (CDN) | Lightweight, no install |

---

## Limitations & Honest Caveats

- **Price predictions are estimates, not forecasts.** The model is a weighted combination of three simple signals. It does not use machine learning or historical back-testing.
- **Data is delayed.** Yahoo Finance typically provides 15-minute delayed prices for NSE stocks during market hours.
- **Sentiment scoring is basic.** TextBlob scores headline polarity but does not understand context, sarcasm, or financial jargon deeply.
- **The LLM can hallucinate.** llama3.2 is a capable small model but may occasionally make factual errors. Always verify before acting.
- **Scanner takes ~60 seconds.** It makes sequential API calls for all 50 stocks — there is no parallelism currently.
- **Not financial advice.** Artha is a research and learning tool. Do not make investment decisions based solely on its output.

---

## File Map

```
backend/
  api.py               — FastAPI server, all endpoints, chat logic
  data_fetcher.py      — Yahoo Finance data fetching
  technical_analysis.py — RSI, MACD, SMA, price prediction, outlook scoring
  news_fetcher.py      — Google News RSS fetch + TextBlob sentiment
  stock_registry.py    — ~120 stock registry, aliases, sector/index maps, fuzzy resolver
  requirements.txt     — Python dependencies
  .env                 — API keys (Gemini key stored here)
  static/
    index.html         — Entire frontend (UI, JS, CSS in one file)
```
