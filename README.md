# Artha (अर्थ) — Indian Stock Market Analyser

> *Artha* means **wealth** and **meaning** in Sanskrit.

A local-first stock analysis tool for Indian markets. No cloud, no API keys, no subscriptions — runs entirely on your machine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.129-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What it does

- **Analyse any NSE stock** — live price, RSI, MACD, moving averages, Bollinger Bands, expected price range, and short/medium/long-term outlook
- **Smart search** — type a company name, abbreviation, sector, or index and it figures out what you mean (fuzzy matching included)
- **News sentiment** — fetches recent headlines from Google News and scores them automatically
- **Nifty 50 Scanner** — one click scans all 50 stocks and surfaces the ones most likely to move, sorted by expected move %
- **AI Chat** — ask questions about any stock in plain English, powered by a local LLM (Ollama) — nothing leaves your machine

---

## Screenshots

| Analyse Stock | Nifty 50 Scanner | Chat |
|---|---|---|
| Live quote, 30-day chart, technical indicators, outlook cards, news | Table of all 50 stocks ranked by expected move, ≥3% movers highlighted | Conversational Q&A with full live stock data as context |

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) (only needed for the Chat feature)

### 2. Clone & install

```bash
git clone https://github.com/your-username/artha-stock-analyser.git
cd artha-stock-analyser

pip install -r requirements.txt
```

### 3. Set up environment (optional)

```bash
cp .env.example .env
# Edit .env if you want to add optional API keys
```

### 4. Start the server

```bash
uvicorn api:app --reload
```

Open **http://localhost:8000** in your browser.

### 5. (Optional) Enable Chat with local AI

```bash
# Install Ollama
brew install ollama          # macOS
# or download from https://ollama.com for other platforms

# Start the service and pull the model (~2GB)
brew services start ollama
ollama pull llama3.2
```

Once the model is downloaded, the Chat tab is ready to use.

---

## How to Search

The search bar accepts natural language — you don't need to know the exact NSE symbol:

| What you type | What happens |
|---|---|
| `RELIANCE` | Exact symbol match |
| `hdfc bank`, `sbi`, `ril` | Common alias → resolves to correct symbol |
| `IT stocks`, `pharma sector` | Returns all stocks in that sector |
| `nifty bank`, `bank nifty` | Returns all Nifty Bank constituent stocks |
| `reliAnc`, `infossys` | Fuzzy match → suggests closest results |

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                    Browser UI                        │
│         (static/index.html — vanilla JS)             │
└────────────────────┬────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────┐
│               FastAPI Server (api.py)                │
│  /api/analyse  /api/scan  /api/search  /api/chat    │
└──────┬──────────────┬──────────────┬────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────────────┐
│ data_fetcher│ │ technical_ │ │   news_fetcher.py  │
│    .py      │ │ analysis.py│ │ Google News + NLP  │
│ Yahoo Fin.  │ │RSI MACD SMA│ │   TextBlob         │
└─────────────┘ └────────────┘ └────────────────────┘
       │
┌──────▼──────────────────┐    ┌───────────────────┐
│  stock_registry.py      │    │  Ollama (local)   │
│  ~120 stocks, aliases,  │    │  llama3.2 model   │
│  sectors, fuzzy search  │    │  Chat endpoint    │
└─────────────────────────┘    └───────────────────┘
```

### Technical Indicators

| Indicator | What it measures |
|---|---|
| **RSI (14-day)** | Overbought (>70) or oversold (<30) conditions |
| **MACD** | Momentum — difference between 12-day and 26-day exponential averages |
| **SMA 5 / SMA 20** | Short vs medium-term trend direction |
| **Bollinger Bands** | Price volatility envelope (used in prediction) |

### Price Prediction Formula

The expected close range combines three signals:

```
combined_signal = (SMA_trend × 0.4) + (RSI_signal × 0.3) + (news_sentiment × 0.3)
expected_move   = combined_signal × volatility × current_price × 2
```

This is a heuristic estimate, not a trained ML model. See [PRODUCT_SPEC.md](PRODUCT_SPEC.md) for full details.

---

## Project Structure

```
artha-stock-analyser/
├── api.py                  # FastAPI server — all endpoints
├── data_fetcher.py         # Yahoo Finance data fetching
├── technical_analysis.py   # RSI, MACD, SMA, price prediction, outlook
├── news_fetcher.py         # Google News RSS + TextBlob sentiment
├── stock_registry.py       # ~120 stock registry, fuzzy resolver
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── PRODUCT_SPEC.md         # Detailed product specification
└── static/
    └── index.html          # Entire frontend (HTML + CSS + JS)
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web UI |
| `/api/analyse/{symbol}` | GET | Full analysis for a stock |
| `/api/search?q=...` | GET | Resolve any text to a stock or list |
| `/api/scan` | GET | Scan Nifty 50 for top movers |
| `/api/chat` | POST | Chat with local LLM about a stock |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Stock data | yfinance (Yahoo Finance) |
| Technical analysis | pandas, numpy, scikit-learn |
| News | Google News RSS + feedparser |
| Sentiment | TextBlob |
| Fuzzy search | rapidfuzz |
| Local LLM | Ollama + llama3.2 |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Charts | Chart.js |

---

## Disclaimer

Artha is a personal research and learning tool. The price predictions and outlooks it generates are based on simple technical heuristics — they are **not** financial advice. Always do your own research before making any investment decisions.

---

## License

MIT — free to use, modify, and share.
