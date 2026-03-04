import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
from urllib.parse import quote_plus
from pathlib import Path
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import StockDataFetcher
from technical_analysis import StockPredictor
from news_fetcher import NewsAnalyzer
from stock_registry import resolve, get_sector_list, get_index_list
from ml_analysis import MLAnalyzer, WalkForwardBacktester
import feedparser
from textblob import TextBlob

app = FastAPI(title="Artha — Indian Stock Analyser")

fetcher      = StockDataFetcher()
predictor    = StockPredictor()
news_analyzer = NewsAnalyzer()
ml_analyzer  = MLAnalyzer()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "WIPRO", "ULTRACEMCO", "NTPC", "POWERGRID",
    "BAJFINANCE", "BAJAJFINSV", "ONGC", "TECHM", "HCLTECH",
    "TATASTEEL", "ADANIENT", "ADANIPORTS", "JSWSTEEL", "COALINDIA",
    "NESTLEIND", "BRITANNIA", "HEROMOTOCO", "BAJAJ-AUTO", "EICHERMOT",
    "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP", "HINDALCO",
    "VEDL", "TATACONSUM", "UPL", "GRASIM", "M&M",
    "INDUSINDBK", "BPCL", "IOC", "SHREECEM", "LTIM",
]


_NEWS_FEEDS = [
    # Google News — 3 query variants
    lambda sym, name: f"https://news.google.com/rss/search?q={quote_plus(f'{sym} stock India')}&hl=en-IN&gl=IN",
    lambda sym, name: f"https://news.google.com/rss/search?q={quote_plus(name or sym)}&hl=en-IN&gl=IN",
    lambda sym, name: f"https://news.google.com/rss/search?q={quote_plus(f'{sym} NSE')}&hl=en-IN&gl=IN",
    # Economic Times
    lambda sym, name: f"https://economictimes.indiatimes.com/rssfeeds/{quote_plus(sym)}.cms",
    # Moneycontrol
    lambda sym, name: f"https://www.moneycontrol.com/rss/results.xml?q={quote_plus(sym)}",
]


import ssl as _ssl
import urllib.request as _urllib_req

_UNVERIFIED_SSL = _ssl.create_default_context()
_UNVERIFIED_SSL.check_hostname = False
_UNVERIFIED_SSL.verify_mode = _ssl.CERT_NONE


def _fetch_feed(url: str):
    """Fetch RSS URL with a browser UA and unverified SSL, return feedparser feed."""
    req = _urllib_req.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
    )
    try:
        resp = _urllib_req.urlopen(req, context=_UNVERIFIED_SSL, timeout=8)
        return feedparser.parse(resp.read())
    except Exception:
        return feedparser.parse(url)  # fallback to default (may work on some systems)


def _get_sentiment(symbol: str, company_name: str = "") -> tuple[float, str, list]:
    """Fetch news from multiple RSS sources, keep only last 7 days, return (score, label, articles)."""
    import time as _time
    cutoff = _time.time() - 7 * 24 * 3600  # epoch seconds 7 days ago

    seen = set()
    articles = []
    for feed_fn in _NEWS_FEEDS:
        if len(articles) >= 8:
            break
        try:
            url = feed_fn(symbol, company_name)
            feed = _fetch_feed(url)
            for entry in feed.entries[:15]:
                # Date filter: skip articles older than 7 days when date is available
                parsed_time = getattr(entry, "published_parsed", None)
                if parsed_time is not None:
                    entry_epoch = _time.mktime(parsed_time)
                    if entry_epoch < cutoff:
                        continue

                key = entry.title.lower()[:60]
                if key in seen:
                    continue
                seen.add(key)
                polarity = TextBlob(entry.title).sentiment.polarity
                sentiment = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", ""),
                    "source": entry.get("source", {}).get("title", "Unknown"),
                    "sentiment": sentiment,
                    "polarity": round(polarity, 3),
                })
        except Exception:
            continue

    score = sum(a["polarity"] for a in articles) / len(articles) if articles else 0.0
    label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
    return score, label, articles


def _safe_float(val):
    """Convert numpy floats / None to plain Python float."""
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def _sanitise_ml_result(obj):
    """Recursively convert numpy scalars to Python native types for JSON serialisation."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _sanitise_ml_result(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_ml_result(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def _compute_consensus(outlook: dict, ml_result: dict) -> dict:
    """Compare heuristic rule-based outlook vs ML ensemble direction."""
    if not outlook or not ml_result or "error" in ml_result:
        return {}

    rule_str = outlook.get("short_term_1_3_days", {}).get("outlook", "").lower()
    rule_dir = 1 if "bull" in rule_str else -1 if "bear" in rule_str else 0

    ens = ml_result.get("ensemble", {})
    ml_dir = ens.get("direction", 0)

    agree = rule_dir == ml_dir
    dir_labels = {1: "Bullish", -1: "Bearish", 0: "Neutral"}

    return {
        "rule_signal": dir_labels.get(rule_dir, "Neutral"),
        "ml_signal":   dir_labels.get(ml_dir, "Neutral"),
        "agree":       agree,
        "message": (
            f"Both rule-based ({dir_labels[rule_dir]}) and ML ({dir_labels[ml_dir]}) agree."
            if agree else
            f"Conflict: rules say {dir_labels[rule_dir]}, ML says {dir_labels[ml_dir]}. "
            "Exercise caution — signals diverge."
        ),
        "confidence": _safe_float(ens.get("confidence", 0)),
    }


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/analyse/{symbol}")
def analyse(symbol: str):
    symbol = symbol.upper().strip()
    try:
        # Fetch 5 years of data (superset of 3mo — no second network call needed)
        df_5y, exchange = fetcher.get_stock_data_with_exchange(symbol, period="5y", interval="1d")
        if df_5y is None or len(df_5y) < 5:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        # Use last 65 trading days (~3 months) for heuristic indicators & sparkline
        df = df_5y.tail(65)

        # Realtime quote
        try:
            quote = fetcher.get_realtime_quote(symbol, exchange=exchange)
            quote = {k: _safe_float(v) if isinstance(v, (int, float)) else v
                     for k, v in quote.items()}
        except Exception:
            quote = {}

        # Fundamentals
        try:
            fundamentals = fetcher.get_fundamentals(symbol, exchange=exchange)
            fundamentals = {k: _safe_float(v) if isinstance(v, float) else v
                            for k, v in fundamentals.items()}
            company_name = fundamentals.get("long_name", "")
        except Exception:
            fundamentals = {}
            company_name = ""

        # Price history for chart — last 6 months (~130 trading days), full OHLCV
        history = []
        for dt, row in df_5y.tail(130).iterrows():
            history.append({
                "date":   str(dt)[:10],
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]) if row["Volume"] else 0,
            })

        # News sentiment
        sentiment_score, sentiment_label, articles = _get_sentiment(symbol, company_name)

        # Heuristic technical prediction (uses 3mo slice)
        pred = predictor.predict_price_range(df, news_sentiment=sentiment_score)
        outlook = predictor.get_outlook(df, news_sentiment=sentiment_score)

        if pred:
            pred["current_price"] = _safe_float(pred["current_price"])
            pred["expected_close_range"]["low"] = _safe_float(pred["expected_close_range"]["low"])
            pred["expected_close_range"]["high"] = _safe_float(pred["expected_close_range"]["high"])
            pred["expected_tomorrow_open_range"]["low"] = _safe_float(pred["expected_tomorrow_open_range"]["low"])
            pred["expected_tomorrow_open_range"]["high"] = _safe_float(pred["expected_tomorrow_open_range"]["high"])
            for k in pred["technical_indicators"]:
                pred["technical_indicators"][k] = _safe_float(pred["technical_indicators"][k])

        if outlook:
            for timeframe in outlook.values():
                timeframe["confidence"] = _safe_float(timeframe["confidence"])

        # ML predictions (uses full 5y data)
        ml_result = None
        try:
            ml_result = ml_analyzer.run_all(df_5y)
            ml_result = _sanitise_ml_result(ml_result)
        except Exception as e:
            ml_result = {"error": str(e)}

        consensus = _compute_consensus(outlook, ml_result)

        return {
            "symbol": symbol,
            "exchange": exchange,
            "quote": quote,
            "history": history,
            "prediction": pred,
            "outlook": outlook,
            "news": {
                "overall_sentiment": round(sentiment_score, 3),
                "sentiment_label": sentiment_label,
                "articles": articles[:8],
            },
            "ml": ml_result,
            "fundamentals": fundamentals,
            "consensus": consensus,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scan")
def scan():
    results = []
    for symbol in NIFTY50:
        try:
            df = fetcher.get_stock_data(symbol, period="3mo", interval="1d")
            if df is None or len(df) < 20:
                continue

            sentiment_score, sentiment_label, _ = _get_sentiment(symbol)

            pred = predictor.predict_price_range(df, news_sentiment=sentiment_score)
            outlook = predictor.get_outlook(df, news_sentiment=sentiment_score)

            if not pred or not outlook:
                continue

            current = _safe_float(pred["current_price"])
            low = _safe_float(pred["expected_close_range"]["low"])
            high = _safe_float(pred["expected_close_range"]["high"])
            rsi = _safe_float(pred["technical_indicators"]["RSI"])

            pct_high = ((high - current) / current) * 100 if current else 0
            pct_low = ((low - current) / current) * 100 if current else 0
            max_move = max(abs(pct_high), abs(pct_low))

            results.append({
                "symbol": symbol,
                "current_price": round(current, 2) if current else None,
                "expected_low": round(low, 2) if low else None,
                "expected_high": round(high, 2) if high else None,
                "pct_low": round(pct_low, 2),
                "pct_high": round(pct_high, 2),
                "max_move_pct": round(max_move, 2),
                "rsi": round(rsi, 1) if rsi else None,
                "short_outlook": outlook["short_term_1_3_days"]["outlook"],
                "confidence": round(_safe_float(outlook["short_term_1_3_days"]["confidence"]), 1),
                "news_sentiment": sentiment_label,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["max_move_pct"], reverse=True)
    return {"stocks": results, "total": len(results)}


# ── Backtest endpoint ─────────────────────────────────────────────────────

@app.get("/api/backtest/{symbol}")
def backtest(symbol: str):
    symbol = symbol.upper().strip()
    try:
        df_5y, exchange = fetcher.get_stock_data_with_exchange(symbol, period="5y", interval="1d")
        if df_5y is None or len(df_5y) < 300:
            raise HTTPException(status_code=404,
                detail=f"Insufficient data for {symbol} — need ~5 years of history")
        result = WalkForwardBacktester().run(df_5y, ml_analyzer)
        result = _sanitise_ml_result(result)
        result["symbol"]   = symbol
        result["exchange"] = exchange
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Search endpoint ────────────────────────────────────────────────────────

@app.get("/api/search")
def search(q: str = Query(..., min_length=1)):
    result = resolve(q)
    return {
        "type": result["type"],
        "match_type": result.get("match_type", "none"),
        "group": result.get("group", ""),
        "results": result["results"],
    }


# ── Chat endpoint ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    context_symbol: str = ""
    history: list[dict] = []


def _build_stock_context(symbol: str) -> str:
    """Fetch quick analysis data and format as plain text for the LLM prompt."""
    try:
        df, exchange = fetcher.get_stock_data_with_exchange(symbol, period="3mo", interval="1d")
        if df is None or len(df) < 5:
            return f"No data available for {symbol}."

        quote = {}
        try:
            quote = fetcher.get_realtime_quote(symbol, exchange=exchange)
        except Exception:
            pass

        sentiment_score, sentiment_label, articles = _get_sentiment(symbol)
        pred = predictor.predict_price_range(df, news_sentiment=sentiment_score)
        outlook = predictor.get_outlook(df, news_sentiment=sentiment_score)

        ti = pred.get("technical_indicators", {}) if pred else {}
        ecr = pred.get("expected_close_range", {}) if pred else {}
        st = outlook.get("short_term_1_3_days", {}) if outlook else {}
        mt = outlook.get("medium_term_1_2_weeks", {}) if outlook else {}
        lt = outlook.get("long_term_1_month", {}) if outlook else {}

        headlines = " | ".join(a["title"] for a in articles[:5]) if articles else "No recent news."

        ctx = f"""
Stock: {symbol}
Date: {date.today()}
Current Price: ₹{_safe_float(quote.get('current_price'))}
Day High: ₹{_safe_float(quote.get('day_high'))}  Day Low: ₹{_safe_float(quote.get('day_low'))}
Previous Close: ₹{_safe_float(quote.get('previous_close'))}
52W High: ₹{_safe_float(quote.get('52_week_high'))}  52W Low: ₹{_safe_float(quote.get('52_week_low'))}
Volume: {quote.get('volume')}
RSI (14): {_safe_float(ti.get('RSI'))}
MACD: {_safe_float(ti.get('MACD'))}
SMA 5: ₹{_safe_float(ti.get('SMA_5'))}  SMA 20: ₹{_safe_float(ti.get('SMA_20'))}
Expected Close Range: ₹{_safe_float(ecr.get('low'))} – ₹{_safe_float(ecr.get('high'))}
Short-term Outlook (1–3 days): {st.get('outlook', 'N/A')} ({st.get('confidence', 0):.1f}% confidence)
Medium-term Outlook (1–2 weeks): {mt.get('outlook', 'N/A')}
Long-term Outlook (1 month): {lt.get('outlook', 'N/A')}
News Sentiment: {sentiment_label}
Recent Headlines: {headlines}
""".strip()
        return ctx
    except Exception as e:
        return f"Could not fetch data for {symbol}: {e}"


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        import ollama as _ollama
    except ImportError:
        raise HTTPException(status_code=503, detail="ollama Python package not installed. Run: pip3 install ollama")

    # Resolve symbol from message if not provided
    symbol = req.context_symbol.upper().strip() if req.context_symbol else ""
    if not symbol:
        result = resolve(req.message)
        if result["type"] == "single":
            symbol = result["results"][0]["symbol"]

    # Build system prompt
    stock_ctx = _build_stock_context(symbol) if symbol else "No specific stock selected."
    system_prompt = f"""You are Artha, an expert Indian stock market analyst assistant.
You help users understand technical indicators, news sentiment, price trends, and investment insights for NSE-listed stocks.
Be concise, direct, and data-driven. Always mention the specific numbers from the data when relevant.
If asked about predictions, caveat that these are technical estimates, not financial advice.

Today's date: {date.today()}

{f"Current stock context:{chr(10)}{stock_ctx}" if symbol else "No stock context loaded — ask the user to search for a stock first."}
"""

    # Build message list
    messages = [{"role": "system", "content": system_prompt}]
    for h in req.history[-10:]:  # last 10 turns for context window
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    try:
        response = _ollama.chat(model="llama3.2", messages=messages)
        reply = response["message"]["content"]
        return {"reply": reply, "symbol_used": symbol}
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "refused" in err.lower():
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Start it with: ollama serve"
            )
        raise HTTPException(status_code=500, detail=err)


# ── Streaming chat endpoint ────────────────────────────────────────────────

@app.get("/api/chat/stream")
async def chat_stream(payload: str = Query(...)):
    req_data = json.loads(payload)
    msg            = req_data.get("message", "")
    context_symbol = req_data.get("context_symbol", "")
    history        = req_data.get("history", [])

    symbol = context_symbol.upper().strip() if context_symbol else ""
    if not symbol:
        result = resolve(msg)
        if result["type"] == "single":
            symbol = result["results"][0]["symbol"]

    stock_ctx = _build_stock_context(symbol) if symbol else "No specific stock selected."
    system_prompt = f"""You are Artha, an expert Indian stock market analyst assistant.
You help users understand technical indicators, news sentiment, price trends, and investment insights for NSE-listed stocks.
Be concise, direct, and data-driven. Always mention the specific numbers from the data when relevant.
If asked about predictions, caveat that these are technical estimates, not financial advice.

Today's date: {date.today()}

{f"Current stock context:{chr(10)}{stock_ctx}" if symbol else "No stock context loaded — ask the user to search for a stock first."}
"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-10:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": msg})

    symbol_used = symbol

    async def event_generator():
        try:
            import ollama as _ollama
            client = _ollama.AsyncClient()
            async for chunk in await client.chat(
                model="llama3.2", messages=messages, stream=True
            ):
                token = chunk["message"]["content"]
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'symbol_used': symbol_used})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
