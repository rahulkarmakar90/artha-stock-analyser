import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class StockDataFetcher:
    def __init__(self):
        self.cache = {}
    
    def get_stock_data_with_exchange(self, symbol: str, period: str = "1mo", interval: str = "1d"):
        """Fetch historical stock data and return (DataFrame, exchange_str).
        Auto-detects exchange: tries NSE first, falls back to BSE."""
        if symbol.endswith('.NS'):
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            return df, "NSE"
        if symbol.endswith('.BO'):
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            return df, "BSE"
        # Auto-detect: try NSE first
        df = yf.Ticker(f"{symbol}.NS").history(period=period, interval=interval)
        if len(df) > 0:
            return df, "NSE"
        # Fallback to BSE
        df = yf.Ticker(f"{symbol}.BO").history(period=period, interval=interval)
        if len(df) > 0:
            return df, "BSE"
        return pd.DataFrame(), "UNKNOWN"

    def get_stock_data(self, symbol: str, period: str = "1mo", interval: str = "1d"):
        """Fetch historical stock data"""
        df, _ = self.get_stock_data_with_exchange(symbol, period, interval)
        return df
    
    def get_realtime_quote(self, symbol: str, exchange: str = "NSE"):
        """Get current price and basic info"""
        if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
            suffix = ".BO" if exchange == "BSE" else ".NS"
            symbol = f"{symbol}{suffix}"

        ticker = yf.Ticker(symbol)
        info = ticker.info

        # yfinance .info is empty for many BSE numeric-code tickers;
        # fall back to the last row of recent history for price fields.
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        if current is None:
            hist = ticker.history(period="2d")
            if len(hist) > 0:
                last = hist.iloc[-1]
                return {
                    "symbol": symbol,
                    "current_price": float(last["Close"]),
                    "day_high": float(last["High"]),
                    "day_low": float(last["Low"]),
                    "open": float(last["Open"]),
                    "previous_close": float(hist.iloc[-2]["Close"]) if len(hist) > 1 else None,
                    "volume": int(last["Volume"]) if last["Volume"] else None,
                    "market_cap": None,
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                }

        return {
            "symbol": symbol,
            "current_price": current,
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "open": info.get("open"),
            "previous_close": info.get("previousClose"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }
    
    def get_intraday_data(self, symbol: str):
        """Get intraday data for candlestick charts"""
        if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        # Get 5-minute interval data for last 5 days
        data = ticker.history(period="5d", interval="5m")
        return data