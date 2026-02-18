import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class StockDataFetcher:
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol: str, period: str = "1mo", interval: str = "1d"):
        """Fetch historical stock data"""
        # Add .NS suffix for NSE if not present
        if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data
    
    def get_realtime_quote(self, symbol: str):
        """Get current price and basic info"""
        if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol,
            "current_price": info.get("currentPrice"),
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