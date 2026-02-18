import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def calculate_technical_indicators(self, df: pd.DataFrame):
        """Calculate technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
        df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])
        
        return df
    
    def predict_price_range(self, df: pd.DataFrame, news_sentiment: float = 0):
        """Predict expected closing price range"""
        df = self.calculate_technical_indicators(df)
        df = df.dropna()
        
        if len(df) < 5:
            return None
        
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        # Calculate volatility
        volatility = df['Close'].pct_change().std()
        
        # Base prediction using technical indicators
        sma_trend = (latest['SMA_5'] - latest['SMA_20']) / latest['SMA_20']
        rsi_signal = (latest['RSI'] - 50) / 100
        
        # Combine signals
        combined_signal = (sma_trend * 0.4 + rsi_signal * 0.3 + news_sentiment * 0.3)
        
        # Expected movement (simplified)
        expected_move = combined_signal * volatility * current_price * 2
        
        # Calculate ranges
        today_close_low = current_price + expected_move - (volatility * current_price)
        today_close_high = current_price + expected_move + (volatility * current_price)
        
        tomorrow_open_low = today_close_low * 0.998
        tomorrow_open_high = today_close_high * 1.002
        
        return {
            "current_price": round(current_price, 2),
            "expected_close_range": {
                "low": round(today_close_low, 2),
                "high": round(today_close_high, 2)
            },
            "expected_tomorrow_open_range": {
                "low": round(tomorrow_open_low, 2),
                "high": round(tomorrow_open_high, 2)
            },
            "technical_indicators": {
                "RSI": round(latest['RSI'], 2),
                "MACD": round(latest['MACD'], 4),
                "SMA_5": round(latest['SMA_5'], 2),
                "SMA_20": round(latest['SMA_20'], 2)
            }
        }
    
    def get_outlook(self, df: pd.DataFrame, news_sentiment: float = 0):
        """Get bullish/bearish outlook for different timeframes"""
        df = self.calculate_technical_indicators(df)
        df = df.dropna()
        
        if len(df) < 20:
            return None
        
        latest = df.iloc[-1]
        
        # Short-term (1-3 days)
        short_term_score = 0
        if latest['SMA_5'] > latest['SMA_20']:
            short_term_score += 1
        if latest['RSI'] < 70 and latest['RSI'] > 30:
            short_term_score += 0.5
        if latest['MACD'] > latest['Signal']:
            short_term_score += 1
        short_term_score += news_sentiment
        
        # Medium-term (1-2 weeks)
        medium_term_score = short_term_score * 0.7
        price_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10]
        medium_term_score += price_trend * 10
        
        # Long-term (1 month)
        long_term_score = 0
        if len(df) >= 20:
            long_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
            long_term_score = long_trend * 5 + (1 if latest['Close'] > latest['SMA_20'] else -1)
        
        def score_to_outlook(score):
            if score > 1.5:
                return "strongly_bullish"
            elif score > 0.5:
                return "bullish"
            elif score > -0.5:
                return "neutral"
            elif score > -1.5:
                return "bearish"
            else:
                return "strongly_bearish"
        
        return {
            "short_term_1_3_days": {
                "outlook": score_to_outlook(short_term_score),
                "confidence": min(abs(short_term_score) / 3 * 100, 100)
            },
            "medium_term_1_2_weeks": {
                "outlook": score_to_outlook(medium_term_score),
                "confidence": min(abs(medium_term_score) / 3 * 100, 100)
            },
            "long_term_1_month": {
                "outlook": score_to_outlook(long_term_score),
                "confidence": min(abs(long_term_score) / 3 * 100, 100)
            }
        }