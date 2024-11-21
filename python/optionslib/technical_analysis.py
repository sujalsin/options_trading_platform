"""
Technical Analysis Module for Options Trading Platform.
Provides advanced technical indicators and analysis tools using TA-Lib.
"""

import numpy as np
import pandas as pd
import talib

class TechnicalAnalyzer:
    """Provides technical analysis capabilities for market data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the technical analyzer with market data.
        
        Args:
            data: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
        """
        self.data = data
        self.open = data['open'].values
        self.high = data['high'].values
        self.low = data['low'].values
        self.close = data['close'].values
        self.volume = data['volume'].values
    
    def calculate_sma(self, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average."""
        sma = pd.Series(talib.SMA(self.close, timeperiod=period), index=self.data.index)
        return sma
    
    def calculate_ema(self, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average."""
        ema = pd.Series(talib.EMA(self.close, timeperiod=period), index=self.data.index)
        return ema
    
    def calculate_bollinger_bands(self, period: int = 20, num_std: float = 2.0) -> tuple:
        """Calculate Bollinger Bands."""
        upper, middle, lower = talib.BBANDS(
            self.close,
            timeperiod=period,
            nbdevup=num_std,
            nbdevdn=num_std,
            matype=0
        )
        return (
            pd.Series(upper, index=self.data.index),
            pd.Series(middle, index=self.data.index),
            pd.Series(lower, index=self.data.index)
        )
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        rsi = pd.Series(talib.RSI(self.close, timeperiod=period), index=self.data.index)
        return rsi
    
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """Calculate MACD (Moving Average Convergence/Divergence)."""
        macd, signal, hist = talib.MACD(
            self.close,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        return (
            pd.Series(macd, index=self.data.index),
            pd.Series(signal, index=self.data.index),
            pd.Series(hist, index=self.data.index)
        )
    
    def calculate_stochastic_oscillator(self, fastk_period: int = 14, slowk_period: int = 3,
                                      slowd_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator."""
        slowk, slowd = talib.STOCH(
            self.high,
            self.low,
            self.close,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0
        )
        return pd.Series(slowk, index=self.data.index), pd.Series(slowd, index=self.data.index)
    
    def calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        atr = pd.Series(talib.ATR(self.high, self.low, self.close, timeperiod=period),
                       index=self.data.index)
        return atr
    
    def identify_candlestick_patterns(self) -> dict:
        """Identify candlestick patterns."""
        patterns = {}
        
        # Single candlestick patterns
        patterns['doji'] = talib.CDLDOJI(self.open, self.high, self.low, self.close)
        patterns['hammer'] = talib.CDLHAMMER(self.open, self.high, self.low, self.close)
        patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(self.open, self.high, self.low, self.close)
        
        # Double candlestick patterns
        patterns['engulfing'] = talib.CDLENGULFING(self.open, self.high, self.low, self.close)
        patterns['harami'] = talib.CDLHARAMI(self.open, self.high, self.low, self.close)
        
        # Triple candlestick patterns
        patterns['morning_star'] = talib.CDLMORNINGSTAR(self.open, self.high, self.low, self.close)
        patterns['evening_star'] = talib.CDLEVENINGSTAR(self.open, self.high, self.low, self.close)
        
        return {name: pd.Series(pattern, index=self.data.index) 
                for name, pattern in patterns.items()}
