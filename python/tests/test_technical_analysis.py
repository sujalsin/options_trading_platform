"""
Test suite for technical analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from optionslib.technical_analysis import TechnicalAnalyzer

@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Generate sample OHLCV data
    data = {
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }
    
    # Ensure high is highest and low is lowest
    for i in range(len(data['high'])):
        values = [data['open'][i], data['high'][i], data['low'][i], data['close'][i]]
        data['high'][i] = max(values)
        data['low'][i] = min(values)
    
    return pd.DataFrame(data, index=dates)

def test_moving_averages(sample_data):
    """Test moving average calculations."""
    analyzer = TechnicalAnalyzer()
    mas = analyzer.calculate_moving_averages(sample_data['close'])
    
    assert isinstance(mas, dict)
    assert 'MA_5' in mas
    assert 'MA_200' in mas
    assert len(mas['MA_5'].dropna()) == len(sample_data) - 4  # First 4 values are NaN
    assert isinstance(mas['MA_5'], pd.Series)

def test_bollinger_bands(sample_data):
    """Test Bollinger Bands calculation."""
    analyzer = TechnicalAnalyzer()
    upper, middle, lower = analyzer.calculate_bollinger_bands(sample_data['close'])
    
    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)
    assert len(upper) == len(sample_data)
    assert all(upper >= middle)
    assert all(middle >= lower)

def test_rsi(sample_data):
    """Test RSI calculation."""
    analyzer = TechnicalAnalyzer()
    rsi = analyzer.calculate_rsi(sample_data['close'])
    
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(sample_data)
    assert all((rsi >= 0) & (rsi <= 100))

def test_macd(sample_data):
    """Test MACD calculation."""
    analyzer = TechnicalAnalyzer()
    macd, signal, hist = analyzer.calculate_macd(sample_data['close'])
    
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert isinstance(hist, pd.Series)
    assert len(macd) == len(sample_data)
    assert len(signal) == len(sample_data)
    assert len(hist) == len(sample_data)

def test_stochastic(sample_data):
    """Test Stochastic Oscillator calculation."""
    analyzer = TechnicalAnalyzer()
    k, d = analyzer.calculate_stochastic(sample_data['high'],
                                       sample_data['low'],
                                       sample_data['close'])
    
    assert isinstance(k, pd.Series)
    assert isinstance(d, pd.Series)
    assert len(k) == len(sample_data)
    assert len(d) == len(sample_data)
    assert all((k >= 0) & (k <= 100))
    assert all((d >= 0) & (d <= 100))

def test_atr(sample_data):
    """Test Average True Range calculation."""
    analyzer = TechnicalAnalyzer()
    atr = analyzer.calculate_atr(sample_data['high'],
                               sample_data['low'],
                               sample_data['close'])
    
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(sample_data)
    assert all(atr >= 0)  # ATR should always be positive

def test_candlestick_patterns(sample_data):
    """Test candlestick pattern identification."""
    analyzer = TechnicalAnalyzer()
    patterns = analyzer.identify_candlestick_patterns(sample_data['open'],
                                                    sample_data['high'],
                                                    sample_data['low'],
                                                    sample_data['close'])
    
    assert isinstance(patterns, dict)
    assert 'doji' in patterns
    assert 'hammer' in patterns
    assert 'engulfing' in patterns
    assert all(isinstance(pattern, pd.Series) for pattern in patterns.values())

def test_volume_indicators(sample_data):
    """Test volume-based indicators calculation."""
    analyzer = TechnicalAnalyzer()
    indicators = analyzer.calculate_volume_indicators(sample_data['close'],
                                                   sample_data['volume'])
    
    assert isinstance(indicators, dict)
    assert 'obv' in indicators
    assert 'ad' in indicators
    assert 'mfi' in indicators
    assert all(isinstance(indicator, pd.Series) for indicator in indicators.values())

def test_momentum_analysis(sample_data):
    """Test comprehensive momentum analysis."""
    analyzer = TechnicalAnalyzer()
    results = analyzer.analyze_price_momentum(sample_data['close'])
    
    assert isinstance(results, dict)
    assert 'rsi_14' in results
    assert 'macd' in results
    assert 'macd_signal' in results
    assert all(isinstance(result, pd.Series) for result in results.values())
