"""
Comprehensive test script for the options trading platform.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from optionslib.technical_analysis import TechnicalAnalyzer
from optionslib.portfolio import Portfolio
from optionslib.market_data import MarketDataFetcher

def generate_sample_data(n_days=100):
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days)
    prices = 100 * (1 + np.random.randn(n_days).cumsum() * 0.02)
    volumes = np.random.randint(1000, 10000, n_days)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_days) * 0.01),
        'high': prices * (1 + abs(np.random.randn(n_days) * 0.02)),
        'low': prices * (1 - abs(np.random.randn(n_days) * 0.02)),
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return df

def test_technical_analysis():
    """Test technical analysis functionality."""
    print("\n=== Testing Technical Analysis ===")
    
    # Generate sample data
    data = generate_sample_data()
    analyzer = TechnicalAnalyzer(data)
    
    # Test moving averages
    print("\nTesting Moving Averages:")
    sma = analyzer.calculate_sma(20)
    ema = analyzer.calculate_ema(20)
    print(f"SMA (last 5 values): {sma.tail().values}")
    print(f"EMA (last 5 values): {ema.tail().values}")
    
    # Test Bollinger Bands
    print("\nTesting Bollinger Bands:")
    upper, middle, lower = analyzer.calculate_bollinger_bands(20)
    print(f"Bollinger Bands (last value) - Upper: {upper.iloc[-1]:.2f}, Middle: {middle.iloc[-1]:.2f}, Lower: {lower.iloc[-1]:.2f}")
    
    # Test RSI
    print("\nTesting RSI:")
    rsi = analyzer.calculate_rsi(14)
    print(f"RSI (last 5 values): {rsi.tail().values}")
    
    # Test MACD
    print("\nTesting MACD:")
    macd, signal, hist = analyzer.calculate_macd()
    print(f"MACD (last value): {macd.iloc[-1]:.2f}")
    print(f"Signal (last value): {signal.iloc[-1]:.2f}")
    
    # Test Stochastic Oscillator
    print("\nTesting Stochastic Oscillator:")
    k, d = analyzer.calculate_stochastic_oscillator()
    print(f"Stochastic %K (last value): {k.iloc[-1]:.2f}")
    print(f"Stochastic %D (last value): {d.iloc[-1]:.2f}")
    
    # Test ATR
    print("\nTesting ATR:")
    atr = analyzer.calculate_atr(14)
    print(f"ATR (last value): {atr.iloc[-1]:.2f}")

def test_portfolio():
    """Test portfolio management functionality."""
    print("\n=== Testing Portfolio Management ===")
    
    portfolio = Portfolio()
    
    # Test adding positions
    print("\nTesting Position Management:")
    portfolio.add_position("AAPL", 100, "black_scholes", {"volatility": 0.2, "risk_free_rate": 0.03})
    portfolio.add_position("GOOGL", 50, "binomial", {"steps": 100, "volatility": 0.25})
    
    positions = portfolio.get_all_positions()
    print(f"Number of positions: {len(positions)}")
    print(f"Positions: {positions}")
    
    # Test portfolio value calculation
    market_data = {
        "AAPL": 150.0,
        "GOOGL": 2800.0
    }
    
    value = portfolio.calculate_portfolio_value(market_data)
    print(f"\nPortfolio Value: ${value:,.2f}")
    
    # Test risk calculation
    risk = portfolio.calculate_portfolio_risk(market_data)
    print(f"Portfolio VaR: ${risk:,.2f}")
    
    # Test position removal
    portfolio.remove_position("AAPL")
    print(f"\nPositions after removal: {portfolio.get_all_positions()}")

def main():
    """Run all tests."""
    try:
        test_technical_analysis()
        test_portfolio()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    main()
