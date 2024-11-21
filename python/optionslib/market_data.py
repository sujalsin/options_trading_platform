import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

class MarketDataFetcher:
    """Fetch and process market data for options and underlying assets."""
    
    def __init__(self):
        """Initialize the market data fetcher."""
        self.cache = {}
        
    def get_option_chain(self, symbol: str) -> pd.DataFrame:
        """
        Fetch option chain data for a given symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            DataFrame containing option chain data
        """
        ticker = yf.Ticker(symbol)
        options = ticker.options
        
        all_options = []
        for date in options:
            chain = ticker.option_chain(date)
            
            # Process calls
            calls = chain.calls.copy()
            calls['option_type'] = 'call'
            
            # Process puts
            puts = chain.puts.copy()
            puts['option_type'] = 'put'
            
            # Combine and add expiration date
            combined = pd.concat([calls, puts])
            combined['expiration'] = pd.to_datetime(date)
            all_options.append(combined)
            
        return pd.concat(all_options, ignore_index=True)
    
    def get_historical_data(self,
                          symbols: Union[str, List[str]],
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical price data for one or more symbols.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval ('1d', '1h', etc.)
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
            
        all_data = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            data['Symbol'] = symbol
            all_data.append(data)
            
        return pd.concat(all_data, axis=0)
    
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate (using 3-month T-bill rate).
        """
        try:
            treasury = yf.Ticker("^IRX")
            return float(treasury.info['regularMarketPrice']) / 100
        except:
            # Default to a reasonable value if fetch fails
            return 0.02
    
    def get_implied_volatility_surface(self,
                                     symbol: str) -> pd.DataFrame:
        """
        Construct implied volatility surface from option chain data.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            DataFrame with implied volatility surface data
        """
        options_data = self.get_option_chain(symbol)
        
        # Calculate time to expiration in years
        now = datetime.now()
        options_data['time_to_expiry'] = (
            (options_data['expiration'] - pd.Timestamp(now)).dt.days / 365
        )
        
        # Calculate moneyness
        ticker = yf.Ticker(symbol)
        current_price = ticker.info['regularMarketPrice']
        options_data['moneyness'] = np.log(options_data['strike'] / current_price)
        
        # Pivot data to create surface
        surface = pd.pivot_table(
            options_data,
            values='impliedVolatility',
            index='moneyness',
            columns='time_to_expiry',
            aggfunc='mean'
        )
        
        return surface
    
    def get_correlation_matrix(self,
                             symbols: List[str],
                             lookback_days: int = 252) -> pd.DataFrame:
        """
        Calculate correlation matrix for given symbols.
        
        Args:
            symbols: List of ticker symbols
            lookback_days: Number of days for historical data
            
        Returns:
            Correlation matrix as DataFrame
        """
        prices = self.get_historical_data(
            symbols,
            start_date=datetime.now() - timedelta(days=lookback_days)
        )
        
        # Calculate returns
        returns = prices.pivot(columns='Symbol', values='Close').pct_change()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        return corr_matrix
    
    def get_market_indicators(self) -> Dict[str, float]:
        """
        Get various market indicators and sentiment measures.
        """
        indicators = {}
        
        try:
            # VIX (Volatility Index)
            vix = yf.Ticker("^VIX")
            indicators['vix'] = vix.info['regularMarketPrice']
            
            # S&P 500
            sp500 = yf.Ticker("^GSPC")
            indicators['sp500'] = sp500.info['regularMarketPrice']
            indicators['sp500_change'] = sp500.info['regularMarketChangePercent']
            
            # 10-Year Treasury Yield
            treasury = yf.Ticker("^TNX")
            indicators['treasury_10y'] = treasury.info['regularMarketPrice']
            
        except Exception as e:
            print(f"Error fetching market indicators: {e}")
            
        return indicators
