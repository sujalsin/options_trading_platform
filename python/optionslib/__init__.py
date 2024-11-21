"""
Options Trading Platform Package.
"""

__version__ = '0.1.0'

# Only import the classes that are needed
from .technical_analysis import TechnicalAnalyzer
from .portfolio import Portfolio
from .risk_analytics import RiskAnalytics
from .optimization import PortfolioOptimizer
from .market_data import MarketDataFetcher
