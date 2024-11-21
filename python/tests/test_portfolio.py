import pytest
import numpy as np
from optionslib.portfolio import Portfolio
from optionslib.optimization import PortfolioOptimizer
from optionslib.risk_analytics import RiskAnalytics

def test_portfolio_optimization():
    # Create sample portfolio
    portfolio = Portfolio()
    
    # Add some test positions
    portfolio.add_position(
        "AAPL",
        100,
        "black_scholes",
        {"spot": 150.0, "strike": 155.0, "volatility": 0.2, "rate": 0.02, "time": 0.5}
    )
    
    portfolio.add_position(
        "GOOGL",
        50,
        "black_scholes",
        {"spot": 2800.0, "strike": 2850.0, "volatility": 0.25, "rate": 0.02, "time": 0.5}
    )
    
    # Test portfolio value calculation
    value = portfolio.calculate_portfolio_value()
    assert value > 0
    
    # Test risk metrics calculation
    risk_metrics = portfolio.calculate_risk_metrics()
    assert isinstance(risk_metrics, dict)
    assert all(metric in risk_metrics for metric in [
        'total_value', 'delta', 'gamma', 'vega', 'theta', 'rho'
    ])
    
def test_optimization():
    # Create sample returns and covariance matrix
    returns = np.array([
        [0.01, -0.02, 0.03],
        [0.02, -0.01, 0.02],
        [0.03, 0.02, 0.01]
    ])
    
    cov_matrix = np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.05, 0.02],
        [0.01, 0.02, 0.03]
    ])
    
    optimizer = PortfolioOptimizer(returns, cov_matrix)
    
    # Test mean-variance optimization
    mv_result = optimizer.optimize_mean_variance(target_return=0.02)
    assert mv_result.success
    assert len(mv_result.weights) == 3
    assert np.abs(np.sum(mv_result.weights) - 1.0) < 1e-6
    
    # Test risk parity optimization
    rp_result = optimizer.optimize_risk_parity()
    assert rp_result.success
    assert len(rp_result.weights) == 3
    assert np.abs(np.sum(rp_result.weights) - 1.0) < 1e-6
    
    # Test maximum diversification
    md_result = optimizer.optimize_maximum_diversification()
    assert md_result.success
    assert len(md_result.weights) == 3
    assert np.abs(np.sum(md_result.weights) - 1.0) < 1e-6
    
def test_risk_analytics():
    # Create sample return series
    returns1 = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    returns2 = np.array([0.02, -0.01, 0.02, -0.02, 0.01])
    
    # Test correlation calculations
    pearson_corr = RiskAnalytics.calculate_correlation(returns1, returns2, 'pearson')
    assert -1.0 <= pearson_corr <= 1.0
    
    spearman_corr = RiskAnalytics.calculate_correlation(returns1, returns2, 'spearman')
    assert -1.0 <= spearman_corr <= 1.0
    
    # Test tail dependence
    lower_tail, upper_tail = RiskAnalytics.calculate_tail_dependence(returns1, returns2)
    assert 0.0 <= lower_tail <= 1.0
    assert 0.0 <= upper_tail <= 1.0
    
    # Test drawdown calculations
    drawdown_stats = RiskAnalytics.calculate_drawdowns(returns1)
    assert isinstance(drawdown_stats, dict)
    assert all(metric in drawdown_stats for metric in [
        'max_drawdown', 'drawdown_duration', 'recovery_time'
    ])
    assert drawdown_stats['max_drawdown'] <= 0.0
    
    # Test VaR and CVaR calculations
    var, cvar = RiskAnalytics.calculate_var_cvar(returns1)
    assert var < 0.0  # VaR should be negative for losses
    assert cvar <= var  # CVaR should be more conservative than VaR
