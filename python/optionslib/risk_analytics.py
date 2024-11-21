import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import pandas as pd

class RiskAnalytics:
    """Advanced risk analytics and statistical analysis."""
    
    @staticmethod
    def calculate_correlation(returns1: np.ndarray, 
                            returns2: np.ndarray,
                            method: str = 'pearson') -> float:
        """
        Calculate correlation between two return series.
        
        Args:
            returns1: First return series
            returns2: Second return series
            method: One of ['pearson', 'spearman', 'kendall']
        """
        if method == 'pearson':
            return stats.pearsonr(returns1, returns2)[0]
        elif method == 'spearman':
            return stats.spearmanr(returns1, returns2)[0]
        elif method == 'kendall':
            return stats.kendalltau(returns1, returns2)[0]
        else:
            raise ValueError(f"Unknown correlation method: {method}")
    
    @staticmethod
    def calculate_tail_dependence(returns1: np.ndarray,
                                returns2: np.ndarray,
                                quantile: float = 0.05) -> Tuple[float, float]:
        """
        Calculate lower and upper tail dependence coefficients.
        
        Args:
            returns1: First return series
            returns2: Second return series
            quantile: Tail probability threshold
        
        Returns:
            Tuple of (lower_tail_dependence, upper_tail_dependence)
        """
        n = len(returns1)
        rank1 = stats.rankdata(returns1) / (n + 1)
        rank2 = stats.rankdata(returns2) / (n + 1)
        
        # Lower tail dependence
        lower_mask = (rank1 <= quantile) & (rank2 <= quantile)
        lower_tail = np.sum(lower_mask) / (n * quantile)
        
        # Upper tail dependence
        upper_mask = (rank1 >= 1 - quantile) & (rank2 >= 1 - quantile)
        upper_tail = np.sum(upper_mask) / (n * quantile)
        
        return lower_tail, upper_tail
    
    @staticmethod
    def calculate_risk_contribution(weights: np.ndarray,
                                  cov_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate risk contribution of each asset to portfolio risk.
        
        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix of returns
        """
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_risk = (cov_matrix @ weights) / port_vol
        risk_contrib = weights * marginal_risk
        return risk_contrib / port_vol
    
    @staticmethod
    def calculate_drawdowns(returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate drawdown statistics.
        
        Args:
            returns: Array of returns
            
        Returns:
            Dictionary with max drawdown, drawdown duration, and time to recovery
        """
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        
        max_drawdown = np.min(drawdowns)
        max_drawdown_idx = np.argmin(drawdowns)
        
        # Find the start of the drawdown period
        peak_idx = np.where(cum_returns[:max_drawdown_idx] == 
                           running_max[max_drawdown_idx])[0][-1]
        
        # Find the end of the drawdown period (recovery)
        try:
            recovery_idx = np.where(cum_returns[max_drawdown_idx:] >= 
                                  cum_returns[peak_idx])[0][0] + max_drawdown_idx
            recovery_time = recovery_idx - max_drawdown_idx
        except IndexError:
            recovery_time = len(returns) - max_drawdown_idx
        
        drawdown_duration = max_drawdown_idx - peak_idx
        
        return {
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'recovery_time': recovery_time
        }
    
    @staticmethod
    def calculate_var_cvar(returns: np.ndarray,
                          confidence_level: float = 0.95,
                          method: str = 'historical') -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: One of ['historical', 'parametric', 'cornish_fisher']
        """
        alpha = 1 - confidence_level
        
        if method == 'historical':
            var = np.percentile(returns, alpha * 100)
            cvar = np.mean(returns[returns <= var])
            
        elif method == 'parametric':
            mu = np.mean(returns)
            sigma = np.std(returns)
            var = stats.norm.ppf(alpha, mu, sigma)
            cvar = mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
            
        elif method == 'cornish_fisher':
            z = stats.norm.ppf(alpha)
            s = stats.skew(returns)
            k = stats.kurtosis(returns)
            
            z_cf = (z + 
                   (z**2 - 1) * s / 6 +
                   (z**3 - 3*z) * (k - 3) / 24 -
                   (2*z**3 - 5*z) * s**2 / 36)
            
            mu = np.mean(returns)
            sigma = np.std(returns)
            
            var = mu + sigma * z_cf
            # Approximation for CVaR using Cornish-Fisher VaR
            cvar = var - sigma * stats.norm.pdf(z) / alpha
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
            
        return var, cvar
