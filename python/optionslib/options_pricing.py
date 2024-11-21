"""
Options Pricing Models Module.
Implements various options pricing models including Black-Scholes, Binomial, and Monte Carlo.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
try:
    from optionslib_cpp import (
        OptionParamsCpp, OptionResultCpp,
        black_scholes_cpp, binomial_tree_cpp, monte_carlo_cpp
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ implementation not available. Using pure Python implementation.")

@dataclass
class OptionParams:
    """Parameters for option pricing."""
    S: float  # Current stock price
    K: float  # Strike price
    T: float  # Time to maturity
    r: float  # Risk-free rate
    sigma: float  # Volatility
    q: float = 0.0  # Dividend yield
    option_type: str = 'call'  # 'call' or 'put'

    def to_cpp_params(self) -> Optional['OptionParamsCpp']:
        if not CPP_AVAILABLE:
            return None
        params = OptionParamsCpp()
        params.S = self.S
        params.K = self.K
        params.T = self.T
        params.r = self.r
        params.sigma = self.sigma
        params.q = self.q
        params.is_call = self.option_type == 'call'
        return params

class OptionsPricing:
    """Options pricing models implementation."""
    
    @staticmethod
    def black_scholes(params: OptionParams) -> Dict[str, float]:
        """Calculate option price and Greeks using Black-Scholes model."""
        if CPP_AVAILABLE:
            cpp_params = params.to_cpp_params()
            result = black_scholes_cpp(cpp_params)
            return {
                'price': result.price,
                'delta': result.delta,
                'gamma': result.gamma,
                'theta': result.theta,
                'vega': result.vega,
                'rho': result.rho
            }
        
        # Pure Python implementation as fallback
        S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
        is_call = params.option_type == 'call'
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        if is_call:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        # Calculate Greeks
        delta = np.exp(-q * T) * (norm.cdf(d1) if is_call else -norm.cdf(-d1))
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T)) -
                r * K * np.exp(-r * T) * (norm.cdf(d2) if is_call else norm.cdf(-d2)) +
                q * S * np.exp(-q * T) * (norm.cdf(d1) if is_call else norm.cdf(-d1)))
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
        rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if is_call else -norm.cdf(-d2))
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def binomial_tree(params: OptionParams, steps: int = 100) -> Dict[str, float]:
        """Calculate option price and Greeks using Binomial Tree model."""
        if CPP_AVAILABLE:
            cpp_params = params.to_cpp_params()
            result = binomial_tree_cpp(cpp_params, steps)
            return {
                'price': result.price,
                'delta': result.delta,
                'gamma': result.gamma,
                'theta': result.theta,
                'vega': result.vega,
                'rho': result.rho
            }
        
        # Pure Python implementation as fallback
        S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
        is_call = params.option_type == 'call'
        
        # Calculate parameters
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Initialize stock price tree
        stock = np.zeros((steps + 1, steps + 1))
        stock[0, 0] = S
        
        for i in range(1, steps + 1):
            stock[i, 0] = stock[i-1, 0] * u
            for j in range(1, i + 1):
                stock[i, j] = stock[i-1, j-1] * d
        
        # Initialize option value tree
        option = np.zeros((steps + 1, steps + 1))
        
        # Calculate option values at expiration
        for j in range(steps + 1):
            if is_call:
                option[steps, j] = max(0, stock[steps, j] - K)
            else:
                option[steps, j] = max(0, K - stock[steps, j])
        
        # Backward induction
        df = np.exp(-r * dt)
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                option[i, j] = df * (p * option[i+1, j] + (1-p) * option[i+1, j+1])
        
        price = option[0, 0]
        
        # Calculate Greeks using finite differences
        delta = (option[1, 0] - option[1, 1]) / (stock[1, 0] - stock[1, 1])
        gamma = ((option[2, 0] - option[2, 1]) / (stock[2, 0] - stock[2, 1]) -
                (option[2, 1] - option[2, 2]) / (stock[2, 1] - stock[2, 2])) / (0.5 * (stock[2, 0] - stock[2, 2]))
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma
        }
    
    @staticmethod
    def monte_carlo(params: OptionParams, num_sims: int = 100000, time_steps: int = 100) -> Tuple[float, float]:
        """Calculate option price using Monte Carlo simulation."""
        if CPP_AVAILABLE:
            cpp_params = params.to_cpp_params()
            return monte_carlo_cpp(cpp_params, num_sims, time_steps)
        
        # Pure Python implementation as fallback
        S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
        is_call = params.option_type == 'call'
        
        dt = T / time_steps
        nudt = (r - q - 0.5 * sigma**2) * dt
        sigsdt = sigma * np.sqrt(dt)
        
        # Generate paths
        Z = np.random.standard_normal((num_sims, time_steps))
        paths = np.zeros((num_sims, time_steps + 1))
        paths[:, 0] = S
        
        for t in range(time_steps):
            paths[:, t+1] = paths[:, t] * np.exp(nudt + sigsdt * Z[:, t])
        
        # Calculate payoffs
        if is_call:
            payoffs = np.maximum(paths[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - paths[:, -1], 0)
        
        # Calculate price
        price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate standard error
        std_error = np.std(payoffs) / np.sqrt(num_sims)
        
        return price, std_error
    
    @staticmethod
    def plot_option_surface(spot_range: np.ndarray, vol_range: np.ndarray, params: OptionParams) -> None:
        """Plot option price surface as a function of spot price and volatility."""
        X, Y = np.meshgrid(spot_range, vol_range)
        Z = np.zeros_like(X)
        
        for i in range(len(vol_range)):
            for j in range(len(spot_range)):
                temp_params = OptionParams(
                    S=spot_range[j],
                    K=params.K,
                    T=params.T,
                    r=params.r,
                    sigma=vol_range[i],
                    q=params.q,
                    option_type=params.option_type
                )
                
                if CPP_AVAILABLE:
                    cpp_params = temp_params.to_cpp_params()
                    result = black_scholes_cpp(cpp_params)
                    Z[i, j] = result.price
                else:
                    result = OptionsPricing.black_scholes(temp_params)
                    Z[i, j] = result['price']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X, Y, Z, cmap='viridis')
        
        ax.set_xlabel('Spot Price')
        ax.set_ylabel('Volatility')
        ax.set_zlabel('Option Price')
        ax.set_title(f'{params.option_type.title()} Option Price Surface')
        
        plt.colorbar(surface)
    
    @staticmethod
    def plot_greeks(spot_range: np.ndarray, params: OptionParams) -> None:
        """Plot option Greeks as a function of spot price."""
        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        values = {greek: [] for greek in greeks}
        
        for spot in spot_range:
            temp_params = OptionParams(
                S=spot,
                K=params.K,
                T=params.T,
                r=params.r,
                sigma=params.sigma,
                q=params.q,
                option_type=params.option_type
            )
            if CPP_AVAILABLE:
                cpp_params = temp_params.to_cpp_params()
                result = black_scholes_cpp(cpp_params)
                for greek in greeks:
                    values[greek].append(getattr(result, greek))
            else:
                result = OptionsPricing.black_scholes(temp_params)
                for greek in greeks:
                    values[greek].append(result[greek])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, greek in enumerate(greeks):
            axes[idx].plot(spot_range, values[greek])
            axes[idx].set_xlabel('Spot Price')
            axes[idx].set_ylabel(greek.title())
            axes[idx].set_title(f'{greek.title()} vs Spot Price')
            axes[idx].grid(True)
        
        plt.tight_layout()
