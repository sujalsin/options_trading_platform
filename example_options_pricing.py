"""
Example script demonstrating options pricing functionality.
"""

import numpy as np
from optionslib.options_pricing import OptionParams, OptionsPricing
import matplotlib.pyplot as plt

def main():
    # Test parameters
    params = OptionParams(
        S=100.0,  # Current stock price
        K=100.0,  # Strike price
        T=1.0,    # Time to maturity (1 year)
        r=0.05,   # Risk-free rate (5%)
        sigma=0.2, # Volatility (20%)
        q=0.02,   # Dividend yield (2%)
        option_type='call'
    )
    
    # Test Black-Scholes
    print("\nBlack-Scholes Model:")
    bs_result = OptionsPricing.black_scholes(params)
    print(f"Price: ${bs_result['price']:.2f}")
    print(f"Delta: {bs_result['delta']:.4f}")
    print(f"Gamma: {bs_result['gamma']:.4f}")
    print(f"Theta: {bs_result['theta']:.4f}")
    print(f"Vega: {bs_result['vega']:.4f}")
    print(f"Rho: {bs_result['rho']:.4f}")
    
    # Test Binomial Tree
    print("\nBinomial Tree Model:")
    bin_result = OptionsPricing.binomial_tree(params)
    print(f"Price: ${bin_result['price']:.2f}")
    print(f"Delta: {bin_result['delta']:.4f}")
    print(f"Gamma: {bin_result['gamma']:.4f}")
    
    # Test Monte Carlo
    print("\nMonte Carlo Simulation:")
    mc_price, mc_std_error = OptionsPricing.monte_carlo(params)
    print(f"Price: ${mc_price:.2f}")
    print(f"Standard Error: {mc_std_error:.4f}")
    
    # Plot option price surface
    spot_range = np.linspace(80, 120, 20)
    vol_range = np.linspace(0.1, 0.4, 20)
    
    print("\nGenerating visualizations...")
    
    # Plot option price surface
    OptionsPricing.plot_option_surface(spot_range, vol_range, params)
    
    # Plot Greeks
    OptionsPricing.plot_greeks(spot_range, params)
    
    plt.show()

if __name__ == "__main__":
    main()
