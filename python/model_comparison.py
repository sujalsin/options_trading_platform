import optionstrader_py as opt
import numpy as np
import matplotlib.pyplot as plt
import time

def compare_pricing_models(spot_range, strike, time_to_maturity, risk_free_rate, volatility, is_call=True):
    # Create option parameters
    params = opt.OptionParameters()
    params.strike_price = strike
    params.time_to_maturity = time_to_maturity
    params.risk_free_rate = risk_free_rate
    params.volatility = volatility
    params.is_call = is_call
    
    # Initialize pricers
    bs_pricer = opt.BlackScholesPricer()
    
    mc_config = opt.MonteCarloConfig()
    mc_config.num_paths = 100000
    mc_config.time_steps = 252
    mc_pricer = opt.MonteCarloPricer(mc_config)
    
    bin_config = opt.BinomialConfig()
    bin_config.num_steps = 1000
    bin_pricer = opt.BinomialPricer(bin_config)
    
    # Arrays for storing results
    spots = np.linspace(spot_range[0], spot_range[1], 50)
    bs_prices = []
    mc_prices = []
    mc_confidence = []
    bin_prices = []
    
    # Calculate prices
    for spot in spots:
        params.spot_price = spot
        
        bs_price = bs_pricer.calculate_price(params)
        bs_prices.append(bs_price)
        
        mc_price, mc_conf = mc_pricer.calculate_price_with_confidence(params)
        mc_prices.append(mc_price)
        mc_confidence.append(mc_conf)
        
        bin_price = bin_pricer.calculate_price(params)
        bin_prices.append(bin_price)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(spots, bs_prices, 'b-', label='Black-Scholes')
    plt.plot(spots, mc_prices, 'r--', label='Monte Carlo')
    plt.plot(spots, bin_prices, 'g:', label='Binomial Tree')
    
    # Add Monte Carlo confidence intervals
    mc_prices = np.array(mc_prices)
    mc_confidence = np.array(mc_confidence)
    plt.fill_between(spots, mc_prices - mc_confidence, mc_prices + mc_confidence,
                    color='r', alpha=0.2, label='MC 95% CI')
    
    plt.title(f'{"Call" if is_call else "Put"} Option Price Comparison\n' +
              f'K={strike}, T={time_to_maturity}, r={risk_free_rate}, σ={volatility}')
    plt.xlabel('Spot Price')
    plt.ylabel('Option Price')
    plt.grid(True)
    plt.legend()
    plt.show()

def compare_performance(params, num_trials=100):
    # Initialize pricers
    bs_pricer = opt.BlackScholesPricer()
    
    mc_config = opt.MonteCarloConfig()
    mc_config.num_paths = 10000
    mc_pricer = opt.MonteCarloPricer(mc_config)
    
    bin_config = opt.BinomialConfig()
    bin_config.num_steps = 100
    bin_pricer = opt.BinomialPricer(bin_config)
    
    # Time each method
    bs_times = []
    mc_times = []
    bin_times = []
    
    for _ in range(num_trials):
        # Black-Scholes
        start = time.time()
        bs_pricer.calculate_price(params)
        bs_times.append(time.time() - start)
        
        # Monte Carlo
        start = time.time()
        mc_pricer.calculate_price(params)
        mc_times.append(time.time() - start)
        
        # Binomial
        start = time.time()
        bin_pricer.calculate_price(params)
        bin_times.append(time.time() - start)
    
    # Plot performance comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot([bs_times, mc_times, bin_times], labels=['Black-Scholes', 'Monte Carlo', 'Binomial'])
    plt.title('Performance Comparison of Pricing Methods')
    plt.ylabel('Computation Time (seconds)')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    spot_range = (30, 70)
    strike = 50
    time_to_maturity = 1.0
    risk_free_rate = 0.05
    volatility = 0.3
    
    # Compare pricing models for call options
    compare_pricing_models(spot_range, strike, time_to_maturity, risk_free_rate, volatility, True)
    
    # Compare pricing models for put options
    compare_pricing_models(spot_range, strike, time_to_maturity, risk_free_rate, volatility, False)
    
    # Performance comparison
    params = opt.OptionParameters()
    params.spot_price = 50
    params.strike_price = 50
    params.time_to_maturity = 1.0
    params.risk_free_rate = 0.05
    params.volatility = 0.3
    params.is_call = True
    
    compare_performance(params)
