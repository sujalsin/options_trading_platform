import optionstrader_py as opt
import numpy as np
import matplotlib.pyplot as plt

def plot_option_greeks(spot_range, strike, time_to_maturity, risk_free_rate, volatility, is_call=True):
    pricer = opt.BlackScholesPricer()
    spots = np.linspace(spot_range[0], spot_range[1], 100)
    
    prices = []
    deltas = []
    gammas = []
    vegas = []
    thetas = []
    rhos = []
    
    for spot in spots:
        params = opt.OptionParameters()
        params.spot_price = spot
        params.strike_price = strike
        params.time_to_maturity = time_to_maturity
        params.risk_free_rate = risk_free_rate
        params.volatility = volatility
        params.is_call = is_call
        
        prices.append(pricer.calculate_price(params))
        deltas.append(pricer.calculate_delta(params))
        gammas.append(pricer.calculate_gamma(params))
        vegas.append(pricer.calculate_vega(params))
        thetas.append(pricer.calculate_theta(params))
        rhos.append(pricer.calculate_rho(params))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{"Call" if is_call else "Put"} Option Greeks (K={strike}, T={time_to_maturity}, r={risk_free_rate}, σ={volatility})')
    
    axes[0, 0].plot(spots, prices)
    axes[0, 0].set_title('Price')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(spots, deltas)
    axes[0, 1].set_title('Delta')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(spots, gammas)
    axes[0, 2].set_title('Gamma')
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(spots, vegas)
    axes[1, 0].set_title('Vega')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(spots, thetas)
    axes[1, 1].set_title('Theta')
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(spots, rhos)
    axes[1, 2].set_title('Rho')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    spot_range = (30, 70)
    strike = 50
    time_to_maturity = 1.0  # 1 year
    risk_free_rate = 0.05   # 5%
    volatility = 0.3        # 30%
    
    # Plot call option greeks
    plot_option_greeks(spot_range, strike, time_to_maturity, risk_free_rate, volatility, True)
    
    # Plot put option greeks
    plot_option_greeks(spot_range, strike, time_to_maturity, risk_free_rate, volatility, False)
