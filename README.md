# Advanced Options Trading Platform

A comprehensive, high-performance cross-language platform for sophisticated options pricing, risk management, and portfolio optimization techniques. The platform combines C++ computational efficiency with Python's flexibility for advanced financial modeling and analysis.

## Features

### Options Pricing Models
- Black-Scholes Model
  - Accurate price calculation
  - Complete Greeks (Delta, Gamma, Theta, Vega, Rho)
  - Dividend yield support
- Binomial Tree Model
  - American and European options
  - Early exercise capability
  - Delta and Gamma calculations
- Monte Carlo Simulation
  - Path-dependent options
  - Standard error estimation
  - Configurable simulation parameters

### Technical Analysis
- Advanced visualization tools
  - Option price surface plots
  - Greeks visualization
  - Interactive analysis tools
- Market data integration (via yfinance)
- Technical indicators (via TA-Lib)

### Cross-Language Implementation
- High-performance C++ core
- Python interface with pybind11
- Automatic fallback to pure Python implementation
- Comprehensive error handling

## Prerequisites

- Python >= 3.8
- C++17 compatible compiler
- CMake >= 3.15
- Required Python packages:
  ```
  numpy>=1.21.0
  pandas>=1.3.0
  scipy>=1.7.0
  matplotlib>=3.4.0
  pybind11>=2.7.0
  cvxopt>=1.2.0
  ta-lib>=0.4.0
  yfinance>=0.1.63
  ```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd options_trading_platform
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build the C++ module:
```bash
./build.sh
```

## Usage Example

```python
from optionslib.options_pricing import OptionParams, OptionsPricing

# Create option parameters
params = OptionParams(
    S=100.0,  # Current stock price
    K=100.0,  # Strike price
    T=1.0,    # Time to maturity (1 year)
    r=0.05,   # Risk-free rate (5%)
    sigma=0.2, # Volatility (20%)
    q=0.02,   # Dividend yield (2%)
    option_type='call'
)

# Calculate option price and Greeks using Black-Scholes
bs_result = OptionsPricing.black_scholes(params)
print(f"Price: ${bs_result['price']:.2f}")
print(f"Delta: {bs_result['delta']:.4f}")
print(f"Gamma: {bs_result['gamma']:.4f}")

# Calculate using Binomial Tree
bin_result = OptionsPricing.binomial_tree(params)
print(f"Binomial Price: ${bin_result['price']:.2f}")

# Monte Carlo simulation
mc_price, mc_std_error = OptionsPricing.monte_carlo(params)
print(f"Monte Carlo Price: ${mc_price:.2f}")
print(f"Standard Error: {mc_std_error:.4f}")
```

## Project Structure

```
options_trading_platform/
├── CMakeLists.txt          # C++ build configuration
├── README.md
├── requirements.txt        # Python dependencies
├── build.sh               # Build script for C++ module
├── cpp/                   # C++ implementation
│   ├── optionslib/       # Core C++ library
│   │   ├── options_pricing.hpp
│   │   └── options_pricing.cpp
│   └── CMakeLists.txt
├── python/               # Python implementation
│   └── optionslib/
│       ├── __init__.py
│       ├── options_pricing.py
│       ├── technical_analysis.py
│       └── optimization.py
├── tests/               # Test files
│   ├── test_options_pricing.py
│   └── test_functionality.py
└── example_options_pricing.py  # Usage examples
```

## Testing

Run the test suite:
```bash
python -m pytest test_options_pricing.py -v
```

## Future Enhancements

- Real-time market data integration
- Advanced portfolio optimization
- Machine learning-based options pricing
- Enhanced risk management features
- Additional exotic options support

## License

This project is licensed under the MIT License - see the LICENSE file for details.
