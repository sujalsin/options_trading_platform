#pragma once

#include <vector>
#include <cmath>
#include <random>

namespace optionslib {

class OptionPricingCpp {
public:
    struct OptionParams {
        double S;      // Current stock price
        double K;      // Strike price
        double T;      // Time to maturity
        double r;      // Risk-free rate
        double sigma;  // Volatility
        double q;      // Dividend yield
        bool is_call;  // True for call, false for put
    };
    
    struct OptionResult {
        double price;
        double delta;
        double gamma;
        double theta;
        double vega;
        double rho;
    };
    
    static OptionResult black_scholes(const OptionParams& params);
    static OptionResult binomial_tree(const OptionParams& params, int steps = 100);
    static std::pair<double, double> monte_carlo(const OptionParams& params, 
                                               int num_sims = 100000,
                                               int time_steps = 100);
    
private:
    static double norm_cdf(double x);
    static double norm_pdf(double x);
};

} // namespace optionslib
