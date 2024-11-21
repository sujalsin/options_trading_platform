#include "options_pricing.hpp"
#include <cmath>

namespace optionslib {

double OptionPricingCpp::norm_cdf(double x) {
    return 0.5 * (1 + std::erf(x / std::sqrt(2)));
}

double OptionPricingCpp::norm_pdf(double x) {
    return (1.0 / std::sqrt(2 * M_PI)) * std::exp(-0.5 * x * x);
}

OptionPricingCpp::OptionResult OptionPricingCpp::black_scholes(const OptionParams& params) {
    double d1 = (std::log(params.S / params.K) + (params.r - params.q + 0.5 * params.sigma * params.sigma) * params.T) / 
                (params.sigma * std::sqrt(params.T));
    double d2 = d1 - params.sigma * std::sqrt(params.T);
    
    OptionResult result;
    if (params.is_call) {
        result.price = params.S * std::exp(-params.q * params.T) * norm_cdf(d1) - 
                      params.K * std::exp(-params.r * params.T) * norm_cdf(d2);
        result.delta = std::exp(-params.q * params.T) * norm_cdf(d1);
    } else {
        result.price = params.K * std::exp(-params.r * params.T) * norm_cdf(-d2) - 
                      params.S * std::exp(-params.q * params.T) * norm_cdf(-d1);
        result.delta = -std::exp(-params.q * params.T) * norm_cdf(-d1);
    }
    
    result.gamma = std::exp(-params.q * params.T) * norm_pdf(d1) / 
                  (params.S * params.sigma * std::sqrt(params.T));
    result.vega = params.S * std::exp(-params.q * params.T) * std::sqrt(params.T) * norm_pdf(d1);
    result.theta = -params.S * std::exp(-params.q * params.T) * norm_pdf(d1) * params.sigma / 
                  (2 * std::sqrt(params.T)) - params.r * params.K * std::exp(-params.r * params.T) * norm_cdf(d2);
    result.rho = params.K * params.T * std::exp(-params.r * params.T) * norm_cdf(d2);
    
    return result;
}

OptionPricingCpp::OptionResult OptionPricingCpp::binomial_tree(const OptionParams& params, int steps) {
    double dt = params.T / steps;
    double u = std::exp(params.sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double p = (std::exp((params.r - params.q) * dt) - d) / (u - d);
    
    std::vector<double> prices(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        prices[i] = params.S * std::pow(u, steps - i) * std::pow(d, i);
    }
    
    std::vector<double> values(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        values[i] = params.is_call ? std::max(0.0, prices[i] - params.K) 
                                 : std::max(0.0, params.K - prices[i]);
    }
    
    for (int j = steps - 1; j >= 0; --j) {
        for (int i = 0; i <= j; ++i) {
            values[i] = std::exp(-params.r * dt) * (p * values[i] + (1 - p) * values[i + 1]);
        }
    }
    
    OptionResult result;
    result.price = values[0];
    
    // Calculate delta using finite difference
    double delta_s = 0.01 * params.S;
    OptionParams up_params = params;
    up_params.S += delta_s;
    OptionParams down_params = params;
    down_params.S -= delta_s;
    
    double up_price = binomial_tree(up_params, steps).price;
    double down_price = binomial_tree(down_params, steps).price;
    result.delta = (up_price - down_price) / (2 * delta_s);
    result.gamma = (up_price - 2 * result.price + down_price) / (delta_s * delta_s);
    
    // Set other Greeks to 0 for now
    result.theta = 0;
    result.vega = 0;
    result.rho = 0;
    
    return result;
}

std::pair<double, double> OptionPricingCpp::monte_carlo(const OptionParams& params, 
                                                       int num_sims, 
                                                       int time_steps) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal(0, 1);
    
    double dt = params.T / time_steps;
    double drift = (params.r - params.q - 0.5 * params.sigma * params.sigma) * dt;
    double vol = params.sigma * std::sqrt(dt);
    double discount = std::exp(-params.r * params.T);
    
    std::vector<double> payoffs(num_sims);
    for (int i = 0; i < num_sims; ++i) {
        double S_t = params.S;
        for (int t = 0; t < time_steps; ++t) {
            S_t *= std::exp(drift + vol * normal(gen));
        }
        
        payoffs[i] = params.is_call ? std::max(0.0, S_t - params.K) 
                                  : std::max(0.0, params.K - S_t);
    }
    
    double sum = 0.0;
    double sum_sq = 0.0;
    for (double payoff : payoffs) {
        sum += payoff;
        sum_sq += payoff * payoff;
    }
    
    double price = discount * sum / num_sims;
    double variance = (sum_sq - sum * sum / num_sims) / (num_sims - 1);
    double std_error = std::sqrt(variance / num_sims);
    
    return {price, std_error};
}

} // namespace optionslib
