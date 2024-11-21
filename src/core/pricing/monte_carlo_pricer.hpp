#pragma once

#include "pricing_engine.hpp"
#include <random>
#include <thread>
#include <vector>

namespace optionstrader {
namespace pricing {

class MonteCarloConfig {
public:
    size_t num_paths = 10000;
    size_t time_steps = 252;  // Daily steps for a year
    size_t num_threads = std::thread::hardware_concurrency();
    bool antithetic = true;   // Use antithetic variates for variance reduction
    bool control_variate = true;  // Use control variates for variance reduction
};

class MonteCarloPricer : public PricingEngine {
public:
    explicit MonteCarloPricer(const MonteCarloConfig& config = MonteCarloConfig());

    double calculate_price(const OptionParameters& params) override;
    double calculate_delta(const OptionParameters& params) override;
    double calculate_gamma(const OptionParameters& params) override;
    double calculate_vega(const OptionParameters& params) override;
    double calculate_theta(const OptionParameters& params) override;
    double calculate_rho(const OptionParameters& params) override;

    // Additional Monte Carlo specific methods
    std::pair<double, double> calculate_price_with_confidence(const OptionParameters& params);
    void set_config(const MonteCarloConfig& config) { config_ = config; }

private:
    std::vector<double> simulate_paths(const OptionParameters& params, size_t batch_size) const;
    double simulate_path(const OptionParameters& params, std::mt19937& gen) const;
    double calculate_payoff(double spot_price, const OptionParameters& params) const;
    
    MonteCarloConfig config_;
    mutable std::mt19937 gen_;
    mutable std::normal_distribution<> normal_{0.0, 1.0};
};

} // namespace pricing
} // namespace optionstrader
