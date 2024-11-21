#pragma once

#include "pricing_engine.hpp"
#include <vector>

namespace optionstrader {
namespace pricing {

class BinomialConfig {
public:
    size_t num_steps = 100;  // Number of time steps in the tree
    bool american = false;   // Whether to price American options
};

class BinomialPricer : public PricingEngine {
public:
    explicit BinomialPricer(const BinomialConfig& config = BinomialConfig());

    double calculate_price(const OptionParameters& params) override;
    double calculate_delta(const OptionParameters& params) override;
    double calculate_gamma(const OptionParameters& params) override;
    double calculate_vega(const OptionParameters& params) override;
    double calculate_theta(const OptionParameters& params) override;
    double calculate_rho(const OptionParameters& params) override;

    // Additional methods for American options
    double calculate_early_exercise_boundary(const OptionParameters& params);
    std::vector<double> calculate_critical_prices(const OptionParameters& params);

private:
    std::vector<double> build_price_tree(const OptionParameters& params) const;
    std::vector<double> build_value_tree(const std::vector<double>& price_tree,
                                       const OptionParameters& params) const;
    double calculate_payoff(double spot_price, const OptionParameters& params) const;
    
    BinomialConfig config_;
};

} // namespace pricing
} // namespace optionstrader
