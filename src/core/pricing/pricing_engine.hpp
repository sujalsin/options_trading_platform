#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <stdexcept>

namespace optionstrader {
namespace pricing {

struct OptionParameters {
    double spot_price;
    double strike_price;
    double time_to_maturity;
    double risk_free_rate;
    double volatility;
    bool is_call;
};

class PricingEngine {
public:
    virtual ~PricingEngine() = default;
    virtual double calculate_price(const OptionParameters& params) = 0;
    virtual double calculate_delta(const OptionParameters& params) = 0;
    virtual double calculate_gamma(const OptionParameters& params) = 0;
    virtual double calculate_vega(const OptionParameters& params) = 0;
    virtual double calculate_theta(const OptionParameters& params) = 0;
    virtual double calculate_rho(const OptionParameters& params) = 0;
};

class BlackScholesPricer : public PricingEngine {
public:
    double calculate_price(const OptionParameters& params) override;
    double calculate_delta(const OptionParameters& params) override;
    double calculate_gamma(const OptionParameters& params) override;
    double calculate_vega(const OptionParameters& params) override;
    double calculate_theta(const OptionParameters& params) override;
    double calculate_rho(const OptionParameters& params) override;

private:
    double normal_cdf(double x) const;
    double normal_pdf(double x) const;
    void calculate_d1_d2(const OptionParameters& params, double& d1, double& d2) const;
};

} // namespace pricing
} // namespace optionstrader
