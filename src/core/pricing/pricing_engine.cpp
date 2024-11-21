#include "pricing_engine.hpp"
#include <cmath>

namespace optionstrader {
namespace pricing {

double BlackScholesPricer::normal_cdf(double x) const {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double BlackScholesPricer::normal_pdf(double x) const {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

void BlackScholesPricer::calculate_d1_d2(const OptionParameters& params, double& d1, double& d2) const {
    d1 = (std::log(params.spot_price / params.strike_price) + 
          (params.risk_free_rate + 0.5 * params.volatility * params.volatility) * 
          params.time_to_maturity) / 
         (params.volatility * std::sqrt(params.time_to_maturity));
    
    d2 = d1 - params.volatility * std::sqrt(params.time_to_maturity);
}

double BlackScholesPricer::calculate_price(const OptionParameters& params) {
    double d1, d2;
    calculate_d1_d2(params, d1, d2);
    
    if (params.is_call) {
        return params.spot_price * normal_cdf(d1) - 
               params.strike_price * std::exp(-params.risk_free_rate * params.time_to_maturity) * 
               normal_cdf(d2);
    } else {
        return params.strike_price * std::exp(-params.risk_free_rate * params.time_to_maturity) * 
               normal_cdf(-d2) - params.spot_price * normal_cdf(-d1);
    }
}

double BlackScholesPricer::calculate_delta(const OptionParameters& params) {
    double d1, d2;
    calculate_d1_d2(params, d1, d2);
    
    if (params.is_call) {
        return normal_cdf(d1);
    } else {
        return normal_cdf(d1) - 1.0;
    }
}

double BlackScholesPricer::calculate_gamma(const OptionParameters& params) {
    double d1, d2;
    calculate_d1_d2(params, d1, d2);
    
    return normal_pdf(d1) / (params.spot_price * params.volatility * 
                            std::sqrt(params.time_to_maturity));
}

double BlackScholesPricer::calculate_vega(const OptionParameters& params) {
    double d1, d2;
    calculate_d1_d2(params, d1, d2);
    
    return params.spot_price * std::sqrt(params.time_to_maturity) * normal_pdf(d1);
}

double BlackScholesPricer::calculate_theta(const OptionParameters& params) {
    double d1, d2;
    calculate_d1_d2(params, d1, d2);
    
    if (params.is_call) {
        return -params.spot_price * normal_pdf(d1) * params.volatility / 
               (2 * std::sqrt(params.time_to_maturity)) -
               params.risk_free_rate * params.strike_price * 
               std::exp(-params.risk_free_rate * params.time_to_maturity) * normal_cdf(d2);
    } else {
        return -params.spot_price * normal_pdf(d1) * params.volatility / 
               (2 * std::sqrt(params.time_to_maturity)) +
               params.risk_free_rate * params.strike_price * 
               std::exp(-params.risk_free_rate * params.time_to_maturity) * normal_cdf(-d2);
    }
}

double BlackScholesPricer::calculate_rho(const OptionParameters& params) {
    double d1, d2;
    calculate_d1_d2(params, d1, d2);
    
    if (params.is_call) {
        return params.strike_price * params.time_to_maturity * 
               std::exp(-params.risk_free_rate * params.time_to_maturity) * normal_cdf(d2);
    } else {
        return -params.strike_price * params.time_to_maturity * 
                std::exp(-params.risk_free_rate * params.time_to_maturity) * normal_cdf(-d2);
    }
}

} // namespace pricing
} // namespace optionstrader
