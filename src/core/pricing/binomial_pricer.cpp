#include "binomial_pricer.hpp"
#include <cmath>
#include <algorithm>

namespace optionstrader {
namespace pricing {

BinomialPricer::BinomialPricer(const BinomialConfig& config)
    : config_(config) {}

std::vector<double> BinomialPricer::build_price_tree(const OptionParameters& params) const {
    const double dt = params.time_to_maturity / config_.num_steps;
    const double u = std::exp(params.volatility * std::sqrt(dt));
    const double d = 1.0 / u;
    
    std::vector<double> price_tree((config_.num_steps + 1) * (config_.num_steps + 2) / 2);
    
    // Initialize stock price at t=0
    price_tree[0] = params.spot_price;
    
    // Build the tree
    size_t index = 0;
    for (size_t step = 0; step < config_.num_steps; ++step) {
        for (size_t node = 0; node <= step; ++node) {
            double current_price = price_tree[index + node];
            price_tree[index + step + 1 + node] = current_price * u;
            price_tree[index + step + 2 + node] = current_price * d;
        }
        index += step + 1;
    }
    
    return price_tree;
}

std::vector<double> BinomialPricer::build_value_tree(
    const std::vector<double>& price_tree, const OptionParameters& params) const {
    
    const double dt = params.time_to_maturity / config_.num_steps;
    const double u = std::exp(params.volatility * std::sqrt(dt));
    const double d = 1.0 / u;
    const double p = (std::exp(params.risk_free_rate * dt) - d) / (u - d);
    const double discount = std::exp(-params.risk_free_rate * dt);
    
    std::vector<double> value_tree = price_tree;
    
    // Initialize option values at maturity
    size_t last_step_start = (config_.num_steps * (config_.num_steps + 1)) / 2;
    for (size_t i = 0; i <= config_.num_steps; ++i) {
        value_tree[last_step_start + i] = calculate_payoff(price_tree[last_step_start + i], params);
    }
    
    // Backward induction
    for (int step = config_.num_steps - 1; step >= 0; --step) {
        size_t step_start = (step * (step + 1)) / 2;
        for (size_t node = 0; node <= static_cast<size_t>(step); ++node) {
            double continuation_value = discount * (
                p * value_tree[step_start + step + 1 + node] +
                (1 - p) * value_tree[step_start + step + 2 + node]);
            
            if (config_.american) {
                double exercise_value = calculate_payoff(price_tree[step_start + node], params);
                value_tree[step_start + node] = std::max(continuation_value, exercise_value);
            } else {
                value_tree[step_start + node] = continuation_value;
            }
        }
    }
    
    return value_tree;
}

double BinomialPricer::calculate_price(const OptionParameters& params) {
    auto price_tree = build_price_tree(params);
    auto value_tree = build_value_tree(price_tree, params);
    return value_tree[0];
}

double BinomialPricer::calculate_payoff(double spot_price, const OptionParameters& params) const {
    if (params.is_call) {
        return std::max(spot_price - params.strike_price, 0.0);
    } else {
        return std::max(params.strike_price - spot_price, 0.0);
    }
}

double BinomialPricer::calculate_delta(const OptionParameters& params) {
    auto price_tree = build_price_tree(params);
    auto value_tree = build_value_tree(price_tree, params);
    
    const double dt = params.time_to_maturity / config_.num_steps;
    const double u = std::exp(params.volatility * std::sqrt(dt));
    const double d = 1.0 / u;
    
    double up_value = value_tree[1];
    double down_value = value_tree[2];
    double up_price = price_tree[1];
    double down_price = price_tree[2];
    
    return (up_value - down_value) / (up_price - down_price);
}

double BinomialPricer::calculate_gamma(const OptionParameters& params) {
    const double h = 0.01 * params.spot_price;
    
    OptionParameters up_params = params;
    up_params.spot_price += h;
    
    OptionParameters down_params = params;
    down_params.spot_price -= h;
    
    double delta_up = calculate_delta(up_params);
    double delta_down = calculate_delta(down_params);
    
    return (delta_up - delta_down) / (2 * h);
}

double BinomialPricer::calculate_vega(const OptionParameters& params) {
    const double h = 0.0001;
    
    OptionParameters up_params = params;
    up_params.volatility += h;
    
    OptionParameters down_params = params;
    down_params.volatility -= h;
    
    double up_price = calculate_price(up_params);
    double down_price = calculate_price(down_params);
    
    return (up_price - down_price) / (2 * h);
}

double BinomialPricer::calculate_theta(const OptionParameters& params) {
    const double h = 1.0 / 365.0;  // One day
    
    OptionParameters next_params = params;
    next_params.time_to_maturity -= h;
    
    double next_price = calculate_price(next_params);
    double current_price = calculate_price(params);
    
    return (next_price - current_price) / h;
}

double BinomialPricer::calculate_rho(const OptionParameters& params) {
    const double h = 0.0001;
    
    OptionParameters up_params = params;
    up_params.risk_free_rate += h;
    
    OptionParameters down_params = params;
    down_params.risk_free_rate -= h;
    
    double up_price = calculate_price(up_params);
    double down_price = calculate_price(down_params);
    
    return (up_price - down_price) / (2 * h);
}

double BinomialPricer::calculate_early_exercise_boundary(const OptionParameters& params) {
    if (!config_.american) {
        throw std::runtime_error("Early exercise boundary only available for American options");
    }
    
    auto price_tree = build_price_tree(params);
    auto value_tree = build_value_tree(price_tree, params);
    
    const double dt = params.time_to_maturity / config_.num_steps;
    double current_time = 0.0;
    std::vector<std::pair<double, double>> boundary_points;
    
    size_t index = 0;
    for (size_t step = 0; step < config_.num_steps; ++step) {
        for (size_t node = 0; node <= step; ++node) {
            double spot = price_tree[index + node];
            double option_value = value_tree[index + node];
            double intrinsic_value = calculate_payoff(spot, params);
            
            if (std::abs(option_value - intrinsic_value) < 1e-10) {
                boundary_points.emplace_back(current_time, spot);
                break;
            }
        }
        index += step + 1;
        current_time += dt;
    }
    
    // Return the average boundary price
    if (boundary_points.empty()) {
        return params.strike_price;
    }
    
    double sum = 0.0;
    for (const auto& point : boundary_points) {
        sum += point.second;
    }
    return sum / boundary_points.size();
}

std::vector<double> BinomialPricer::calculate_critical_prices(const OptionParameters& params) {
    if (!config_.american) {
        throw std::runtime_error("Critical prices only available for American options");
    }
    
    auto price_tree = build_price_tree(params);
    auto value_tree = build_value_tree(price_tree, params);
    
    std::vector<double> critical_prices;
    critical_prices.reserve(config_.num_steps + 1);
    
    size_t index = 0;
    for (size_t step = 0; step < config_.num_steps; ++step) {
        bool found = false;
        for (size_t node = 0; node <= step; ++node) {
            double spot = price_tree[index + node];
            double option_value = value_tree[index + node];
            double intrinsic_value = calculate_payoff(spot, params);
            
            if (std::abs(option_value - intrinsic_value) < 1e-10) {
                critical_prices.push_back(spot);
                found = true;
                break;
            }
        }
        if (!found) {
            critical_prices.push_back(params.strike_price);
        }
        index += step + 1;
    }
    
    return critical_prices;
}

} // namespace pricing
} // namespace optionstrader
