#include "monte_carlo_pricer.hpp"
#include <future>
#include <numeric>
#include <cmath>

namespace optionstrader {
namespace pricing {

MonteCarloPricer::MonteCarloPricer(const MonteCarloConfig& config)
    : config_(config), gen_(std::random_device{}()) {}

double MonteCarloPricer::calculate_price(const OptionParameters& params) {
    const size_t paths_per_thread = config_.num_paths / config_.num_threads;
    std::vector<std::future<std::vector<double>>> futures;

    // Launch simulation threads
    for (size_t i = 0; i < config_.num_threads; ++i) {
        futures.push_back(std::async(std::launch::async,
            &MonteCarloPricer::simulate_paths, this, params, paths_per_thread));
    }

    // Collect results
    std::vector<double> all_prices;
    all_prices.reserve(config_.num_paths);
    
    for (auto& future : futures) {
        auto batch_prices = future.get();
        all_prices.insert(all_prices.end(), batch_prices.begin(), batch_prices.end());
    }

    // Calculate mean price
    double sum = std::accumulate(all_prices.begin(), all_prices.end(), 0.0);
    double mean = sum / all_prices.size();
    
    // Apply discount factor
    return std::exp(-params.risk_free_rate * params.time_to_maturity) * mean;
}

std::vector<double> MonteCarloPricer::simulate_paths(const OptionParameters& params, size_t batch_size) const {
    std::vector<double> prices;
    prices.reserve(batch_size);
    
    std::mt19937 local_gen(std::random_device{}());  // Thread-local random generator
    
    for (size_t i = 0; i < batch_size; ++i) {
        double path_price = simulate_path(params, local_gen);
        prices.push_back(path_price);
        
        if (config_.antithetic) {
            // Add antithetic path
            normal_.param(std::normal_distribution<>::param_type(-normal_.mean(), normal_.stddev()));
            double antithetic_price = simulate_path(params, local_gen);
            prices.push_back(antithetic_price);
            normal_.param(std::normal_distribution<>::param_type(0.0, 1.0));
        }
    }
    
    return prices;
}

double MonteCarloPricer::simulate_path(const OptionParameters& params, std::mt19937& gen) const {
    const double dt = params.time_to_maturity / config_.time_steps;
    const double drift = (params.risk_free_rate - 0.5 * params.volatility * params.volatility) * dt;
    const double vol_sqrt_dt = params.volatility * std::sqrt(dt);
    
    double spot = params.spot_price;
    
    for (size_t i = 0; i < config_.time_steps; ++i) {
        double z = normal_(gen);
        spot *= std::exp(drift + vol_sqrt_dt * z);
    }
    
    return calculate_payoff(spot, params);
}

double MonteCarloPricer::calculate_payoff(double spot_price, const OptionParameters& params) const {
    if (params.is_call) {
        return std::max(spot_price - params.strike_price, 0.0);
    } else {
        return std::max(params.strike_price - spot_price, 0.0);
    }
}

std::pair<double, double> MonteCarloPricer::calculate_price_with_confidence(const OptionParameters& params) {
    const size_t paths_per_thread = config_.num_paths / config_.num_threads;
    std::vector<std::future<std::vector<double>>> futures;

    // Launch simulation threads
    for (size_t i = 0; i < config_.num_threads; ++i) {
        futures.push_back(std::async(std::launch::async,
            &MonteCarloPricer::simulate_paths, this, params, paths_per_thread));
    }

    // Collect results
    std::vector<double> all_prices;
    all_prices.reserve(config_.num_paths);
    
    for (auto& future : futures) {
        auto batch_prices = future.get();
        all_prices.insert(all_prices.end(), batch_prices.begin(), batch_prices.end());
    }

    // Calculate mean and standard error
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (double price : all_prices) {
        sum += price;
        sum_sq += price * price;
    }
    
    double mean = sum / all_prices.size();
    double variance = (sum_sq / all_prices.size()) - (mean * mean);
    double std_error = std::sqrt(variance / all_prices.size());
    
    // Apply discount factor
    double discount = std::exp(-params.risk_free_rate * params.time_to_maturity);
    return {discount * mean, discount * std_error};
}

double MonteCarloPricer::calculate_delta(const OptionParameters& params) {
    const double h = params.spot_price * 0.01;  // 1% bump
    
    OptionParameters up_params = params;
    up_params.spot_price += h;
    
    OptionParameters down_params = params;
    down_params.spot_price -= h;
    
    double up_price = calculate_price(up_params);
    double down_price = calculate_price(down_params);
    
    return (up_price - down_price) / (2 * h);
}

double MonteCarloPricer::calculate_gamma(const OptionParameters& params) {
    const double h = params.spot_price * 0.01;  // 1% bump
    
    OptionParameters up_params = params;
    up_params.spot_price += h;
    
    OptionParameters down_params = params;
    down_params.spot_price -= h;
    
    double up_price = calculate_price(up_params);
    double center_price = calculate_price(params);
    double down_price = calculate_price(down_params);
    
    return (up_price - 2 * center_price + down_price) / (h * h);
}

double MonteCarloPricer::calculate_vega(const OptionParameters& params) {
    const double h = 0.01;  // 1% bump in volatility
    
    OptionParameters up_params = params;
    up_params.volatility += h;
    
    OptionParameters down_params = params;
    down_params.volatility -= h;
    
    double up_price = calculate_price(up_params);
    double down_price = calculate_price(down_params);
    
    return (up_price - down_price) / (2 * h);
}

double MonteCarloPricer::calculate_theta(const OptionParameters& params) {
    const double h = 1.0 / 365.0;  // One day
    
    OptionParameters forward_params = params;
    forward_params.time_to_maturity -= h;
    
    double forward_price = calculate_price(forward_params);
    double spot_price = calculate_price(params);
    
    return (forward_price - spot_price) / h;
}

double MonteCarloPricer::calculate_rho(const OptionParameters& params) {
    const double h = 0.0001;  // 1 basis point
    
    OptionParameters up_params = params;
    up_params.risk_free_rate += h;
    
    OptionParameters down_params = params;
    down_params.risk_free_rate -= h;
    
    double up_price = calculate_price(up_params);
    double down_price = calculate_price(down_params);
    
    return (up_price - down_price) / (2 * h);
}

} // namespace pricing
} // namespace optionstrader
