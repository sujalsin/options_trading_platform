#include "portfolio.hpp"
#include "../pricing/monte_carlo_pricer.hpp"
#include "../pricing/binomial_pricer.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace optionstrader {
namespace risk {

Portfolio::Portfolio() {}

void Portfolio::add_position(const Position& position) {
    positions_[position.instrument_id] = position;
}

void Portfolio::remove_position(const std::string& instrument_id) {
    positions_.erase(instrument_id);
}

void Portfolio::update_position(const std::string& instrument_id, double new_quantity) {
    if (positions_.find(instrument_id) != positions_.end()) {
        positions_[instrument_id].quantity = new_quantity;
    }
}

std::shared_ptr<pricing::PricingEngine> Portfolio::create_pricer(const std::string& model) const {
    if (model == "black_scholes") {
        return std::make_shared<pricing::BlackScholesPricer>();
    } else if (model == "monte_carlo") {
        return std::make_shared<pricing::MonteCarloPricer>();
    } else if (model == "binomial") {
        return std::make_shared<pricing::BinomialPricer>();
    }
    throw std::runtime_error("Unknown pricing model: " + model);
}

PortfolioRisk Portfolio::calculate_risk() const {
    PortfolioRisk risk = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    for (const auto& [id, position] : positions_) {
        auto pricer = create_pricer(position.pricing_model);
        
        double price = pricer->calculate_price(position.params);
        double delta = pricer->calculate_delta(position.params);
        double gamma = pricer->calculate_gamma(position.params);
        double vega = pricer->calculate_vega(position.params);
        double theta = pricer->calculate_theta(position.params);
        double rho = pricer->calculate_rho(position.params);
        
        risk.total_value += position.quantity * price;
        risk.delta += position.quantity * delta;
        risk.gamma += position.quantity * gamma;
        risk.vega += position.quantity * vega;
        risk.theta += position.quantity * theta;
        risk.rho += position.quantity * rho;
    }
    
    risk.value_at_risk = calculate_var();
    risk.expected_shortfall = calculate_expected_shortfall();
    
    return risk;
}

std::vector<double> Portfolio::simulate_returns(int num_scenarios) const {
    std::vector<double> returns(num_scenarios);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal(0.0, 1.0);
    
    for (int i = 0; i < num_scenarios; ++i) {
        double portfolio_return = 0.0;
        
        for (const auto& [id, position] : positions_) {
            // Simulate price movement
            double z = normal(gen);
            double price_return = position.params.risk_free_rate -
                                0.5 * position.params.volatility * position.params.volatility +
                                position.params.volatility * z;
            
            portfolio_return += position.quantity * price_return;
        }
        
        returns[i] = portfolio_return;
    }
    
    return returns;
}

double Portfolio::calculate_var(double confidence_level, int days) const {
    std::vector<double> returns = simulate_returns(10000);
    std::sort(returns.begin(), returns.end());
    
    size_t var_index = static_cast<size_t>((1.0 - confidence_level) * returns.size());
    double daily_var = -returns[var_index];
    
    return daily_var * std::sqrt(static_cast<double>(days));
}

double Portfolio::calculate_expected_shortfall(double confidence_level, int days) const {
    std::vector<double> returns = simulate_returns(10000);
    std::sort(returns.begin(), returns.end());
    
    size_t var_index = static_cast<size_t>((1.0 - confidence_level) * returns.size());
    double sum = 0.0;
    
    for (size_t i = 0; i < var_index; ++i) {
        sum += returns[i];
    }
    
    double daily_es = -sum / var_index;
    return daily_es * std::sqrt(static_cast<double>(days));
}

std::vector<double> Portfolio::stress_test_market_crash() const {
    std::vector<double> scenario_values;
    const std::vector<double> price_shocks = {-0.05, -0.10, -0.20, -0.30}; // 5%, 10%, 20%, 30% drops
    
    for (double shock : price_shocks) {
        Portfolio stress_portfolio = *this;
        
        // Apply market crash scenario
        for (auto& [id, position] : stress_portfolio.positions_) {
            position.params.spot_price *= (1.0 + shock);
            position.params.volatility *= 1.5;  // Volatility typically spikes in crashes
            position.params.risk_free_rate *= 0.8;  // Rates typically fall in crashes
        }
        
        auto risk = stress_portfolio.calculate_risk();
        scenario_values.push_back(risk.total_value);
    }
    
    return scenario_values;
}

std::vector<double> Portfolio::stress_test_volatility_spike() const {
    std::vector<double> scenario_values;
    const std::vector<double> vol_multipliers = {1.5, 2.0, 3.0, 5.0}; // 50%, 100%, 200%, 400% vol increase
    
    for (double mult : vol_multipliers) {
        Portfolio stress_portfolio = *this;
        
        // Apply volatility spike scenario
        for (auto& [id, position] : stress_portfolio.positions_) {
            position.params.volatility *= mult;
            position.params.spot_price *= (1.0 - 0.05 * (mult - 1.0)); // Price typically falls with vol spikes
        }
        
        auto risk = stress_portfolio.calculate_risk();
        scenario_values.push_back(risk.total_value);
    }
    
    return scenario_values;
}

std::vector<double> Portfolio::stress_test_interest_rate_shock() const {
    std::vector<double> scenario_values;
    const std::vector<double> rate_shocks = {0.01, 0.02, 0.03, 0.05}; // 100, 200, 300, 500 bps
    
    for (double shock : rate_shocks) {
        Portfolio stress_portfolio = *this;
        
        // Apply interest rate shock scenario
        for (auto& [id, position] : stress_portfolio.positions_) {
            position.params.risk_free_rate += shock;
            position.params.spot_price *= (1.0 - shock * 5.0); // Equity prices typically fall with rate hikes
        }
        
        auto risk = stress_portfolio.calculate_risk();
        scenario_values.push_back(risk.total_value);
    }
    
    return scenario_values;
}

std::vector<Portfolio::ScenarioResult> Portfolio::run_monte_carlo_scenarios(int num_scenarios) const {
    std::vector<ScenarioResult> results(num_scenarios);
    
    // Use TBB for parallel processing
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_scenarios),
        [&](const tbb::blocked_range<size_t>& range) {
            // Thread-local random number generators
            thread_local std::random_device rd;
            thread_local std::mt19937 gen(rd());
            
            // Initialize distributions for each risk factor
            std::normal_distribution<> price_change(0.0, 0.01);  // 1% daily volatility
            std::normal_distribution<> vol_change(0.0, 0.005);   // 0.5% vol change
            std::normal_distribution<> rate_change(0.0, 0.001);  // 0.1% rate change
            std::normal_distribution<> correlation_shock(0.0, 0.1); // 10% correlation shock
            
            for (size_t i = range.begin(); i != range.end(); ++i) {
                Portfolio scenario_portfolio = *this;
                
                // Generate correlated market shocks
                std::vector<double> market_shocks;
                double correlation_factor = 1.0 + correlation_shock(gen);
                
                // Apply random shocks to each position with correlation
                for (auto& [id, position] : scenario_portfolio.positions_) {
                    // Generate base shocks
                    double base_shock = price_change(gen);
                    double vol_shock = vol_change(gen);
                    double rate_shock = rate_change(gen);
                    
                    // Apply correlation to shocks
                    double correlated_price_shock = base_shock * correlation_factor;
                    double correlated_vol_shock = vol_shock * correlation_factor;
                    
                    // Apply shocks with realistic market dynamics
                    position.params.spot_price *= std::exp(correlated_price_shock);
                    position.params.volatility = std::max(0.01, position.params.volatility * (1.0 + correlated_vol_shock));
                    position.params.risk_free_rate = std::max(0.0, position.params.risk_free_rate + rate_shock);
                    
                    // Apply term structure effects
                    if (position.params.time_to_maturity > 0) {
                        position.params.time_to_maturity -= 1.0/252.0; // Daily decay
                    }
                }
                
                // Calculate scenario risk metrics with enhanced measures
                auto risk = scenario_portfolio.calculate_risk();
                results[i].portfolio_value = risk.total_value;
                results[i].risk_metrics = risk;
            }
        });
    
    return results;
}

void Portfolio::optimize_hedge_ratios() {
    // Implement portfolio optimization using quadratic programming
    // This is a placeholder for more sophisticated optimization
    double total_delta = 0.0;
    
    for (const auto& [id, position] : positions_) {
        auto pricer = create_pricer(position.pricing_model);
        total_delta += position.quantity * pricer->calculate_delta(position.params);
    }
    
    // Simple delta-neutral adjustment
    if (std::abs(total_delta) > 0.01) {
        for (auto& [id, position] : positions_) {
            auto pricer = create_pricer(position.pricing_model);
            double delta = pricer->calculate_delta(position.params);
            if (std::abs(delta) > 0.01) {
                position.quantity -= total_delta / delta;
                break;
            }
        }
    }
}

void Portfolio::rebalance_portfolio(const std::vector<double>& target_weights) {
    if (target_weights.size() != positions_.size()) {
        throw std::runtime_error("Number of target weights must match number of positions");
    }
    
    // Calculate current portfolio value and position values
    double total_value = 0.0;
    std::vector<double> position_values;
    std::vector<double> prices;
    
    for (const auto& [id, position] : positions_) {
        auto pricer = create_pricer(position.pricing_model);
        double price = pricer->calculate_price(position.params);
        double value = std::abs(position.quantity) * price;
        total_value += value;
        position_values.push_back(value);
        prices.push_back(price);
    }
    
    // Calculate and apply new quantities
    size_t i = 0;
    for (auto& [id, position] : positions_) {
        double target_value = target_weights[i] * total_value;
        double current_value = position_values[i];
        
        if (current_value > 0) {  // Avoid division by zero
            // Keep the sign (long/short) but adjust the quantity
            int sign = (position.quantity >= 0) ? 1 : -1;
            position.quantity = sign * (target_value / prices[i]);
        }
        ++i;
    }
}

double Portfolio::calculate_correlation(const std::vector<double>& returns1,
                                     const std::vector<double>& returns2,
                                     bool use_rank) const {
    if (returns1.size() != returns2.size() || returns1.empty()) {
        throw std::invalid_argument("Return series must be non-empty and of equal length");
    }
    
    if (use_rank) {
        // Calculate Spearman rank correlation
        std::vector<double> rank1(returns1.size()), rank2(returns2.size());
        std::iota(rank1.begin(), rank1.end(), 0.0);
        std::iota(rank2.begin(), rank2.end(), 0.0);
        
        std::sort(rank1.begin(), rank1.end(),
                 [&](double i, double j) { return returns1[i] < returns1[j]; });
        std::sort(rank2.begin(), rank2.end(),
                 [&](double i, double j) { return returns2[i] < returns2[j]; });
                 
        return calculate_correlation(rank1, rank2, false);
    }
    
    // Calculate Pearson correlation
    double sum1 = 0.0, sum2 = 0.0, sum12 = 0.0;
    double sum1_sq = 0.0, sum2_sq = 0.0;
    
    for (size_t i = 0; i < returns1.size(); ++i) {
        sum1 += returns1[i];
        sum2 += returns2[i];
        sum12 += returns1[i] * returns2[i];
        sum1_sq += returns1[i] * returns1[i];
        sum2_sq += returns2[i] * returns2[i];
    }
    
    double n = static_cast<double>(returns1.size());
    double num = n * sum12 - sum1 * sum2;
    double den = std::sqrt((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2));
    
    return den == 0.0 ? 0.0 : num / den;
}

double Portfolio::calculate_tail_dependence(const std::vector<double>& returns1,
                                         const std::vector<double>& returns2) const {
    if (returns1.size() != returns2.size() || returns1.empty()) {
        throw std::invalid_argument("Return series must be non-empty and of equal length");
    }
    
    // Calculate empirical tail dependence coefficient
    const double tail_threshold = 0.05; // 5% tail
    const size_t tail_size = static_cast<size_t>(returns1.size() * tail_threshold);
    
    std::vector<size_t> extreme_events = 0;
    std::vector<std::pair<double, double>> paired_returns;
    
    for (size_t i = 0; i < returns1.size(); ++i) {
        paired_returns.push_back({returns1[i], returns2[i]});
    }
    
    // Sort by first series
    std::sort(paired_returns.begin(), paired_returns.end());
    
    // Count joint tail events
    size_t joint_extremes = 0;
    for (size_t i = 0; i < tail_size; ++i) {
        if (paired_returns[i].second <= paired_returns[tail_size - 1].second) {
            ++joint_extremes;
        }
    }
    
    return static_cast<double>(joint_extremes) / tail_size;
}

void Portfolio::optimize_portfolio(const OptimizationConfig& config) {
    switch (config.objective) {
        case OptimizationConfig::Objective::MIN_VARIANCE:
            optimize_mean_variance(config.target_return);
            break;
        case OptimizationConfig::Objective::MAX_SHARPE_RATIO:
            optimize_maximum_sharpe_ratio(config.risk_free_rate);
            break;
        case OptimizationConfig::Objective::MIN_TRACKING_ERROR:
            optimize_tracking_error(config.benchmark_weights);
            break;
        case OptimizationConfig::Objective::MAX_UTILITY:
            optimize_maximum_utility(config.risk_free_rate);
            break;
    }
    
    // Apply risk constraints after optimization
    if (!check_risk_constraints(config.constraints)) {
        // If constraints are violated, adjust positions
        adjust_for_risk_constraints(config.constraints);
    }
}

void Portfolio::optimize_mean_variance(double target_return) {
    // Calculate covariance matrix and expected returns
    auto covariance_matrix = calculate_covariance_matrix();
    auto expected_returns = calculate_expected_returns();
    
    // Implement quadratic programming optimization
    size_t n = positions_.size();
    std::vector<double> optimal_weights(n);
    
    // Use quadratic programming solver (placeholder)
    // In practice, you would use a library like OSQP or Gurobi
    // This is a simplified implementation
    double lambda = 0.5; // Risk aversion parameter
    
    // Solve for optimal weights using gradient descent
    std::vector<double> gradient(n, 0.0);
    const int max_iterations = 1000;
    const double learning_rate = 0.01;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Calculate gradient
        for (size_t i = 0; i < n; ++i) {
            gradient[i] = 0.0;
            for (size_t j = 0; j < n; ++j) {
                gradient[i] += covariance_matrix[i][j] * optimal_weights[j];
            }
            gradient[i] = 2 * lambda * gradient[i] - expected_returns[i];
        }
        
        // Update weights
        for (size_t i = 0; i < n; ++i) {
            optimal_weights[i] -= learning_rate * gradient[i];
        }
        
        // Project onto simplex (ensure weights sum to 1)
        double sum = std::accumulate(optimal_weights.begin(), optimal_weights.end(), 0.0);
        for (double& w : optimal_weights) {
            w /= sum;
        }
    }
    
    // Apply optimal weights
    size_t i = 0;
    for (auto& [id, position] : positions_) {
        auto pricer = create_pricer(position.pricing_model);
        double price = pricer->calculate_price(position.params);
        position.quantity = optimal_weights[i] * calculate_risk().total_value / price;
        ++i;
    }
}

void Portfolio::optimize_risk_parity() {
    // Implement risk parity optimization
    auto covariance_matrix = calculate_covariance_matrix();
    size_t n = positions_.size();
    std::vector<double> risk_contributions(n, 1.0 / n); // Target equal risk contribution
    std::vector<double> optimal_weights(n, 1.0 / n);    // Start with equal weights
    
    const int max_iterations = 1000;
    const double tolerance = 1e-6;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Calculate portfolio volatility
        double portfolio_vol = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                portfolio_vol += optimal_weights[i] * optimal_weights[j] * covariance_matrix[i][j];
            }
        }
        portfolio_vol = std::sqrt(portfolio_vol);
        
        // Calculate current risk contributions
        std::vector<double> current_rc(n);
        for (size_t i = 0; i < n; ++i) {
            current_rc[i] = optimal_weights[i] * 
                           (std::accumulate(optimal_weights.begin(), optimal_weights.end(), 0.0,
                                          [&](double sum, double w) {
                                              return sum + w * covariance_matrix[i][&w - &optimal_weights[0]];
                                          })) / portfolio_vol;
        }
        
        // Update weights
        bool converged = true;
        for (size_t i = 0; i < n; ++i) {
            double adjustment = risk_contributions[i] / current_rc[i];
            double new_weight = optimal_weights[i] * std::sqrt(adjustment);
            
            if (std::abs(new_weight - optimal_weights[i]) > tolerance) {
                converged = false;
            }
            
            optimal_weights[i] = new_weight;
        }
        
        // Normalize weights
        double sum = std::accumulate(optimal_weights.begin(), optimal_weights.end(), 0.0);
        for (double& w : optimal_weights) {
            w /= sum;
        }
        
        if (converged) break;
    }
    
    // Apply optimal weights
    size_t i = 0;
    for (auto& [id, position] : positions_) {
        auto pricer = create_pricer(position.pricing_model);
        double price = pricer->calculate_price(position.params);
        position.quantity = optimal_weights[i] * calculate_risk().total_value / price;
        ++i;
    }
}

void Portfolio::optimize_maximum_diversification() {
    // Calculate asset volatilities and correlation matrix
    auto covariance_matrix = calculate_covariance_matrix();
    size_t n = positions_.size();
    std::vector<double> volatilities(n);
    
    for (size_t i = 0; i < n; ++i) {
        volatilities[i] = std::sqrt(covariance_matrix[i][i]);
    }
    
    // Calculate diversification ratio for given weights
    auto calc_div_ratio = [&](const std::vector<double>& weights) {
        double weighted_vol = 0.0;
        double portfolio_vol = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            weighted_vol += weights[i] * volatilities[i];
            for (size_t j = 0; j < n; ++j) {
                portfolio_vol += weights[i] * weights[j] * covariance_matrix[i][j];
            }
        }
        
        portfolio_vol = std::sqrt(portfolio_vol);
        return weighted_vol / portfolio_vol;
    };
    
    // Optimize using gradient ascent
    std::vector<double> optimal_weights(n, 1.0 / n);
    const int max_iterations = 1000;
    const double learning_rate = 0.01;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<double> gradient(n);
        
        // Numerical gradient
        for (size_t i = 0; i < n; ++i) {
            const double h = 1e-7;
            std::vector<double> weights_plus = optimal_weights;
            std::vector<double> weights_minus = optimal_weights;
            
            weights_plus[i] += h;
            weights_minus[i] -= h;
            
            gradient[i] = (calc_div_ratio(weights_plus) - calc_div_ratio(weights_minus)) / (2 * h);
        }
        
        // Update weights
        for (size_t i = 0; i < n; ++i) {
            optimal_weights[i] += learning_rate * gradient[i];
        }
        
        // Project onto simplex
        double sum = std::accumulate(optimal_weights.begin(), optimal_weights.end(), 0.0);
        for (double& w : optimal_weights) {
            w = std::max(0.0, w / sum);
        }
    }
    
    // Apply optimal weights
    size_t i = 0;
    for (auto& [id, position] : positions_) {
        auto pricer = create_pricer(position.pricing_model);
        double price = pricer->calculate_price(position.params);
        position.quantity = optimal_weights[i] * calculate_risk().total_value / price;
        ++i;
    }
}

bool Portfolio::check_risk_constraints(const RiskConstraints& constraints) const {
    auto risk = calculate_risk();
    double total_value = std::abs(risk.total_value);
    
    // Check position size constraints
    for (const auto& [id, position] : positions_) {
        auto pricer = create_pricer(position.pricing_model);
        double position_value = std::abs(position.quantity * pricer->calculate_price(position.params));
        
        if (position_value > constraints.max_position_size) {
            return false;
        }
    }
    
    // Check portfolio-level constraints
    if (total_value > constraints.max_portfolio_value ||
        std::abs(risk.delta) > constraints.max_delta ||
        std::abs(risk.gamma) > constraints.max_gamma ||
        std::abs(risk.vega) > constraints.max_vega ||
        calculate_sharpe_ratio(0.02) < constraints.min_sharpe_ratio ||
        calculate_var(0.99, 10) > constraints.max_var) {
        return false;
    }
    
    return true;
}

void Portfolio::adjust_for_risk_constraints(const RiskConstraints& constraints) {
    // First, adjust individual position sizes
    for (auto& [id, position] : positions_) {
        auto pricer = create_pricer(position.pricing_model);
        double price = pricer->calculate_price(position.params);
        double position_value = std::abs(position.quantity * price);
        
        if (position_value > constraints.max_position_size) {
            double scale = constraints.max_position_size / position_value;
            position.quantity *= scale;
        }
    }
    
    // Then, adjust for portfolio-level constraints
    auto risk = calculate_risk();
    
    // Adjust for delta constraint
    if (std::abs(risk.delta) > constraints.max_delta) {
        double scale = constraints.max_delta / std::abs(risk.delta);
        for (auto& [id, position] : positions_) {
            position.quantity *= scale;
        }
    }
    
    // Adjust for gamma constraint
    if (std::abs(risk.gamma) > constraints.max_gamma) {
        double scale = constraints.max_gamma / std::abs(risk.gamma);
        for (auto& [id, position] : positions_) {
            position.quantity *= scale;
        }
    }
    
    // Adjust for vega constraint
    if (std::abs(risk.vega) > constraints.max_vega) {
        double scale = constraints.max_vega / std::abs(risk.vega);
        for (auto& [id, position] : positions_) {
            position.quantity *= scale;
        }
    }
    
    // Final portfolio value check
    if (std::abs(risk.total_value) > constraints.max_portfolio_value) {
        double scale = constraints.max_portfolio_value / std::abs(risk.total_value);
        for (auto& [id, position] : positions_) {
            position.quantity *= scale;
        }
    }
}

} // namespace risk
} // namespace optionstrader
