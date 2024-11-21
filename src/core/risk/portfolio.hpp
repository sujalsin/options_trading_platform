#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>
#include "../pricing/pricing_engine.hpp"

namespace optionstrader {
namespace risk {

struct Position {
    std::string instrument_id;
    double quantity;
    pricing::OptionParameters params;
    std::string pricing_model;  // "black_scholes", "monte_carlo", or "binomial"
};

struct PortfolioRisk {
    double total_value;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
    double value_at_risk;  // 95% VaR
    double expected_shortfall;  // 95% ES/CVaR
};

class Portfolio {
public:
    Portfolio();
    
    // Position management
    void add_position(const Position& position);
    void remove_position(const std::string& instrument_id);
    void update_position(const std::string& instrument_id, double new_quantity);
    
    // Accessor methods
    const std::map<std::string, Position>& get_positions() const { return positions_; }
    
    // Risk calculations
    PortfolioRisk calculate_risk() const;
    double calculate_var(double confidence_level = 0.95, int days = 10) const;
    double calculate_expected_shortfall(double confidence_level = 0.95, int days = 10) const;
    
    // Stress testing
    std::vector<double> stress_test_market_crash() const;
    std::vector<double> stress_test_volatility_spike() const;
    std::vector<double> stress_test_interest_rate_shock() const;
    
    // Scenario analysis
    struct ScenarioResult {
        double portfolio_value;
        PortfolioRisk risk_metrics;
    };
    std::vector<ScenarioResult> run_monte_carlo_scenarios(int num_scenarios = 1000) const;
    
    // Portfolio optimization
    void optimize_hedge_ratios();
    void rebalance_portfolio(const std::vector<double>& target_weights);
    
    // Pricing engine creation
    std::shared_ptr<pricing::PricingEngine> create_pricer(const std::string& model) const;

    // Enhanced portfolio optimization and risk management structures and methods
    struct RiskMetrics {
        double total_value;
        double delta;
        double gamma;
        double vega;
        double theta;
        double rho;
        double sharpe_ratio;
        double max_drawdown;
        double tracking_error;
    };

    struct RiskConstraints {
        double max_position_size;      // Maximum size for any single position
        double max_portfolio_value;    // Maximum total portfolio value
        double max_delta;             // Maximum absolute delta exposure
        double max_gamma;             // Maximum absolute gamma exposure
        double max_vega;              // Maximum absolute vega exposure
        double min_sharpe_ratio;      // Minimum required Sharpe ratio
        double max_var;               // Maximum Value at Risk
        double max_leverage;          // Maximum leverage ratio
    };

    struct OptimizationConfig {
        enum class Objective {
            MIN_VARIANCE,           // Minimize portfolio variance
            MAX_SHARPE_RATIO,      // Maximize Sharpe ratio
            MIN_TRACKING_ERROR,    // Minimize tracking error to benchmark
            MAX_UTILITY            // Maximize utility function
        };

        Objective objective;
        RiskConstraints constraints;
        double risk_free_rate;
        double target_return;
        std::vector<double> benchmark_weights;  // For tracking error minimization
    };

    // Enhanced optimization methods
    void optimize_portfolio(const OptimizationConfig& config);
    void optimize_mean_variance(double target_return);
    void optimize_risk_parity();
    void optimize_maximum_diversification();
    bool check_risk_constraints(const RiskConstraints& constraints) const;
    
    // Enhanced risk metrics
    double calculate_portfolio_variance() const;
    double calculate_sharpe_ratio(double risk_free_rate) const;
    double calculate_tracking_error(const std::vector<double>& benchmark_weights) const;
    double calculate_max_drawdown() const;
    double calculate_information_ratio(const std::vector<double>& benchmark_weights) const;

private:
    std::map<std::string, Position> positions_;
    RiskConstraints risk_constraints_;
    
    // Helper methods for optimization
    std::vector<std::vector<double>> calculate_covariance_matrix() const;
    std::vector<double> calculate_expected_returns() const;
    double calculate_portfolio_utility(const std::vector<double>& weights, double risk_aversion) const;
    
    // Enhanced correlation methods
    double calculate_correlation(const std::vector<double>& returns1,
                               const std::vector<double>& returns2,
                               bool use_rank = false) const;
    double calculate_tail_dependence(const std::vector<double>& returns1,
                                   const std::vector<double>& returns2) const;
};

} // namespace risk
} // namespace optionstrader
