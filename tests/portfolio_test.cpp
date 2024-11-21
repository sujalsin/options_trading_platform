#include <gtest/gtest.h>
#include "../src/core/risk/portfolio.hpp"
#include <cmath>

using namespace optionstrader::risk;
using namespace optionstrader::pricing;

class PortfolioTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a test portfolio with a call and put option
        OptionParameters call_params{
            100.0,  // spot_price
            100.0,  // strike_price
            0.05,   // risk_free_rate
            0.2,    // volatility
            1.0,    // time_to_maturity
            true    // is_call
        };

        OptionParameters put_params{
            100.0,  // spot_price
            100.0,  // strike_price
            0.05,   // risk_free_rate
            0.2,    // volatility
            1.0,    // time_to_maturity
            false   // is_call
        };

        Position call_position{
            "AAPL_CALL_100",
            100.0,
            call_params,
            "black_scholes"
        };

        Position put_position{
            "AAPL_PUT_100",
            -50.0,
            put_params,
            "black_scholes"
        };

        portfolio.add_position(call_position);
        portfolio.add_position(put_position);
    }

    Portfolio portfolio;
};

TEST_F(PortfolioTest, CalculateRisk) {
    auto risk = portfolio.calculate_risk();
    
    // Basic sanity checks
    EXPECT_GT(risk.total_value, 0.0);
    EXPECT_NE(risk.delta, 0.0);
    EXPECT_NE(risk.gamma, 0.0);
    EXPECT_GT(risk.vega, 0.0);
    EXPECT_NE(risk.theta, 0.0);
    EXPECT_NE(risk.rho, 0.0);
}

TEST_F(PortfolioTest, ValueAtRisk) {
    double var = portfolio.calculate_var(0.95, 10);
    double es = portfolio.calculate_expected_shortfall(0.95, 10);
    
    // VaR should be positive (we're measuring potential losses)
    EXPECT_GT(var, 0.0);
    // ES should be greater than VaR
    EXPECT_GT(es, var);
}

TEST_F(PortfolioTest, StressTests) {
    auto market_crash = portfolio.stress_test_market_crash();
    auto vol_spike = portfolio.stress_test_volatility_spike();
    auto rate_shock = portfolio.stress_test_interest_rate_shock();
    
    // Check we have the expected number of scenarios
    EXPECT_EQ(market_crash.size(), 4);
    EXPECT_EQ(vol_spike.size(), 4);
    EXPECT_EQ(rate_shock.size(), 4);
    
    // Market crash should lead to decreasing portfolio values
    for (size_t i = 1; i < market_crash.size(); ++i) {
        EXPECT_LT(market_crash[i], market_crash[i-1]);
    }
    
    // Volatility spike should affect option values
    for (size_t i = 1; i < vol_spike.size(); ++i) {
        EXPECT_NE(vol_spike[i], vol_spike[i-1]);
    }
}

TEST_F(PortfolioTest, MonteCarloScenarios) {
    int num_scenarios = 1000;
    auto scenarios = portfolio.run_monte_carlo_scenarios(num_scenarios);
    
    EXPECT_EQ(scenarios.size(), num_scenarios);
    
    // Check that we have some variation in the results
    double sum = 0.0;
    double sum_sq = 0.0;
    double prev_value = 0.0;
    bool has_different_values = false;
    
    for (const auto& scenario : scenarios) {
        sum += scenario.portfolio_value;
        sum_sq += scenario.portfolio_value * scenario.portfolio_value;
        
        if (scenario.portfolio_value != prev_value && prev_value != 0.0) {
            has_different_values = true;
        }
        prev_value = scenario.portfolio_value;
    }
    
    double mean = sum / num_scenarios;
    double variance = (sum_sq / num_scenarios) - (mean * mean);
    
    // We expect some variation in the Monte Carlo results
    EXPECT_TRUE(has_different_values) << "All scenario values are identical!";
    EXPECT_GT(variance, 0.0) << "Variance is zero! Mean: " << mean << ", Sum_sq: " << sum_sq;
}

TEST_F(PortfolioTest, PortfolioOptimization) {
    // Test initial delta
    auto initial_risk = portfolio.calculate_risk();
    double initial_delta = initial_risk.delta;
    
    // Optimize hedge ratios
    portfolio.optimize_hedge_ratios();
    
    // Check that delta has been reduced
    auto final_risk = portfolio.calculate_risk();
    EXPECT_LT(std::abs(final_risk.delta), std::abs(initial_delta));
}

TEST_F(PortfolioTest, PortfolioRebalancing) {
    std::vector<double> target_weights = {0.7, 0.3};  // 70-30 portfolio
    
    // Store initial quantities
    auto initial_risk = portfolio.calculate_risk();
    double initial_call_quantity = 0.0;
    double initial_put_quantity = 0.0;
    
    for (const auto& [id, position] : portfolio.get_positions()) {
        if (position.instrument_id == "AAPL_CALL_100") {
            initial_call_quantity = position.quantity;
        } else if (position.instrument_id == "AAPL_PUT_100") {
            initial_put_quantity = position.quantity;
        }
    }
    
    // Rebalance portfolio
    portfolio.rebalance_portfolio(target_weights);
    
    // Check that position quantities have changed
    bool quantities_changed = false;
    for (const auto& [id, position] : portfolio.get_positions()) {
        if (position.instrument_id == "AAPL_CALL_100") {
            quantities_changed |= (std::abs(position.quantity - initial_call_quantity) > 1e-10);
        } else if (position.instrument_id == "AAPL_PUT_100") {
            quantities_changed |= (std::abs(position.quantity - initial_put_quantity) > 1e-10);
        }
    }
    
    EXPECT_TRUE(quantities_changed) << "Position quantities did not change after rebalancing!";
    
    // Check that the portfolio value proportions match target weights
    auto final_positions = portfolio.get_positions();
    double total_value = 0.0;
    std::vector<double> position_values;
    
    for (const auto& [id, position] : final_positions) {
        auto pricer = portfolio.create_pricer(position.pricing_model);
        double value = std::abs(position.quantity * pricer->calculate_price(position.params));
        total_value += value;
        position_values.push_back(value);
    }
    
    std::vector<double> actual_weights;
    for (double value : position_values) {
        actual_weights.push_back(value / total_value);
    }
    
    // Check weights are close to target weights (within 1%)
    for (size_t i = 0; i < target_weights.size(); ++i) {
        EXPECT_NEAR(actual_weights[i], target_weights[i], 0.01) 
            << "Weight mismatch at position " << i 
            << ". Expected: " << target_weights[i] 
            << ", Actual: " << actual_weights[i];
    }
}

TEST_F(PortfolioTest, OptimizationAndConstraints) {
    // Set up risk constraints
    Portfolio::RiskConstraints constraints;
    constraints.max_position_size = 1000000.0;
    constraints.max_portfolio_value = 2000000.0;
    constraints.max_delta = 100.0;
    constraints.max_gamma = 10.0;
    constraints.max_vega = 1000.0;
    constraints.min_sharpe_ratio = 0.5;
    constraints.max_var = 100000.0;
    constraints.max_leverage = 2.0;
    
    // Test mean-variance optimization
    portfolio.optimize_mean_variance(0.10); // Target 10% return
    auto risk_after_mv = portfolio.calculate_risk();
    EXPECT_GT(risk_after_mv.total_value, 0.0);
    
    // Test risk parity optimization
    portfolio.optimize_risk_parity();
    auto risk_after_rp = portfolio.calculate_risk();
    EXPECT_GT(risk_after_rp.total_value, 0.0);
    
    // Test maximum diversification
    portfolio.optimize_maximum_diversification();
    auto risk_after_md = portfolio.calculate_risk();
    EXPECT_GT(risk_after_md.total_value, 0.0);
    
    // Test portfolio optimization with constraints
    Portfolio::OptimizationConfig config;
    config.objective = Portfolio::OptimizationConfig::Objective::MIN_VARIANCE;
    config.constraints = constraints;
    config.risk_free_rate = 0.02;
    config.target_return = 0.10;
    
    portfolio.optimize_portfolio(config);
    
    // Verify constraints are satisfied
    EXPECT_TRUE(portfolio.check_risk_constraints(constraints));
    
    // Test individual position constraints
    for (const auto& [id, position] : portfolio.get_positions()) {
        auto pricer = portfolio.create_pricer(position.pricing_model);
        double position_value = std::abs(position.quantity * pricer->calculate_price(position.params));
        EXPECT_LE(position_value, constraints.max_position_size);
    }
    
    // Test portfolio-level constraints
    auto final_risk = portfolio.calculate_risk();
    EXPECT_LE(std::abs(final_risk.total_value), constraints.max_portfolio_value);
    EXPECT_LE(std::abs(final_risk.delta), constraints.max_delta);
    EXPECT_LE(std::abs(final_risk.gamma), constraints.max_gamma);
    EXPECT_LE(std::abs(final_risk.vega), constraints.max_vega);
}

TEST_F(PortfolioTest, EnhancedRiskMetrics) {
    // Test enhanced correlation calculations
    std::vector<double> returns1 = {0.01, -0.02, 0.03, -0.01, 0.02};
    std::vector<double> returns2 = {0.02, -0.01, 0.02, -0.02, 0.01};
    
    // Test Pearson correlation
    double pearson_corr = portfolio.calculate_correlation(returns1, returns2, false);
    EXPECT_GE(pearson_corr, -1.0);
    EXPECT_LE(pearson_corr, 1.0);
    
    // Test Spearman rank correlation
    double spearman_corr = portfolio.calculate_correlation(returns1, returns2, true);
    EXPECT_GE(spearman_corr, -1.0);
    EXPECT_LE(spearman_corr, 1.0);
    
    // Test tail dependence
    double tail_dep = portfolio.calculate_tail_dependence(returns1, returns2);
    EXPECT_GE(tail_dep, 0.0);
    EXPECT_LE(tail_dep, 1.0);
}

TEST_F(PortfolioTest, StressTestingScenarios) {
    // Test market crash scenarios
    auto crash_results = portfolio.stress_test_market_crash();
    EXPECT_EQ(crash_results.size(), 4);
    
    // Values should decrease in crash scenario
    for (size_t i = 1; i < crash_results.size(); ++i) {
        EXPECT_LT(crash_results[i], crash_results[i-1]);
    }
    
    // Test volatility spike scenarios
    auto vol_results = portfolio.stress_test_volatility_spike();
    EXPECT_EQ(vol_results.size(), 4);
    
    // Test rate shock scenarios
    auto rate_results = portfolio.stress_test_interest_rate_shock();
    EXPECT_EQ(rate_results.size(), 4);
    
    // Values should change in rate shock scenario
    for (size_t i = 1; i < rate_results.size(); ++i) {
        EXPECT_NE(rate_results[i], rate_results[0]);
    }
}

TEST_F(PortfolioTest, ParallelMonteCarloSimulation) {
    const int num_scenarios = 10000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto scenarios = portfolio.run_monte_carlo_scenarios(num_scenarios);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_EQ(scenarios.size(), num_scenarios);
    
    // Calculate statistics
    double sum = 0.0, sum_sq = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    
    for (const auto& scenario : scenarios) {
        sum += scenario.portfolio_value;
        sum_sq += scenario.portfolio_value * scenario.portfolio_value;
        min_val = std::min(min_val, scenario.portfolio_value);
        max_val = std::max(max_val, scenario.portfolio_value);
    }
    
    double mean = sum / num_scenarios;
    double variance = (sum_sq / num_scenarios) - (mean * mean);
    
    // Check statistical properties
    EXPECT_GT(variance, 0.0);
    EXPECT_LT(min_val, mean);
    EXPECT_GT(max_val, mean);
    
    // Check performance (should be relatively fast due to parallel processing)
    std::cout << "Monte Carlo simulation with " << num_scenarios 
              << " scenarios took " << duration.count() << "ms" << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
