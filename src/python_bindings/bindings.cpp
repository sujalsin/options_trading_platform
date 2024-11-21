#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/pricing/pricing_engine.hpp"
#include "core/pricing/monte_carlo_pricer.hpp"
#include "core/pricing/binomial_pricer.hpp"

namespace py = pybind11;
using namespace optionstrader::pricing;

PYBIND11_MODULE(optionstrader_py, m) {
    m.doc() = "Options Trading Platform - Core C++ functionality exposed to Python";

    py::class_<OptionParameters>(m, "OptionParameters")
        .def(py::init<>())
        .def_readwrite("spot_price", &OptionParameters::spot_price)
        .def_readwrite("strike_price", &OptionParameters::strike_price)
        .def_readwrite("time_to_maturity", &OptionParameters::time_to_maturity)
        .def_readwrite("risk_free_rate", &OptionParameters::risk_free_rate)
        .def_readwrite("volatility", &OptionParameters::volatility)
        .def_readwrite("is_call", &OptionParameters::is_call);

    py::class_<BlackScholesPricer>(m, "BlackScholesPricer")
        .def(py::init<>())
        .def("calculate_price", &BlackScholesPricer::calculate_price)
        .def("calculate_delta", &BlackScholesPricer::calculate_delta)
        .def("calculate_gamma", &BlackScholesPricer::calculate_gamma)
        .def("calculate_vega", &BlackScholesPricer::calculate_vega)
        .def("calculate_theta", &BlackScholesPricer::calculate_theta)
        .def("calculate_rho", &BlackScholesPricer::calculate_rho);

    py::class_<MonteCarloConfig>(m, "MonteCarloConfig")
        .def(py::init<>())
        .def_readwrite("num_paths", &MonteCarloConfig::num_paths)
        .def_readwrite("time_steps", &MonteCarloConfig::time_steps)
        .def_readwrite("num_threads", &MonteCarloConfig::num_threads)
        .def_readwrite("antithetic", &MonteCarloConfig::antithetic)
        .def_readwrite("control_variate", &MonteCarloConfig::control_variate);

    py::class_<MonteCarloPricer>(m, "MonteCarloPricer")
        .def(py::init<const MonteCarloConfig&>())
        .def("calculate_price", &MonteCarloPricer::calculate_price)
        .def("calculate_delta", &MonteCarloPricer::calculate_delta)
        .def("calculate_gamma", &MonteCarloPricer::calculate_gamma)
        .def("calculate_vega", &MonteCarloPricer::calculate_vega)
        .def("calculate_theta", &MonteCarloPricer::calculate_theta)
        .def("calculate_rho", &MonteCarloPricer::calculate_rho)
        .def("calculate_price_with_confidence", &MonteCarloPricer::calculate_price_with_confidence);

    py::class_<BinomialConfig>(m, "BinomialConfig")
        .def(py::init<>())
        .def_readwrite("num_steps", &BinomialConfig::num_steps)
        .def_readwrite("american", &BinomialConfig::american);

    py::class_<BinomialPricer>(m, "BinomialPricer")
        .def(py::init<const BinomialConfig&>())
        .def("calculate_price", &BinomialPricer::calculate_price)
        .def("calculate_delta", &BinomialPricer::calculate_delta)
        .def("calculate_gamma", &BinomialPricer::calculate_gamma)
        .def("calculate_vega", &BinomialPricer::calculate_vega)
        .def("calculate_theta", &BinomialPricer::calculate_theta)
        .def("calculate_rho", &BinomialPricer::calculate_rho)
        .def("calculate_early_exercise_boundary", &BinomialPricer::calculate_early_exercise_boundary)
        .def("calculate_critical_prices", &BinomialPricer::calculate_critical_prices);
}
