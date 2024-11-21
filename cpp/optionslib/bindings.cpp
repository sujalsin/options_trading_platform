#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "options_pricing.hpp"

namespace py = pybind11;

PYBIND11_MODULE(optionslib_cpp, m) {
    m.doc() = "C++ implementation of options pricing models"; // optional module docstring
    
    py::class_<optionslib::OptionPricingCpp::OptionParams>(m, "OptionParamsCpp")
        .def(py::init<>())
        .def_readwrite("S", &optionslib::OptionPricingCpp::OptionParams::S)
        .def_readwrite("K", &optionslib::OptionPricingCpp::OptionParams::K)
        .def_readwrite("T", &optionslib::OptionPricingCpp::OptionParams::T)
        .def_readwrite("r", &optionslib::OptionPricingCpp::OptionParams::r)
        .def_readwrite("sigma", &optionslib::OptionPricingCpp::OptionParams::sigma)
        .def_readwrite("q", &optionslib::OptionPricingCpp::OptionParams::q)
        .def_readwrite("is_call", &optionslib::OptionPricingCpp::OptionParams::is_call);
    
    py::class_<optionslib::OptionPricingCpp::OptionResult>(m, "OptionResultCpp")
        .def(py::init<>())
        .def_readonly("price", &optionslib::OptionPricingCpp::OptionResult::price)
        .def_readonly("delta", &optionslib::OptionPricingCpp::OptionResult::delta)
        .def_readonly("gamma", &optionslib::OptionPricingCpp::OptionResult::gamma)
        .def_readonly("theta", &optionslib::OptionPricingCpp::OptionResult::theta)
        .def_readonly("vega", &optionslib::OptionPricingCpp::OptionResult::vega)
        .def_readonly("rho", &optionslib::OptionPricingCpp::OptionResult::rho);
    
    m.def("black_scholes_cpp", &optionslib::OptionPricingCpp::black_scholes, 
          "Calculate option price and Greeks using Black-Scholes model");
    
    m.def("binomial_tree_cpp", &optionslib::OptionPricingCpp::binomial_tree, 
          "Calculate option price and Greeks using Binomial Tree model",
          py::arg("params"), py::arg("steps") = 100);
    
    m.def("monte_carlo_cpp", &optionslib::OptionPricingCpp::monte_carlo,
          "Calculate option price using Monte Carlo simulation",
          py::arg("params"), py::arg("num_sims") = 100000, py::arg("time_steps") = 100);
}
