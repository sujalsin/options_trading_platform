add_library(options_core
    core/pricing/pricing_engine.cpp
    core/pricing/monte_carlo_pricer.cpp
    core/pricing/binomial_pricer.cpp
    core/risk/portfolio.cpp
)

target_include_directories(options_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(options_core
    PUBLIC
        TBB::tbb
)

# Create Python module using pybind11
pybind11_add_module(optionstrader_py 
    python_bindings/bindings.cpp
)

target_link_libraries(optionstrader_py 
    PRIVATE 
        options_core
)
