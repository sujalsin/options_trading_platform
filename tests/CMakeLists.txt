find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(portfolio_test portfolio_test.cpp)
target_link_libraries(portfolio_test 
    PRIVATE 
        options_core
        GTest::GTest 
        GTest::Main
)

add_test(NAME portfolio_test COMMAND portfolio_test)
