cmake_minimum_required(VERSION 3.18)

# Find GTest package
find_package(GTest REQUIRED)
include(GoogleTest)

# Add test executables
add_executable(test_ntxent 
    test_forward.cpp
    test_backward.cpp
)

target_link_libraries(test_ntxent
    PRIVATE
        ntxent_cuda
        GTest::GTest
        GTest::Main
)

# Register tests
gtest_discover_tests(test_ntxent)
