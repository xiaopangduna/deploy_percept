add_executable(test_smoke test_stdlib.cpp)
target_link_libraries(test_smoke PRIVATE GTest::gtest_main)

add_test(NAME test_smoke COMMAND test_smoke)
set_tests_properties(test_smoke PROPERTIES LABELS "smoke")

percept_install_test(test_smoke)
