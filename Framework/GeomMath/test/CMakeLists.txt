add_executable(GeomMathTest test.cpp)
target_link_libraries(GeomMathTest GeomMath)

add_test(NAME Test_GeomMath COMMAND GeomMathTest)

