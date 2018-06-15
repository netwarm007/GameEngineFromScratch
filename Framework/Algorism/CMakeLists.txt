add_library(Algorism quickhull.cpp)
target_include_directories(Algorism PUBLIC GeomMath)
target_link_libraries(Algorism PUBLIC GeomMath)
