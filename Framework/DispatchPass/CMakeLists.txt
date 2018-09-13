add_library(DispatchPass 
BRDFIntegrator.cpp
)
target_include_directories(DispatchPass PUBLIC GeomMath)
target_link_libraries(DispatchPass PUBLIC GeomMath)