add_library(Geometries Geometry.cpp Box.cpp Sphere.cpp Plane.cpp)
add_dependencies(Geometries GeomMath)
