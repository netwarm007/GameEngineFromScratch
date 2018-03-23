add_library(Geometries Geometry.cpp Box.cpp Sphere.cpp Plane.cpp Polyhedron.cpp)
add_dependencies(Geometries GeomMath)
