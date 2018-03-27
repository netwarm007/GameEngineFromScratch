add_library(Geometries Box.cpp Geometry.cpp Sphere.cpp Plane.cpp Polyhedron.cpp)
add_dependencies(Geometries GeomMath)
