add_library(Geometries Box.cpp Geometry.cpp Sphere.cpp Plane.cpp Polyhedron.cpp)
target_include_directories(Geometries PUBLIC GeomMath)
target_link_libraries(Geometries PUBLIC GeomMath)
