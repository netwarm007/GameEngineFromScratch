#pragma once
#include "geommath.hpp"

namespace My {
class Hit {
public:
  // CONSTRUCTOR & DESTRUCTOR
  Hit(float _t, Vector3f c) { t = _t; color = c; }
  ~Hit() {}

  // ACCESSORS
  float getT() const { return t; }
  Vector3f getColor() const { return color; }
  
  // MODIFIER
  void set(float _t, Vector3f c) { t = _t; color = c; }

private: 

  // REPRESENTATION
  float t;
  Vector3f color;
};
}