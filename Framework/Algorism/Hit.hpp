#pragma once
#include "geommath.hpp"

namespace My {
template <typename T>
class Hit {
public:
  // CONSTRUCTOR & DESTRUCTOR
  Hit() : t(std::numeric_limits<T>::infinity()), color({0.0, 0.0, 0.0}) {}
  Hit(T _t, Vector3<T> c) { t = _t; color = c; }
  ~Hit() {}

  // ACCESSORS
  T getT() const { return t; }
  Vector3<T> getColor() const { return color; }
  
  // MODIFIER
  void set(T _t, Vector3<T> c) { t = _t; color = c; }

private: 

  // REPRESENTATION
  T t;
  Vector3<T> color;
};
}