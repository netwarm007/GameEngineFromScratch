#pragma once
#include "geommath.hpp"

namespace My {
template <typename T>
class Hit {
public:
  // CONSTRUCTOR & DESTRUCTOR
  Hit() : t(std::numeric_limits<T>::infinity()) {}
  Hit(T _t, Vector3<T> c) { t = _t; color = c; }
  ~Hit() {}

  // ACCESSORS
  T getT() const { return t; }
  inline Vector3<T> getColor() const { return color; }
  inline Vector3<T> getNormal() const { return normal; }
  
  // MODIFIER
  void set(const T _t, const Vector3<T>& _n, const Vector3<T>& _c) { t = _t; normal = _n, color = _c; }

private: 

  // REPRESENTATION
  T t;
  Vector3<T> normal;
  Vector3<T> color;
};
}