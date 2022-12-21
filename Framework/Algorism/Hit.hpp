#pragma once
#include "geommath.hpp"

class material;

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
  inline auto getColor() const { return color; }
  inline auto getNormal() const { return normal; }
  inline auto getMaterial() const { return mat_ptr; }
  
  // MODIFIER
  void set(const T _t, const Vector3<T>& _n, const Vector3<T>& _c) { t = _t; normal = _n, color = _c; }
  void set(const T _t, const Vector3<T>& _n, const std::shared_ptr<material> _m) { t = _t; normal = _n, mat_ptr = _m; }

private: 

  // REPRESENTATION
  T t;
  Vector3<T> normal;
  Vector3<T> color;
  std::shared_ptr<material> mat_ptr;
};
}