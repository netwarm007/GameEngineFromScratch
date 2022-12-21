#pragma once
#include "geommath.hpp"

class material;

namespace My {
template <typename T>
class Hit {
public:
  // CONSTRUCTOR & DESTRUCTOR
  Hit() : t(std::numeric_limits<T>::infinity()) {}
  ~Hit() {}

  // ACCESSORS
  T getT() const { return t; }
  inline auto getNormal() const { return normal; }
  inline auto getMaterial() const { return mat_ptr; }
  inline bool isFrontFace() const { return front_face; }
  
  // MODIFIER
  void set(const T _t, const Vector3<T>& _n, const bool _front_face, const std::shared_ptr<material> _m) { t = _t; normal = _n, front_face = _front_face, mat_ptr = _m; }

private: 

  // REPRESENTATION
  T t;
  Vector3<T> normal;
  std::shared_ptr<material> mat_ptr;
  bool front_face;
};
}