#pragma once
#include "geommath.hpp"

namespace My {
template <class T>
class Hit {
public:
  // CONSTRUCTOR & DESTRUCTOR
  __device__ Hit() : t(std::numeric_limits<T>::infinity()) {}

  // ACCESSORS
  __device__ T getT() const { return t; }
  __device__ Point<T> getP() const { return p; }
  __device__ auto getNormal() const { return normal; }
  __device__ auto getAppDataPtr() const { return app_data_ptr; }
  __device__ bool isFrontFace() const { return front_face; }
  
  // MODIFIER
  __device__ void set(const T _t, const Point<T>& _p, const Vector3<T>& _n, const bool _front_face, const intptr_t _app_data_ptr) { t = _t; p = _p; normal = _n, front_face = _front_face, app_data_ptr = _app_data_ptr; }

private: 

  // REPRESENTATION
  T t;
  Point<T> p;
  Vector3<T> normal;
  intptr_t app_data_ptr;
  bool front_face;
};
}