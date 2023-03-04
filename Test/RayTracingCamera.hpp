#include "geommath.hpp"
#include "Ray.hpp"
#include "random.hpp"

namespace My {
template <class T>
class RayTracingCamera {
   public:
    __host__ __device__ RayTracingCamera(Point<T> lookfrom, Point<T> lookat, Vector3<T> vup,
           T vfov, T aspect_ratio, T aperture, T focus_dist) {
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta / (T)2.0);
        T viewport_height = (T)2.0 * h;
        T viewport_width = aspect_ratio * viewport_height;

        w = lookfrom - lookat;
        Normalize(w);
        u = CrossProduct(vup, w);
        Normalize(u);
        v = CrossProduct(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner =
            origin - horizontal / (T)2.0 - vertical / (T)2.0 - focus_dist * w;

        lens_radius = aperture / (T)2.0;
    }

#ifdef __CUDACC__
    __device__ Ray<T> get_ray(T s, T t, curandStateMRG32k3a_t *local_rand_state) const {
        Vector3<T> rd = lens_radius * random_in_unit_disk<T>(local_rand_state);
#else
    Ray<T> get_ray(T s, T t) const {
        Vector3<T> rd = lens_radius * random_in_unit_disk<T>();
#endif
        Vector3<T> offset = u * rd[0] + v * rd[1];

        return Ray(origin + offset, lower_left_corner + s * horizontal +
                                        t * vertical - origin - offset);
    }

   private:
    Point<T> origin;
    Point<T> lower_left_corner;
    Vector3<T> horizontal;
    Vector3<T> vertical;
    Vector3<T> u, v, w;
    T lens_radius;
};
}