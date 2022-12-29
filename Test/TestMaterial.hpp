#pragma once
#include "Ray.hpp"
#include "Hit.hpp"
#include "geommath.hpp"
#include "random.hpp"

using float_precision = float;
using ray = My::Ray<float_precision>;
using color = My::Vector3<float_precision>;
using point3 = My::Point<float_precision>;
using vec3 = My::Vector3<float_precision>;
using hit_record = My::Hit<float_precision>;

// Material
class material {
   public:
#ifdef __CUDACC__
    __device__ virtual bool scatter(const ray& r_in, const hit_record& hit,
                         color& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
#else
    virtual bool scatter(const ray& r_in, const hit_record& hit,
                         color& attenuation, ray& scattered) const = 0;
#endif
};

class lambertian : public material {
   public:
    __device__ lambertian(const color& a) : albedo(a) {}

#ifdef __CUDACC__
    __device__ bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered, curandState *local_rand_state) const override {
        auto scatter_direction =
            hit.getNormal() + My::random_unit_vector<float_precision, 3>(local_rand_state);
#else
    bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered) const override {
        auto scatter_direction =
            hit.getNormal() + My::random_unit_vector<float_precision, 3>();
#endif

        if (My::isNearZero(scatter_direction)) {
            scatter_direction = hit.getNormal();
        }

        scattered = ray(hit.getP(), scatter_direction);
        attenuation = albedo;
        return true;
    }

   public:
    color albedo;
};

class metal : public material {
   public:
    __device__ metal(const color& a, float_precision f) : albedo(a), fuzz(f < 1 ? f : 1) {}

#ifdef __CUDACC__
    __device__ bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered, curandState *local_rand_state) const override {
        vec3 reflected = My::Reflect(r_in.getDirection(), hit.getNormal());
        scattered = ray(
            hit.getP(),
            reflected + fuzz * My::random_in_unit_sphere<float_precision, 3>(local_rand_state));
#else
    bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered) const override {
        vec3 reflected = My::Reflect(r_in.getDirection(), hit.getNormal());
        scattered = ray(
            hit.getP(),
            reflected + fuzz * My::random_in_unit_sphere<float_precision, 3>());
#endif
        attenuation = albedo;
        return My::DotProduct(scattered.getDirection(), hit.getNormal()) >
               0;  // absorb scarted rays below the surface
    }

   public:
    color albedo;
    float_precision fuzz;
};

class dielectric : public material {
   public:
    __device__ dielectric(float_precision index_of_refraction) : ir(index_of_refraction) {}

#ifdef __CUDACC__
    __device__ bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered, curandState *local_rand_state) const override {
#else
    bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered) const override {
#endif
        attenuation = color({1.0, 1.0, 1.0});
        float_precision refraction_ratio = hit.isFrontFace() ? (1.0 / ir) : ir;
        auto v = r_in.getDirection();
        auto n = hit.isFrontFace() ? hit.getNormal() : -hit.getNormal();
        float_precision cos_theta = fmin(DotProduct(-v, n), 1.0);
        float_precision sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        bool cannot_refract = refraction_ratio * sin_theta > 1.0;

        vec3 direction;
        if ((cannot_refract) ||
            (schlick_reflectance_approximation(cos_theta, refraction_ratio) >
#ifdef __CUDACC__
             My::random_f<float_precision>(local_rand_state))) {
#else
             My::random_f<float_precision>())) {
#endif
            direction = My::Reflect(v, n);
        } else {
            direction = My::Refract(v, n, refraction_ratio);
        }
        scattered = ray(hit.getP(), direction);

        return true;
    }

   public:
    float_precision ir;  // Index of Refraction

   private:
    __device__ static float_precision schlick_reflectance_approximation(
        float_precision cosine, float_precision ref_idx) {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};