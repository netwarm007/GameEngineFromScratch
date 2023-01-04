#pragma once
#include "Hit.hpp"
#include "Ray.hpp"
#include "geommath.hpp"
#include "random.hpp"

#ifndef float_precision
#define float_precision float
#endif

using ray = My::Ray<float_precision>;
using color = My::Vector3<float_precision>;
using point3 = My::Point<float_precision>;
using vec3 = My::Vector3<float_precision>;
using hit_record = My::Hit<float_precision>;

enum Material { MAT_DIFFUSE = 0, MAT_METAL, MAT_DIELECTRIC, MAT_COUNT };

// Material
class material {
   public:
#ifdef __CUDACC__
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& hit, color& attenuation,
        ray& scattered, curandStateMRG32k3a_t* local_rand_state) const = 0;
#else
    virtual bool scatter(const ray& r_in, const hit_record& hit,
                         color& attenuation, ray& scattered) const = 0;
#endif
};

class lambertian : public material {
   public:
    __device__ lambertian(const color& a) : albedo(a) {}

#ifdef __CUDACC__
    static __device__ bool scatter_static(const ray& r_in, const hit_record& hit,
                                   color& attenuation, ray& scattered,
                                   curandStateMRG32k3a_t* local_rand_state,
                                   const color& base_color) {
        auto scatter_direction =
            hit.getNormal() +
            My::random_unit_vector<float_precision, 3>(local_rand_state);

        if (My::isNearZero(scatter_direction)) {
            scatter_direction = hit.getNormal();
        }

        scattered = ray(hit.getP(), scatter_direction);
        attenuation = base_color;
        return true;
    }

    __device__ bool scatter(
        const ray& r_in, const hit_record& hit, color& attenuation,
        ray& scattered,
        curandStateMRG32k3a_t* local_rand_state) const override {
        return scatter_static(r_in, hit, attenuation, scattered, local_rand_state,
                       albedo);
    }
#else
    bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered) const override {
        auto scatter_direction =
            hit.getNormal() + My::random_unit_vector<float_precision, 3>();

        if (My::isNearZero(scatter_direction)) {
            scatter_direction = hit.getNormal();
        }

        scattered = ray(hit.getP(), scatter_direction);
        attenuation = albedo;
        return true;
    }
#endif

   public:
    color albedo;
};

class metal : public material {
   public:
    __device__ metal(const color& a, float_precision f)
        : albedo(a), fuzz(f < (float_precision)1 ? f : (float_precision)1) {}

#ifdef __CUDACC__
    static __device__ bool scatter_static(const ray& r_in, const hit_record& hit,
                                   color& attenuation, ray& scattered,
                                   curandStateMRG32k3a_t* local_rand_state,
                                   const color& base_color,
                                   float_precision fuzz) {
        vec3 reflected = My::Reflect(r_in.getDirection(), hit.getNormal());
        scattered = ray(
            hit.getP(),
            reflected + fuzz * My::random_in_unit_sphere<float_precision, 3>(
                                   local_rand_state));

        attenuation = base_color;
        return My::DotProduct(scattered.getDirection(), hit.getNormal()) >
               (float_precision)0;  // absorb scarted rays below the surface
    }

    __device__ bool scatter(
        const ray& r_in, const hit_record& hit, color& attenuation,
        ray& scattered,
        curandStateMRG32k3a_t* local_rand_state) const override {
        return scatter_static(r_in, hit, attenuation, scattered, local_rand_state,
                       albedo, fuzz);
    }

#else
    bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered) const override {
        vec3 reflected = My::Reflect(r_in.getDirection(), hit.getNormal());
        scattered = ray(
            hit.getP(),
            reflected + fuzz * My::random_in_unit_sphere<float_precision, 3>());

        attenuation = albedo;
        return My::DotProduct(scattered.getDirection(), hit.getNormal()) >
               (float_precision)0;  // absorb scarted rays below the surface
    }
#endif

   public:
    color albedo;
    float_precision fuzz;
};

class dielectric : public material {
   public:
    __device__ dielectric(float_precision index_of_refraction)
        : ir(index_of_refraction) {}

#ifdef __CUDACC__
    static __device__ bool scatter_static(const ray& r_in, const hit_record& hit,
                        color& attenuation, ray& scattered,
                        curandStateMRG32k3a_t* local_rand_state,
                        float_precision ir) {
        attenuation = color({1.0, 1.0, 1.0});
        float_precision refraction_ratio = hit.isFrontFace() ? ((float_precision)1.0 / ir) : ir;
        auto v = r_in.getDirection();
        auto n = hit.isFrontFace() ? hit.getNormal() : -hit.getNormal();
        float_precision cos_theta = fminf(DotProduct(-v, n), (float_precision)1.0);
        float_precision sin_theta = sqrt((float_precision)1.0 - cos_theta * cos_theta);
        bool cannot_refract = refraction_ratio * sin_theta > (float_precision)1.0;

        vec3 direction;
        if ((cannot_refract) ||
            (schlick_reflectance_approximation(cos_theta, refraction_ratio) >

             My::random_f<float_precision>(local_rand_state))) {
            direction = My::Reflect(v, n);
        } else {
            direction = My::Refract(v, n, refraction_ratio);
        }
        scattered = ray(hit.getP(), direction);

        return true;
    }

    __device__ bool scatter(const ray& r_in, const hit_record& hit,
                            color& attenuation, ray& scattered,
                            curandStateMRG32k3a_t* local_rand_state) const override {
        scatter_static(r_in, hit, attenuation, scattered, local_rand_state, ir);
    }
#else
    bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered) const override {
        attenuation = color({1.0, 1.0, 1.0});
        float_precision refraction_ratio = hit.isFrontFace() ? ((float_precision)1.0 / ir) : ir;
        auto v = r_in.getDirection();
        auto n = hit.isFrontFace() ? hit.getNormal() : -hit.getNormal();
        float_precision cos_theta = fminf(DotProduct(-v, n), (float_precision)1.0);
        float_precision sin_theta = sqrt((float_precision)1.0 - cos_theta * cos_theta);
        bool cannot_refract = refraction_ratio * sin_theta > (float_precision)1.0;

        vec3 direction;
        if ((cannot_refract) ||
            (schlick_reflectance_approximation(cos_theta, refraction_ratio) >
             My::random_f<float_precision>())) {
            direction = My::Reflect(v, n);
        } else {
            direction = My::Refract(v, n, refraction_ratio);
        }
        scattered = ray(hit.getP(), direction);

        return true;
    }
#endif

   public:
    float_precision ir;  // Index of Refraction

   private:
    __device__ static float_precision schlick_reflectance_approximation(
        float_precision cosine, float_precision ref_idx) {
        auto r0 = ((float_precision)1 - ref_idx) / ((float_precision)1 + ref_idx);
        r0 = r0 * r0;
        return r0 + ((float_precision)1 - r0) * pow(((float_precision)1 - cosine), (float_precision)5);
    }
};