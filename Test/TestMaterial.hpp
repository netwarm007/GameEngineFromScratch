#pragma once
#include "Ray.hpp"
#include "Hit.hpp"
#include "geommath.hpp"

#include <climits>

using float_precision = float;
using ray = My::Ray<float_precision>;
using color = My::Vector3<float_precision>;
using point3 = My::Vector3<float_precision>;
using vec3 = My::Vector3<float_precision>;
using hit_record = My::Hit<float_precision>;

// Material
class material {
   public:
    virtual bool scatter(const ray& r_in, const hit_record& hit,
                         color& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
   public:
    lambertian(const color& a) : albedo(a) {}

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

   public:
    color albedo;
};

class metal : public material {
   public:
    metal(const color& a, float_precision f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered) const override {
        vec3 reflected = My::Reflect(r_in.getDirection(), hit.getNormal());
        scattered = ray(
            hit.getP(),
            reflected + fuzz * My::random_in_unit_sphere<float_precision, 3>());
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
    dielectric(float_precision index_of_refraction) : ir(index_of_refraction) {}

    bool scatter(const ray& r_in, const hit_record& hit, color& attenuation,
                 ray& scattered) const override {
        attenuation = color({1.0, 1.0, 1.0});
        float_precision refraction_ratio = hit.isFrontFace() ? (1.0 / ir) : ir;
        auto v = r_in.getDirection();
        auto n = hit.isFrontFace() ? hit.getNormal() : -hit.getNormal();
        auto cos_theta = fmin(DotProduct(-v, n), 1.0);
        auto sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        bool cannot_refract = refraction_ratio * sin_theta > 1.0;

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

   public:
    float_precision ir;  // Index of Refraction

   private:
    static float_precision schlick_reflectance_approximation(
        float_precision cosine, float_precision ref_idx) {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};