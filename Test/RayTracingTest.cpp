#include "Encoder/PPM.hpp"
#include "Image.hpp"
#include "Ray.hpp"
#include "geommath.hpp"
#include "portable.hpp"
#include "random.hpp"

#include "Sphere.hpp"

#include <future>
#include <memory>
#include <queue>
#include <thread>

using float_precision = float;

inline int to_unorm(float_precision f) {
    return My::clamp(f, decltype(f)(0.0), decltype(f)(0.999)) * 256;
}

using ray = My::Ray<float_precision>;
using color = My::Vector3<float_precision>;
using point3 = My::Vector3<float_precision>;
using vec3 = My::Vector3<float_precision>;
using image = My::Image;
using hit_record = My::Hit<float_precision>;
constexpr auto infinity = std::numeric_limits<float_precision>::infinity();
constexpr auto epsilon = std::numeric_limits<float_precision>::epsilon();

// Material
class material {
   public:
    virtual bool scatter(const ray& r_in, const hit_record hit,
                         color& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
   public:
    lambertian(const color& a) : albedo(a) {}

    bool scatter(const ray& r_in, const hit_record hit, color& attenuation,
                 ray& scattered) const override {
        auto scatter_direction =
            hit.getNormal() + My::random_unit_vector<float_precision, 3>();

        if (My::isNearZero(scatter_direction)) {
            scatter_direction = hit.getNormal();
        }

        scattered = ray(r_in.pointAtParameter(hit.getT()), scatter_direction);
        attenuation = albedo;
        return true;
    }

   public:
    color albedo;
};

class metal : public material {
   public:
    metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    bool scatter(const ray& r_in, const hit_record hit, color& attenuation,
                 ray& scattered) const override {
        vec3 reflected = My::Reflect(r_in.getDirection(), hit.getNormal());
        scattered = ray(
            r_in.pointAtParameter(hit.getT()),
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
    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    bool scatter(const ray& r_in, const hit_record hit, color& attenuation,
                 ray& scattered) const override {
        attenuation = color({1.0, 1.0, 1.0});
        float_precision refraction_ratio = hit.isFrontFace() ? (1.0 / ir) : ir;
        auto v = r_in.getDirection();
        auto n = hit.getNormal();
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
        scattered = ray(r_in.pointAtParameter(hit.getT()), direction);

        return true;
    }

   public:
    double ir;  // Index of Refraction

   private:
    static float_precision schlick_reflectance_approximation(
        float_precision cosine, float_precision ref_idx) {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

// Utilities
color ray_color(const ray& r, int depth, My::IntersectableList<float_precision>& world) {
    hit_record hit;

    if (depth <= 0) {
        return color({0, 0, 0});
    }

    if (world.Intersect(r, hit, 0.001, infinity)) {
        ray scattered;
        color attenuation;
        auto p = r.pointAtParameter(hit.getT());

        if (hit.getMaterial()->scatter(r, hit, attenuation, scattered)) {
            return attenuation * ray_color(scattered, depth - 1, world);
        }

        return color({0, 0, 0});
    }

    // background
    auto unit_direction = r.getDirection();
    auto t = 0.5 * (unit_direction[1] + 1.0);
    return (1.0 - t) * color({1.0, 1.0, 1.0}) + t * color({0.5, 0.7, 1.0});
}

// Camera
template <class T>
class camera {
   public:
    camera(
        My::Point<T>    lookfrom,
        My::Point<T>    lookat,
        My::Vector3<T>  vup,
        T vfov, 
        T aspect_ratio,
        T aperture,
        T focus_dist
        ) {
        auto theta = My::degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        T viewport_height = 2.0 * h;
        T viewport_width = aspect_ratio * viewport_height;

        w = lookfrom - lookat;
        My::Normalize(w);
        u = My::CrossProduct(vup, w);
        My::Normalize(u);
        v = My::CrossProduct(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;

        lens_radius = aperture / 2;
    }

    ray get_ray(T s, T t) const {
        My::Vector3<T> rd = lens_radius * My::random_in_unit_disk<T>();
        My::Vector3<T> offset = u * rd[0] + v * rd[1];
        
        return ray(origin + offset,
                   lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

   private:
    My::Point<T> origin;
    My::Point<T> lower_left_corner;
    My::Vector3<T> horizontal;
    My::Vector3<T> vertical;
    My::Vector3<T> u, v, w;
    T    lens_radius;
};

// World
auto random_scene() {
    My::IntersectableList<float_precision> world;

    auto material_ground = std::make_shared<lambertian>(color({0.5, 0.5, 0.5}));
    world.emplace_back(std::make_shared<My::Sphere<float_precision>>(
        1000, point3({0, -1000, -1.0}), material_ground));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = My::random_f<float_precision>();
            point3 center({a + (float_precision)0.9 * My::random_f<float_precision>(), (float_precision)0.2, b + (float_precision)0.9 * My::random_f<float_precision>()});

            if (Length(center - point3({4, 0.2, 0})) > 0.9) {
                std::shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    color albedo = My::random_v<float_precision, 3>() * My::random_v<float_precision, 3>();
                    sphere_material = std::make_shared<lambertian>(albedo);
                    world.emplace_back(std::make_shared<My::Sphere<float_precision>>(0.2, center, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = My::random_v<float_precision, 3>(0.5, 1);
                    auto fuzz = My::random_f<float_precision>(0, 0.5);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                    world.emplace_back(std::make_shared<My::Sphere<float_precision>>(0.2, center, sphere_material));
                } else {
                    // glass
                    sphere_material = std::make_shared<dielectric>(1.5);
                    world.emplace_back(std::make_shared<My::Sphere<float_precision>>(0.2, center, sphere_material));
                }
            }
        }
    }

    auto material_1 = std::make_shared<dielectric>(1.5);
    world.emplace_back(std::make_shared<My::Sphere<float_precision>>(
        1.0, point3({0, 1, 0}), material_1));

    auto material_2 = std::make_shared<lambertian>(color({0.4, 0.2, 0.1}));
    world.emplace_back(std::make_shared<My::Sphere<float_precision>>(
        1.0, point3({-4, 1, 0}), material_2));

    auto material_3 = std::make_shared<metal>(color({0.7, 0.6, 0.5}), 0.1);
    world.emplace_back(std::make_shared<My::Sphere<float_precision>>(
        1.0, point3({4, 1, 0}), material_3));

    return world;
}

// Main
int main(int argc, char** argv) {
    // Image
    const float_precision aspect_ratio = 16.0 / 9.0;
    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 500;
    const int max_depth = 50;

    // World
    auto world = random_scene();

    // Camera
    point3 lookfrom({13, 2, 3});
    point3 lookat({0, 0, 0});
    vec3 vup({0, 1, 0});
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera<float_precision> cam(lookfrom, lookat, vup, (float_precision)20.0, aspect_ratio, aperture, dist_to_focus);

    // Canvas
    image img;
    img.Width = image_width;
    img.Height = image_height;
    img.bitcount = 24;
    img.bitdepth = 8;
    img.pixel_format = My::PIXEL_FORMAT::RGB8;
    img.pitch = (img.bitcount >> 3) * img.Width;
    img.compressed = false;
    img.compress_format = My::COMPRESSED_FORMAT::NONE;
    img.data_size = img.Width * img.Height * (img.bitcount >> 3);
    img.data = new uint8_t[img.data_size];

    // Render
    int concurrency = std::thread::hardware_concurrency();
    std::cerr << "Concurrent ray tracing with (" << concurrency << ") threads." << std::endl;

    std::queue<std::future<void>> raytrace_tasks;
    for (auto j = 0; j < img.Height; j++) {
        std::cerr << "\rScanlines remaining: " << img.Height - j << ' ' << std::flush;
        raytrace_tasks.emplace(std::async(std::launch::async, [j, cam, &img, &world] {
            for (auto i = 0; i < img.Width; i++) {
                color pixel_color(0);
                for (auto s = 0; s < samples_per_pixel; s++) {
                    auto u =
                        (i + My::random_f<float_precision>()) / (img.Width - 1);
                    auto v = (j + My::random_f<float_precision>()) /
                             (img.Height - 1);

                    auto r = cam.get_ray(u, v);
                    pixel_color += ray_color(r, max_depth, world);
                }

                pixel_color = pixel_color * (1.0 / samples_per_pixel);

                // Gamma-correction for gamma = 2.0
                pixel_color = My::sqrt(pixel_color);

                img.SetR(i, j, to_unorm(pixel_color[0]));
                img.SetG(i, j, to_unorm(pixel_color[1]));
                img.SetB(i, j, to_unorm(pixel_color[2]));
            }
        }));

        if (raytrace_tasks.size() >= concurrency) {
            while (raytrace_tasks.size() > 0) {
                raytrace_tasks.front().wait();
                raytrace_tasks.pop();
            }
        }
    }


    std::cerr << "\r";

    My::PpmEncoder encoder;
    encoder.Encode(img);

    return 0;
}
