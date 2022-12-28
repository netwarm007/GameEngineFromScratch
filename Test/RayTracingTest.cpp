#include "Encoder/PPM.hpp"
#include "Image.hpp"
#include "Ray.hpp"
#include "geommath.hpp"
#include "portable.hpp"
#include "random.hpp"
#include "BVH.hpp"

#include "TestMaterial.hpp"
#include "TestScene.hpp"

#include <chrono>
#include <future>
#include <list>
#include <memory>
#include <thread>

using namespace std::chrono_literals;

using float_precision = float;

inline int to_unorm(float_precision f) {
    return My::clamp(f, decltype(f)(0.0), decltype(f)(0.999)) * 256;
}

using image = My::Image;
using bvh = My::BVHNode<float_precision>;
constexpr auto infinity = std::numeric_limits<float_precision>::infinity();
constexpr auto epsilon = std::numeric_limits<float_precision>::epsilon();

// Utilities
const color white({1.0, 1.0, 1.0});
const color black({0.0, 0.0, 0.0});
const color bg_color({0.5, 0.7, 1.0});

color ray_color(const ray& r, int depth,
                bvh& world) {
    hit_record hit;

    if (depth <= 0) {
        return black;
    }

    if (world.Intersect(r, hit, 0.001, infinity)) {
        ray scattered;
        color attenuation;

        intptr_t pAppDataPtr = hit.getAppDataPtr();
        std::shared_ptr<material> pMat = *(std::shared_ptr<material>*)pAppDataPtr;

        if (pMat->scatter(r, hit, attenuation, scattered)) {
            return attenuation * ray_color(scattered, depth - 1, world);
        }

        return black;
    }

    // background
    auto& unit_direction = r.getDirection();
    auto t = 0.5 * (unit_direction[1] + 1.0);
    return (1.0 - t) * white + t * bg_color;
}

// Camera
template <class T>
class camera {
   public:
    camera(My::Point<T> lookfrom, My::Point<T> lookat, My::Vector3<T> vup,
           T vfov, T aspect_ratio, T aperture, T focus_dist) {
        auto theta = My::degrees_to_radians(vfov);
        auto h = std::tan(theta / 2);
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
        lower_left_corner =
            origin - horizontal / (T)2.0 - vertical / (T)2.0 - focus_dist * w;

        lens_radius = aperture / 2;
    }

    ray get_ray(T s, T t) const {
        My::Vector3<T> rd = lens_radius * My::random_in_unit_disk<T>();
        My::Vector3<T> offset = u * rd[0] + v * rd[1];

        return ray(origin + offset, lower_left_corner + s * horizontal +
                                        t * vertical - origin - offset);
    }

   private:
    My::Point<T> origin;
    My::Point<T> lower_left_corner;
    My::Vector3<T> horizontal;
    My::Vector3<T> vertical;
    My::Vector3<T> u, v, w;
    T lens_radius;
};

// Raytrace
// Main
int main(int argc, char** argv) {
    // Render Settings
    const float_precision aspect_ratio = 16.0 / 9.0;
    const int image_width = 1920;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 500;
    const int max_depth = 50;

    // World
    auto world = random_scene();
    bvh world_bvh(world);

    // Camera
    point3 lookfrom({13, 2, 3});
    point3 lookat({0, 0, 0});
    vec3 vup({0, 1, 0});
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera<float_precision> cam(lookfrom, lookat, vup, (float_precision)20.0,
                                aspect_ratio, aperture, dist_to_focus);

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
    std::cerr << "Concurrent ray tracing with (" << concurrency << ") threads."
              << std::endl;

    std::list<std::future<int>> raytrace_tasks;

    auto f_raytrace = [samples_per_pixel, max_depth, &cam, &world_bvh, &img](
                          int x, int y) -> int {
        color pixel_color(0);
        for (auto s = 0; s < samples_per_pixel; s++) {
            auto u = (x + My::random_f<float_precision>()) / (img.Width - 1);
            auto v = (y + My::random_f<float_precision>()) / (img.Height - 1);

            auto r = cam.get_ray(u, v);
            pixel_color += ray_color(r, max_depth, world_bvh);
        }

        pixel_color = pixel_color * (1.0 / samples_per_pixel);

        // Gamma-correction for gamma = 2.2
        const auto gamma = 2.2;
        pixel_color = My::pow(pixel_color, 1.0 / gamma);

        img.SetR(x, y, to_unorm(pixel_color[0]));
        img.SetG(x, y, to_unorm(pixel_color[1]));
        img.SetB(x, y, to_unorm(pixel_color[2]));

        return 0;
    };

    for (auto j = 0; j < img.Height; j++) {
        std::cerr << "\rScanlines remaining: " << img.Height - j << ' '
                  << std::flush;
        for (auto i = 0; i < img.Width; i++) {
            while (raytrace_tasks.size() >= concurrency) {
                // wait for at least one task finish
                raytrace_tasks.remove_if([](std::future<int>& task) {
                    return task.wait_for(1ms) == std::future_status::ready;
                });
            }
            color pixel_color(0);
            raytrace_tasks.emplace_back(
                std::async(std::launch::async, f_raytrace, i, j));
        }
    }

    while (!raytrace_tasks.empty()) {
        // wait for at least one task finish
        raytrace_tasks.remove_if([](std::future<int>& task) {
            return task.wait_for(1s) == std::future_status::ready;
        });
    }

    std::cerr << "\r";

#if 0
    My::PpmEncoder encoder;
    encoder.Encode(img);
#endif
    My::adjust_image(img);
    img.SaveTGA("raytraced.tga");

    return 0;
}
