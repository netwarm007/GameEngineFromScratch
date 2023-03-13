#pragma once
#include "BVH.hpp"
#include "Hit.hpp"
#include "Image.hpp"
#include "geommath.hpp"
#include "ColorSpaceConversion.hpp"

#include <chrono>
#include <concepts>
#include <future>
#include <iostream>
#include <list>
#include <limits>
#include <sstream>

using namespace std::chrono_literals;

namespace My {
// Raytrace
struct raytrace_config {
    int samples_per_pixel;
    int max_depth;
};

template <class T>
class PathTracing {
   private:
    // Utilities
    using color = Vector3<T>;
    static inline const color white = {1.0, 1.0, 1.0};
    static inline const color black = {0.0, 0.0, 0.0};
    static inline const color bg_color = {0.5, 0.7, 1.0};

    static Vector3<T> ray_color(const Ray<T> &r, int depth, BVHNode<T> &world) {
        Hit<T> hit;

        if (depth <= 0) {
            return black;
        }

        if (world.Intersect(r, hit, 0.001, std::numeric_limits<T>::infinity())) {
            Ray<T> scattered;
            color attenuation;

            const std::shared_ptr<material> pMat =
                *reinterpret_cast<const std::shared_ptr<material> *>(
                    hit.getMaterial());

            if (pMat->scatter(r, hit, attenuation, scattered)) {
                if (My::LengthSquared(attenuation) < 0.0002f)
                    return black;  // roughly squre of (1.0 / 256)
                return attenuation * ray_color(scattered, depth - 1, world);
            }

            return black;
        }

        // background
        auto &unit_direction = r.getDirection();
        T t = 0.5 * (unit_direction[1] + 1.0);
        return ((T)1.0 - t) * white + t * bg_color;
    }

   public:
    static void raytrace(const raytrace_config cfg, BVHNode<T> &world_bvh,
                         const RayTracingCamera<T> &cam, Image &img, std::ostringstream &out, bool &cancel) {
        int concurrency = std::thread::hardware_concurrency();

        out << "Concurrent ray tracing with (" << concurrency
                  << ") threads." << std::endl;

        std::list<std::future<int>> raytrace_tasks;

        auto f_raytrace = [&cfg, &cam, &world_bvh, &img](int x, int y) -> int {
            color pixel_color(0);
            for (auto s = 0; s < cfg.samples_per_pixel; s++) {
                auto u =
                    (x + My::random_f<T>()) / (img.Width - 1);
                auto v =
                    (y + My::random_f<T>()) / (img.Height - 1);

                auto r = cam.get_ray(u, v);
                pixel_color += ray_color(r, cfg.max_depth, world_bvh);
            }

            pixel_color =
                pixel_color * ((T)1.0 / cfg.samples_per_pixel);

            // Gamma-correction for gamma = 2.4
            My::RGB8 pixel_color_unorm =
                My::QuantizeUnsigned8Bits(My::Linear2SRGB(pixel_color));

            img.SetR(x, y, pixel_color_unorm[0]);
            img.SetG(x, y, pixel_color_unorm[1]);
            img.SetB(x, y, pixel_color_unorm[2]);
            img.SetA(x, y, 255);

            return 0;
        };

        for (auto j = 0; j < img.Height && !cancel; j++) {
            out << "\rScanlines remaining: " << img.Height - j << ' '
                      << std::flush;
            for (auto i = 0; i < img.Width && !cancel; i++) {
                while (raytrace_tasks.size() >= concurrency) {
                    // wait for at least one task finish
                    raytrace_tasks.remove_if([](std::future<int> &task) {
                        return task.wait_for(1ms) == std::future_status::ready;
                    });
                }

                raytrace_tasks.emplace_back(
                    std::async(std::launch::async, f_raytrace, i, j));
            }
        }

        while (!raytrace_tasks.empty()) {
            // wait for at least one task finish
            raytrace_tasks.remove_if([](std::future<int> &task) {
                return task.wait_for(1s) == std::future_status::ready;
            });
        }
    }
};
}  // namespace My