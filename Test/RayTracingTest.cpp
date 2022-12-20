#include "Encoder/PPM.hpp"
#include "Image.hpp"
#include "Ray.hpp"
#include "geommath.hpp"
#include "portable.hpp"

#include "Box.hpp"
#include "Sphere.hpp"

#include <vector>

using float_precision = float;

inline int to_unorm(float_precision f) { return f * 255.999; }

using ray = My::Ray<float_precision>;
using color = My::Vector3<float_precision>;
using point3 = My::Vector3<float_precision>;
using vec3 = My::Vector3<float_precision>;
using image = My::Image;
constexpr auto infinity = std::numeric_limits<float_precision>::infinity();

My::IntersectableList<float_precision> scene_objects;

color ray_color(const ray& r) {
    My::Hit<float_precision> hit;
    if (scene_objects.Intersect(r, hit, 0, infinity)) {
        return hit.getColor();
    }

    // background
    auto unit_direction = r.getDirection();
    auto t = 0.5 * (unit_direction[1] + 1.0);
    return (1.0 - t) * color({1.0, 1.0, 1.0}) + t * color({0.5, 0.7, 1.0});
}

int main(int argc, char** argv) {
    scene_objects.emplace_back(std::make_shared<My::Sphere<float_precision>>(
        1, point3({0, 0, -2.0}), color({1.0, 0, 0})));
    scene_objects.emplace_back(std::make_shared<My::Sphere<float_precision>>(
        1, point3({-2.0, 0, -2.0}), color({0.5, 0, 0})));
    scene_objects.emplace_back(std::make_shared<My::Sphere<float_precision>>(
        1, point3({2.0, 0, -2.0}), color({0.0, 0.5, 0})));

    // Image
    const float_precision aspect_ratio = 16.0 / 9.0;
    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Camera
    float_precision viewport_height = 2.0;
    float_precision viewport_width = aspect_ratio * viewport_height;
    float_precision focal_length = 1.0;

    auto origin = point3({0, 0, 0});
    auto horizontal = vec3({viewport_width, 0, 0});
    auto vertical = vec3({0, viewport_height, 0});
    auto lower_left_corner =
        origin - horizontal / 2 - vertical / 2 - vec3({0, 0, focal_length});

    // Image
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
    for (auto j = 0; j < img.Height; j++) {
        for (auto i = 0; i < img.Width; i++) {
            auto u = double(i) / (img.Width - 1);
            auto v = double(j) / (img.Height - 1);
            ray r(origin,
                  lower_left_corner + u * horizontal + v * vertical - origin);
            color pixel_color = ray_color(r);

            img.SetR(i, j, to_unorm(pixel_color[0]));
            img.SetG(i, j, to_unorm(pixel_color[1]));
            img.SetB(i, j, to_unorm(pixel_color[2]));
        }
    }

    My::PpmEncoder encoder;
    encoder.Encode(img);

    return 0;
}
