#include "Encoder/PPM.hpp"
#include "Image.hpp"
#include "geommath.hpp"
#include "portable.hpp"
#include "BVH.hpp"
#include "RayTracingCamera.hpp"

#define float_precision float
#include "TestMaterial.hpp"
#include "TestScene.hpp"
#include "PathTracing.hpp"

#include <chrono>

using image = My::Image;
using bvh = My::BVHNode<float_precision>;
using camera = My::RayTracingCamera<float_precision>;

// Main
int main(int argc, char** argv) {
    // Render Settings
    const float_precision aspect_ratio = 16.0 / 9.0;
    const int image_width = 1920;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const My::raytrace_config config = {.samples_per_pixel =  512, .max_depth =  50};

    // World
    auto world = random_scene();
    bvh world_bvh(world);

    // Camera
    point3 lookfrom({13, 2, 3});
    point3 lookat({0, 0, 0});
    vec3 vup({0, 1, 0});
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, (float_precision)20.0,
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
    auto start = std::chrono::steady_clock::now();
    My::PathTracing<float_precision>::raytrace(config, world_bvh, cam, img);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end - start;

    std::cerr << "\r";
    std::cout << "Rendering time: " << diff.count() << " s\n";

#if 0
    My::PpmEncoder encoder;
    encoder.Encode(img);
#endif
    img.SaveTGA("raytraced.tga");

    return 0;
}
