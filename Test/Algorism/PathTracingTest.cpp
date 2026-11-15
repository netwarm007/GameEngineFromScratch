#include <iostream>
#include <memory>
#include <sstream>

#define float_precision float
#include "BVH.hpp"
#include "Image.hpp"
#include "RayTracingCamera.hpp"
#include "Sphere.hpp"
#include "TestMaterial.hpp"
// PathTracing.hpp relies on the camera and material types declared above.
#include "PathTracing.hpp"

using namespace My;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

namespace {
void InitImage(Image& img, uint32_t size) {
    img.Width = size;
    img.Height = size;
    img.bitcount = 32;
    img.bitdepth = 8;
    img.pixel_format = PIXEL_FORMAT::RGBA8;
    img.pitch = (img.bitcount >> 3) * img.Width;
    img.compressed = false;
    img.compress_format = COMPRESSED_FORMAT::NONE;
    img.data_size = img.pitch * img.Height;
    img.data = new uint8_t[img.data_size]();
}
}  // namespace

int main() {
    // A large black (zero-albedo) sphere covers the middle of the view and
    // absorbs every ray, so the center pixel is black while the top row, which
    // looks above the sphere, shows sky.
    auto black = std::make_shared<lambertian>(color({0.0f, 0.0f, 0.0f}));
    HitableList<float> world;
    world.add(std::make_shared<Sphere<float, std::shared_ptr<material>>>(
        2.0f, point3({0, 0, -3}), black));
    BVHNode<float> bvh(world);

    RayTracingCamera<float> cam(point3({0, 0, 0}), point3({0, 0, -1}),
                                vec3({0, 1, 0}), 90.0f, 1.0f, 0.0f, 1.0f);

    const uint32_t size = 5;
    Image img;
    InitImage(img, size);

    raytrace_config cfg = {.samples_per_pixel = 2, .max_depth = 3};
    std::ostringstream out;
    bool cancel = false;
    PathTracing<float>::raytrace(cfg, bvh, cam, img, out, cancel);

    CHECK(out.str().find("Scanlines remaining") != std::string::npos);
    CHECK(img.GetA(2, 2) == 255 && img.GetA(0, 4) == 255);
    CHECK(img.GetR(2, 2) == 0 && img.GetG(2, 2) == 0 && img.GetB(2, 2) == 0);

    // Sky is bluish: blue dominates red in the top row.
    CHECK(img.GetB(2, 4) > img.GetR(2, 4));
    CHECK(img.GetB(2, 4) > 200);

    // A pre-cancelled render leaves the image untouched.
    Image untouched;
    InitImage(untouched, size);
    std::ostringstream out2;
    bool stop = true;
    PathTracing<float>::raytrace(cfg, bvh, cam, untouched, out2, stop);
    CHECK(untouched.GetA(2, 4) == 0 && untouched.GetB(2, 4) == 0);

    return 0;
}
