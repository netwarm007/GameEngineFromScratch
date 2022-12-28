#include <iostream>
#include <limits>
#include "geommath.hpp"
#include "Image.hpp"
#include "Ray.hpp"
#include "Sphere.hpp"
#include "HitableList.hpp"

using ray = My::Ray<float>;
using color = My::Vector3<float>;
using point3 = My::Vector3<float>;
using vec3 = My::Vector3<float>;
using image = My::Image;

constexpr auto infinity = std::numeric_limits<float>::infinity();

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ float hit_sphere(const point3& center, float radius, const ray& r) {
    vec3 oc = r.getOrigin() - center;
    auto half_b = My::DotProduct(oc, r.getDirection());
    auto c = My::LengthSquared(oc) - radius * radius;
    auto discriminant = half_b * half_b - c;

    if (discriminant < 0.0f) {
        return -1.0f;
    } else {
        return (-half_b - sqrt(discriminant));
    }
}

__device__ color ray_color(const ray& r ) {
    auto t = hit_sphere(point3({0, 0, -1}), 0.5f, r);
    if (t > 0.0f) {
        vec3 N = r.pointAtParameter(t) - vec3({0, 0, -1});
        return 0.5f * color({N[0] + 1, N[1] + 1, N[2] + 1});
    }

    vec3 unit_direction = r.getDirection();
    t = 0.5f * (unit_direction[1] + 1.0f);
    return (1.0f - t) * color({1.0, 1.0, 1.0}) + t * color({0.5, 0.7, 1.0});
}

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i < max_x) && (j < max_y)) {
        int pixel_index = j * max_x + i;
        float u = float(i) / float(max_x);
        float v = float(j) / float(max_y);
        ray r(origin, lower_left_corner + u * horizontal + v * vertical);
        fb[pixel_index] = ray_color(r);
    }
}

int main() {
    // Render Settings
    const float aspect_ratio = 16.0 / 9.0;
    const int image_width = 1920;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    int tile_width = 8;
    int tile_height = 8;

    // Canvas
    image img;
    img.Width = image_width;
    img.Height = image_height;
    img.bitcount = 96;
    img.bitdepth = 32;
    img.pixel_format = My::PIXEL_FORMAT::RGB32;
    img.pitch = (img.bitcount >> 3) * img.Width;
    img.compressed = false;
    img.compress_format = My::COMPRESSED_FORMAT::NONE;
    img.data_size = img.Width * img.Height * (img.bitcount >> 3);

    checkCudaErrors(cudaMallocManaged((void **)&img.data, img.data_size));

    // Camera
    auto viewport_height = 2.0f;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0f;

    auto origin = point3({0, 0, 0});
    auto horizontal = vec3({viewport_width, 0, 0});
    auto vertical = vec3({0, viewport_height, 0});
    auto lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - vec3({0, 0, focal_length});

    // Rendering
    dim3 blocks(image_width / tile_width + 1, image_height / tile_height + 1);
    dim3 threads(tile_width, tile_height);
    render<<<blocks, threads>>>(reinterpret_cast<vec3 *>(img.data), image_width, image_height, lower_left_corner, horizontal, vertical, origin);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    img.SaveTGA("raytracing_cuda.tga");
    
    checkCudaErrors(cudaFree(img.data));
    img.data = nullptr; // to avoid double free

    return 0;
}