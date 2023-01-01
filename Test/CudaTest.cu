#include <curand_kernel.h>
#include "geommath.hpp"
#include "BVH.hpp"
#include "Sphere.hpp"

#include "TestMaterial.hpp"

using color = My::Vector3<float>;
using point3 = My::Point<float>;
using vec3 = My::Vector3<float>;

// help functions 
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " (" << cudaGetErrorString(result) << ") "
                  << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(2023, 0, 0, rand_state);
    }
}

__global__ void test(curandState *local_rand_state) {
    const int scene_obj_num = 1;
    My::Hitable<float>** pList = new My::Hitable<float>*[scene_obj_num];
    for (int i = 0; i < scene_obj_num; i++) {
        pList[i] = new My::Sphere<float, material *>(1000.0f, point3({0, -1000, -1}),
                              new lambertian(vec3({0.5, 0.5, 0.5})));
    }

    My::SimpleBVHNode<float>* pWorld = new My::SimpleBVHNode<float>(pList, 0, scene_obj_num, local_rand_state);
    delete pList;
    delete pWorld;
}

int main() {
    curandState *d_rand_state_1;

    checkCudaErrors(cudaMalloc((void **)&d_rand_state_1, sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state_1);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    test<<<1, 1>>>(d_rand_state_1);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_rand_state_1));

    return 0;
}