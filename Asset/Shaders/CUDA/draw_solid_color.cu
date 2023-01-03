#include <optix.h>

#include "OptixTest.h"
#include "ColorSpaceConversion.hpp"

extern "C" {
__constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color() {
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    params.image[launch_index.y * params.image_width + launch_index.x] = My::Linear2SRGB({rtData->r, rtData->g, rtData->b});
}
