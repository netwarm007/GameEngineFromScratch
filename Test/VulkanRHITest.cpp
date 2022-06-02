#include "Vulkan/VulkanRHI.hpp"

using namespace My;

int main() {
    VulkanRHI rhi;

    rhi.createInstance();
    rhi.setupDebugMessenger();

    return 0;
}