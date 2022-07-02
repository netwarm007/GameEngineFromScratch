#include "VulkanApplication.hpp"

using namespace My;

using VkXcbSurfaceCreateFlagsKHR = VkFlags;

struct VkXcbSurfaceCreateInfoKHR {
    VkStructureType sType;
    const void* pNext;
    VkXcbSurfaceCreateFlagsKHR flags;
    xcb_connection_t* connection;
    xcb_window_t* window;
};

typedef VkResult(VKAPI_PTR* PFN_vkCreateXcbSurfaceKHR)(
    VkInstance, const VkXcbSurfaceCreateInfoKHR*, const VkAllocationCallbacks*,
    VkSurfaceKHR*);

VkResult VulkanApplication::CreateWindowSurface(vk::Instance instance,
                                                VkSurfaceKHR& surface) {
    VkResult err;
    VkXcbSurfaceCreateInfoKHR sci;
    PFN_vkCreateXcbSurfaceKHR vkCreateXcbSurfaceKHR;

    vkCreateXcbSurfaceKHR = (PFN_vkCreateXcbSurfaceKHR)vkGetInstanceProcAddr(
        instance, "vkCreateXcbSurfaceKHR");

    if (!vkCreateXcbSurfaceKHR) {
        fprintf(stderr,
                "Xcb: Vulkan instance missing VK_KHR_xcb_surface extension");
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    memset(&sci, 0, sizeof(sci));
    sci.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
    sci.connection = m_pConn;
    sci.window = &m_XWindow;

    err = vkCreateXcbSurfaceKHR(instance, &sci, nullptr, &surface);
    assert(!err && "Xcb: Failed to create Vulkan surface.");

    return err;
}

void VulkanApplication::CreateMainWindow() {
    XcbApplication::CreateMainWindow();

    std::vector<const char*> extensions = {"VK_KHR_surface",
                                           "VK_KHR_xcb_surface"};

    // 设置回调函数
    auto getFramebufferSize = [this](int& width, int& height) {
        GetFramebufferSize(width, height);
    };

    // 设置回调函数
    m_Rhi.setFramebufferSizeQueryCB(getFramebufferSize);

    // 创建实例
    m_Rhi.createInstance(extensions);

    // 开启调试层（对发行版不起作用）
    m_Rhi.setupDebugMessenger();

    // 创建（连接）画布
    {
        auto createWindowSurface = [this](const vk::Instance& instance,
                                          vk::SurfaceKHR& surface) {
            VkSurfaceKHR _surface;
            assert(instance);

            if (CreateWindowSurface(instance, _surface) != VK_SUCCESS) {
                throw std::runtime_error("failed to create window surface!");
            }

            surface = _surface;
        };

        m_Rhi.createSurface(createWindowSurface);
    }

    // 枚举物理设备并连接
    m_Rhi.pickPhysicalDevice();

    // 创建逻辑设备
    m_Rhi.createLogicalDevice();

    // 创建 SwapChain
    m_Rhi.createSwapChain();

    // 创建 Uniform Buffers
    m_Rhi.createUniformBuffers();

    // 创建同步对象
    m_Rhi.createSyncObjects();
}

void VulkanApplication::Finalize() {
    // 销毁 Vulkan 上下文
    m_Rhi.destroyAll();

    XcbApplication::Finalize();
}
