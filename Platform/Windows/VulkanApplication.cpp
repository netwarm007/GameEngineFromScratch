#include "VulkanApplication.hpp"

using namespace My;

using VkWin32SurfaceCreateFlagsKHR = VkFlags;

typedef struct VkWin32SurfaceCreateInfoKHR {
    VkStructureType sType;
    const void* pNext;
    VkWin32SurfaceCreateFlagsKHR flags;
    HINSTANCE hinstance;
    HWND hwnd;
} VkWin32SurfaceCreateInfoKHR;

typedef VkResult(APIENTRY* PFN_vkCreateWin32SurfaceKHR)(
    VkInstance, const VkWin32SurfaceCreateInfoKHR*,
    const VkAllocationCallbacks*, VkSurfaceKHR*);

VkResult VulkanApplication::CreateWindowSurface(vk::Instance instance,
                                                VkSurfaceKHR& surface) {
    VkResult err;
    VkWin32SurfaceCreateInfoKHR sci;
    PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR;

    vkCreateWin32SurfaceKHR =
        (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(
            instance, "vkCreateWin32SurfaceKHR");
    if (!vkCreateWin32SurfaceKHR) {
        fprintf(
            stderr,
            "Win32: Vulkan instance missing VK_KHR_win32_surface extension");
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    memset(&sci, 0, sizeof(sci));
    sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hinstance = m_hInstance;
    sci.hwnd = m_hWnd;

    err = vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface);
    assert(!err && "Win32: Failed to create Vulkan surface.");

    return err;
}

void VulkanApplication::CreateMainWindow() {
    WindowsApplication::CreateMainWindow();

    std::vector<const char*> extensions = {"VK_KHR_surface",
                                           "VK_KHR_win32_surface"};

    // 设置回调函数
    auto getFramebufferSize = [this](uint32_t& width, uint32_t& height) {
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
                throw std::runtime_error("faild to create window surface!");
            }

            surface = _surface;
        };

        m_Rhi.createSurface(createWindowSurface);
    }

    // 枚举物理设备并选择
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
    // 销毁Vulkan上下文
    m_Rhi.destroyAll();

    WindowsApplication::Finalize();
}
