#ifdef _WIN32
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#else
#define VK_USE_PLATFORM_XCB_KHR
#endif

#include "Vulkan/VulkanRHI.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class Window {
   public:
    Window() {
#ifdef _WIN32
        SetProcessDPIAware();
        WNDCLASSW type{};
        type.lpfnWndProc = DefWindowProcW;
        type.hInstance = GetModuleHandleW(nullptr);
        type.lpszClassName = L"VulkanFrameLoopRegression";
        Require(RegisterClassW(&type) != 0, "Win32 test class registration failed");
        m_window = CreateWindowExW(0, type.lpszClassName, L"Vulkan frame regression",
                                   WS_OVERLAPPEDWINDOW, 0, 0, 160, 160, nullptr,
                                   nullptr, type.hInstance, nullptr);
        Require(m_window != nullptr, "Win32 test window creation failed");
        ShowWindow(m_window, SW_SHOWNOACTIVATE);
        UpdateWindow(m_window);
        MSG message;
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
#else
        m_connection = xcb_connect(nullptr, nullptr);
        Require(m_connection && !xcb_connection_has_error(m_connection),
                "Xcb test requires a working display");
        m_screen = xcb_setup_roots_iterator(xcb_get_setup(m_connection)).data;
        m_window = xcb_generate_id(m_connection);
        xcb_create_window(m_connection, XCB_COPY_FROM_PARENT, m_window,
                           m_screen->root, 0, 0, 160, 160, 0,
                           XCB_WINDOW_CLASS_INPUT_OUTPUT, m_screen->root_visual,
                           0, nullptr);
        xcb_map_window(m_connection, m_window);
        xcb_flush(m_connection);
#endif
    }

    ~Window() {
#ifdef _WIN32
        DestroyWindow(m_window);
        UnregisterClassW(L"VulkanFrameLoopRegression", GetModuleHandleW(nullptr));
#else
        xcb_disconnect(m_connection);
#endif
    }

    void Resize(uint32_t width, uint32_t height) {
#ifdef _WIN32
        RECT rect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        Require(AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE) != 0,
                "Win32 test window rectangle adjustment failed");
        Require(SetWindowPos(m_window, nullptr, 0, 0, rect.right - rect.left,
                              rect.bottom - rect.top,
                              SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE) != 0,
                "Win32 test resize failed");
        MSG message;
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
#else
        const uint32_t values[] = {width, height};
        xcb_configure_window(m_connection, m_window,
                              XCB_CONFIG_WINDOW_WIDTH | XCB_CONFIG_WINDOW_HEIGHT,
                              values);
        xcb_flush(m_connection);
#endif
    }

    void Size(uint32_t& width, uint32_t& height) const {
#ifdef _WIN32
        RECT rect{};
        Require(GetClientRect(m_window, &rect) != 0, "Win32 test size query failed");
        width = static_cast<uint32_t>(rect.right);
        height = static_cast<uint32_t>(rect.bottom);
#else
        const auto cookie = xcb_get_geometry(m_connection, m_window);
        const std::unique_ptr<xcb_get_geometry_reply_t, decltype(&std::free)> reply(
            xcb_get_geometry_reply(m_connection, cookie, nullptr), std::free);
        Require(reply != nullptr, "Xcb test size query failed");
        width = reply->width;
        height = reply->height;
#endif
    }

    std::vector<const char*> Extensions() const {
#ifdef _WIN32
        return {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
#else
        return {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_XCB_SURFACE_EXTENSION_NAME};
#endif
    }

    void Surface(const vk::Instance& instance, vk::SurfaceKHR& surface) const {
#ifdef _WIN32
        const auto create = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
            instance.getProcAddr("vkCreateWin32SurfaceKHR"));
        VkWin32SurfaceCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        info.hinstance = GetModuleHandleW(nullptr);
        info.hwnd = m_window;
#else
        const auto create = reinterpret_cast<PFN_vkCreateXcbSurfaceKHR>(
            instance.getProcAddr("vkCreateXcbSurfaceKHR"));
        VkXcbSurfaceCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
        info.connection = m_connection;
        info.window = m_window;
#endif
        Require(create != nullptr, "Native Vulkan surface entry point unavailable");
        VkSurfaceKHR native = VK_NULL_HANDLE;
        Require(create(instance, &info, nullptr, &native) == VK_SUCCESS,
                "Native Vulkan test surface creation failed");
        surface = native;
    }

   private:
#ifdef _WIN32
    HWND m_window = nullptr;
#else
    xcb_connection_t* m_connection = nullptr;
    xcb_screen_t* m_screen = nullptr;
    xcb_window_t m_window = XCB_WINDOW_NONE;
#endif
};

void Run() {
#ifdef NDEBUG
    throw std::runtime_error("Run the Vulkan frame regression in Debug with validation enabled");
#endif
    Window window;
    vk::RenderPass renderPass;
    My::VulkanRHI rhi;
    rhi.destroyAll();
    rhi.destroyAll();

    rhi.setFramebufferSizeQueryCB([&](uint32_t& width, uint32_t& height) {
        window.Size(width, height);
    });
    bool surfaceFailureObserved = false;
    try {
        rhi.Initialize(window.Extensions(), [](const vk::Instance&, vk::SurfaceKHR&) {
            throw std::runtime_error("injected surface failure");
        });
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()) != "injected surface failure") throw;
        surfaceFailureObserved = true;
    }
    Require(surfaceFailureObserved && !rhi.GetDevice(), "Surface failure did not roll back");

    rhi.setFramebufferSizeQueryCB([](uint32_t&, uint32_t&) {
        throw std::runtime_error("injected device-stage failure");
    });
    bool deviceFailureObserved = false;
    try {
        rhi.Initialize(window.Extensions(), [&](const vk::Instance& instance, vk::SurfaceKHR& surface) {
            window.Surface(instance, surface);
        });
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()) != "injected device-stage failure") throw;
        deviceFailureObserved = true;
    }
    Require(deviceFailureObserved && !rhi.GetDevice(), "Device-stage failure did not roll back");

    bool minimized = true;
    rhi.setFramebufferSizeQueryCB([&](uint32_t& width, uint32_t& height) {
        if (minimized) width = height = 0;
        else window.Size(width, height);
    });
    rhi.CreateSwapChainCB([&] {
        renderPass = rhi.createRenderPass();
        rhi.createColorResources();
        rhi.createDepthResources();
        rhi.createFramebuffers(renderPass);
    });
    rhi.DestroySwapChainCB([&] {
        rhi.GetDevice().destroyRenderPass(renderPass);
        renderPass = nullptr;
    });
    rhi.Initialize(window.Extensions(), [&](const vk::Instance& instance, vk::SurfaceKHR& surface) {
        window.Surface(instance, surface);
    });
    Require(!rhi.BeginFrame(), "A minimized window must defer acquisition");

    minimized = false;
    window.Resize(128, 256);
    unsigned presented = 0;
    for (unsigned attempt = 0; attempt < 40 && presented < 20; ++attempt) {
        if (presented == 6) window.Resize(256, 128);
        if (!rhi.BeginFrame()) continue;
        auto command = rhi.GetFrameCommandBuffer();
        const std::array<vk::ClearValue, 2> clear = {
            vk::ClearColorValue(std::array<float, 4>{0.25f, 0.5f, 0.75f, 1.0f}),
            vk::ClearDepthStencilValue(1.0f, 0)};
        const vk::RenderPassBeginInfo pass(renderPass, rhi.GetFrameFramebuffer(),
            vk::Rect2D({0, 0}, rhi.GetSwapChainExtent()), clear);
        command.beginRenderPass(pass, vk::SubpassContents::eInline);
        command.endRenderPass();
        rhi.EndFrame();
        ++presented;
        if (presented == 12) {
            const auto generation = rhi.GetSwapChainGeneration();
            minimized = true;
            Require(!rhi.BeginFrame() && !rhi.BeginFrame(),
                    "Minimize must not acquire or reset a submission fence");
            Require(rhi.GetSwapChainGeneration() == generation,
                    "Minimize must not continuously recreate the swapchain");
            minimized = false;
            window.Resize(128, 256);
        }
    }
    Require(presented == 20, "The frame loop failed to make bounded progress");
    Require(rhi.GetSwapChainGeneration() >= 3, "Resize/restore did not rebuild the swapchain");
    rhi.WaitGPU();
    rhi.destroyAll();
    rhi.destroyAll();
    Require(rhi.GetValidationErrorCount() == 0,
            "Vulkan lifecycle/frame regression reported validation errors");
    std::cout << "Vulkan partial initialization, retry, 20 presented frames, resize, "
                 "minimize/restore and repeated shutdown passed\n";
}
}  // namespace

int main() {
    try {
        Run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan frame regression failed: " << error.what() << '\n';
        return 1;
    }
}
