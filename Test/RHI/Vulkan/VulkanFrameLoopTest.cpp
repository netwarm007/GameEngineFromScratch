#ifdef _WIN32
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#else
#define VK_USE_PLATFORM_XCB_KHR
#endif

#include "Vulkan/VulkanGraphicsManager.hpp"
#include "GuiContext.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <filesystem>

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
        static unsigned identity = 0;
        m_className = L"VulkanFrameLoopRegression" + std::to_wstring(++identity);
        type.lpszClassName = m_className.c_str();
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
#elif defined(__APPLE__)
        m_window = [[NSWindow alloc]
            initWithContentRect:NSMakeRect(0, 0, 160, 160)
                      styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskResizable
                        backing:NSBackingStoreBuffered defer:NO];
        Require(m_window != nil, "Cocoa test window creation failed");
        m_window.releasedWhenClosed = NO;
        m_window.title = @"Vulkan frame regression";
        m_window.contentView.wantsLayer = YES;
        m_window.contentView.layer = [CAMetalLayer layer];
        [m_window orderFront:nil];
        PumpEvents();
        std::cout << "Cocoa Vulkan native backing scale: " << m_window.backingScaleFactor << '\n';
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
        UnregisterClassW(m_className.c_str(), GetModuleHandleW(nullptr));
#elif defined(__APPLE__)
        [m_window close];
        [m_window release];
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
#elif defined(__APPLE__)
        [m_window setContentSize:NSMakeSize(width, height)];
        PumpEvents();
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
#elif defined(__APPLE__)
        PumpEvents();
        NSView* view = m_window.contentView;
        const auto size = [view convertRectToBacking:view.bounds].size;
        width = static_cast<uint32_t>(size.width);
        height = static_cast<uint32_t>(size.height);
        auto* layer = static_cast<CAMetalLayer*>(view.layer);
        layer.contentsScale = m_window.backingScaleFactor;
        layer.drawableSize = CGSizeMake(width, height);
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
#elif defined(__APPLE__)
        return {VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_METAL_SURFACE_EXTENSION_NAME,
                VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
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
#elif defined(__APPLE__)
        uint32_t width = 0, height = 0;
        Size(width, height);
        const auto create = reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(
            instance.getProcAddr("vkCreateMetalSurfaceEXT"));
        VkMetalSurfaceCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        info.pLayer = static_cast<CAMetalLayer*>(m_window.contentView.layer);
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
    std::wstring m_className;
#elif defined(__APPLE__)
    static void PumpEvents() {
        while (NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
            untilDate:nil inMode:NSDefaultRunLoopMode dequeue:YES]) {
            [NSApp sendEvent:event];
        }
        [NSApp updateWindows];
    }
    NSWindow* m_window = nil;
#else
    xcb_connection_t* m_connection = nullptr;
    xcb_screen_t* m_screen = nullptr;
    xcb_window_t m_window = XCB_WINDOW_NONE;
#endif
};

class GuiApplication final : public My::IApplication {
   public:
    explicit GuiApplication(Window& window) : m_window(window) {}
    void SetCommandLineParameters(int, char**) final {}
    int GetCommandLineArgumentsCount() const final { return 0; }
    const char* GetCommandLineArgument(int) const final { return nullptr; }
    bool IsQuit() const final { return false; }
    void RequestQuit() final {}
    void CreateMainWindow() final { throw std::logic_error("Native GUI test owns its window"); }
    void* GetMainWindowHandler() final { return nullptr; }
    const My::GfxConfiguration GetConfiguration() const final {
        My::GfxConfiguration config;
        config.msaaSamples = 1;
        return config;
    }
    void GetFramebufferSize(uint32_t& width, uint32_t& height) final { m_window.Size(width, height); }

   private:
    Window& m_window;
};

class GuiGraphics final : public My::VulkanGraphicsManager {
   public:
    using VulkanGraphicsManager::VulkanGraphicsManager;
    using VulkanGraphicsManager::BeginFrame;
    using VulkanGraphicsManager::EndFrame;
    My::Frame& CurrentFrame() {
        auto& frame = m_Frames.at(m_nFrameIndex);
        frame.frameIndex = m_nFrameIndex;
        frame.frameContext.numLights = 0;
        return frame;
    }
};

void RunNativeGui() {
    Window window, floating;
    GuiApplication app(window);
    My::VulkanRHI rhi;
    rhi.setFramebufferSizeQueryCB([&](uint32_t& width, uint32_t& height) { window.Size(width, height); });
    rhi.SetGetGfxConfigCB([&] { return app.GetConfiguration(); });
    rhi.Initialize(window.Extensions(), [&](const vk::Instance& instance, vk::SurfaceKHR& surface) {
        window.Surface(instance, surface);
    });
    My::GuiContext context;
    Require(context.Initialize() == 0, "Native Vulkan GUI context initialization failed");
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DeltaTime = 1.0f / 60;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::GetPlatformIO().Platform_CreateVkSurface =
        [](ImGuiViewport* viewport, ImU64 instance, const void*, ImU64* output) -> int {
        vk::SurfaceKHR surface;
        static_cast<Window*>(viewport->PlatformUserData)->Surface(
            reinterpret_cast<VkInstance>(instance), surface);
        *output = reinterpret_cast<ImU64>(static_cast<VkSurfaceKHR>(surface));
        return VK_SUCCESS;
    };
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
        GuiGraphics graphics(rhi);
        graphics.SetAppPointer(&app);
        Require(graphics.Initialize() == 0, "Native Vulkan GUI graphics initialization failed");
        My::Image image;
        image.Width = image.Height = 1;
        image.bitcount = 32;
        image.bitdepth = 8;
        image.pixel_format = My::PIXEL_FORMAT::RGBA8;
        image.pitch = image.data_size = 4;
        image.data = new uint8_t[4]{220, 80, 20, 255};
        auto texture = graphics.CreateTexture(image);
        ImGuiViewport viewport;
        viewport.ID = 0x1234;
        viewport.Size = {160, 160};
        viewport.PlatformUserData = &floating;
        auto& platform = ImGui::GetPlatformIO();
        struct ViewportCleanup {
            ImGuiViewport& viewport;
            ~ViewportCleanup() {
                if (viewport.RendererUserData) ImGui::GetPlatformIO().Renderer_DestroyWindow(&viewport);
                viewport.PlatformUserData = nullptr;
            }
        } cleanup{viewport};
        platform.Renderer_CreateWindow(&viewport);
        Require(viewport.RendererUserData != nullptr, "Native Vulkan secondary viewport was not initialized");
        Require(rhi.GetValidationErrorCount() == 0,
                "Secondary viewport creation produced validation errors before first acquire");
        const auto generation = rhi.GetSwapChainGeneration();
        for (unsigned frame = 0; frame < 12; ++frame) {
            if (frame == 3) window.Resize(256, 128);
            if (frame == 5) {
                floating.Resize(224, 96);
                viewport.Size = {224, 96};
                platform.Renderer_SetWindowSize(&viewport, viewport.Size);
                Require(rhi.GetValidationErrorCount() == 0,
                        "Secondary viewport resize produced validation errors before reacquire");
            }
            if (frame == 8) {
                platform.Renderer_DestroyWindow(&viewport);
                Require(!viewport.RendererUserData, "Native Vulkan viewport destruction retained renderer data");
                platform.Renderer_CreateWindow(&viewport);
                Require(rhi.GetValidationErrorCount() == 0,
                        "Secondary viewport recreation produced validation errors before reacquire");
            }
            uint32_t width = 0, height = 0;
            window.Size(width, height);
            io.DisplaySize = {static_cast<float>(width), static_cast<float>(height)};
            auto& current = graphics.CurrentFrame();
            graphics.BeginFrame(current);
            ImGui::NewFrame();
            ImGui::GetBackgroundDrawList()->AddImage(
                ImTextureRef(static_cast<ImTextureID>(texture.handler)), {8, 8}, {80, 80});
            ImGui::GetBackgroundDrawList()->AddText({8, 90}, IM_COL32_WHITE, "Native Vulkan GUI");
            ImGui::Render();
            const auto capture = std::filesystem::path("vulkan-images") /
                ("native-gui-" + std::to_string(attempt) + ".ppm");
            if (frame == 10) {
                std::filesystem::create_directories(capture.parent_path());
                graphics.RequestScreenshot(capture);
            }
            graphics.EndFrame(current);
            if (frame == 10) {
                std::ifstream input(capture, std::ios::binary);
                std::string magic;
                unsigned captureWidth = 0, captureHeight = 0, maxColor = 0;
                input >> magic >> captureWidth >> captureHeight >> maxColor;
                Require(input.good() && magic == "P6" && captureWidth == width &&
                            captureHeight == height && maxColor == 255,
                        "Native Vulkan application capture has invalid dimensions or PPM header");
                Require(input.get() == '\n', "Native Vulkan capture header has no pixel separator");
                std::vector<uint8_t> pixels(size_t{width} * height * 3);
                input.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
                Require(input.good() && input.peek() == std::char_traits<char>::eof(),
                        "Native Vulkan application capture was truncated or contains extra pixels");
                const auto pixel = (size_t{16} * width + 16) * 3;
                Require(pixels[pixel] == 220 && pixels[pixel + 1] == 80 && pixels[pixel + 2] == 20,
                        "Native Vulkan capture changed image channels, top-left placement, or display gamma");
                Require(pixels[0] < 20 && pixels[1] < 20 && pixels[2] < 20,
                        "Native Vulkan capture was not the application-only composed GUI framebuffer");
            }
            viewport.DrawData = ImGui::GetDrawData();
            auto* originalOwner = viewport.DrawData->OwnerViewport;
            viewport.DrawData->OwnerViewport = &viewport;
            struct RestoreOwner {
                ImDrawData& data;
                ImGuiViewport* owner;
                ~RestoreOwner() { data.OwnerViewport = owner; }
            } restore{*viewport.DrawData, originalOwner};
            platform.Renderer_RenderWindow(&viewport, nullptr);
            platform.Renderer_SwapBuffers(&viewport, nullptr);
            Require(rhi.GetValidationErrorCount() == 0,
                    "Secondary viewport acquire/render/present produced validation errors");
        }
        Require(rhi.GetSwapChainGeneration() > generation || attempt != 0,
                "Native Vulkan GUI resize did not recreate main composition targets");
        platform.Renderer_DestroyWindow(&viewport);
        graphics.Finalize();
        graphics.Finalize();
        Require(!io.BackendRendererUserData, "Native Vulkan GUI renderer shutdown retained borrowed context state");
    }
    context.Finalize();
    rhi.destroyAll();
    Require(rhi.GetValidationErrorCount() == 0, "Native Vulkan GUI/presentation produced validation errors");
    std::cout << "Vulkan native GUI composition, engine texture IDs, main/secondary resize, "
                 "strict secondary presentation retirement, recreate, retry, and cleanup passed\n";
}

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
    unsigned submitted = 0;
    for (unsigned attempt = 0; attempt < 40 && submitted < 20; ++attempt) {
        if (submitted == 6) window.Resize(256, 128);
        if (!rhi.BeginFrame()) continue;
        if (submitted == 3) {
            const auto generation = rhi.GetSwapChainGeneration();
            rhi.AbortFrame();
            Require(rhi.BeginFrame() && rhi.GetSwapChainGeneration() > generation,
                    "Abandoned acquisition did not permit a safe frame retry");
        }
        auto command = rhi.GetFrameCommandBuffer();
        const std::array<vk::ClearValue, 2> clear = {
            vk::ClearColorValue(std::array<float, 4>{0.25f, 0.5f, 0.75f, 1.0f}),
            vk::ClearDepthStencilValue(1.0f, 0)};
        const vk::RenderPassBeginInfo pass(renderPass, rhi.GetFrameFramebuffer(),
            vk::Rect2D({0, 0}, rhi.GetSwapChainExtent()), clear);
        command.beginRenderPass(pass, vk::SubpassContents::eInline);
        command.endRenderPass();
        rhi.EndFrame();
        ++submitted;
        if (submitted == 12) {
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
    Require(submitted == 20, "The frame loop failed to make bounded progress");
    Require(rhi.GetSwapChainGeneration() >= 3, "Resize/restore did not rebuild the swapchain");
    rhi.WaitGPU();
    bool acquired = false;
    for (unsigned attempt = 0; attempt < 10 && !acquired; ++attempt) acquired = rhi.BeginFrame();
    Require(acquired, "Could not acquire the frame used to test aborted recording cleanup");
    rhi.destroyAll();
    rhi.destroyAll();
    Require(rhi.GetValidationErrorCount() == 0,
            "Vulkan lifecycle/frame regression reported validation errors");
    std::cout << "Vulkan partial initialization, retry, 20 native frame submissions, resize, "
                 "minimize/restore, presentation-fence retirement, aborted recording, and repeated shutdown passed\n";
}
}  // namespace

int main() {
#ifdef __APPLE__
    @autoreleasepool {
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp finishLaunching];
#endif
    try {
        Run();
        RunNativeGui();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan frame regression failed: " << error.what() << '\n';
        return 1;
    }
#ifdef __APPLE__
    }
#endif
}
