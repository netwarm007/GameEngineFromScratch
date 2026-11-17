#include "Vulkan/VulkanGraphicsManager.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using namespace My;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Error, typename Action>
void Reject(Action action, const char* message) {
    bool rejected = false;
    try {
        action();
    } catch (const Error&) {
        rejected = true;
    }
    Require(rejected, message);
}

struct Context {
    ImGuiContext* value = ImGui::CreateContext();
    ~Context() { ImGui::DestroyContext(value); }
};

struct Texture {
    VulkanRHI& rhi;
    VulkanRHI::Texture value;
    ~Texture() { rhi.DestroyTexture(value); }
};

class TestApplication final : public IApplication {
   public:
    void SetCommandLineParameters(int, char**) final {}
    int GetCommandLineArgumentsCount() const final { return 0; }
    const char* GetCommandLineArgument(int) const final { return nullptr; }
    bool IsQuit() const final { return false; }
    void RequestQuit() final {}
    void CreateMainWindow() final { throw std::logic_error("Headless GUI test has no native window"); }
    void* GetMainWindowHandler() final { return nullptr; }
    const GfxConfiguration GetConfiguration() const final {
        GfxConfiguration config;
        config.msaaSamples = 1;
        return config;
    }
    void GetFramebufferSize(uint32_t& width, uint32_t& height) final { width = height = 16; }
};

class TestGraphics final : public VulkanGraphicsManager {
   public:
    using VulkanGraphicsManager::VulkanGraphicsManager;
    using VulkanGraphicsManager::InitializeGui;
    using VulkanGraphicsManager::PrepareGui;
    using VulkanGraphicsManager::RenderGui;
    using VulkanGraphicsManager::BeginFrame;
    using VulkanGraphicsManager::EndFrame;
    Frame& CurrentFrame() { return m_Frames.at(m_nFrameIndex); }
};

void RunManager(VulkanRHI& rhi, VulkanRHI::Texture& target, vk::RenderPass pass, vk::Framebuffer framebuffer) {
    TestApplication app;
    TestGraphics graphics(rhi);
    graphics.SetAppPointer(&app);
    Require(graphics.Initialize() == 0, "Vulkan GUI graphics-manager initialization failed");
    graphics.ResizeCanvas(17, 16);
    Image image;
    image.Width = image.Height = 2;
    image.bitcount = 32;
    image.bitdepth = 8;
    image.pixel_format = PIXEL_FORMAT::RGBA8;
    image.pitch = 12;
    image.data_size = 24;
    image.data = new uint8_t[24]{};
    const auto fill = [&](size_t channel) {
        std::memset(image.data, 0, image.data_size);
        for (size_t y = 0; y < 2; ++y) {
            for (size_t x = 0; x < 2; ++x) {
                image.data[y * image.pitch + x * 4 + channel] = 255;
                image.data[y * image.pitch + x * 4 + 3] = 255;
            }
        }
    };
    fill(0);
    auto texture = graphics.CreateTexture(image);
    const auto identity = texture.handler;
    graphics.InitializeGui(pass);
    Texture2DArray array;
    array.width = array.height = 4;
    array.size = 2;
    array.pixel_format = PIXEL_FORMAT::D32;
    graphics.GenerateTextureArray(array);
    Texture2D view;
    graphics.CreateTextureView(view, array, 1, 0);
    const auto depth = graphics.ReadTexture(view);
    graphics.ReleaseTexture(array);
    Require(graphics.ReadTexture(view) == depth, "Vulkan GUI view did not retain its released parent image");
    const auto oldTarget = graphics.CurrentFrame().colorTextures.front();
    for (uint32_t frame = 0; frame < 2; ++frame) {
        if (frame) {
            fill(1);
            graphics.UpdateTexture(texture, image);
            Require(texture.handler == identity && graphics.ReadTexture(texture)[1] == 255,
                    "Vulkan GUI image update changed its identity or ignored padded rows");
            image.Width = 3;
            Reject<std::invalid_argument>([&] { graphics.UpdateTexture(texture, image); },
                                          "Mismatched Vulkan GUI update dimensions were accepted");
            image.Width = 2;
            Require(texture.handler == identity && graphics.ReadTexture(texture)[1] == 255,
                    "Rejected Vulkan GUI update damaged the original image");
        }
        auto& current = graphics.CurrentFrame();
        current.frameIndex = frame;
        current.frameContext.numLights = 0;
        graphics.BeginFrame(current);
        if (!frame) {
            auto icon = graphics.CreateTexture(image);
            Require(icon.handler != identity, "Lazy Vulkan GUI icon reused an existing texture identity");
            graphics.ResizeCanvas(20, 12);
            Require(current.colorTextures.front().width == 17,
                    "Vulkan GUI resize destroyed the canvas during an active recording");
        } else {
            Require(current.colorTextures.front().width == 20 && current.depthTexture.height == 12,
                    "Vulkan GUI deferred resize was not applied before the next recording");
            graphics.ReleaseTexture(view);
        }
        ImGui::NewFrame();
        ImGui::GetBackgroundDrawList()->AddImage(ImTextureRef(static_cast<ImTextureID>(identity)),
                                                {0, 0}, {32, 32});
        ImGui::Render();
        graphics.EndFrame(current);
        if (frame) {
            Reject<std::invalid_argument>([&] { graphics.ReadTexture(oldTarget); },
                                          "Vulkan GUI resize retained a stale canvas handle");
        }
        graphics.PrepareGui();
        rhi.ExecuteCommands([&](vk::CommandBuffer command) {
            const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            rhi.TransitionTexture(command, target, range, vk::ImageLayout::eColorAttachmentOptimal);
            const vk::ClearValue clear(vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1}));
            command.beginRenderPass({pass, framebuffer, {{0, 0}, {96, 64}}, clear}, vk::SubpassContents::eInline);
            graphics.RenderGui(*ImGui::GetDrawData(), command);
            command.endRenderPass();
            rhi.TransitionTexture(command, target, range, vk::ImageLayout::eShaderReadOnlyOptimal);
        });
        const auto pixels = rhi.ReadTexture(target);
        Require(pixels[(8 * 96 + 8) * 4 + frame] == 255 &&
                    pixels[(8 * 96 + 8) * 4 + (1 - frame)] == 0,
                "Vulkan GUI did not sample the graphics manager's current engine texture");
    }
    graphics.ReleaseTexture(texture);
    graphics.Finalize();
    graphics.Finalize();
    Require(!ImGui::GetIO().BackendRendererUserData, "Graphics finalization retained the Vulkan GUI renderer");
}

void Run(VulkanRHI& rhi) {
    Texture target{rhi, rhi.CreateTexture(96, 64, vk::Format::eR8G8B8A8Unorm, 1, 1,
        vk::ImageViewType::e2D, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment)};
    const vk::AttachmentDescription attachment({}, target.value.format, vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal);
    const vk::AttachmentReference color(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::SubpassDescription subpass;
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.setColorAttachments(color);
    const auto pass = rhi.GetDevice().createRenderPassUnique({{}, attachment, subpass});
    const auto framebuffer = rhi.GetDevice().createFramebufferUnique(
        {{}, *pass, target.value.descriptor, target.value.width, target.value.height, 1});
    Texture source{rhi, rhi.CreateTexture(2, 2, vk::Format::eR8G8B8A8Unorm)};
    const std::array<uint8_t, 16> texels{255, 0, 0, 255, 0, 255, 0, 255,
                                       0, 0, 255, 255, 255, 255, 255, 255};
    rhi.UpdateTexture(source.value, texels.data(), texels.size());
    Texture depth{rhi, rhi.CreateTexture(2, 2, vk::Format::eD32Sfloat)};
    const std::array<float, 4> depths{0.5f, 0.5f, 0.5f, 0.5f};
    rhi.UpdateTexture(depth.value, depths.data(), sizeof(depths));
    {
        VulkanGuiRenderer missingContext(rhi);
        Reject<std::logic_error>([&] { missingContext.Initialize(*pass, 2); },
                                 "Vulkan GUI created a context instead of requiring its owner");
    }
    Context context;
    VulkanGuiRenderer renderer(rhi);
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {96, 64};
    io.DeltaTime = 1.0f / 60;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    int borrowed = 0;
    io.BackendRendererUserData = &borrowed;
    Reject<std::logic_error>([&] { renderer.Initialize(*pass, 2); },
                             "Vulkan GUI replaced a borrowed renderer");
    Require(io.BackendRendererUserData == &borrowed, "Rejected initialization mutated borrowed state");
    io.BackendRendererUserData = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    Reject<std::runtime_error>([&] { renderer.Initialize(*pass, 2); },
                               "Unsafe secondary swapchain helpers were accepted");
    Require(!io.BackendRendererUserData && (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable),
            "Viewport rejection silently disabled the requested behavior");
    io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;
    constexpr ImTextureID identity = 12345;
    for (uint32_t attempt = 0; attempt < 2; ++attempt) {
        renderer.Initialize(*pass, 2);
        auto* backend = io.BackendRendererUserData;
        Require(backend && ImGui::GetCurrentContext() == context.value &&
                    (io.BackendFlags & ImGuiBackendFlags_RendererHasTextures),
                "Vulkan renderer failed to bind the borrowed context and dynamic texture support");
        Reject<std::logic_error>([&] { renderer.Initialize(*pass, 2); },
                                 "Duplicate GUI initialization was accepted");
        Require(io.BackendRendererUserData == backend, "Duplicate initialization replaced renderer ownership");
        renderer.RegisterTexture(identity, source.value.descriptor, vk::ImageLayout::eShaderReadOnlyOptimal,
                                  source.value.format);
        renderer.RegisterTexture(identity + 1, depth.value.descriptor, vk::ImageLayout::eShaderReadOnlyOptimal,
                                  depth.value.format);
        Reject<std::invalid_argument>([&] {
            renderer.RegisterTexture(identity, source.value.descriptor, vk::ImageLayout::eShaderReadOnlyOptimal,
                                      source.value.format);
        }, "Duplicate GUI texture identity was accepted");
        for (uint32_t frame = 0; frame < 3; ++frame) {
            if (frame == 1) {
                ImFontConfig config;
                config.SizePixels = 24;
                io.Fonts->AddFontDefault(&config);
            }
            renderer.NewFrame();
            ImGui::NewFrame();
            auto* draw = ImGui::GetBackgroundDrawList();
            draw->AddImage(ImTextureRef(identity), {0, 0}, {32, 32});
            draw->PushClipRect({40, 4}, {56, 20});
            draw->AddImage(ImTextureRef(identity), {32, 0}, {64, 32});
            draw->PopClipRect();
            draw->AddImage(ImTextureRef(identity + 1), {64, 0}, {96, 32});
            draw->AddText({2, 40}, IM_COL32_WHITE, "Vulkan UI");
            ImGui::Render();
            auto& data = *ImGui::GetDrawData();
            renderer.PrepareTextures();
            Require(io.Fonts->TexRef.GetTexID() != ImTextureID_Invalid,
                    "Dynamic font atlas was not uploaded");
            const auto render = [&](vk::CommandBuffer command) {
                const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
                rhi.TransitionTexture(command, target.value, range, vk::ImageLayout::eColorAttachmentOptimal);
                const vk::ClearValue clear(vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1}));
                command.beginRenderPass({*pass, *framebuffer, {{0, 0}, {96, 64}}, clear}, vk::SubpassContents::eInline);
                renderer.Render(data, command);
                command.endRenderPass();
                rhi.TransitionTexture(command, target.value, range, vk::ImageLayout::eShaderReadOnlyOptimal);
            };
            rhi.ExecuteCommands(render);
            const auto pixels = rhi.ReadTexture(target.value);
            const auto pixel = [&](uint32_t x, uint32_t y, uint32_t channel) {
                return pixels[(y * 96 + x) * 4 + channel];
            };
            Require(pixel(4, 4, 0) > 250 && pixel(4, 4, 1) < 3 &&
                        pixel(27, 4, 1) > 250 && pixel(4, 27, 2) > 250 &&
                        pixel(27, 27, 0) > 250 && pixel(27, 27, 1) > 250,
                    "Vulkan GUI sampled image identity or top-left texture orientation is incorrect");
            Require(pixel(36, 10, 0) == 0 && pixel(43, 8, 0) > 0 && pixel(60, 10, 1) == 0,
                    "Vulkan GUI ignored its draw-command clip rectangle");
            Require(pixel(80, 16, 0) >= 127 && pixel(80, 16, 0) <= 128,
                    "Vulkan GUI could not display a depth texture with capability-safe filtering");
            size_t glyphPixels = 0;
            for (uint32_t y = 40; y < 62; ++y) {
                for (uint32_t x = 2; x < 90; ++x) glyphPixels += pixel(x, y, 0) > 40;
            }
            Require(glyphPixels > 40, "Vulkan GUI dynamic font text was not rendered");
            bool originalIdentity = false;
            for (const auto* list : data.CmdLists) {
                for (const auto& command : list->CmdBuffer) {
                    if (!command.TexRef._TexData && command.ElemCount) {
                        Require(command.GetTexID() == identity || command.GetTexID() == identity + 1,
                                "GUI submission permanently rewrote shared texture IDs");
                        originalIdentity = true;
                    }
                }
            }
            Require(originalIdentity, "GUI texture fixture did not produce image draw commands");
        }
        renderer.ReleaseTexture(identity);
        Reject<std::invalid_argument>([&] { renderer.ReleaseTexture(identity); },
                                      "Released GUI descriptor remained registered");
        Reject<std::invalid_argument>([&] {
            rhi.ExecuteCommands([&](vk::CommandBuffer command) { renderer.Render(*ImGui::GetDrawData(), command); });
        }, "Stale image draw data reached the Vulkan driver");
        Require(io.BackendRendererUserData == backend, "Rejected stale texture destroyed the renderer");
        renderer.Finalize();
        renderer.Finalize();
        Require(!io.BackendRendererUserData && !ImGui::GetMainViewport()->RendererUserData &&
                    (io.ConfigFlags & ImGuiConfigFlags_DockingEnable),
                "Vulkan GUI cleanup retained its backend or changed docking preferences");
    }
    RunManager(rhi, target.value, *pass, *framebuffer);
}
}  // namespace

int main() {
    try {
        My::VulkanRHI rhi;
        rhi.InitializeHeadless();
        Run(rhi);
        rhi.destroyAll();
        Require(rhi.GetValidationErrorCount() == 0, "Vulkan GUI produced validation errors");
        std::cout << "Vulkan ImGui borrowed ownership, dynamic font atlas, sampled textures, clipping, "
                     "stale descriptors, retry, and repeated cleanup passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan GUI regression failed: " << error.what() << '\n';
        return 1;
    }
}
