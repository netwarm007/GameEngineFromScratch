#include "VulkanGraphicsManager.hpp"

#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <string_view>

using namespace My;

namespace {
vk::ImageLayout SampledLayout(const VulkanRHI::Texture& texture) {
    return texture.aspect == vk::ImageAspectFlagBits::eColor
        ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eDepthStencilReadOnlyOptimal;
}
}  // namespace

void VulkanGraphicsManager::RequestScreenshot(const std::filesystem::path& path) {
    if (!m_rhi.HasSurface() || !m_screenshot.empty() || path.empty() || path.extension() != ".ppm") {
        throw std::invalid_argument("Vulkan screenshot requires an idle capture request and a native-window .ppm output path");
    }
    m_screenshot = path;
    m_screenshotDue = std::chrono::steady_clock::now();
}

void VulkanGraphicsManager::InitializeCapture() {
    if (const auto* path = std::getenv("GEFS_SCREENSHOT_PATH")) RequestScreenshot(path);
    for (int argument = 1; argument < m_pApp->GetCommandLineArgumentsCount(); ++argument) {
        if (std::string_view(m_pApp->GetCommandLineArgument(argument)) == "-screenshot") {
            if (++argument >= m_pApp->GetCommandLineArgumentsCount()) {
                throw std::invalid_argument("Vulkan -screenshot requires a .ppm output path");
            }
            RequestScreenshot(m_pApp->GetCommandLineArgument(argument));
        }
    }
    if (!m_screenshot.empty()) {
        m_quitAfterCapture = true;
        m_screenshotDue = std::chrono::steady_clock::now() + std::chrono::seconds(45);
        if (const auto* path = std::getenv("GEFS_SCREENSHOT_WAIT_FOR")) m_screenshotGate = path;
    }
}

void VulkanGraphicsManager::SubmitGuiFrame() {
    const bool ready = !m_screenshot.empty() &&
        (m_screenshotGate.empty() ? std::chrono::steady_clock::now() >= m_screenshotDue
                                  : std::filesystem::exists(m_screenshotGate));
    if (!ready) {
        m_rhi.EndFrame();
        return;
    }
    auto image = m_rhi.EndFrameWithCapture();
    std::ofstream output(m_screenshot, std::ios::binary);
    output << "P6\n" << image.Width << ' ' << image.Height << "\n255\n";
    for (size_t pixel = 0; pixel < image.data_size; pixel += 4) {
        output.write(reinterpret_cast<const char*>(image.data + pixel), 3);
    }
    output.close();
    if (!output) throw std::runtime_error("Cannot write Vulkan application screenshot: " + m_screenshot.string());
    std::cout << "Vulkan application framebuffer capture saved: " << image.Width << 'x' << image.Height << '\n';
    m_screenshot.clear();
    if (m_quitAfterCapture) m_pApp->RequestQuit();
}

TextureHandler VulkanGraphicsManager::StoreTexture(std::shared_ptr<OwnedTexture> texture) {
    if (m_nextTexture == (std::numeric_limits<TextureHandler>::max)()) {
        throw std::overflow_error("Vulkan texture identity exhausted");
    }
    const auto id = m_nextTexture + 1;
    m_ownedTextures.emplace(id, texture);
    try {
        if (m_gui && texture->guiVisible) {
            m_gui->RegisterTexture(static_cast<ImTextureID>(id), texture->texture.descriptor,
                                   SampledLayout(texture->texture), texture->texture.format);
        }
    } catch (...) {
        m_ownedTextures.erase(id);
        throw;
    }
    m_nextTexture = id;
    return id;
}

void VulkanGraphicsManager::InitializeGui(vk::RenderPass renderPass) {
    if (!IsInitialized() || m_gui || m_frameRecording) {
        throw std::logic_error("Vulkan GUI binding requires an idle initialized graphics manager");
    }
    auto gui = std::make_unique<VulkanGuiRenderer>(m_rhi);
    gui->Initialize(renderPass, GfxConfiguration::kMaxInFlightFrameCount);
    for (const auto& [identity, texture] : m_ownedTextures) {
        if (texture->guiVisible) {
            gui->RegisterTexture(static_cast<ImTextureID>(identity), texture->texture.descriptor,
                                  SampledLayout(texture->texture), texture->texture.format);
        }
    }
    m_gui = std::move(gui);
}

void VulkanGraphicsManager::PrepareGui() {
    if (!m_gui) throw std::logic_error("Vulkan GUI renderer is not initialized");
    m_gui->PrepareTextures();
}

void VulkanGraphicsManager::RenderGui(ImDrawData& drawData, vk::CommandBuffer command) {
    if (!m_gui) throw std::logic_error("Vulkan GUI renderer is not initialized");
    m_gui->Render(drawData, command);
}

void VulkanGraphicsManager::CreateGuiTargets() {
    m_guiTargets.clear();
    const auto format = m_rhi.GetSwapChainFormat();
    if (format != m_guiFormat) {
        if (m_gui) m_gui->Finalize();
        m_gui.reset();
        m_guiPass.reset();
        const vk::AttachmentDescription attachment({}, format, vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);
        const vk::AttachmentReference color(0, vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subpass;
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.setColorAttachments(color);
        const vk::SubpassDependency dependency(VK_SUBPASS_EXTERNAL, 0,
            vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {}, vk::AccessFlagBits::eColorAttachmentWrite);
        m_guiPass = m_rhi.GetDevice().createRenderPassUnique({{}, attachment, subpass, dependency});
        InitializeGui(*m_guiPass);
        m_guiFormat = format;
    }
    const auto size = m_rhi.GetSwapChainExtent();
    for (const auto view : m_rhi.GetSwapChainViews()) {
        m_guiTargets.push_back(m_rhi.GetDevice().createFramebufferUnique(
            {{}, *m_guiPass, view, size.width, size.height, 1}));
    }
}

void VulkanGraphicsManager::ComposeGui() {
    if (!m_gui || !m_guiPass || !ImGui::GetDrawData()) {
        throw std::logic_error("Native Vulkan composition requires a completed application GUI frame");
    }
    PrepareGui();
    const vk::ClearValue clear(vk::ClearColorValue(std::array<float, 4>{0.05f, 0.05f, 0.05f, 1}));
    const auto command = Command();
    command.beginRenderPass({*m_guiPass, *m_guiTargets.at(m_rhi.GetCurrentImageIndex()),
        {{0, 0}, m_rhi.GetSwapChainExtent()}, clear}, vk::SubpassContents::eInline);
    RenderGui(*ImGui::GetDrawData(), command);
    command.endRenderPass();
}

void VulkanGraphicsManager::ResizeCanvas(int32_t width, int32_t height) {
    if (width <= 0 || height <= 0) return;
    if (m_frameRecording) {
        m_pendingWidth = width;
        m_pendingHeight = height;
    } else {
        GraphicsManager::ResizeCanvas(width, height);
    }
}

void VulkanGraphicsManager::CreateTextureView(Texture2D& descriptor,
    const TextureArrayBase& source, uint32_t slice, uint32_t mip) {
    const auto& original = GetTextureResource(source);
    if (descriptor.handler || descriptor.buff || slice >= original.layers || mip >= original.mipLevels ||
        original.samples != vk::SampleCountFlagBits::e1 ||
        m_ownedTextures.at(source.handler)->parent) {
        throw std::invalid_argument("Invalid Vulkan GUI texture view range or output descriptor");
    }
    auto owned = std::make_shared<OwnedTexture>(m_rhi);
    owned->parent = m_ownedTextures.at(source.handler);
    owned->slice = slice;
    owned->mip = mip;
    owned->texture = original;
    owned->texture.width = (std::max)(1u, original.width >> mip);
    owned->texture.height = (std::max)(1u, original.height >> mip);
    owned->texture.layers = owned->texture.mipLevels = 1;
    owned->texture.layoutBaseMipLevel = mip;
    owned->texture.layoutBaseArrayLayer = slice;
    owned->view = m_rhi.GetDevice().createImageViewUnique(
        {{}, original.image, vk::ImageViewType::e2D, original.format, {},
         {original.aspect, mip, 1, slice, 1}});
    owned->texture.descriptor = *owned->view;
    owned->guiVisible = true;
    Texture2D result;
    result.width = owned->texture.width;
    result.height = owned->texture.height;
    result.pixel_format = source.pixel_format;
    result.format = source.format;
    result.handler = StoreTexture(std::move(owned));
    descriptor = result;
}

void VulkanGraphicsManager::UpdateTexture(Texture2D& descriptor, Image& image) {
    if (m_frameRecording) throw std::logic_error("Vulkan texture update requires a completed graphics frame");
    auto& texture = GetTextureResource(descriptor);
    const auto& owner = m_ownedTextures.at(descriptor.handler);
    if (owner->parent || texture.width != image.Width || texture.height != image.Height ||
        texture.layers != 1 || texture.mipLevels != 1 || texture.samples != vk::SampleCountFlagBits::e1 ||
        !image.data || image.compressed || !image.mipmaps.empty()) {
        throw std::invalid_argument("Vulkan GUI updates require a matching uncompressed single-level owned image");
    }
    m_rhi.WaitGPU();
    // Reuse the production upload adapter for padded rows and RGB expansion.
    OwnedTexture upload(m_rhi);
    upload.texture = m_rhi.createTextureImage(image);
    if (upload.texture.format != texture.format) {
        throw std::invalid_argument("Vulkan texture update format does not match the owned image");
    }
    m_rhi.ExecuteCommands([&](vk::CommandBuffer command) {
        const vk::ImageSubresourceRange range(texture.aspect, 0, 1, 0, 1);
        m_rhi.TransitionTexture(command, upload.texture, range, vk::ImageLayout::eTransferSrcOptimal);
        m_rhi.TransitionTexture(command, texture, range, vk::ImageLayout::eTransferDstOptimal);
        const vk::ImageSubresourceLayers layer(texture.aspect, 0, 0, 1);
        command.copyImage(upload.texture.image, vk::ImageLayout::eTransferSrcOptimal,
            texture.image, vk::ImageLayout::eTransferDstOptimal,
            vk::ImageCopy(layer, {0, 0, 0}, layer, {0, 0, 0}, {texture.width, texture.height, 1}));
        m_rhi.TransitionTexture(command, texture, range, SampledLayout(texture));
    });
}
