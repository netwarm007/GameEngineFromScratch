#pragma once
#include <array>
#include <functional>
#include <string>
#include <vector>

#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.hpp>
#include "Buffer.hpp"
#include "GfxConfiguration.hpp"
#include "Image.hpp"
#include "geommath.hpp"

namespace My {

class VulkanRHI {
   public:
    using QueryFrameBufferSizeCBFunc = std::function<void(uint32_t&, uint32_t&)>;
    using CreateSurfaceCBFunc =
        std::function<void(const vk::Instance&, vk::SurfaceKHR&)>;
    using GetGfxConfigCBFunc = std::function<const GfxConfiguration&()>;
    using CreateSwapChainCBFunc = std::function<void()>;
    using DestroySwapChainCBFunc = std::function<void()>;

    struct IndexBuffer {
        vk::Buffer buffer;
        vk::DeviceMemory heap;

        uint32_t indexCount;
    };

    struct VertexBuffer {
        vk::Buffer buffer;
        vk::DeviceMemory heap;
    };

    struct Texture {
        vk::Image image;
        vk::ImageView descriptor;
        vk::DeviceMemory heap;
    };

    struct UniformBuffer {
        vk::Buffer buffer;
        vk::DeviceMemory heap;
        size_t size;
    };

    struct Shader {
        Shader(const vk::ShaderModule &&module, vk::ShaderStageFlagBits stage, const char *entry_point) {
            this->module = module;
            this->stage = stage;
            this->entry_point = entry_point;
        }
        vk::ShaderModule module;
        vk::ShaderStageFlagBits stage;
        std::string entry_point;
    };

   public:
    VulkanRHI();
    ~VulkanRHI();

   public:
    void setFramebufferSizeQueryCB(const QueryFrameBufferSizeCBFunc& func) {
        m_fQueryFramebufferSizeCB = func;
    }
    void SetGetGfxConfigCB(const GetGfxConfigCBFunc& func) {
        m_fGetGfxConfigCB = func;
    }
    void CreateSwapChainCB(const CreateSwapChainCBFunc& func) {
        m_fCreateSwapChainCB = func;
    }
    void DestroySwapChainCB(const DestroySwapChainCBFunc& func) {
        m_fDestroySwapChainCB = func;
    }

    void CreateSwapChain();

    void createInstance(std::vector<const char*> extensions);
    void setupDebugMessenger();
    void createSurface(const CreateSurfaceCBFunc& func);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void getDeviceQueues();
    vk::RenderPass createRenderPass();
    vk::DescriptorSetLayout createDescriptorSetLayout();
    void createGraphicsPipeline(
        const VulkanRHI::Shader* shaders, size_t shaderCount,
        const vk::VertexInputBindingDescription& bindingDescription,
        const vk::VertexInputAttributeDescription* attributeDescriptions,
        size_t attributeCount,
        const vk::DescriptorSetLayout& descriptorSetLayout,
        const vk::RenderPass& renderPass, vk::PipelineLayout& pipelineLayout,
        vk::Pipeline& pipeline);
    void createCommandPool();
    void createColorResources();
    void createDepthResources();
    void createFramebuffers(const vk::RenderPass& renderPass);
    VulkanRHI::Texture createTextureImage(Image& image);
    vk::Sampler createTextureSampler();
    void createVertexBuffer(void* buffer, vk::DeviceSize bufferSize,
                            VulkanRHI::VertexBuffer& vertexBuffer);
    void createIndexBuffer(void* buffer, vk::DeviceSize bufferSize,
                           VulkanRHI::IndexBuffer& indexBuffer);
    void createUniformBuffers(
        std::vector<VulkanRHI::UniformBuffer>& uniformBuffers,
        vk::DeviceSize bufferSize);
    void UpdateUniformBuffer(const std::vector<VulkanRHI::UniformBuffer> &uniformBuffers,
                            void* ubo, size_t uboSize);
    vk::ShaderModule createShaderModule(const Buffer& code);
    void createDescriptorPool();
    void createDescriptorSets(
        const vk::DescriptorSetLayout& descriptorSetLayout,
        const std::vector<VulkanRHI::UniformBuffer>& uniformBuffers,
        const std::vector<VulkanRHI::Texture>& textures,
        const vk::Sampler& textureSampler);
    void createCommandBuffers();
    void createSyncObjects();
    void drawFrame(const vk::RenderPass& renderPass,
                   const vk::Pipeline& pipeline,
                   const vk::PipelineLayout& pipelineLayout,
                   const vk::Buffer& vertexBuffer,
                   const vk::Buffer& indexBuffer, uint32_t indicesCount);
    void cleanupSwapChain();
    void recreateSwapChain(const vk::RenderPass& renderPass);

    void destroyAll();

    inline vk::Device GetDevice() { return m_vkDevice; }

   private:
    vk::Instance m_vkInstance;
    vk::DebugUtilsMessengerEXT m_vkDebugMessenger;
    vk::SurfaceKHR m_vkSurface;

    vk::PhysicalDevice m_vkPhysicalDevice;
    vk::Device m_vkDevice;

    vk::SwapchainKHR m_vkSwapChain;
    std::vector<vk::Image> m_vkSwapChainImages;
    vk::Format m_vkSwapChainImageFormat;
    vk::Extent2D m_vkSwapChainExtent;
    std::vector<vk::ImageView> m_vkSwapChainImageViews;

    vk::Queue m_vkGraphicsQueue;
    vk::Queue m_vkComputeQueue;
    vk::Queue m_vkPresentQueue;
    vk::Queue m_vkTransferQueue;

    std::vector<vk::Framebuffer> m_vkSwapChainFramebuffers;
    vk::CommandPool m_vkCommandPool;
    vk::CommandPool m_vkCommandPoolTransfer;

    vk::DescriptorPool m_vkDescriptorPool;
    std::vector<vk::DescriptorSet> m_vkDescriptorSets;

    std::vector<vk::CommandBuffer> m_vkCommandBuffers;
    std::vector<vk::Semaphore> m_vkImageAvailableSemaphores;
    std::vector<vk::Semaphore> m_vkRenderFinishedSemaphores;
    std::vector<vk::Fence> m_vkInFlightFences;

    vk::Image m_vkColorImage;
    vk::DeviceMemory m_vkColorImageMemory;
    vk::ImageView m_vkColorImageView;

    vk::Image m_vkDepthImage;
    vk::DeviceMemory m_vkDepthImageMemory;
    vk::ImageView m_vkDepthImageView;

    vk::SampleCountFlagBits m_vkMsaaSamples = vk::SampleCountFlagBits::e1;

    QueryFrameBufferSizeCBFunc m_fQueryFramebufferSizeCB;
    GetGfxConfigCBFunc m_fGetGfxConfigCB;
    CreateSwapChainCBFunc m_fCreateSwapChainCB;
    DestroySwapChainCBFunc m_fDestroySwapChainCB;

   private:
    void recordCommandBuffer(const vk::RenderPass& renderPass,
                             const vk::Pipeline& pipeline,
                             const vk::PipelineLayout& pipelineLayout,
                             const vk::Buffer& vertexBuffer,
                             const vk::Buffer& indexBuffer,
                             const vk::CommandBuffer& commandBuffer,
                             uint32_t indicesCount, uint32_t imageIndex);
    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags properties, vk::Buffer& buffer,
                      vk::DeviceMemory& bufferMemory);
    void createImage(Image& image, vk::ImageTiling tiling,
                     vk::ImageUsageFlags usage, vk::SharingMode sharing_mode,
                     vk::MemoryPropertyFlags properties, vk::Image& vk_image,
                     vk::DeviceMemory& vk_image_memory,
                     uint32_t queueFamilyIndexCount,
                     uint32_t* queueFamilyIndices);
    void createImage(uint32_t width, uint32_t height,
                     vk::SampleCountFlagBits numSamples, vk::Format format,
                     vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                     vk::SharingMode sharing_mode,
                     vk::MemoryPropertyFlags properties, vk::Image& vk_image,
                     vk::DeviceMemory& vk_image_memory,
                     uint32_t queueFamilyIndexCount = 1,
                     uint32_t* queueFamilyIndices = nullptr);
    vk::ImageView createImageView(vk::Image image, vk::Format format,
                                  vk::ImageAspectFlags aspectFlags);
    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                    vk::DeviceSize size);
    void copyBufferToImage(vk::Buffer srcBuffer, vk::Image image,
                           uint32_t width, uint32_t height);
    uint32_t findMemoryType(uint32_t typeFilter,
                            vk::MemoryPropertyFlags properties);
    void getTextureFormat(const Image& img, vk::Format& internal_format);
    vk::CommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(vk::CommandBuffer& commandBuffer);
    void transitionImageLayout(
        vk::Image image, vk::Format format, vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);
    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
                                   vk::ImageTiling tiling,
                                   vk::FormatFeatureFlags features);
    vk::Format findDepthFormat();
    vk::SampleCountFlagBits getMaxUsableSampleCount();

   private:
    uint32_t m_nCurrentFrame = 0;
    VkBool32 m_bUseSampleShading = VK_TRUE;
};
}  // namespace My