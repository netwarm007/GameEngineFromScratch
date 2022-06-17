#pragma once
#include <vector>
#include <vulkan/vulkan.hpp>
#include "Buffer.hpp"
#include "Image.hpp"
#include "a2v.hpp"

namespace My {

class VulkanRHI {
   public:
    VulkanRHI();
    ~VulkanRHI();

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

    vk::RenderPass m_vkRenderPass;
    vk::Pipeline m_vkGraphicPipeline;
    vk::DescriptorSetLayout m_vkDescriptorSetLayout;
    vk::PipelineLayout m_vkPipelineLayout;

    std::vector<vk::Framebuffer> m_vkSwapChainFramebuffers;
    vk::CommandPool m_vkCommandPool;
    vk::CommandPool m_vkCommandPoolTransfer;

    vk::DescriptorPool m_vkDescriptorPool;
    std::vector<vk::DescriptorSet> m_vkDescriptorSets;

    std::vector<vk::CommandBuffer> m_vkCommandBuffers;
    std::vector<vk::Semaphore> m_vkImageAvailableSemaphores;
    std::vector<vk::Semaphore> m_vkRenderFinishedSemaphores;
    std::vector<vk::Fence> m_vkInFlightFences;

    std::vector<Vertex> m_Vertices;
    std::vector<uint32_t> m_Indices;
    vk::Buffer m_vkVertexBuffer;
    vk::DeviceMemory m_vkVertexBufferMemory;
    vk::Buffer m_vkIndexBuffer;
    vk::DeviceMemory m_vkIndexBufferMemory;

    vk::Image m_vkTextureImage;
    vk::DeviceMemory m_vkTextureImageMemory;
    vk::ImageView m_vkTextureImageView;
    vk::Sampler m_vkTextureSampler;

    vk::Image m_vkColorImage;
    vk::DeviceMemory m_vkColorImageMemory;
    vk::ImageView m_vkColorImageView;

    vk::Image m_vkDepthImage;
    vk::DeviceMemory m_vkDepthImageMemory;
    vk::ImageView m_vkDepthImageView;

    std::vector<vk::Buffer> m_vkUniformBuffers;
    std::vector<vk::DeviceMemory> m_vkUniformBuffersMemory;

    vk::SampleCountFlagBits m_vkMsaaSamples = vk::SampleCountFlagBits::e1;

    vk::ShaderModule m_vkVertShaderModule;
    vk::ShaderModule m_vkFragShaderModule;

    using QueryFrameBufferSizeFunc = std::function<void(int&, int&)>;
    QueryFrameBufferSizeFunc m_fQueryFramebufferSize;

   public:
    void setFramebufferSizeQueryCB(const QueryFrameBufferSizeFunc& func) {
        m_fQueryFramebufferSize = func;
    }
    void createInstance(std::vector<const char*> extensions);
    void setupDebugMessenger();
    void createSurface(
        const std::function<void(const vk::Instance&, vk::SurfaceKHR&)>&);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void getDeviceQueues();
    void createRenderPass();
    void createDescriptorSetLayout();
    void setShaders(const Buffer& vertShaderCode, const Buffer& fragShaderCode);
    void createGraphicsPipeline();
    void createCommandPool();
    void createColorResources();
    void createDepthResources();
    void createFramebuffers();
    void createTextureImage(Image& image);
    void createTextureImageView(Image& image);
    void createTextureSampler();
    void setModel(const std::vector<Vertex>& vertices,
                  const std::vector<uint32_t>& indices);
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void recordCommandBuffer(vk::CommandBuffer& commandBuffer,
                             uint32_t imageIndex);
    void createSyncObjects();
    void drawFrame();
    void cleanupSwapChain();
    void recreateSwapChain();

    void destroyAll();

   private:
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
    void updateUniformBufer(uint32_t currentImage);
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