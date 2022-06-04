#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "Buffer.hpp"
#include "Image.hpp"

namespace My {
    class VulkanRHI {
    public:
        VulkanRHI();
        ~VulkanRHI();

    private:
        VkInstance               m_vkInstance;
        VkDebugUtilsMessengerEXT m_vkDebugMessenger;
        VkSurfaceKHR             m_vkSurface;

        VkPhysicalDevice         m_vkPhysicalDevice;
        VkDevice                 m_vkDevice;

        VkSwapchainKHR           m_vkSwapChain;
        std::vector<VkImage>     m_vkSwapChainImages;
        VkFormat                 m_vkSwapChainImageFormat;
        VkExtent2D               m_vkSwapChainExtent;
        std::vector<VkImageView> m_vkSwapChainImageViews;

        VkQueue                  m_vkGraphicsQueue;
        VkQueue                  m_vkComputeQueue;
        VkQueue                  m_vkPresentQueue;
        VkQueue                  m_vkTransferQueue;

        VkRenderPass             m_vkRenderPass;
        VkPipeline               m_vkGraphicPipeline;
        VkDescriptorSetLayout    m_vkDescriptorSetLayout;
        VkPipelineLayout         m_vkPipelineLayout;

        std::vector<VkFramebuffer> m_vkSwapChainFramebuffers;
        VkCommandPool            m_vkCommandPool;
        VkCommandPool            m_vkCommandPoolTransfer;

        VkDescriptorPool         m_vkDescriptorPool;
        std::vector<VkDescriptorSet> m_vkDescriptorSets;
        
        std::vector<VkCommandBuffer> m_vkCommandBuffers;
        std::vector<VkSemaphore>    m_vkImageAvailableSemaphores;
        std::vector<VkSemaphore>    m_vkRenderFinishedSemaphores;
        std::vector<VkFence>        m_vkInFlightFences;

        VkBuffer                 m_vkVertexBuffer;
        VkDeviceMemory           m_vkVertexBufferMemory;
        VkBuffer                 m_vkIndexBuffer;
        VkDeviceMemory           m_vkIndexBufferMemory;

        VkImage                  m_vkTextureImage;
        VkDeviceMemory           m_vkTextureImageMemory;
        VkImageView              m_vkTextureImageView;

        VkSampler                m_vkTextureSampler;

        std::vector<VkBuffer>    m_vkUniformBuffers;
        std::vector<VkDeviceMemory> m_vkUniformBuffersMemory;

        VkShaderModule m_vkVertShaderModule;
        VkShaderModule m_vkFragShaderModule;

        using QueryFrameBufferSizeFunc = std::function<void(int&, int&)>;
        QueryFrameBufferSizeFunc m_fQueryFramebufferSize;

    public:
        void setFramebufferSizeQueryCB(const QueryFrameBufferSizeFunc& func) { m_fQueryFramebufferSize = func; }
        void createInstance(std::vector<const char*> extensions); 
        void setupDebugMessenger();
        void createSurface(const std::function<void(const VkInstance&, VkSurfaceKHR&)>&);
        void pickPhysicalDevice();
        void createLogicalDevice();
        void createSwapChain ();
        void createImageViews();
        void getDeviceQueues();
        void createRenderPass();
        void createDescriptorSetLayout();
        void setShaders(const Buffer& vertShaderCode, const Buffer& fragShaderCode);
        void createGraphicsPipeline();
        void createFramebuffers();
        void createCommandPool();
        void createTextureImage(Image& image);
        void createTextureImageView(Image& image);
        void createTextureSampler();
        void createVertexBuffer();
        void createIndexBuffer();
        void createUniformBuffers();
        void createDescriptorPool();
        void createDescriptorSets();
        void createCommandBuffers();
        void recordCommandBuffer(VkCommandBuffer& commandBuffer, uint32_t imageIndex);
        void createSyncObjects();
        void drawFrame();
        void cleanupSwapChain();
        void recreateSwapChain();

    private:
        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
        void createImage(Image& image, VkImageTiling tiling, VkImageUsageFlags usage, VkSharingMode sharing_mode, VkMemoryPropertyFlags properties, VkImage& vk_image, VkDeviceMemory& vk_image_memory, uint32_t queueFamilyIndexCount, uint32_t* queueFamilyIndices);
        VkImageView createImageView(VkImage image, VkFormat format);
        void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
        void copyBufferToImage(VkBuffer srcBuffer, VkImage image, uint32_t width, uint32_t height);
        uint32_t findMemoryType (uint32_t typeFilter, VkMemoryPropertyFlags properties);
        void updateUniformBufer(uint32_t currentImage);
        void getTextureFormat(const Image& img, VkFormat& internal_format);
        VkCommandBuffer beginSingleTimeCommands();
        void endSingleTimeCommands(VkCommandBuffer commandBuffer);
        void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, 
                uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

    private:
        uint32_t m_nCurrentFrame = 0;
    };
}