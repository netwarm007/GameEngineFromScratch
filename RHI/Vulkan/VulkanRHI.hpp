#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "Buffer.hpp"

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

        VkRenderPass             m_vkRenderPass;
        VkPipeline               m_vkGraphicPipeline;
        VkPipelineLayout         m_vkPipelineLayout;

        std::vector<VkFramebuffer> m_vkSwapChainFramebuffers;
        VkCommandPool            m_vkCommandPool;
        
        std::vector<VkCommandBuffer> m_vkCommandBuffers;
        std::vector<VkSemaphore>    m_vkImageAvailableSemaphores;
        std::vector<VkSemaphore>    m_vkRenderFinishedSemaphores;
        std::vector<VkFence>        m_vkInFlightFences;

    public:
        void createInstance(std::vector<const char*> extensions); 
        void setupDebugMessenger();
        void createSurface(const std::function<void(const VkInstance&, VkSurfaceKHR&)>&);
        void pickPhysicalDevice();
        void createLogicalDevice();
        void createSwapChain (const int width, const int height);
        void createImageViews();
        void getDeviceQueues();
        void createRenderPass();
        void createGraphicsPipeline(const Buffer& vertShaderCode, const Buffer& fragShaderCode);
        void createFramebuffers();
        void createCommandPool();
        void createCommandBuffers();
        void recordCommandBuffer(VkCommandBuffer& commandBuffer, uint32_t imageIndex);
        void createSyncObjects();
        void drawFrame();

    private:
        uint32_t m_nCurrentFrame = 0;
    };
}