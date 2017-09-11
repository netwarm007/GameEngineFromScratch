/*
 * Copyright (C) 2016 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SHELL_H
#define SHELL_H

#include <queue>
#include <vector>
#include <stdexcept>
#include <vulkan/vulkan.h>

#include "Game.h"

class Game;

class Shell {
   public:
    Shell(const Shell &sh) = delete;
    Shell &operator=(const Shell &sh) = delete;
    virtual ~Shell() {}

    struct BackBuffer {
        uint32_t image_index;

        VkSemaphore acquire_semaphore;
        VkSemaphore render_semaphore;

        // signaled when this struct is ready for reuse
        VkFence present_fence;
    };

    struct Context {
        VkInstance instance;
        VkDebugReportCallbackEXT debug_report;

        VkPhysicalDevice physical_dev;
        uint32_t game_queue_family;
        uint32_t present_queue_family;

        VkDevice dev;
        VkQueue game_queue;
        VkQueue present_queue;

        std::queue<BackBuffer> back_buffers;

        VkSurfaceKHR surface;
        VkSurfaceFormatKHR format;

        VkSwapchainKHR swapchain;
        VkExtent2D extent;

        BackBuffer acquired_back_buffer;
    };
    const Context &context() const { return ctx_; }

    enum LogPriority {
        LOG_DEBUG,
        LOG_INFO,
        LOG_WARN,
        LOG_ERR,
    };
    virtual void log(LogPriority priority, const char *msg) const;

    virtual void run() = 0;
    virtual void quit() = 0;

   protected:
    Shell(Game &game);

    void init_vk();
    void cleanup_vk();

    void create_context();
    void destroy_context();

    void resize_swapchain(uint32_t width_hint, uint32_t height_hint);

    void add_game_time(float time);

    void acquire_back_buffer();
    void present_back_buffer();

    Game &game_;
    const Game::Settings &settings_;

    std::vector<const char *> instance_layers_;
    std::vector<const char *> instance_extensions_;

    std::vector<const char *> device_extensions_;

   private:
    bool debug_report_callback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT obj_type, uint64_t object, size_t location,
                               int32_t msg_code, const char *layer_prefix, const char *msg);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report_callback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT obj_type,
                                                                uint64_t object, size_t location, int32_t msg_code,
                                                                const char *layer_prefix, const char *msg, void *user_data) {
        Shell *shell = reinterpret_cast<Shell *>(user_data);
        return shell->debug_report_callback(flags, obj_type, object, location, msg_code, layer_prefix, msg);
    }

    void assert_all_instance_layers() const;
    void assert_all_instance_extensions() const;

    bool has_all_device_layers(VkPhysicalDevice phy) const;
    bool has_all_device_extensions(VkPhysicalDevice phy) const;

    // called by init_vk
    virtual PFN_vkGetInstanceProcAddr load_vk() = 0;
    virtual bool can_present(VkPhysicalDevice phy, uint32_t queue_family) = 0;
    void init_instance();
    void init_debug_report();
    void init_physical_dev();

    // called by create_context
    void create_dev();
    void create_back_buffers();
    void destroy_back_buffers();
    virtual VkSurfaceKHR create_surface(VkInstance instance) = 0;
    void create_swapchain();
    void destroy_swapchain();

    void fake_present();

    Context ctx_;

    const float game_tick_;
    float game_time_;
};

#endif  // SHELL_H
