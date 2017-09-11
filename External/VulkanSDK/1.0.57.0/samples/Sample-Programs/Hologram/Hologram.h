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

#ifndef HOLOGRAM_H
#define HOLOGRAM_H

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "Simulation.h"
#include "Game.h"

class Meshes;

class Hologram : public Game {
   public:
    Hologram(const std::vector<std::string> &args);
    ~Hologram();

    void attach_shell(Shell &sh);
    void detach_shell();

    void attach_swapchain();
    void detach_swapchain();

    void on_key(Key key);
    void on_tick();

    void on_frame(float frame_pred);

   private:
    class Worker {
       public:
        Worker(Hologram &hologram, int index, int object_begin, int object_end);

        void start();
        void stop();
        void update_simulation();
        void draw_objects(VkFramebuffer fb);
        void wait_idle();

        Hologram &hologram_;

        const int index_;
        const int object_begin_;
        const int object_end_;

        const float tick_interval_;

        VkFramebuffer fb_;

       private:
        enum State {
            INIT,
            IDLE,
            STEP,
            DRAW,
        };

        void update_loop();

        static void thread_loop(Worker *worker) { worker->update_loop(); }

        std::thread thread_;
        std::mutex mutex_;
        std::condition_variable state_cv_;
        State state_;
    };

    struct Camera {
        glm::vec3 eye_pos;
        glm::mat4 view_projection;

        Camera(float eye) : eye_pos(eye) {}
    };

    struct FrameData {
        // signaled when this struct is ready for reuse
        VkFence fence;

        VkCommandBuffer primary_cmd;
        std::vector<VkCommandBuffer> worker_cmds;

        VkBuffer buf;
        uint8_t *base;
        VkDescriptorSet desc_set;
    };

    // called by the constructor
    void init_workers();

    bool multithread_;
    bool use_push_constants_;

    // called mostly by on_key
    void update_camera();

    bool sim_paused_;
    bool sim_fade_;
    Simulation sim_;
    Camera camera_;

    std::vector<std::unique_ptr<Worker>> workers_;

    // called by attach_shell
    void create_render_pass();
    void create_shader_modules();
    void create_descriptor_set_layout();
    void create_pipeline_layout();
    void create_pipeline();

    void create_frame_data(int count);
    void destroy_frame_data();
    void create_fences();
    void create_command_buffers();
    void create_buffers();
    void create_buffer_memory();
    void create_descriptor_sets();

    VkPhysicalDevice physical_dev_;
    VkDevice dev_;
    VkQueue queue_;
    uint32_t queue_family_;
    VkFormat format_;
    VkDeviceSize aligned_object_data_size;

    VkPhysicalDeviceProperties physical_dev_props_;
    std::vector<VkMemoryPropertyFlags> mem_flags_;

    const Meshes *meshes_;

    VkRenderPass render_pass_;
    VkShaderModule vs_;
    VkShaderModule fs_;
    VkDescriptorSetLayout desc_set_layout_;
    VkPipelineLayout pipeline_layout_;
    VkPipeline pipeline_;

    VkCommandPool primary_cmd_pool_;
    std::vector<VkCommandPool> worker_cmd_pools_;
    VkDescriptorPool desc_pool_;
    VkDeviceMemory frame_data_mem_;
    std::vector<FrameData> frame_data_;
    int frame_data_index_;

    VkClearValue render_pass_clear_value_;
    VkRenderPassBeginInfo render_pass_begin_info_;

    VkCommandBufferBeginInfo primary_cmd_begin_info_;
    VkPipelineStageFlags primary_cmd_submit_wait_stages_;
    VkSubmitInfo primary_cmd_submit_info_;

    // called by attach_swapchain
    void prepare_viewport(const VkExtent2D &extent);
    void prepare_framebuffers(VkSwapchainKHR swapchain);

    VkExtent2D extent_;
    VkViewport viewport_;
    VkRect2D scissor_;

    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;
    std::vector<VkFramebuffer> framebuffers_;

    // called by workers
    void update_simulation(const Worker &worker);
    void draw_object(const Simulation::Object &obj, FrameData &data, VkCommandBuffer cmd) const;
    void draw_objects(Worker &worker);
};

#endif  // HOLOGRAM_H
