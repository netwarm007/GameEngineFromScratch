/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
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

/*
VULKAN_SAMPLE_SHORT_DESCRIPTION
Use basic events
*/

#include <util_init.hpp>
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include "cube_data.h"

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    struct sample_info info = {};
    char sample_title[] = "Events";

    init_global_layer_properties(info);
    init_instance_extension_names(info);
    init_device_extension_names(info);
    init_instance(info, sample_title);
    init_enumerate_device(info);
    init_device(info);

    init_command_pool(info);
    init_command_buffer(info);
    execute_begin_command_buffer(info);
    init_device_queue(info);

    /* VULKAN_KEY_START */

    // Start with a trivial command buffer and make sure fence wait doesn't time out
    info.viewport.height = 10.0;
    info.viewport.width = 10.0;
    info.viewport.minDepth = (float)0.0f;
    info.viewport.maxDepth = (float)1.0f;
    info.viewport.x = 0;
    info.viewport.y = 0;
    vkCmdSetViewport(info.cmd, 0, NUM_VIEWPORTS, &info.viewport);
    execute_end_command_buffer(info);

    VkFence fence;
    VkFenceCreateInfo fenceInfo;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = NULL;
    fenceInfo.flags = 0;
    vkCreateFence(info.device, &fenceInfo, NULL, &fence);

    VkPipelineStageFlags pipe_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkCommandBuffer cmd_bufs[] = {info.cmd};
    VkSubmitInfo submit_info[1] = {};
    submit_info[0].pNext = NULL;
    submit_info[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info[0].waitSemaphoreCount = 0;
    submit_info[0].pWaitSemaphores = NULL;
    submit_info[0].pWaitDstStageMask = &pipe_stage_flags;
    submit_info[0].commandBufferCount = 1;
    submit_info[0].pCommandBuffers = cmd_bufs;
    submit_info[0].signalSemaphoreCount = 0;
    submit_info[0].pSignalSemaphores = NULL;

    res = vkQueueSubmit(info.graphics_queue, 1, submit_info, fence);
    assert(res == VK_SUCCESS);

    // Make sure timeout is long enough for a simple command buffer without
    // waiting for an event
    int timeouts = -1;
    do {
        res = vkWaitForFences(info.device, 1, &fence, VK_TRUE, FENCE_TIMEOUT);
        timeouts++;
    } while (res == VK_TIMEOUT);
    assert(res == VK_SUCCESS);
    if (timeouts != 0) {
        std::cout << "Unsuitable timeout value, exiting\n";
        exit(-1);
    }

    vkResetCommandBuffer(info.cmd, 0);

    // Now create an event and wait for it on the GPU
    VkEvent event;
    VkEventCreateInfo eventInfo = {};
    eventInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    eventInfo.pNext = NULL;
    eventInfo.flags = 0;
    vkCreateEvent(info.device, &eventInfo, NULL, &event);

    execute_begin_command_buffer(info);
    vkCmdWaitEvents(info.cmd, 1, &event, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, nullptr, 0, nullptr,
                    0, nullptr);
    execute_end_command_buffer(info);
    vkResetFences(info.device, 1, &fence);

    // Note that stepping through this code in the debugger is a bad idea because the
    // GPU can TDR waiting for the event.  Execute the code from vkQueueSubmit through
    // vkSetEvent without breakpoints
    pipe_stage_flags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    res = vkQueueSubmit(info.graphics_queue, 1, submit_info, fence);
    assert(res == VK_SUCCESS);

    // We should timeout waiting for the fence because the GPU should be waiting
    // on the event
    res = vkWaitForFences(info.device, 1, &fence, VK_TRUE, FENCE_TIMEOUT);
    if (res != VK_TIMEOUT) {
        std::cout << "Didn't get expected timeout in vkWaitForFences, exiting\n";
        exit(-1);
    }

    // Set the event from the CPU and wait for the fence.  This should succeed
    // since we set the event
    vkSetEvent(info.device, event);
    do {
        res = vkWaitForFences(info.device, 1, &fence, VK_TRUE, FENCE_TIMEOUT);
    } while (res == VK_TIMEOUT);
    assert(res == VK_SUCCESS);

    vkResetCommandBuffer(info.cmd, 0);
    vkResetFences(info.device, 1, &fence);
    vkResetEvent(info.device, event);

    // Now set the event from the GPU and wait on the CPU
    execute_begin_command_buffer(info);
    vkCmdSetEvent(info.cmd, event, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    execute_end_command_buffer(info);

    // Look for the event on the CPU. It should be RESET since we haven't sent
    // the command buffer yet.
    res = vkGetEventStatus(info.device, event);
    assert(res == VK_EVENT_RESET);

    // Send the command buffer and loop waiting for the event
    res = vkQueueSubmit(info.graphics_queue, 1, submit_info, fence);
    assert(res == VK_SUCCESS);

    int polls = 0;
    do {
        res = vkGetEventStatus(info.device, event);
        polls++;
    } while (res != VK_EVENT_SET);
    printf("%d polls to find the event set\n", polls);

    do {
        res = vkWaitForFences(info.device, 1, &fence, VK_TRUE, FENCE_TIMEOUT);
    } while (res == VK_TIMEOUT);
    assert(res == VK_SUCCESS);

    vkDestroyEvent(info.device, event, NULL);
    vkDestroyFence(info.device, fence, NULL);
    destroy_command_buffer(info);
    destroy_command_pool(info);
    destroy_device(info);
    destroy_instance(info);
    return 0;
}
