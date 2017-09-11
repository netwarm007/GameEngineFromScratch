# Layer Description and Status

## Overview

Layer libraries can be written to intercept or hook VK entry points for various
debug and validation purposes.  One or more VK entry points can be defined in your Layer
library.  Undefined entrypoints in the Layer library will be passed to the next Layer which
may be the driver.  Multiple layer libraries can be chained (actually a hierarchy) together.
vkEnumerateInstanceLayerProperties and vkEnumerateDeviceLayerProperties can be called to list the
available layers and their properties. Layers can intercept Vulkan instance level entry points
in which case they are called an Instance Layer.  Layers can intercept device entry  points
in which case they are called a Device Layer. Instance level entry points are those with VkInstance
or VkPhysicalDevice as first parameter.  Device level entry points are those with VkDevice, VkCommandBuffer,
or VkQueue as the first parameter. Layers that want to intercept both instance and device
level entrypoints are called Global Layers. vkXXXXGetProcAddr is used internally by the Layers and
Loader to initialize dispatch tables. Device Layers are activated at vkCreateDevice time. Instance
Layers are activated at vkCreateInstance time.  Layers can also be activated via environment variable:
(VK_INSTANCE_LAYERS).

### Layer library example code

Note that some layers are code-generated and will therefore exist in the directory (build_dir)/layers

-include/vkLayer.h  - header file for layer code.

### Print API Calls and Parameter Values
(build dir)/layers/api_dump.cpp (name=VK_LAYER_LUNARG_api_dump) - print out API calls along with parameter values

### Capture Screenshots
layersvt/screenshot.cpp (name='VK_LAYER_LUNARG_screenshot') - utility layer used to capture and save screenshots of running applications. 
To specify frames to be captured, the environment variable 'VK_SCREENSHOT_FRAMES' can be set to a comma-separated list of frame numbers (ex: 4,8,15,16,23,42).

### View Frames Per Second
layersvt/monitor.cpp - utility layer that will display an applications FPS in the title bar of a windowed application.

### Device Simulation
layersvt/device_simulation.cpp (name='VK_LAYER_LUNARG_device_simulation') - A utility layer to simulate a device with different capabilities than the actual hardware in the system.  See device_simulation.md for details.

## Using Layers

1. Build VK loader using normal steps (cmake and make)
2. Place libVkLayer_<name>.so or VkLayer_<name>.dll in the same directory as your Vulkan test or application, or use the VK\_LAYER\_PATH environment variable to specify where the layer libraries reside.
3. Create a vk_layer_settings.txt file in the same directory to specify how your layers should behave.

    Model it after the following example:  [*vk_layer_settings.txt*](vk_layer_settings.txt)

4. Specify which Layers to activate by using
vkCreateDevice and/or vkCreateInstance or environment variables.

    export VK\_INSTANCE\_LAYERS=VK_LAYER_LUNARG_api_dump
    cd build/tests; ./vkinfo

## Status


### Current known issues
