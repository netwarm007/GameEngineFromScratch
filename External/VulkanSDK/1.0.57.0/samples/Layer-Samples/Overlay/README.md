Overlay layer example

This is a little different from the other samples in that it is implemented as a layer, rather than a vulkan client app. This carries some extra requirements:

- Build directory should be added to VK_LAYER_PATH.
- The overlay layer name (currently "VK_LAYER_LUNARG_overlay") should be added to VK_INSTANCE_LAYERS and VK_DEVICE_LAYERS.
