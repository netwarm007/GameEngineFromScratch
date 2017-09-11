# Create a Depth Buffer

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Code file for this section is `06-init_depth_buffer.cpp`

Depth buffers are optional, but you will need a depth buffer
to render a 3D cube in the final sample.
And you need only one for rendering each frame,
even if the swapchain has more than one image.
This is because you can reuse the same depth buffer while using
each image in the swapchain.

Unlike `vkCreateSwapchainKHR()`, where each of the images
in the swapchain were created for you, you need to
create and allocate your own image to represent the depth buffer.

The steps are:

1. Create the depth buffer image object
1. Allocate the depth buffer device memory
1. Bind the memory to the image object
1. Create the depth buffer image view

You end up with something that looks like this:

![CreateDepth](../images/DepthBufferBindView.png)

## Create the Depth Buffer Image Object

To create the image object for the depth buffer, we fill in the
familiar create info structure:

    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.pNext = NULL;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_D16_UNORM;
    image_info.extent.width = info.width;
    image_info.extent.height = info.height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = NUM_SAMPLES;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    image_info.queueFamilyIndexCount = 0;
    image_info.pQueueFamilyIndices = NULL;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.flags = 0;

    vkCreateImage(info.device, &image_info, NULL, &info.depth.image);

This just creates an object for the image.
There is no memory allocated yet for the depth buffer image memory itself,
but note that you have filled in the information to indicate that you
want a depth buffer that matches the window size.

## Allocate the Memory for the Depth Buffer

At this point, even though you know the width, height, and the
size of a buffer element, you still cannot determine exactly
how much memory to allocate.
This is because there may be alignment constraints placed by
the GPU hardware, for example.
And these constraints vary from device to device.

Vulkan provides a function that you use to find out everything
you need to allocate the memory for an image:

    vkGetImageMemoryRequirements(info.device, info.depth.image, &mem_reqs);

You then use this memory requirement info to fill in a request
structure for a memory allocation.
One of the key pieces of this requirements information is the memory size,
which you use to fill in the allocation request structure, as shown below.

    VkMemoryAllocateInfo mem_alloc = {};
    mem_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mem_alloc.pNext = NULL;
    mem_alloc.allocationSize = mem_reqs.size;
    mem_alloc.memoryTypeIndex = memory_type_from_properties();

While it is tempting to guess at the allocationSize by multiplying
width by height by sizeof(depth texel), there is the possibility
of alignment constraints requiring additional padding.
So it is better to not try to guess the amount needed and request
the required size instead.

The `memory_type_from_properties()` function is a samples utility
function that determines the appropriate memory type index to use.
Since you are allocating device memory here, memory that is directly
accessible from the GPU, there are additional considerations
such as alignment and cache behavior.
The *Device Memory* section in the Vulkan spec is a good reference
for the many details.

Finally, you can allocate the memory:

    vkAllocateMemory(info.device, &mem_alloc, NULL, &info.depth.mem);

## Bind the Memory to the Depth Buffer

Now you have an image object for the depth buffer and some device
memory allocated for the depth buffer.

Next, you associate the memory with the object by binding them:

    vkBindImageMemory(info.device, info.depth.image, info.depth.mem, 0);

## Create the Image View

As with the swapchain images, you need to create an image view
to indicate some of the specifics about how you will be using the
depth buffer.
For example, the image view you create below indicates that the image
will be used as a depth buffer and has a 16-bit format.

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.pNext = NULL;
    view_info.image = info.depth.image;
    view_info.format = VK_FORMAT_D16_UNORM;
    view_info.components.r = VK_COMPONENT_SWIZZLE_R;
    view_info.components.g = VK_COMPONENT_SWIZZLE_G;
    view_info.components.b = VK_COMPONENT_SWIZZLE_B;
    view_info.components.a = VK_COMPONENT_SWIZZLE_A;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.flags = 0;

    res = vkCreateImageView(info.device, &view_info, NULL, &info.depth.view);

<table border="1" width="100%">
    <tr>
        <td align="center" width="33%">Previous: <a href="05-init_swapchain.html" title="Prev">Swapchain</a></td>
        <td align="center" width="33%">Back to: <a href="index.html" title="Index">Index</a></td>
        <td align="center" width="33%">Next: <a href="07-init_uniform_buffer.html" title="Next">Uniform Buffer</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>
