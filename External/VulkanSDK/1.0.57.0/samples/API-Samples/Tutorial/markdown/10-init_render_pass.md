# Create a Render Pass

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Code file for this section is `10-init_render_pass.cpp`

A render pass describes the scope of a rendering operation by
specifying the collection of attachments, subpasses, and dependencies
used during the rendering operation.
A render pass consists of at least one subpass.
The communication of this information to the driver allows the
driver to know what to expect when rendering begins and to set
up the hardware optimally for the rendering operation.

You begin by using `vkCreateRenderPass()` to define the render pass,
and then later insert a render pass instance into a command buffer
using `vkCmdBeginRenderPass()` and `vkCmdEndRenderPass()`.

In this section, you will only be creating and defining the render pass
and not using it in a command buffer until later.

Examples of render pass attachments in the context of this sample include
the color attachment, which is the image from the swapchain,
and the depth/stencil attachment, which is the depth buffer
you allocated in a previous sample.

Image attachments must be prepared for use when they are
used as attachments in a render pass instance
that is executed in a command buffer.
This preparation involves transitioning image layouts from their
initial undefined states to states that are optimal for their
use in the render pass.
Since these layout transitions are rather involved, you
will learn about them here, before continuing on with creating the
render pass.

## Image Layout Transition

### The Need for Alternate Memory Access Patterns

The layout of an image refers to how image texels are mapped from
a grid coordinate representation to an offset in the image memory.
Typically, image data is mapped in this way linearly, which for a 2D
image might mean that a row of texels is stored in contiguous memory,
with the next row stored contiguously after that row, and so on.

Put another way:

    offset = rowCoord * pitch + colCoord

where pitch is the size of a row.  The pitch is often the same as the
width of the image, but may include some additional bytes of padding
to ensure that each row of the image begins at a memory address
that meets the alignment requirements of the GPU.

A linear layout is fine for successive texel read or writes along a
single row, by changing the `colCoord`.
But most graphics operations involve accessing texels across
multiple adjacent rows as well, by changing the `rowCoord`.
If the image width is fairly wide, then these adjacent row accesses
introduce rather large jumps across the linear memory address space.
This can cause performance problems such as slower memory address
translation due to TLB misses and cache misses in multilevel
cache memory systems.

To combat these inefficiencies, many GPU hardware implementations
support an "optimal" or tiled memory access scheme.
In an optimal layout, a rectangle of texels appearing in the
middle of an image is stored in memory such that all the
texels are in one continuous stretch of memory.
For example, the texels that make up a rectangle with the
upper-left corner at [16,32] and lower-right corner at [31,47]
might see this 16 x 16 block of texel stored contiguously
starting at one address.  There are no long gaps between rows.

If the GPU wanted to fill this block, with a solid color for example,
it is able to write to this 256 texel block with a low amount of
memory system overhead.

Here's an example of a simple 2x2 tiling scheme.  Note that the blue texels
can be far apart from each other in the linear scheme, while they are adjacent in
the tiled scheme.

![Image Memory Layout](../images/ImageMemoryLayout.png)

Most implementations use a more complex tiling pattern and tile sizes that are
larger than 2x2.

#### Vulkan Control Over the Layout

GPU hardware often favors optimal layouts for more efficient rendering,
as explained just above.
Optimal layouts are often "opaque", which means that the details
of the optimal layout format are not published or made known
to other components that need to read or write image data.

For example, you may want the GPU to render to an image
using an optimal layout.
But if you wish to copy the resulting image to a buffer that you
can read and understand using the CPU, you change the layout
from optimal to general before trying to read it.

Conversions from one layout to another are called layout transitions
in Vulkan.
You have control over these transitions and can invoke them
in one of three ways:

1. Memory Barrier Command (via `vkCmdPipelineBarrier`)
1. Render Pass final layout specification
1. Render Pass subpass layout specification

The memory barrier command is an explicit layout transition command
that you place in a command buffer.
You would use this command to synchronize memory accesses in more complex
situations, for example.
Since you will be using the other two methods to perform layout
transitions here, you won't be using this barrier command in this tutorial.

The more common situation is the need to perform layout transitions on images used for
rendering before rendering begins and after rendering is complete.
The first transition prepares the image for rendering by the GPU and
the last one prepares the image for presentation to the display.
In these cases, you specify the layout
transitions as part of the render pass definition.
You will see how to do this later in this section.

A layout transition may or may not trigger an actual GPU layout
conversion operation.
For example, if the old layout is undefined and the new layout
is optimal, the GPU would have to do no work other than perhaps
program the GPU hardware to access memory in the optimal pattern.
This is because the contents of the image is undefined and does
not need to be preserved by converting it.
On the other hand, if the old layout was general (non-optimal) and there is
an indication that the data in the image needs to be preserved,
the transition to optimal probably involves some work by the GPU to
shuffle the texels around.

Even if you know or think that a layout transition won't actually do any work,
it is always best practice to do them anyway, since it gives the driver
more information and helps ensure that your application works on more devices.

#### Image Layout Transitions in the Samples

The sample code uses subpass definitions and render pass definitions to
specify the required image layout transitions, instead of using memory barrier commands.

![Render Pass Layout](../images/RenderPassLayouts.png)

The initial render pass layout is undefined and means *don't care* because
when the render pass starts, you don't care what is already in the image
because you are going to draw over it anyway.
Here, you are just telling the driver what layout is in effect
for the image at the start of the render pass.
The driver won't do any transitions until the subpass and/or until the end
of the render pass.

The subpass layout is set to optimal for the color buffer, which indicates
that the driver should transition the layout to optimal for the duration
of the rendering operations that occur during the subpass.
The sample code sets a similar setting for the depth buffer.

The final render pass layout of present tells the driver to transition the
layout to the best layout for presenting to the display.

### Create the Render Pass

Now that you know how to get the image layouts into the correct state,
you can proceed with defining the rest of the render pass.

#### Attachments

There are two attachments, one for color and one for depth:

    VkAttachmentDescription attachments[2];
    attachments[0].format = info.format;
    attachments[0].samples = NUM_SAMPLES;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachments[0].flags = 0;

    attachments[1].format = info.depth.format;
    attachments[1].samples = NUM_SAMPLES;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].flags = 0;

Setting the `loadOp` member to CLEAR in both attachments indicates that you want the buffer to be
cleared at the start of the render pass instance.

Setting the `storeOp` member to STORE for the color attachment means that you
want to leave the rendering result in this buffer, so it can be presented to
the display.

Setting the `storeOp` member to DONT_CARE for the depth attachment means that
you don't need the contents of the buffer when the render pass instance is complete.
Telling the driver that you don't care about the contents of a buffer after it is used
can be useful because it allows the driver to
discard or page out that memory if it needed to without saving the contents.

For image layouts, you specify both the color and depth buffers to start with undefined
layouts, as discussed earlier.

The subpass, which occurs between the initial and final layouts,
takes the color attachment to `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`
and the depth attachment to `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL`.

For the color attachment, you specify the final layout to be the
`VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` layout, which is the one that is
appropriate for the present operation that occurs after the render pass is complete.
You can leave the depth layout the same as the layout set by the subpass
since the depth buffer is not used as part of the present operation.

#### Subpass

The subpass definition is straightforward and would be more interesting
if you were doing multiple subpasses.
And you might be interested in doing multiple subpasses if you are doing
some pre-processing or post-processing of your graphics data, perhaps
for ambient occlusion or some other effect.
But here, the subpass definition is useful for indicating which attachments
are active during the subpass and also for specifying the layouts to be used
while rendering during the subpass.

    VkAttachmentReference color_reference = {};
    color_reference.attachment = 0;
    color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_reference = {};
    depth_reference.attachment = 1;
    depth_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

The `attachment` member is the index of the attachment in the
attachments array you just defined above for the render pass.

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.flags = 0;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = NULL;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_reference;
    subpass.pResolveAttachments = NULL;
    subpass.pDepthStencilAttachment = &depth_reference;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = NULL;

The `pipelineBindPoint` member is meant to indicate if this is a graphics
or a compute subpass.
Currently, only the graphics subpass is valid.

#### Render Pass

Now you have all you need to define the render pass:

    VkRenderPassCreateInfo rp_info = {};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.pNext = NULL;
    rp_info.attachmentCount = 2;
    rp_info.pAttachments = attachments;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 0;
    rp_info.pDependencies = NULL;
    res = vkCreateRenderPass(info.device, &rp_info, NULL, &info.render_pass);

You'll be using the render pass in several of the upcoming samples.

<table border="1" width="100%">
    <tr>
        <td align="center" width="33%"><a href="09-init_descriptor_set.html" title="Prev">Descriptor Set</a></td>
        <td align="center" width="33%"><a href="index.html" title="Index">Index</a></td>
        <td align="center" width="33%"><a href="11-init_shaders.html" title="Next">Shaders</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>
