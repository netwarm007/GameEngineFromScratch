# Create a Vertex Buffer

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Code file for this section is `13-init_vertex_buffer.cpp`

A vertex buffer is a CPU-visible and GPU-visible buffer
that contains the vertex data that describes
the geometry of the object(s) you wish to render.
In general, the vertex data consists of position (x,y,z) data and the optional
color, normal, or other information.
Like other 3D APIs, the approach here is to fill the buffer with this vertex data and pass
it to the GPU during a draw operation.

## Creating the Vertex Buffer Object

Creating a vertex buffer is nearly the same as creating a uniform buffer, and it begins
with creating a buffer object, as you did in the uniform sample.

    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.pNext = NULL;
    buf_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    buf_info.size = sizeof(g_vb_solid_face_colors_Data);
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = NULL;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buf_info.flags = 0;
    res = vkCreateBuffer(info.device, &buf_info, NULL, &info.vertex_buffer.buf);

The only real difference between creating an uniform buffer object and a vertex buffer
object is the setting of the `usage` field.

The cube data (`g_vb_solid_face_colors_Data`) consists of 36 vertices that
define 12 triangles, 2 on each of the 6 cube faces.
Each triangle also has a face color associated with it.
You can inspect the `cube_data.h` file to see the actual data.

## Allocating the Vertex Buffer Memory

Again, the steps are very similar to the steps taken to allocate the uniform buffer.
First make a query to get the memory requirements, which include taking into
account machine restrictions such as alignment.
Look at the code in the sample to see that the process is very similar to the steps taken in the
uniform sample.

## Store the Vertex Data in the Buffer

Once allocated, the memory is mapped, initialized with the vertex data, and unmapped:

    uint8_t *pData;
    res = vkMapMemory(info.device, info.vertex_buffer.mem, 0,
                      mem_reqs.size, 0, (void **)&pData);

    memcpy(pData, g_vb_solid_face_colors_Data,
           sizeof(g_vb_solid_face_colors_Data));

    vkUnmapMemory(info.device, info.vertex_buffer.mem);

As a final step, the memory is bound to the buffer object:

    res = vkBindBufferMemory(info.device, info.vertex_buffer.buf,
                             info.vertex_buffer.mem, 0);

## Describing the Input Vertex Data

The sample lays out the vertex data in the following arrangement:

    struct Vertex {
        float posX, posY, posZ, posW; // Position data
        float r, g, b, a;             // Color
    };

You need to create a vertex input binding to describe the data arrangement to the GPU.
The following `vi_binding` and `vi_attribs` members are set up here, but
are used later in a subsequent sample as part of creating the graphics pipeline.
But since you are looking at the vertex data format, it is good to examine this
code here:

    info.vi_binding.binding = 0;
    info.vi_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    info.vi_binding.stride = sizeof(g_vb_solid_face_colors_Data[0]);

    info.vi_attribs[0].binding = 0;
    info.vi_attribs[0].location = 0;
    info.vi_attribs[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    info.vi_attribs[0].offset = 0;
    info.vi_attribs[1].binding = 0;
    info.vi_attribs[1].location = 1;
    info.vi_attribs[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    info.vi_attribs[1].offset = 16;

The `stride` is the size of one vertex, or the amount needed to add to a pointer to
get to the next vertex.

The `binding` and `location` members refer to their respective values in the GLSL
shader source code.
You can review the shader source code in the shaders sample to see the correspondence.

Even though the first attribute is position data, a 4-byte float color format
is used to describe it in the `format` member for attribute 0.
The `format` for attribute 1 is more clearly a color format, since attribute 1 is a color.

The `offset` members are strightforward indicators of where each attribute is found in the
vertex data.

## Binding the Vertex Buffer to a Render Pass

You can skip over most of the code that sets up the render pass
since you will see it in a later sample.
But for now, find the code that binds the vertex buffer to the render pass:

    vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindVertexBuffers(info.cmd, 0,             /* Start Binding */
                           1,                       /* Binding Count */
                           &info.vertex_buffer.buf, /* pBuffers */
                           offsets);                /* pOffsets */

    vkCmdEndRenderPass(info.cmd);

Note that you can only connect the vertex buffer with a render pass only inside of
a render pass; that is, between a `vkCmdBeginRenderPass` and a `vkCmdEndRenderPass` while
recording a command buffer.
This essentially tells the GPU what vertex buffer to use when drawing.

<table border="1" width="100%">
    <tr>
        <td align="center" width="33%"><a href="12-init_frame_buffers.html" title="Prev">Framebuffers</a></td>
        <td align="center" width="33%"><a href="index.html" title="Index">Index</a></td>
        <td align="center" width="33%"><a href="14-init_pipeline.html" title="Next">Pipeline</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>
