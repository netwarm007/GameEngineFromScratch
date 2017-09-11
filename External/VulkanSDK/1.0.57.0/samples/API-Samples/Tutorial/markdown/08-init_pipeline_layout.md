# Descriptor Set Layouts and Pipeline Layouts

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Code file for this section is `08-init_pipeline_layout.cpp`

In the previous sample, you created a uniform buffer, but you have not
done anything about describing how it is to be used by the shader.
You know that the buffer contains a uniform variable for the MVP
transform and that it
will be used by just the vertex shader, but Vulkan doesn't know
any of this yet.

We accomplish this by using a descriptor.

## Descriptors and Descriptor Sets

A descriptor is a special opaque shader variable that
shaders use to access
buffer and image resources in an indirect fashion.
It can be thought of as a "pointer" to a resource.
The Vulkan API allows these variables to be changed between
draw operations so that the shaders can access different resources
for each draw.

In the sample example, you have only one uniform buffer.
But you could create two uniform buffers, each with a
different MVP to give different views of the scene.
You could then easily change the descriptor to point to
either uniform buffer to switch back and forth between
the MVP matrices.

A descriptor set is called a "set" because it can refer to an
array of homogenous resources that can be described with the
same layout binding.
(The "layout binding" will be explained shortly.)

You are not using textures in this sample, but one possible
way to use multiple descriptors is to construct a
descriptor set with two descriptors, with each descriptor
referencing a separate texture.
Both textures are therefore available during a draw.
A command in a command buffer could then select the texture
to use by specifying the index of the desired texture.

It is important to note that you are just working on *describing*
the descriptor set here and are not actually allocating or
creating the descriptor set itself, which you will do later,
in the descriptor_set sample.

To describe a descriptor set, you use a descriptor set layout.

## Descriptor Set Layouts

A descriptor set layout is used to describe the content of a
list of descriptor sets.
You also need one layout binding for each descriptor set,
which you use to describe each descriptor set:

    VkDescriptorSetLayoutBinding layout_binding = {};
    layout_binding.binding = 0;
    layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layout_binding.descriptorCount = 1;
    layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    layout_binding.pImmutableSamplers = NULL;

* You happen to be making only one descriptor set, so the only
choice for the `binding` member is 0.
* Since this descriptor is referencing a uniform buffer, you set the
`descriptorType` appropriately.
* You have only one descriptor in this descriptor set,
which is indicated by the `descriptorCount` member.
* You indicate that this uniform buffer resource is to be bound to
the shader vertex stage.

With the binding for our one descriptor set defined, you are ready to
create the descriptor set layout:

    #define NUM_DESCRIPTOR_SETS 1
    VkDescriptorSetLayoutCreateInfo descriptor_layout = {};
    descriptor_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout.pNext = NULL;
    descriptor_layout.bindingCount = 1;
    descriptor_layout.pBindings = &layout_binding;
    info.desc_layout.resize(NUM_DESCRIPTOR_SETS);
    res = vkCreateDescriptorSetLayout(info.device, &descriptor_layout, NULL,
                                      info.desc_layout.data());

## Pipeline Layouts

A pipeline layout contains a list of descriptor set layouts.
It also can contain a list of push constant ranges, which
is an alternate way to pass constants to a shader and will
not be covered here.

As with the descriptor sets, you are just defining the layout.
The actual descriptor set is allocated and
filled in with the uniform buffer reference later.

    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
    pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pPipelineLayoutCreateInfo.pNext = NULL;
    pPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pPipelineLayoutCreateInfo.pPushConstantRanges = NULL;
    pPipelineLayoutCreateInfo.setLayoutCount = NUM_DESCRIPTOR_SETS;
    pPipelineLayoutCreateInfo.pSetLayouts = info.desc_layout.data();

    res = vkCreatePipelineLayout(info.device, &pPipelineLayoutCreateInfo, NULL,
                                 &info.pipeline_layout);

You will use the pipeline layout later to create the graphics pipeline.

## Shader Referencing of Descriptors

It is worth pointing out that the shader explicitly references these
descriptors in the shader language.

For example, in GLSL:

     layout (set=M, binding=N) uniform sampler2D variableNameArray[I];

* M refers the the M'th descriptor set layout in the `pSetLayouts` member
of the pipeline layout
* N refers to the N'th descriptor set (binding) in M's `pBindings` member
of the descriptor set layout
* I is the index into the array of descriptors in N's descriptor set

The layout code for the uniform buffer in the vertex shader that you will use
looks like:

    layout (std140, binding = 0) uniform bufferVals {
        mat4 mvp;
    } myBufferVals;

This maps the uniform buffer contents to the `myBufferVals` structure.
"set=M" was not specified and defaults to 0.

"std140" is a standard to describe how data is packed in uniform blocks.
You may wish to refer to it if you wish to put more data in a uniform block.
See <a href="https://www.opengl.org/registry/specs/ARB/uniform_buffer_object.txt" target="_blank">this doc</a>
for more information.

<table border="1" width="100%">
    <tr>
        <td align="center" width="33%"><a href="07-init_uniform_buffer.html" title="Prev">Uniform Buffer</a></td>
        <td align="center" width="33%"><a href="index.html" title="Index">Index</a></td>
        <td align="center" width="33%"><a href="09-init_descriptor_set.html" title="Next">Descriptor Set</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>
