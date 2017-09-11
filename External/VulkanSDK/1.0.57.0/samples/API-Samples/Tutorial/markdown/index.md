# Welcome to the Vulkan Samples Tutorial

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

This tutorial is organized into sections that walk you
through the steps to create a simple Vulkan program.
Each tutorial section corresponds directly to a sample program
in the LunarG samples progression
and is designed to be read as you look at and experiment with real
code from the progression.

## Tutorial Index

* [Introduction](00-intro.html)
* [Instance](01-init_instance.html)
* [Enumerate Devices](02-enumerate_devices.html)
* [Device](03-init_device.html)
* [Command Buffer](04-init_command_buffer.html)
* [Swapchain](05-init_swapchain.html)
* [Depth Buffer](06-init_depth_buffer.html)
* [Uniform Buffer](07-init_uniform_buffer.html)
* [Pipeline Layout](08-init_pipeline_layout.html)
* [Descriptor Set](09-init_descriptor_set.html)
* [Render Pass](10-init_render_pass.html)
* [Shaders](11-init_shaders.html)
* [Framebuffers](12-init_frame_buffers.html)
* [Vertex Buffer](13-init_vertex_buffer.html)
* [Pipeline](14-init_pipeline.html)
* [Draw Cube](15-draw_cube.html)

## The Goal

The final section in the tutorial produces a program that displays this:

![Draw Cube](../images/drawcube.png)
<footer>&copy; Copyright 2016 LunarG, Inc</footer>

## Change Log

* 26 Aug 2016 - Initial Revision
* 26 Oct 2016 - Image layout transitions are now specified
in render pass and subpass definitions, instead of using memory barriers