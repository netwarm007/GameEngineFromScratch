# Introduction to Vulkan

![Vulkan Logo](../images/vulkanlogo.png)
<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Vulkan is an advanced graphics API developed by the Khronos group.
Other graphics APIs (like OpenGL and Direct3D) require the driver
to perform the necessary translations
from the high-level API into something suitable for the hardware.
The intent at the time was to keep developers from having to manage
the more complex details of the graphics hardware.

As those older graphics APIs continued to evolve,
they slowly exposed more and more low-level
hardware functionality directly to the programmers.
The programmers demanded lower-level access to the hardware,
trading off the convenience and safety of the hand-holding
functions that had higher overhead and lower performance.

Vulkan was designed to avoid the higher overhead found in
higher-function APIs.
As a result, the Vulkan programmer needs to take care of many
more details when building a Vulkan application.
But this allows the programmer to manage the application resources
and the GPU hardware more efficiently.
This is because the programmer has more knowledge of the application's
resource usage patterns and so doesn't have to make the costly
assumptions that other APIs are forced to make.

Additionally, Vulkan also aims to be a more cross-platform API
than other graphics APIs by targeting not only high-end systems,
but also low-end mobile devices.

## About this Tutorial

### Purpose

The purpose of this tutorial is to step you through the process
of creating a simple Vulkan application, learning the basics
of Vulkan along the way.
This tutorial is synchronized with the
code making up the samples progression developed by LunarG.
As you work through this tutorial, you are pointed to the actual code
in these sample programs that illustrate each of the steps
needed to develop a simple Vulkan application.
At the end of the tutorial, you will have a complete Vulkan program
that you can use as a starting point for learning more about Vulkan.

To get an idea of what this samples progression looks like, please visit the
<a href="https://vulkan.lunarg.com/doc/sdk/latest/windows/samples_index.html" target="_blank">LunarG Samples Progression Index</a>.

### How to use the Samples Tutorial

This tutorial is most effective when it is used side-by-side
with the code in the samples progression.
We suggest that you set up your development environment to enable you
to download and build the
<a href="https://github.com/LunarG/VulkanSamples" target="_blank">LunarG Vulkan Samples GitHub repository</a>.
Please use the instructions in the top-level README file in
that repository to install the necessary tools and packages to build
the samples.
The samples are found in the `API-Samples` folder of the repository.
Once you have the samples built and running, you can proceed with this tutorial.

It is also useful to be able to easily access the
`vulkan.h` header file for reference as you look at the sample code.
This file can be found in the Samples repository, under the `include` directory.

The Vulkan specification document is an invaluable source for Vulkan information.
You can find the specification at the
<a href="https://vulkan.lunarg.com" target="_blank">LunarG LunarXchange website</a>
or in the <a href="https://www.khronos.org/registry/vulkan/" target="_blank">Khronos Vulkan Registry</a>.
Although it is not strictly required to refer to the specification as you
work through the samples, you may find the specification useful to get
a deeper understanding of Vulkan.

#### Samples Coding Strategy

The samples are implemented to focus on a specific topic by showing only
the code that is related to that topic.
Code related to previously-covered topics is generally organized into
functions that are called from the main program so that code for the
"old" topics does not get in the way of code related to the current topic.
You can always go back to these functions to refresh your understanding
of any previous topic.

The portion of a sample program that is focusing on a specific topic is
delineated with comments in between the functions that cover the "old" topics as follows:

    init_instance(info, sample_title);
    init_enumerate_device(info);
    init_window_size(info, 500, 500);
    init_connection(info);
    init_window(info);
    init_swapchain_extension(info);
    init_device(info);
    ...
    /* VULKAN_KEY_START */

    ... code of interest

    /* VULKAN_KEY_END */
    ...
    destroy_device(info);
    destroy_window(info);
    destroy_instance(info);

Look for these comments in the source files to quickly locate the
code relating to the topic under discussion.

<table border="1" width="100%">
    <tr>
        <td align="center" width="50%">Back to: <a href="index.html" title="Index">Index</a></td>
        <td align="center" width="50%">Next: <a href="01-init_instance.html" title="Next">Instance</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>
