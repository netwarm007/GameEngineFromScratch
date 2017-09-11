# Create a Swapchain

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Code file for this section is `05-init_swapchain.cpp`

This section describes how to create the swapchain, which is a
list of image buffers that are eventually displayed to the user.
This is one of the first steps needed to set up all the buffers
needed for rendering.

Here's a view of the Vulkan swapchain, in relation to various other parts
of the system.
Some of these parts are familiar and the rest you will learn about in this
section.

![Swapchain](../images/Swapchain.png)

## Vulkan and the Windowing System

Like other graphics APIs, Vulkan keeps window system aspects
separated from the core graphics API.
In Vulkan, the windowing system particulars are exposed via the
WSI (Window System Integration) extensions.
You can find the documentation for these extensions in a Vulkan
specification document that was prepared with the WSI content enabled.
The Vulkan specification found on the
<a href="https://vulkan.lunarg.com" target="_blank">LunarG LunarXchange website</a>
and in the <a href="https://www.khronos.org/registry/vulkan/" target="_blank">Khronos Vulkan Registry</a>
is built with this content.

The WSI extensions contain support for various platforms.
The extensions are activated for a particular platform by defining:

* `VK_USE_PLATFORM_ANDROID_KHR` - Android
* `VK_USE_PLATFORM_MIR_KHR` - Mir
* `VK_USE_PLATFORM_WAYLAND_KHR` - Wayland
* `VK_USE_PLATFORM_WIN32_KHR` - Microsoft Windows
* `VK_USE_PLATFORM_XCB_KHR` - X Window System, using the XCB library
* `VK_USE_PLATFORM_XLIB_KHR` - X Window System, using the Xlib library

The "KHR" portion of the name indicates that the symbol is defined
in a Khronos extension.

### Surface Abstraction

Vulkan uses the `VkSurfaceKHR` object to abstract the native platform
surface or window.
This symbol is defined as part of the VK\_KHR\_surface extension.
The various functions in the WSI extensions are used to create, manipulate,
and destroy these surface objects.

## Revisiting Instance and Device Extensions

Since you postponed using extensions in this tutorial in previous sections,
now is the time to revisit them so you can activate the WSI
extensions that you need to use to interface with the windowing
system.

### Instance Extensions

In order to use the WSI extensions, you need to activate the
general surface extension.
Find the code in the `init_instance_extension_names()` function
in the samples that adds the `VK_KHR_SURFACE_EXTENSION_NAME` to the list
of instance extensions to load.

    info.instance_extension_names.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

Note also that this same function determines a platform-specific extension,
depending on the platform for which the code is being built.
For example, if building for Windows, the function adds
`VK_KHR_WIN32_SURFACE_EXTENSION_NAME`
to the list of instance extensions to load.

    info.instance_extension_names.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);

These extensions are loaded when the instance is created,
in the `init_instance()` function.

### Device Extensions

A swapchain is a list of image buffers that the GPU draws into and are also
presented to the display hardware for scan-out to the display.
Since it is the GPU hardware that is writing to these images,
a device-level extension is needed to work with the swapchain.
Therefore the sample code adds the `VK_KHR_SWAPCHAIN_EXTENSION_NAME`
device extension to the list of device extensions to load in
`init_device_extension_names()`.

    info.device_extension_names.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

This extension is used a bit later in this section to create
the swapchain.

To recap:

* This sample uses the `init_instance_extension_names()` function
to load the general surface extension and the platform-specific surface extension
as instance extensions.
* This sample uses the `init_device_extension_names()` function
to load a swapchain extension as a device extension.

You can learn more about instance and device extensions by visiting the
documentation section on the <a href="https://vulkan.lunarg.com" target="_blank">LunarG LunarXchange website</a>.

## Queue Family and Present

The "present" operation consists of getting one of the swapchain images
onto the physical display so that it can be viewed.
When the application wants to present an image to the display,
it puts a present request onto one of the GPU's queues, using the
`vkQueuePresentKHR()` function.
Therefore, the queue referenced by this function must be able to support
present requests, or graphics and present requests.
The sample checks for this as follows:

    // Iterate over each queue to learn whether it supports presenting:
    VkBool32 *pSupportsPresent =
        (VkBool32 *)malloc(info.queue_family_count * sizeof(VkBool32));
    for (uint32_t i = 0; i < info.queue_family_count; i++) {
        vkGetPhysicalDeviceSurfaceSupportKHR(info.gpus[0], i, info.surface,
                                             &pSupportsPresent[i]);
    }

    // Search for a graphics and a present queue in the array of queue
    // families, try to find one that supports both
    info.graphics_queue_family_index = UINT32_MAX;
    info.present_queue_family_index = UINT32_MAX;
    for (uint32_t i = 0; i < info.queue_family_count; ++i) {
        if ((info.queue_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            if (info.graphics_queue_family_index == UINT32_MAX)
                info.graphics_queue_family_index = i;

            if (pSupportsPresent[i] == VK_TRUE) {
                info.graphics_queue_family_index = i;
                info.present_queue_family_index = i;
                break;
            }
        }
    }

    if (info.present_queue_family_index == UINT32_MAX) {
        // If didn't find a queue that supports both graphics and present, then
        // find a separate present queue.
        for (size_t i = 0; i < info.queue_family_count; ++i)
            if (pSupportsPresent[i] == VK_TRUE) {
                info.present_queue_family_index = i;
                break;
            }
    }
    free(pSupportsPresent);

This code reuses the `info.queue_family_count` obtained earlier,
since `vkGetPhysicalDeviceSurfaceSupportKHR()` returns
one flag for each queue family.
It then searches for a queue family that supports both
present and graphics.
If there is no queue family that supports both, the program
remembers a queue family that supports graphics and then
searches again for a queue that supports present.
Since both `graphics_queue_family_index`
and `present_queue_family_index` are set by this code,
later on the samples must use a queue from `graphics_queue_family_index`
for graphics commands and a queue from `present_queue_family_index` for presents.

Yes, this is a bit redundant with the previous search performed in init_device
for a queue family that supported just graphics,
which was done only for illustrative purposes.
A real application may do these steps in a different order to
avoid the repetition.

If no such queue family was found, the sample simply exits.

## Swapchain Create Info

Much of the effort in the remainder of this section is directed
towards filling in this create info structure, used to create the swapchain:

    typedef struct VkSwapchainCreateInfoKHR {
        VkStructureType                  sType;
        const void*                      pNext;
        VkSwapchainCreateFlagsKHR        flags;
        VkSurfaceKHR                     surface;
        uint32_t                         minImageCount;
        VkFormat                         imageFormat;
        VkColorSpaceKHR                  imageColorSpace;
        VkExtent2D                       imageExtent;
        uint32_t                         imageArrayLayers;
        VkImageUsageFlags                imageUsage;
        VkSharingMode                    imageSharingMode;
        uint32_t                         queueFamilyIndexCount;
        const uint32_t*                  pQueueFamilyIndices;
        VkSurfaceTransformFlagBitsKHR    preTransform;
        VkCompositeAlphaFlagBitsKHR      compositeAlpha;
        VkPresentModeKHR                 presentMode;
        VkBool32                         clipped;
        VkSwapchainKHR                   oldSwapchain;
    } VkSwapchainCreateInfoKHR;

### Creating a Surface

With the WSI extensions loaded in the instance and the device, you
can now create a `VkSurface` so that you can proceed with making the
swapchain.
You again need to use platform-specific code, appearing near the top
of the `05-init_swapchain.cpp` file:

    #ifdef _WIN32
        VkWin32SurfaceCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.pNext = NULL;
        createInfo.hinstance = info.connection;
        createInfo.hwnd = info.window;
        res = vkCreateWin32SurfaceKHR(info.inst, &createInfo, NULL, &info.surface);
    #endif

The `info.connection` and `info.window` values are set up in more
platform-specific code contained in the `init_connection()` and
`init_window()` functions, which you can also find in the sample code.

Note that the `init_connection()` and `init_window()` functions also
took care of the platform-specific operations of connecting to the
display and creating the actual window.
The `VkSurfaceKHR` surface you just created with the
`vkCreateWin32SurfaceKHR()` function is represented by a handle that is
usable by Vulkan for the platform window object.

You then add the resulting surface to the swapchain create info structure:

    VkSwapchainCreateInfoKHR swapchain_ci = {};
    swapchain_ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_ci.pNext = NULL;
    swapchain_ci.surface = info.surface;

#### Device Surface Formats

You also need to specify the format of the surface
when creating the swapchain.
In this context, the "format" refers to the pixel formats
as described by the `VkFormat` enumeration, where
`VK_FORMAT_B8G8R8A8_UNORM` is one common example for a
device surface format.

The next section of code in the sample gets a list of
`VkSurfaceFormatKHR` structures which contain the
`VkFormat` formats that are supported by the display,
along with other info.
Since the samples really do not care what format is used
for the display images and surfaces, the sample just
picks the first available format, falling back to an
arbitrary, but common, format if none are specified.

Take a look at the code in the sample and see that it ends up
setting the format in the create info structure:

    swapchain_ci.imageFormat = info.format;

### Surface Capabilities

In order to continue filling out the swapchain create info structure,
the sample calls `vkGetPhysicalDeviceSurfaceCapabilitiesKHR()`
and `vkGetPhysicalDeviceSurfacePresentModesKHR()` to
get the needed information.
It can then fill in the following fields:

    uint32_t desiredNumberOfSwapChainImages = surfCapabilities.minImageCount;

    swapchain_ci.minImageCount = desiredNumberOfSwapChainImages;
    swapchain_ci.imageExtent.width = swapChainExtent.width;
    swapchain_ci.imageExtent.height = swapChainExtent.height;
    swapchain_ci.preTransform = preTransform;
    swapchain_ci.presentMode = swapchainPresentMode;

You should set the `minImageCount` member to a value that represents the
buffering strategy that your application uses, double-buffering or
triple-buffering for example.
This sample queries the minimum number of images that can be in a
swapchain using the `vkGetPhysicalDeviceSurfaceCapabilitiesKHR()`
function, storing the result in `surfCapabilities`.
Asking for this minimum number of images ensures that we can acquire
one presentable image as long as we present it before attempting
to acquire another.
This represents a double-buffered configuration, since you will be able
to acquire one image to render into, while the other image is being presented.
If you wanted to triple-buffer, then you would ask for one more image and then
be able to acquire two buffers to render into before you present them.

## Different Queue Families for Graphics and Present

You determined above the queue families for graphics and present queues.
If they are different, you need to do some additional work to allow the
images to  be shared between queue families.

    swapchain_ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_ci.queueFamilyIndexCount = 0;
    swapchain_ci.pQueueFamilyIndices = NULL;
    uint32_t queueFamilyIndices[2] = {
        (uint32_t)info.graphics_queue_family_index,
        (uint32_t)info.present_queue_family_index};
    if (info.graphics_queue_family_index != info.present_queue_family_index) {
        // If the graphics and present queues are from different queue families,
        // we either have to explicitly transfer ownership of images between the
        // queues, or we have to create the swapchain with imageSharingMode
        // as VK_SHARING_MODE_CONCURRENT
        swapchain_ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_ci.queueFamilyIndexCount = 2;
        swapchain_ci.pQueueFamilyIndices = queueFamilyIndices;
    }

The above fields supply more of the basic info needed to create
the swapchain.
You can inspect the rest of the code in the sample to see
how this information was obtained and how the rest of
the create info structure is filled out.

## Create Swapchain

With the swapchain create info structure filled out,
you can now create the swapchain:

    res = vkCreateSwapchainKHR(info.device, &swapchain_ci, NULL, &info.swap_chain);

This call creates a set of images that make up the swapchain.
At some point, you need the handles to the individual images so that
you can tell the GPU which images to use for rendering.
The `vkCreateSwapchainKHR()` function creates the images itself
and so has kept track of the handles.
The sample obtains the image handles by using the familiar pattern
of first querying how many handles exist by calling

    vkGetSwapchainImagesKHR(info.device, info.swap_chain,
                            &info.swapchainImageCount, NULL);

to get the count and then calling the function again
to get a list of image handles and storing them in `info.buffers[]`.
Now you have a list of target image handles.

## Create Image Views

You need to communicate to Vulkan how you intend to use these swapchain
images by creating image views.
A "view" is essentially additional information attached to a resource
that describes how that resource is used.

The region of memory belonging to an image can be arranged in
a large variety of ways, depending on the intended use of the image.
For example, an image can be 1D, 2D, or 3D.
Or there can be an array of images, etc.
It is also useful to describe the format of the image (`VK_FORMAT_R8G8B8A8_UNORM`,
for example), the order of components, and layer information.
All this information is the image metadata contained in a `VkImageView`.

Creating an image view is straightforward.
Find the `VkImageViewCreateInfo` structure in the `05-init_swapchain.cpp` file
and see that it is filled in with the values you would expect for a
2D framebuffer.
Note that the image handle itself is stored in the image view.

You then finish up the creation of the swapchain
by storing the image view handles in the `info` data structure for use later on.

<table border="1" width="100%">
    <tr>
        <td align="center" width="33%">Previous: <a href="04-init_command_buffer.html" title="Prev">Command Buffer</a></td>
        <td align="center" width="33%">Back to: <a href="index.html" title="Index">Index</a></td>
        <td align="center" width="33%">Next: <a href="06-init_depth_buffer.html" title="Next">Depth Buffer</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>
