# Enumerate Physical Devices

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Code file for this section is `02-enumerate_devices.cpp`

The next step in the samples progression is determining the
physical devices present on the system.

After you have created an instance, the loader knows how many Vulkan
physical devices are available, but your application doesn't know this yet.
The application learns how many devices are available by asking the
Vulkan API for a list of physical devices.

![PhysicalDevices](../images/PhysicalDevices.png)

Physical devices are related to the instance as shown in the above diagram.

## Getting Lists of Objects from Vulkan

Obtaining list of objects is a fairly common operation in Vulkan,
and the API has a consistent pattern for doing so.
API functions that return lists have count and pointer arguments.
The count argument is a pointer to an integer so that the API can
set its value.  The steps are:

1. Call the function with a valid pointer to an integer for the count argument
and a NULL for the pointer argument.
1. The API fills in the count argument with the number of objects in the list.
1. The application allocates enough space to store the list.
1. The application calls the function again with the pointer argument pointing
to the space just allocated.

You will see this pattern often in the Vulkan API.

## vkEnumeratePhysicalDevices Function

The `vkEnumeratePhysicalDevices` function returns only a list of handles
for each physical device on the system.
A physical device might be a graphics card that one plugs into a desktop computer, some sort
of GPU core on an SoC, etc.
If there are multiple devices available, the application must decide somehow
which of them it will use.

Our sample code enumerates the physical devices as follows:

    // Get the number of devices (GPUs) available.
    VkResult res = vkEnumeratePhysicalDevices(info.inst, &gpu_count, NULL);
    // Allocate space and get the list of devices.
    info.gpus.resize(gpu_count);
    res = vkEnumeratePhysicalDevices(info.inst, &gpu_count, info.gpus.data());

Note that the `info.gpus` variable is a vector of type `VkPhysicalDevice`,
which is a handle.

All `enumerate` does is get the list of physical device handles.
The `device` program, which is the next step in the progression,
looks at this list to decide which device to use.

## The Samples `info` Structure

You'll notice the use of an `info` variable in the above code.
Each sample program uses the global `info` structure
variable to track Vulkan information and application state.
This facilitates using more compact function calls to perform
steps that have already been covered in this tutorial.
For example, see the line of code in the `enumerate` program:

    `init_instance(info, "vulkansamples_enumerate");`

which performs the steps discussed on the `instance` page
of this tutorial.
`init_instance()` creates the instance and stores the handle in `info`.
And then `vkEnumeratePhysicalDevices()` uses `info.inst` in the call to
`vkEnumeratePhysicalDevices()`.

Now that you have the list of devices (GPUs), it is now time to select a GPU
and create a Vulkan logical device object so that you can start working with
that GPU.

<table border="1" width="100%">
    <tr>
        <td align="center" width="33%">Previous: <a href="01-init_instance.html" title="Prev">Instance</a></td>
        <td align="center" width="33%">Back to: <a href="index.html" title="Index">Index</a></td>
        <td align="center" width="33%">Next: <a href="03-init_device.html" title="Next">Device</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>
