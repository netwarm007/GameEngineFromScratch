# Create a Vulkan Instance

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Code file for this section is `01-init_instance.cpp`

The first step in the samples progression is creating a
Vulkan instance.
Find the `01-init_instance.cpp` program in the `API-Samples` folder
of the LunarG Vulkan Sample repository and get ready to look
at it as you read this section about Vulkan instances.

## Vulkan Instances

The Vulkan API uses the `vkInstance` object to store all
per-application state.
The application must create a Vulkan instance
before performing any other Vulkan operations.

The basic Vulkan architecture looks like:

![Basic App Loader](../images/BasicAppLoader.png)

The above diagram shows that a Vulkan application is
linked to a Vulkan library which is commonly referred to
as the **loader**.
Creating an instance initializes the loader.
The loader also loads and initializes the
low-level graphics driver,
usually provided by the vendor of the GPU hardware.

Note that there are layers depicted in this diagram,
which are also loaded by the loader.
Layers are commonly used for validation, which is the error
checking that would normally be performed by the driver.
In Vulkan, the drivers are much more lightweight than in
other APIs such as OpenGL partly because they delegate this
validation function to the validation layers.
Layers are optional and can be loaded selectively each
time the application creates an instance.

Vulkan layers are outside the scope of this tutorial
and of the samples progression.
So this tutorial does not cover the layers further.
Further information about layers can be found on the
<a href="https://vulkan.lunarg.com" target="_blank">LunarG LunarXchange website</a>.

## `vkCreateInstance`

Take a look at the `01-init_instance.cpp` source code and find the
`vkCreateInstance` function call, which has the following
prototype:

    VkResult vkCreateInstance(
        const VkInstanceCreateInfo*                 pCreateInfo,
        const VkAllocationCallbacks*                pAllocator,
        VkInstance*                                 pInstance);

Taking this apart, bit by bit:

**`VkResult`** -
This is the return status of the function.
You may wish to open up the `vulkan.h` header file and
inspect some of these definitions as we encounter them.
You can find the file in the `include` directory within the
VulkanSamples repository.

**`VkInstanceCreateInfo`** -
This structure contains any additional information that is
needed to create the instance.
Since this is an important item, you will take a closer look at it later.

**`VkAllocationCallbacks`** -
Functions that allocate host memory are equipped with an argument that
allows the application to perform its own host memory management.
Otherwise, the Vulkan implementation uses the default system
memory management facilities.
An application might want to manage its own host memory
in order to log memory allocations, for example.

The samples do not use this feature, so you will always see NULL passed
for this argument in this and other functions throughout the samples.

**`VkInstance`** -
This is simply a handle returned by this function if the instance
creation was successful.
This is an opaque handle, so do not try to de-reference it.
Many Vulkan functions that create objects return handles for the
objects they create in this manner.

## A Closer Look at the VkInstanceCreateInfo Structure

Vulkan functions that create objects usually have a Vk*Object*CreateInfo
argument.
Find the code in the sample that initializes this structure:

    typedef struct VkInstanceCreateInfo {
        VkStructureType             sType;
        const void*                 pNext;
        VkInstanceCreateFlags       flags;
        const VkApplicationInfo*    pApplicationInfo;
        uint32_t                    enabledLayerCount;
        const char* const*          ppEnabledLayerNames;
        uint32_t                    enabledExtensionCount;
        const char* const*          ppEnabledExtensionNames;
    } VkInstanceCreateInfo;

The first two members are commonly found in many Vulkan `CreateInfo` structures.

**`sType`** -
The `sType` field indicates the type of the structure.
In this case, you set it to `VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO`,
since it is an *VkInstanceCreateInfo* structure.
This may seem redundant since only a structure of this type can be
passed as the first argument of `vkCreateInstance()`.
But it has some value for the following reasons:

* A driver, validation layer, or other consumer of the structure can perform a simple validity
check and perhaps fail the request if the `sType` is not as expected.

* Structures can be passed via type-less (void) pointers to consumers
via interfaces that are not fully defined to the point where the
interface arguments are typed.  For example, if a driver supports an extension to
instance creation, it could look at such a structure passed via
the type-less `pNext` pointer (discussed next).  In this case, the `sType`
member would be set to a value that the extension recognizes.

Since this member is always the first member in the structure, the
consumer can easily determine the type of the structure and decide
how to handle it.

**`pNext`** -
You set `pNext` to NULL more often than not.
This void pointer is sometimes used to pass extension-specific
information in a typed structure where the `sType` member
is set to an extension-defined value.
As discussed above, extensions can analyze any structures that are passed in
along this chain of `pNext` pointers to find structures that they
recognize.

**`flags`** -
There are currently no flags defined, so set this to zero.

**`pApplicationInfo`** -
This is a pointer to another structure that you also need to fill out.
You will come back to this one in a moment.

**`enabledLayerCount`** and **`ppEnabledLayerNames`** -
The samples in this tutorial do not use layers, so these members are cleared.

**`enabledExtensionCount`** and **`ppEnabledExtensionNames`** -
At this point, the samples in this tutorial do not use extensions.
Later on, another sample will show the usage of extensions.

## The VkApplicationInfo Structure

This structure provides the Vulkan implementation with some
basic information about the application:

    typedef struct VkApplicationInfo {
        VkStructureType    sType;
        const void*        pNext;
        const char*        pApplicationName;
        uint32_t           applicationVersion;
        const char*        pEngineName;
        uint32_t           engineVersion;
        uint32_t           apiVersion;
    } VkApplicationInfo;

**`sType`** and **`pNext`** -
These have the same meanings as those in the `vkInstanceCreateInfo` structure.

**`pApplicationName`, `applicationVersion`, `pEngineName`, `engineVersion`** -
These are free-form fields that the application may provide if desired.
Some implementations of tools, loaders, layers, or drivers may
use these fields to provide information while debugging or collecting
information for reports, etc.
It may even be possible for drivers to change their behaviors depending
on the application that is running.

**`apiVersion`** -
This field communicates the major, minor, and patch levels of the
Vulkan API header used to compile the application.
If you are using Vulkan 1.0, major should be 1 and minor should be 0.
Using the `VK_API_VERSION_1_0` macro from `vulkan.h` accomplishes this,
with a patch level of 0.
Differences in the patch level should not affect the full compatibility
between versions that differ only in the patch level.
Generally, you should set this field to `VK_API_VERSION_1_0` unless
you have a good reason to do otherwise.

## Back to the Code

Once the structures are populated, the sample app creates the instance:

    VkInstance inst;
    VkResult res;

    res = vkCreateInstance(&inst_info, NULL, &inst);
    if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {
        std::cout << "cannot find a compatible Vulkan ICD\n";
        exit(-1);
    } else if (res) {
        std::cout << "unknown error\n";
        exit(-1);
    }

    vkDestroyInstance(inst, NULL);

In the code above, the application makes a quick check for the mostly likely error
and reports it, or some other error, according to the result.
Note that a success (`VK_SUCCESS`) has a value of zero,
so many applications take the shortcut of interpreting
a non-zero result as an error, which is the case here.

Finally, the application destroys the instance before it exits.

Now that you have created a Vulkan instance, it is now time to discover what graphics
devices are available to your Vulkan instance.

<table border="1" width="100%">
    <tr>
        <td align="center" width="33%">Previous: <a href="00-intro.html" title="Prev">Introduction</a></td>
        <td align="center" width="33%">Back to: <a href="index.html" title="Index">Index</a></td>
        <td align="center" width="33%">Next: <a href="02-enumerate_devices.html" title="Next">Enumerate</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>
