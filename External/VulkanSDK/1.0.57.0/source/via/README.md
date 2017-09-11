#![LunarG's Vulkan Installation Analyzer (VIA)](images/lunarg_via_title.png)
This document is an overview of how to use the [LunarG Vulkan Installation Analyzer (VIA)](https://vulkan.lunarg.com/doc/sdk/latest/windows/via.html).
VIA is a tool that can:
 1. Determine the state of Vulkan components on your system
 2. Validate that your Vulkan Loader and drivers are installed properly
 3. Capture your system state in a form that can be used as an attachment when submitting bugs

 This document describes where to find the source for VIA, building it, runnning it, and how to understand the resulting command line output that is generated.

<BR />


## Building
Many components of the LunarG Vulkan SDK are Open Source, including VIA.  VIA is currently part of the LunarG
[VulkanTools](https://github.com/LunarG/VulkanTools) GitHub respository.

**Windows Note:** VIA is already pre-built as part of the LunarG Windows Vulkan SDK, but should you wish to build a
debug version or find the source, this section should provide you with the information you need.  Otherwise, simply
skip down to the "Running" section below.

#### Building VIA in VulkanTools
Because it is part of a group of tools, you build it from the top folder by
following the instructions in the [BuilidVT.md](https://github.com/LunarG/VulkanTools/blob/master/BUILDVT.md)
file at the top of the source tree.

#### Building VIA in the Linux Vulkan SDK
The source for VIA can also be found in the LunarG Linux [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) in the "source/via" directory.
 1. Download and install the Linux SDK
 2. Run "source setup-env.sh" in the SDK root directory
 3. Run "./build_tools.sh"

<BR />

## Running
You will find the VIA binary in a different location depending on which OS you are using and whether you have built it, or installed it as part of the SDK.  The following information states where to find the proper executable.

Please note that if you are trying to diagnose a troublesome application, the **best way** to run VIA to assist in diagnosis is to change to the location of the application, and run via in that folder locally (by typing in a relative or absolute path to the via executable).

#### In the Windows Vulkan SDK
VIA is installed into your start menu as part of the Windows Vulkan SDK.  Simply open your start menu, search for the "Vulkan SDK" and click "via".  This will output the resulting via.html directly to your desktop.

If you need to run via from the command-line, you will find it in your SDK folder (defined by the environment variable "VULKAN_SDK") under the "Bin" folder for 64-bit executables, and "Bin32" folder for 32-bit executables.  From there, simply run:
```
via.exe
```

#### In the Linux Vulkan SDK
Once built, VIA can be found in the x86_64/bin directory.  You can simply execute it from there using:

```
via
```

<BR />


#### If Built from VulkanTools
Go into the folder where you generated the build items from the above building step.

**Linux**

Simply run:
```
via
```


**Windows**

Windows has an additional required step to perform if you are building from VulkanTools before you can run VIA in the proper way.  This is because the CMake scripts have an important folder being copied one level too low.  To run, the first time you will have to copy this folder into the appropriate location.

Steps to run the first time:
 1. Go into the folder you built, and then go into the "via" sub-folder.
 2. In this folder you should see an "images" folder.  Copy this into either or both of your "Debug" or "Release" folders.
 3. Go into the "Debug" or "Release" folder (whichever you are desiring to work with) 
 4. Run 
```
via.exe
```

After the first time, you just need to go into the folder and re-run "via.exe".

<BR />

### Resulting Output
VIA outputs two things:
 - A command-line output indicating the overall status
 - An HTML file (called via.html) containing the details which will be output to one of two locations:
  1. If the current directory is writeable, the HTML will be placed in that location.
  2. Otherwise, it will be saved to your home folder, except for the Windows Start Menu short-cut which writes the file to your desktop.

Your home folder is the following location (based on your OS):
 - Windows: Wherever your environment variables %HOMEDRIVE%\%HOMEPATH% point to.
 - Linux: It will be placed in your home folder ("~/.").

<BR />

#### Additional command-line arguments
There are additional command-line parameters which can be used.  These simply augment existing behavior and do not capture any more information.
The available command-line arguments are:

##### --unique_output
The --unique_output argument, if provided, will cause the output html to be generated with a date/time suffix.  This will allow you to perform
multiple state captures on your system without accidentally erasing previous results.  The new file has the following format:

_via_YYYY_MM_DD_HH_MM.html_

Where each component stands for the numeric values for year (YYYY), month (MM), day (DD), hour (HH), and minute (MM).

#### --output_path
The --output_path arument allows the user to specify a location for the output html file. For
example, if the user runs `via --output_path /home/me/Documents`, then the output file will be
`/home/me/Documents/via.html`.

<BR />

## Common Command-Line Outputs

#### "SUCCESS: Validation completed properly"

##### Problem:
LunarG's VIA could detect no problems with your setup.

##### Possible Reason:
Your system is likely setup properly.  If you have trouble running Vulkan from another location, it could be that your environment variables aren't setup properly.

##### Next Step:
Re-run VIA from the location your Vulkan application/game is supposed to run.


#### "ERROR: Failed to find Vulkan Driver JSON in registry"

##### Problem:
This is a Windows-specific error that indicates that no Vulkan Driver JSON files were referenced in the appropriate place in the Windows Registry.

##### Possible Reason:
This can indicate that a Vulkan driver failed to install properly.

##### Next Step:
See the [Vulkan Graphics Driver Problems](#vulkan-graphics-driver-problems) section for more info.


#### "ERROR: Failed to find Vulkan Driver JSON"

##### Problem:
The Vulkan loader on your system failed to find any Vulkan Driver JSON files in the appropriate places.

##### Possible Reason:
This can indicate that a Vulkan driver failed to install properly.

##### Next Step:
See the [Vulkan Graphics Driver Problems](#vulkan-graphics-driver-problems) section for more info.


#### "ERROR: Failed to properly parse Vulkan Driver JSON"

##### Problem:
The Vulkan loader on your system found at least one Vulkan Driver JSON file, but it contained invalid data.

##### Possible Reason:
This can indicate that a Vulkan driver failed to install properly.

##### Next Step:
See the [Vulkan Graphics Driver Problems](#vulkan-graphics-driver-problems) section for more info.


#### "ERROR: Failed to find Vulkan Driver Lib"

##### Problem:
The Vulkan loader on your system found at least one Vulkan Driver JSON file, but the Driver library referenced in that file appears invalid.

##### Possible Reason:
This can indicate that a Vulkan driver failed to install properly.

##### Next Step:
See the [Vulkan Graphics Driver Problems](#vulkan-graphics-driver-problems) section for more info.


#### "ERROR: Failed to find Vulkan Layer JSON"

##### Problem:
Normally a layer isn't a requirement, but in this case a layer was attempted to be used by VIA, which should have existed, and did not exist
in the appropriate location.

##### Possible Reason:
If the layer is associated with a driver vendor (for example Nvidia), it likely failed during the driver install.  If the layer is associated
with an application (for example RenderDoc), then it likely failed during application install.  Otherwise, it could be an invalid runtime
or SDK install.

##### Next Step:
Look at the HTML file to determine which layer caused the problem, and then contact the appropriate person.  If it's an SDK layer, or you
are having a hard time figuring out exactly what the problem is, follow the instructions in the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.


#### "ERROR: Failed to properly parse Vulkan Layer JSON"

##### Problem:
Normally a layer isn't a requirement, but in this case a layer was attempted to be used by VIA, and the JSON file was found to be invalid.

##### Possible Reason:
If the layer is associated with a driver vendor (for example Nvidia), it likely failed during the driver install.  If the layer is associated
with an application (for example RenderDoc), then it likely failed during application install.  Otherwise, it could be an invalid runtime
or SDK install.

##### Next Step:
Look at the HTML file to determine which layer caused the problem, and then contact the appropriate person.  If it's an SDK layer, or you
are having a hard time figuring out exactly what the problem is, follow the instructions in the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.


#### "ERROR: Failed to find Vulkan Layer Lib"

##### Problem:
Normally a layer isn't a requirement, but in this case a layer was attempted to be used by VIA, and the JSON file was found to point to an
invalid layer library file.

##### Possible Reason:
If the layer is associated with a driver vendor (for example Nvidia), it likely failed during the driver install.  If the layer is associated
with an application (for example RenderDoc), then it likely failed during application install.  Otherwise, it could be an invalid runtime
or SDK install.

##### Next Step:
Look at the HTML file to determine which layer caused the problem, and then contact the appropriate person.  If it's an SDK layer, or you
are having a hard time figuring out exactly what the problem is, follow the instructions in the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.



#### "ERROR: Vulkan failed to find a compatible driver"

##### Problem:
All the components appeared to be in place to allow Vulkan to load (from an external stand-point), but the loader still failed to find
a Vulkan compatible driver.

##### Possible Reason:
This can indicate that either a Vulkan driver failed to install properly, or the run-time is failing for some reason.

##### Next Step:
See the [Vulkan Graphics Driver Problems](#vulkan-graphics-driver-problems) section for more info.


#### "ERROR: Vulkan failed to find a Vulkan Runtime to use"

##### Problem:
The Vulkan loader "vulkan-1.dll" couldn't be found on your system.  This file is typically installed with some Vulkan driver installs,
some Vulkan-capable games, or the LunarG Vulkan SDK.

##### Possible Reason:
The last Vulkan Runtime install that executed on your system failed to behave properly.  Or, you have never installed a Vulkan loader
on your system.

##### Next Step:
See the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.


#### "ERROR: Unknown error while attempting to create Vulkan Instance"

##### Problem:
Some error occurred when trying to call Vulkan's vkCreateInstance function.

##### Possible Reason:
This can usually indicate that there's something about the installed driver that the Vulkan Loader does not like.

##### Next Step:
See the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.


#### "ERROR: Unknown error while attempting to create Vulkan Device"

##### Problem:
Some error occurred when trying to call Vulkan's vkCreateDevice function.

##### Possible Reason:
This can usually indicate that there's something about the installed driver that the Vulkan Loader does not like.

##### Next Step:
See the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.


#### "ERROR: Failed to find expected Vulkan Extensions"

##### Problem:
Some error occurred when trying to query the available drivers for the Vulkan Extensions that are supported.

##### Possible Reason:
This can usually indicate that there's something about the installed driver that the Vulkan Loader does not like.

##### Next Step:
See the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.


#### "ERROR: Vulkan Loader, Layer, or Driver ran out of memory"

##### Problem:
The Vulkan Loader, a Layer, or a Vulkan Driver failed to be able to allocate memory for a required item.

##### Possible Reason:
Check to see how much memory is available on your system.  It could also be caused by low hard-drive space.

##### Next Step:
Attempt to free up some hard-drive space if you're currently using more that 90% of your hard-drive.

Other things to try:
 * Follow the instructions in [Vulkan SDK Issues](#vulkan-sdk-issues) section below.
 * If that doesn't help, try [installing a new Vulkan Driver](#vulkan-graphics-driver-problems)

 
#### "ERROR: Unknown Test failure occurred"

##### Problem:
One of the Vulkan tests failed for some reason.

##### Possible Reason:
This could just indicate an error in setting up the test environment.  Currently, the only tests used are in the SDK, so
it's possible the SDK did not install properly.

##### Next Step:
See the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.


#### [WINDOWS] Dialog box pop's up indicating "vulkan-1.dll is missing from your computer."

##### Problem:
The Vulkan loader "vulkan-1.dll" couldn't be found on your system.  This file is typically installed with some Vulkan driver installs,
some Vulkan-capable games, or the LunarG Vulkan SDK.

##### Possible Reason:
The last Vulkan Runtime install that executed on your system failed to behave properly.  Or, you have never installed a Vulkan loader
on your system.

##### Next Step:
See the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.


#### [LINUX] Message indicating "error while loading shared libraries: libvulkan.so.1"

##### Problem:
The Vulkan loader "libvulkan.so.1" couldn't be found on your system.  This file is typically installed with some Vulkan driver installs,
some Vulkan-capable games, or the LunarG Vulkan SDK.

##### Possible Reason:
 1. There is no Vulkan loader installed to a system folder.
 2. The user didn't set up the LD_LIBRARY_PATH environment variable to point to a local folder containing libvulkan.so.1.
```
 export LD_LIBRARY_PATH=/home/<user>/sdk/VulkanSDK/1.0.39.0/x64_86/lib
```


##### Next Step:
See the [Vulkan SDK Issues](#vulkan-sdk-issues) section below.

<BR />


## Vulkan SDK Issues
If the problem you've encountered is possibly related to an SDK issue.  Visit [LunarXchange](https://vulkan.lunarg.com/), and install
the latest Vulkan SDK.  If that does not help, attempt to install a new [Vulkan Driver](#vulkan-graphics-driver-problems).  If that still
fails, file an issue on [LunarXchange](https://vulkan.lunarg.com/).  To file an issue, you may be required to create a free account
(only requires an email address).

<BR />


## Vulkan Graphics Driver Problems
If the problem is possibly related to your Graphics Driver, it could be for several reasons:
 1. The hardware you have doesn't support Vulkan.
 2. Your hardware supports Vulkan, but you haven't yet installed a driver with Vulkan support.
 3. There is no Vulkan driver with support for the OS on which you are currently running.
    - Sometimes, the company may provide Vulkan support for some devices on one Operatings System (say Windows), while still waiting to complete Vulkan on other systems.
 4. Everything supports Vulkan, but the driver failed to install properly.

Before approaching your Graphics driver vendor, it would help if you verified that your current driver for your current hardware on your current
operating system **does** support Vulkan.
