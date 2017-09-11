# Be sure to run "Set-ExecutionPolicy RemoteSigned" before running powershell scripts

Param([switch]$Debug)

if ($Debug) {
    $dPath = "Debug"
    } else {
    $dPath = "Release"
}

trap
{
    Write-Error $_
    exit 1
}

function Exec
{
    param ($Command)
    & $dPath\$Command
    if ($LastExitCode -ne 0) {
        throw "Error running sample"
    }
}

if (Test-Path ..\loader\$dPath\vulkan-1.dll) {
    Copy-Item -force ..\loader\$dPath\vulkan-1.dll $dPath
}

echo "Initialize Instance"
Exec "01-init_instance"
echo "Enumerate Devices"
Exec "02-enumerate_devices"
echo "Initialize Device"
Exec "03-init_device"
echo "Initialize Command Buffer"
Exec "04-init_command_buffer"
echo "Initialize Swapchain"
Exec "05-init_swapchain"
echo "Initialize Depth Buffer"
Exec "06-init_depth_buffer"
echo "Initialize Uniform Buffer"
Exec "07-init_uniform_buffer"
echo "Initialize Pipeline Layout"
Exec "08-init_pipeline_layout"
echo "Initialize Descriptor Set"
Exec "09-init_descriptor_set"
echo "Initialize Render Pass"
Exec "10-init_render_pass"
echo "Initialize Shaders"
Exec "11-init_shaders"
echo "Initialize Frame Buffers"
Exec "12-init_frame_buffers"
echo "Initialize Vertex Buffer"
Exec "13-init_vertex_buffer"
echo "Initialize Pipeline"
Exec "14-init_pipeline"
echo "Draw Cube"
Exec "15-draw_cube"
echo "Instance Layer Properties"
Exec "instance_layer_properties"
echo "Instance Extension Properties"
Exec "instance_extension_properties"
echo "Instance Layer ExtensionProperties"
Exec "instance_layer_extension_properties"
echo "Enable Validation and Debug Message Callback"
Exec "enable_validation_with_callback"
echo "Events"
Exec "events"
echo "Create Debug Report Callback"
Exec "create_debug_report_callback"
echo "Initialize Texture"
Exec "init_texture"
echo "Copy/Blit Image"
Exec "copy_blit_image"
echo "Draw Textured Cube"
Exec "draw_textured_cube"
echo "Draw Cubes with Dynamic Uniform Buffer"
Exec "dynamic_uniform"
echo "Texel Buffer"
Exec "texel_buffer"
echo "Immutable Sampler"
Exec "immutable_sampler"
echo "Input Attachment"
Exec "input_attachment"
echo "Memory Barriers"
Exec "memory_barriers"
echo "Multiple Descriptor Sets"
Exec "multiple_sets"
echo "Multithreaded Command Buffers"
Exec "multithreaded_command_buffers"
echo "Push Constants"
Exec "push_constants"
echo "Push Descriptors"
Exec "push_descriptors"
echo "Separate image sampler"
Exec "separate_image_sampler"
echo "Draw Sub-passes"
Exec "draw_subpasses"
echo "Occlusion Query"
Exec "occlusion_query"
echo "Pipeline Cache"
Exec "pipeline_cache"
echo "Pipeline Derivative"
Exec "pipeline_derivative"
echo "Secondary Command Buffers"
Exec "secondary_command_buffer"
echo "SPIR-V Assembly"
Exec "spirv_assembly"
echo "SPIR-V Specialization"
Exec "spirv_specialization"
