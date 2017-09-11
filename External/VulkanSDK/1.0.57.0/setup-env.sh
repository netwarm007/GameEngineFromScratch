# source this file into an existing shell.

export VULKAN_SDK="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/x86_64
export PATH="$VULKAN_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
export VK_LAYER_PATH="$VULKAN_SDK/etc/explicit_layer.d"

