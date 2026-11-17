#!/bin/bash
set -eu

output=${1:-vulkan-applications}
mkdir -p "$output"
output=$(cd "$output" && pwd)
export LD_LIBRARY_PATH="$PWD/External/Linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for application in Viewer BilliardGame; do
    if [ "$application" = Viewer ]; then
        executable=./build-vulkan-apps/Viewer/Viewer
    else
        executable=./build-vulkan-apps/Game/Billiard/BilliardGame
    fi
    timeout 240s xvfb-run -a "$executable" -screenshot "$output/$application.ppm" \
        > "$output/$application.log" 2>&1
    test -s "$output/$application.ppm"
    test "$(head -c 2 "$output/$application.ppm")" = P6
    grep -q "Vulkan application framebuffer capture saved:" "$output/$application.log"
    if grep -Eq 'Validation Error|VUID-' "$output/$application.log"; then
        echo "$application Vulkan capture reported validation errors" >&2
        exit 1
    fi
done
