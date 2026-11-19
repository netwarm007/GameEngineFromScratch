#!/bin/bash
set -eu

output=${1:-vulkan-applications}
mkdir -p "$output"
output=$(cd "$output" && pwd)
platform=$(uname -s)
if [ "$platform" != Darwin ]; then
    export LD_LIBRARY_PATH="$PWD/External/Linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

for application in Viewer BilliardGame; do
    if [ "$application" = Viewer ]; then
        executable=./build-vulkan-apps/Viewer/Viewer
    else
        executable=./build-vulkan-apps/Game/Billiard/BilliardGame
    fi
    if [ "$platform" = Darwin ]; then
        executable="$executable.app/Contents/MacOS/$application"
        "$executable" -screenshot "$output/$application.ppm" > "$output/$application.log" 2>&1 &
        application_pid=$!
        trap 'kill "$application_pid" 2>/dev/null || true' EXIT
        for attempt in $(seq 1 120); do
            sleep 2
            kill -0 "$application_pid" 2>/dev/null || break
        done
        if kill -0 "$application_pid" 2>/dev/null; then
            echo "$application Vulkan capture timed out" >&2
            exit 1
        fi
        wait "$application_pid"
        trap - EXIT
    else
        timeout 240s xvfb-run -a "$executable" -screenshot "$output/$application.ppm" \
            > "$output/$application.log" 2>&1
    fi
    test -s "$output/$application.ppm"
    test "$(head -c 2 "$output/$application.ppm")" = P6
    grep -q "Vulkan application framebuffer capture saved:" "$output/$application.log"
    if grep -Eq "Error opening file 'Materials/|Texture unavailable: Materials/|Texture decoding failed: Materials/" "$output/$application.log"; then
        echo "$application Vulkan capture used missing or invalid scene materials" >&2
        exit 1
    fi
    if grep -Eq 'Validation Error|VUID-' "$output/$application.log"; then
        echo "$application Vulkan capture reported validation errors" >&2
        exit 1
    fi
done
