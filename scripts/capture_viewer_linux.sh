#!/bin/bash
set -eu

output_dir=${1:-screenshots}
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
export LD_LIBRARY_PATH="$PWD/External/Linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

glxinfo -B > "$output_dir/diagnostics.txt" 2>&1
stdbuf -oL -eL ./build/Viewer/Viewer "$@" > "$output_dir/viewer.log" 2>&1 &
viewer_pid=$!
cleanup() {
    kill "$viewer_pid" 2>/dev/null || true
    wait "$viewer_pid" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 24); do
    sleep 5
    if ! kill -0 "$viewer_pid" 2>/dev/null; then
        echo "Viewer exited before capture; see viewer.log" >&2
        exit 1
    fi
    if [ -n "${GEFS_SCREENSHOT_WAIT_FOR:-}" ]; then
        if [ -s "$GEFS_SCREENSHOT_WAIT_FOR" ]; then break; fi
    elif [ "$i" -eq 12 ]; then
        break
    fi
done

if [ -n "${GEFS_SCREENSHOT_WAIT_FOR:-}" ] && [ ! -s "$GEFS_SCREENSHOT_WAIT_FOR" ]; then
    echo "Viewer readiness marker timed out; see viewer.log" >&2
    exit 1
fi

import -window Viewer "$output_dir/viewer-render.png"
test -s "$output_dir/viewer-render.png"
identify "$output_dir/viewer-render.png"
