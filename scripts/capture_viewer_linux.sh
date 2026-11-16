#!/bin/bash
set -eu

output_dir=${1:-screenshots}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
export LD_LIBRARY_PATH="$PWD/External/Linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

glxinfo -B > "$output_dir/diagnostics.txt" 2>&1
viewer_command=(./build/Viewer/Viewer)
if [[ "${GEFS_CAPTURE_GDB:-0}" == 1 ]]; then
    viewer_command=(gdb --quiet --batch -ex 'set pagination off' -ex run \
        -ex 'thread apply all bt' --args "${viewer_command[0]}")
fi
stdbuf -oL -eL "${viewer_command[@]}" > "$output_dir/viewer.log" 2>&1 &
viewer_pid=$!
cleanup() {
    kill "$viewer_pid" 2>/dev/null || true
    wait "$viewer_pid" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 12); do
    sleep 5
    if ! kill -0 "$viewer_pid" 2>/dev/null; then
        echo "Viewer exited before capture; see viewer.log" >&2
        exit 1
    fi
done

import -window Viewer "$output_dir/viewer-render.png"
test -s "$output_dir/viewer-render.png"
identify "$output_dir/viewer-render.png"
