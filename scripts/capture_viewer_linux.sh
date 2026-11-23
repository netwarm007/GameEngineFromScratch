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

# GEFS_CAPTURE_WAIT_LOOPS extends the fixed render window (5s per loop,
# capture at half); switch runs with slow software rasterizers need more
# time for the splash/main-scene sequence to complete after reconstruction.
wait_loops=${GEFS_CAPTURE_WAIT_LOOPS:-24}
for i in $(seq 1 "$wait_loops"); do
    sleep 5
    if ! kill -0 "$viewer_pid" 2>/dev/null; then
        echo "Viewer exited before capture; see viewer.log" >&2
        exit 1
    fi
    if [ -n "${GEFS_SCREENSHOT_WAIT_FOR:-}" ]; then
        if [ -s "$GEFS_SCREENSHOT_WAIT_FOR" ]; then break; fi
    elif [ "$i" -eq $((wait_loops / 2)) ]; then
        break
    fi
done

if [ -n "${GEFS_SCREENSHOT_WAIT_FOR:-}" ] && [ ! -s "$GEFS_SCREENSHOT_WAIT_FOR" ]; then
    echo "Viewer readiness marker timed out; see viewer.log" >&2
    exit 1
fi

window_id=""
for i in $(seq 1 10); do
    # Match the main render window by its title prefix "Viewer -" (the live
    # backend title), NOT the WM_CLASS "Viewer": after a failed switch the
    # failure dialog is a separate SDL viewport window titled "切换失败" that
    # shares the "Viewer" WM_CLASS and stacks topmost, so a class/prefix match
    # on "Viewer" alone grabs the 150x164 dialog instead of the scene.
    window_id=$(xwininfo -root -tree 2>/dev/null | grep -m1 '"Viewer -' | awk '{print $1}') || true
    [ -n "$window_id" ] && break
    sleep 1
done
if [ -z "$window_id" ]; then
    echo "Viewer window not found; see viewer.log" >&2
    echo "=== process state ===" >&2
    if kill -0 "$viewer_pid" 2>/dev/null; then echo "viewer alive" >&2; else echo "viewer EXITED" >&2; fi
    echo "=== xwininfo -root -tree ===" >&2
    xwininfo -root -tree 2>&1 | head -40 >&2 || true
    exit 1
fi
import -window "$window_id" "$output_dir/viewer-render.png"
test -s "$output_dir/viewer-render.png"
identify "$output_dir/viewer-render.png"
