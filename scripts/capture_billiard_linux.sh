#!/bin/bash
set -eu

output_dir=${1:-billiard-screenshots}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
export LD_LIBRARY_PATH="$PWD/External/Linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

stdbuf -oL -eL ./build/Game/Billiard/BilliardGame -physicsRegression "$output_dir" \
    > "$output_dir/billiard.log" 2>&1 &
billiard_pid=$!
cleanup() {
    kill "$billiard_pid" 2>/dev/null || true
    wait "$billiard_pid" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 6000); do
    sleep 0.1
    if ! kill -0 "$billiard_pid" 2>/dev/null; then
        echo "Billiard exited before capture; see billiard.log" >&2
        exit 1
    fi
    test ! -s "$output_dir/physics-passed" || break
done
test -s "$output_dir/physics-passed"
import -window Billiard "$output_dir/billiard-after.png"
test -s "$output_dir/billiard-after.png"
identify "$output_dir/billiard-after.png"
touch "$output_dir/capture-complete"

for i in $(seq 1 12); do
    sleep 5
    kill -0 "$billiard_pid" 2>/dev/null || break
done
if kill -0 "$billiard_pid" 2>/dev/null; then
    echo "Billiard did not exit after capture; see billiard.log" >&2
    exit 1
fi
wait "$billiard_pid"
trap - EXIT
