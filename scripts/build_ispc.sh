#!/bin/bash
set -e
cmake -P "$(dirname "$0")/install-ispc.cmake"

