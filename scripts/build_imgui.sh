#!/bin/bash
set -e
git submodule update --init External/src/imgui
# source only module, no need to build