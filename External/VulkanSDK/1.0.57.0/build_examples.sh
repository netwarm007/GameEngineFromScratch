#!/bin/bash

set -ex

# Build the samples.
pushd examples
cmake -H. -Bbuild
make -j`nproc` -C build
popd
