#!/bin/bash
set -e
mkdir -p build/Asset/Materials
pushd build/Asset/Materials
../../Utility/MaterialBaker
popd