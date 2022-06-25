#!/bin/bash
set -e
git submodule update --init External/src/Smouking
cp -v External/src/Smouking/half.h External/`uname -s`/include/