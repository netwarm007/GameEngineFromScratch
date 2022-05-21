#!/bin/bash
./External/Darwin/bin/flatc --cpp --cpp-std 'c++17' -o ./Framework/Common/ ./Asset/Schema/RenderDefinitions.fbs
./External/Darwin/bin/flatc -b -o ./Asset/Data --schema ./Asset/Schema/RenderDefinitions.fbs
