add_library(Common
        AudioClip.cpp
        Image.cpp
)

find_package(Threads REQUIRED)

target_link_libraries(Common
        Threads::Threads
)

