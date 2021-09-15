add_library(Common
        Image.cpp
        main.cpp
)

find_package(Threads REQUIRED)

target_link_libraries(Common
        Threads::Threads
)

