add_library(OpenGEX
OpenGEX.cpp
)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/../OpenDDL
)

set_target_properties(OpenGEX
PROPERTIES PUBLIC_HEADER OpenGEX.h
)

install(TARGETS OpenGEX
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

