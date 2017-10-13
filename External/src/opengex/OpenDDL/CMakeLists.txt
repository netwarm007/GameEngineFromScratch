add_library(OpenDDL
ODDLMap.cpp
ODDLString.cpp
ODDLTree.cpp
OpenDDL.cpp
)

set_target_properties(OpenDDL
PROPERTIES PUBLIC_HEADER OpenDDL.h
)

install(TARGETS OpenDDL
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

