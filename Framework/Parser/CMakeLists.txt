find_package(BISON)
BISON_TARGET(MGEMXParser MGEMX.y ${CMAKE_CURRENT_BINARY_DIR}/MGEMX.parser.generated.cpp)

find_package(FLEX)
if(OS_WINDOWS)
set(FLEX_COMPILE_FLAGS "--wincompat")
endif(OS_WINDOWS)

FLEX_TARGET(MGEMXScanner MGEMX.l ${CMAKE_CURRENT_BINARY_DIR}/MGEMX.scanner.generated.cpp
    COMPILE_FLAGS ${FLEX_COMPILE_FLAGS}
)

add_library(Parser OGEX.cpp
    ${BISON_MGEMXParser_OUTPUTS}
    ${FLEX_MGEMXScanner_OUTPUTS})

target_link_libraries(Parser
    ${OPENGEX_LIBRARY}
    ${OPENDDL_LIBRARY}
)
