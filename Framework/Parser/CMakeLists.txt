add_library(Parser OGEX.cpp 
    MGEMX.scanner.generated.cpp 
    MGEMX.parser.generated.cpp)

set_property(SOURCE MGEMX.scanner.generated.cpp PROPERTY GENERATED)
set_property(SOURCE MGEMX.parser.generated.cpp PROPERTY GENERATED)

IF(WIN32)
    set(FLEX win_flex.exe --wincompat)
    set(BISON win_bison.exe)
ELSE(WIN32)
    set(FLEX flex)
    set(BISON bison)
ENDIF(WIN32)

add_custom_command(OUTPUT MGEMX.scanner.generated.cpp 
	COMMAND ${FLEX} ${CMAKE_CURRENT_LIST_DIR}/MGEMX.l
    DEPENDS MGEMX.l
)

add_custom_command(OUTPUT MGEMX.parser.generated.cpp 
    COMMAND ${BISON} ${CMAKE_CURRENT_LIST_DIR}/MGEMX.y
    DEPENDS MGEMX.y
)

target_link_libraries(Parser
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
)
