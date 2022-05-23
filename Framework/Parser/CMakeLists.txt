add_library(Parser OGEX.cpp MGEMX.scanner.generated.cpp MGEMX.parser.generated.cpp)

IF(WIN32)
    set(FLEX win_flex.exe)
    set(BISON win_bison.exe)
ELSE(WIN32)
    set(FLEX flex)
    set(BISON bison)
ENDIF(WIN32)

add_custom_command(OUTPUT MGEMX.scanner.generated.cpp 
	COMMAND ${FLEX} MGEMX.l
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_command(OUTPUT MGEMX.parser.generated.cpp 
    COMMAND ${BISON} MGEMX.y
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(Parser
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
)
