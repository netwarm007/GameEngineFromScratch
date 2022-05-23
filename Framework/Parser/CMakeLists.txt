add_library(Parser OGEX.cpp MGEMX.scanner.cpp MGEMX.parser.cpp)

target_link_libraries(Parser
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
)
