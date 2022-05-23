add_library(Parser OGEX.cpp MGEMX.scanner.generated.cpp MGEMX.parser.generated.cpp)

target_link_libraries(Parser
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
)
