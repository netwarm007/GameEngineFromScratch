add_library(Parser OGEX.cpp)

target_link_libraries(Parser
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
)
