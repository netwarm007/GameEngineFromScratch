add_executable(Empty EmptyApplication.cpp)
target_link_libraries(Empty Common  ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY} ${XG_LIBRARY})
__add_xg_platform_dependencies(Empty)
