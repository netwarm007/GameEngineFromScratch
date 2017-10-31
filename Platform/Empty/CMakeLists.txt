add_executable(Empty EmptyApplication.cpp)
target_link_libraries(Empty Common ${XG_LIBRARY} ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY})
__add_xg_platform_dependencies(Empty)
