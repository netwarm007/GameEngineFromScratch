add_library(MyGameEngine SHARED AndroidApplication.cpp)
target_link_libraries(MyGameEngine Common ${XG_LIBRARY} ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY} ${ZLIB_LIBRARY})
__add_xg_platform_dependencies(MyGameEngine)

add_subdirectory(app/src/main/cpp)

