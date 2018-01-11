add_executable(MyGameEngineOrbis.elf OrbisApplication.cpp)
target_link_libraries(MyGameEngineOrbis.elf Common ${XG_LIBRARY} ${OPENGEX_LIBRARY} ${OPENDDL_LIBRARY} ${ZLIB_LIBRARY})
__add_xg_platform_dependencies(MyGameEngineOrbis.elf)
