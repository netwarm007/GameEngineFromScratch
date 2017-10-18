add_executable(MyGameEngine OrbisApplication.cpp)
target_link_libraries(MyGameEngine Common)
set_target_properties(MyGameEngine PROPERTIES SUFFIX ".elf")
