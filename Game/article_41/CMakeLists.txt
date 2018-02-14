IF(ANDROID)
add_library(article_41 SHARED 
    article_41_logic.cpp article_41_config.cpp)
ELSEIF(APPLE)
add_executable(article_41 MACOSX_BUNDLE
    article_41_logic.cpp article_41_config.cpp)
ELSE(ANDROID)
add_executable(article_41
    article_41_logic.cpp article_41_config.cpp)
ENDIF(ANDROID)
target_link_libraries(article_41
                        MyGameEngine
                        MyPhysics
                     )
IF(WIN32)
add_custom_command(TARGET article_41 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:article_41>)
ENDIF(WIN32)


