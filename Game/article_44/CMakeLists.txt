IF(ANDROID)
add_library(article_44 SHARED 
    article_44_logic.cpp article_44_config.cpp)
ELSEIF(APPLE)
add_executable(article_44 MACOSX_BUNDLE
    article_44_logic.cpp article_44_config.cpp)
ELSE(ANDROID)
add_executable(article_44
    article_44_logic.cpp article_44_config.cpp)
ENDIF(ANDROID)
target_link_libraries(article_44
                        MyPhysics
                        MyGameEngine
                     )
IF(WIN32)
add_custom_command(TARGET article_44 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:article_44>)
ENDIF(WIN32)


