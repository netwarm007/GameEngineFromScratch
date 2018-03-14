IF(ANDROID)
add_library(article_45 SHARED 
    article_45_logic.cpp article_45_config.cpp)
ELSEIF(APPLE)
add_executable(article_45 MACOSX_BUNDLE
    article_45_logic.cpp article_45_config.cpp)
ELSE(ANDROID)
add_executable(article_45
    article_45_logic.cpp article_45_config.cpp)
ENDIF(ANDROID)
IF(WIN32)
target_link_libraries(article_45
                        MyPhysics
                        MyGameEngineOpenGL
                     )
add_custom_command(TARGET article_45 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:article_45>)
ELSE(WIN32)
target_link_libraries(article_45
                        MyPhysics
                        MyGameEngine
                     )
ENDIF(WIN32)


