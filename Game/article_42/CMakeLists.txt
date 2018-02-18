IF(ANDROID)
add_library(article_42 SHARED 
    article_42_logic.cpp article_42_config.cpp)
ELSEIF(APPLE)
add_executable(article_42 MACOSX_BUNDLE
    article_42_logic.cpp article_42_config.cpp)
ELSE(ANDROID)
add_executable(article_42
    article_42_logic.cpp article_42_config.cpp)
ENDIF(ANDROID)
target_link_libraries(article_42
                        MyPhysics
                        MyGameEngine
                     )
IF(WIN32)
add_custom_command(TARGET article_42 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:article_42>)
ENDIF(WIN32)


