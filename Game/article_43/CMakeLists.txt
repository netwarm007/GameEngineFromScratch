IF(ANDROID)
add_library(article_43 SHARED 
    article_43_logic.cpp article_43_config.cpp)
ELSEIF(APPLE)
add_executable(article_43 MACOSX_BUNDLE
    article_43_logic.cpp article_43_config.cpp)
ELSE(ANDROID)
add_executable(article_43
    article_43_logic.cpp article_43_config.cpp)
ENDIF(ANDROID)
target_link_libraries(article_43
                        MyPhysics
                        MyGameEngine
                     )
IF(WIN32)
add_custom_command(TARGET article_43 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/External/Windows/bin/zlib.dll"
        $<TARGET_FILE_DIR:article_43>)
ENDIF(WIN32)


