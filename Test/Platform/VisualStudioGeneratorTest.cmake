set(mock_dir "${BINARY_DIR}/VisualStudioGeneratorTest")
file(MAKE_DIRECTORY "${mock_dir}")
file(WRITE "${mock_dir}/cmake.cmd" [=[
@echo off
if defined CMAKE_GENERATOR (
  echo * Ninja = mock
) else if "%MOCK_VS%"=="none" (
  echo * Ninja = mock
) else (
  echo * Visual Studio %MOCK_VS% = mock
)
]=])
file(TO_NATIVE_PATH "${SOURCE_DIR}/scripts/select-vs-generator.bat" selector)
file(TO_NATIVE_PATH "${mock_dir}/run.cmd" runner)
file(WRITE "${runner}"
    "@echo off\ncall \"${selector}\"\nif errorlevel 1 exit /b 1\necho selected=%GEFS_VS_GENERATOR%\n")

function(check_generator version)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "PATH=${mock_dir};$ENV{PATH}"
            "CMAKE_GENERATOR=Ninja"
            "MOCK_VS=${version}"
            "$ENV{ComSpec}" /d /c "${runner}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(version STREQUAL "none")
        if(result STREQUAL "0" OR NOT error MATCHES "could not find")
            message(FATAL_ERROR "Missing Visual Studio was not reported: ${output}\n${error}")
        endif()
    elseif(NOT result STREQUAL "0" OR NOT output MATCHES "selected=Visual Studio ${version}")
        message(FATAL_ERROR "Visual Studio ${version} was not selected: ${output}\n${error}")
    endif()
endfunction()

check_generator("16 2019")
check_generator("17 2022")
check_generator("18 2026")
check_generator("none")
