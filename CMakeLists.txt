cmake_minimum_required(VERSION 3.24)

project(unlimitedocr
    VERSION 0.1.0
    DESCRIPTION "Metal-first C inference core for baidu/Unlimited-OCR"
    LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(UOCR_ENABLE_METAL "AUTO" CACHE STRING "Enable Metal backend: AUTO, ON, or OFF")
set_property(CACHE UOCR_ENABLE_METAL PROPERTY STRINGS AUTO ON OFF)
option(UOCR_ENABLE_CUDA "Enable CUDA backend (placeholder; not implemented)" OFF)
option(UOCR_ENABLE_CPU_REF "Enable CPU reference backend" ON)
option(UOCR_BUILD_TOOLS "Build native tools" ON)
option(UOCR_BUILD_TESTS "Build native tests" ON)
option(UOCR_METAL_RUNTIME_COMPILE "Compile Metal sources at runtime during development" ON)
option(UOCR_METAL_PRECOMPILE "Precompile Metal sources into a metallib" OFF)
option(UOCR_SANITIZE "Enable address/undefined sanitizers for supported compilers" OFF)

string(TOUPPER "${UOCR_ENABLE_METAL}" UOCR_ENABLE_METAL_UPPER)
if(NOT UOCR_ENABLE_METAL_UPPER STREQUAL "AUTO" AND
   NOT UOCR_ENABLE_METAL_UPPER STREQUAL "ON" AND
   NOT UOCR_ENABLE_METAL_UPPER STREQUAL "OFF")
    message(FATAL_ERROR "UOCR_ENABLE_METAL must be AUTO, ON, or OFF")
endif()

set(UOCR_METAL_ENABLED OFF)
if(UOCR_ENABLE_METAL_UPPER STREQUAL "ON")
    if(NOT APPLE)
        message(FATAL_ERROR "UOCR_ENABLE_METAL=ON requires macOS/Apple platforms")
    endif()
    set(UOCR_METAL_ENABLED ON)
elseif(UOCR_ENABLE_METAL_UPPER STREQUAL "AUTO" AND APPLE)
    set(UOCR_METAL_ENABLED ON)
endif()

if(UOCR_ENABLE_CUDA)
    message(FATAL_ERROR "CUDA backend is reserved for a later milestone")
endif()

function(uocr_apply_common_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang,GNU>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow>
        $<$<COMPILE_LANG_AND_ID:OBJC,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)
endfunction()

function(uocr_apply_sanitizers target)
    if(UOCR_SANITIZE AND CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

add_library(uocr_core STATIC
    src/core/uocr_alloc.c
    src/core/uocr_result.c
    src/model/uocr_model_file.c
    src/runtime/uocr_logits_processor.c
    src/runtime/uocr_memory.c
    src/runtime/uocr_request_validation.c
    src/runtime/uocr_sequence.c)
target_include_directories(uocr_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(uocr_core
    PUBLIC
        UOCR_HAVE_CPU_REF=$<BOOL:${UOCR_ENABLE_CPU_REF}>
        UOCR_HAVE_METAL=$<BOOL:${UOCR_METAL_ENABLED}>)
set_target_properties(uocr_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
uocr_apply_common_warnings(uocr_core)
uocr_apply_sanitizers(uocr_core)

if(UOCR_ENABLE_CPU_REF)
    target_sources(uocr_core PRIVATE src/backend/cpu_ref/uocr_cpu_ref.c)
endif()

if(UOCR_METAL_ENABLED)
    enable_language(OBJC)
    add_library(uocr_metal STATIC
        src/backend/metal/uocr_metal.m)
    target_include_directories(uocr_metal
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(uocr_metal
        PRIVATE
            UOCR_METAL_RUNTIME_COMPILE=$<BOOL:${UOCR_METAL_RUNTIME_COMPILE}>
            UOCR_DEFAULT_RESOURCE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal\")
    target_link_libraries(uocr_metal
        PUBLIC
            "-framework Foundation"
            "-framework Metal")
    set_target_properties(uocr_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
    uocr_apply_common_warnings(uocr_metal)
    uocr_apply_sanitizers(uocr_metal)
endif()

add_library(unlimitedocr SHARED
    src/core/uocr_api.c)
target_include_directories(unlimitedocr
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(unlimitedocr
    PRIVATE
        UOCR_BUILDING_SHARED=1)
target_link_libraries(unlimitedocr PRIVATE uocr_core)
if(UOCR_METAL_ENABLED)
    target_link_libraries(unlimitedocr PRIVATE uocr_metal)
endif()
set_target_properties(unlimitedocr PROPERTIES
    OUTPUT_NAME unlimitedocr
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON)
uocr_apply_common_warnings(unlimitedocr)
uocr_apply_sanitizers(unlimitedocr)

if(UOCR_BUILD_TOOLS)
    add_executable(uocr-dump tools/uocr_dump.c)
    target_include_directories(uocr-dump PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(uocr-dump PRIVATE unlimitedocr uocr_core)
    uocr_apply_common_warnings(uocr-dump)
    uocr_apply_sanitizers(uocr-dump)
endif()

if(UOCR_BUILD_TESTS)
    enable_testing()
    add_executable(test_api tests/test_api.c)
    target_include_directories(test_api PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_api PRIVATE unlimitedocr)
    uocr_apply_common_warnings(test_api)
    uocr_apply_sanitizers(test_api)
    add_test(NAME uocr_api COMMAND test_api)

    add_executable(test_uocr_model_file tests/test_uocr_model_file.c)
    target_include_directories(test_uocr_model_file PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_uocr_model_file PRIVATE uocr_core)
    uocr_apply_common_warnings(test_uocr_model_file)
    uocr_apply_sanitizers(test_uocr_model_file)
    add_test(NAME uocr_model_file COMMAND test_uocr_model_file)

    add_executable(test_alloc tests/test_alloc.c)
    target_include_directories(test_alloc PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_alloc PRIVATE uocr_core)
    uocr_apply_common_warnings(test_alloc)
    uocr_apply_sanitizers(test_alloc)
    add_test(NAME uocr_alloc COMMAND test_alloc)

    add_executable(test_memory tests/test_memory.c)
    target_include_directories(test_memory PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_memory PRIVATE uocr_core)
    uocr_apply_common_warnings(test_memory)
    uocr_apply_sanitizers(test_memory)
    add_test(NAME uocr_memory COMMAND test_memory)

    add_executable(test_result tests/test_result.c)
    target_include_directories(test_result PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_result PRIVATE uocr_core)
    uocr_apply_common_warnings(test_result)
    uocr_apply_sanitizers(test_result)
    add_test(NAME uocr_result COMMAND test_result)

    add_executable(test_logits_processor tests/test_logits_processor.c)
    target_include_directories(test_logits_processor PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_logits_processor PRIVATE uocr_core)
    uocr_apply_common_warnings(test_logits_processor)
    uocr_apply_sanitizers(test_logits_processor)
    add_test(NAME uocr_logits_processor COMMAND test_logits_processor)

    add_executable(test_sequence tests/test_sequence.c)
    target_include_directories(test_sequence PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_sequence PRIVATE uocr_core)
    uocr_apply_common_warnings(test_sequence)
    uocr_apply_sanitizers(test_sequence)
    add_test(NAME uocr_sequence COMMAND test_sequence)

    if(UOCR_METAL_ENABLED AND UOCR_METAL_RUNTIME_COMPILE)
        add_executable(test_metal tests/test_metal.c)
        target_include_directories(test_metal PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_compile_definitions(test_metal PRIVATE
            UOCR_TEST_METAL_RESOURCE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal\")
        target_link_libraries(test_metal PRIVATE unlimitedocr uocr_metal uocr_core)
        uocr_apply_common_warnings(test_metal)
        uocr_apply_sanitizers(test_metal)
        add_test(NAME uocr_metal COMMAND test_metal)
    endif()
endif()

install(TARGETS unlimitedocr
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib)
install(FILES include/unlimitedocr.h DESTINATION include)
install(DIRECTORY src/backend/metal/kernels DESTINATION share/unlimitedocr/metal FILES_MATCHING PATTERN "*.metal")
