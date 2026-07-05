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
option(UOCR_ENABLE_PROFILING "Enable profiling instrumentation in the Metal backend" OFF)
option(UOCR_ENABLE_CUDA "Enable CUDA backend (placeholder; not implemented)" OFF)
option(UOCR_ENABLE_CPU_REF "Enable CPU reference backend" ON)
option(UOCR_BUILD_TOOLS "Build native tools" ON)
option(UOCR_BUILD_TESTS "Build native tests" ON)
option(UOCR_METAL_RUNTIME_COMPILE "Compile Metal sources at runtime during development" ON)
option(UOCR_METAL_PRECOMPILE "Precompile Metal sources into a metallib" OFF)
option(UOCR_METAL_ENABLE_MPP_TENSOROPS "Build the experimental Metal 4 MPP TensorOps matmul prototype" OFF)
set(UOCR_METAL_MATH_MODE "FAST" CACHE STRING "Metal math mode: FAST, RELAXED, or SAFE")
set_property(CACHE UOCR_METAL_MATH_MODE PROPERTY STRINGS FAST RELAXED SAFE)
option(UOCR_SANITIZE "Enable address/undefined sanitizers for supported compilers" OFF)

string(TOUPPER "${UOCR_ENABLE_METAL}" UOCR_ENABLE_METAL_UPPER)
if(NOT UOCR_ENABLE_METAL_UPPER STREQUAL "AUTO" AND
   NOT UOCR_ENABLE_METAL_UPPER STREQUAL "ON" AND
   NOT UOCR_ENABLE_METAL_UPPER STREQUAL "OFF")
    message(FATAL_ERROR "UOCR_ENABLE_METAL must be AUTO, ON, or OFF")
endif()

string(TOUPPER "${UOCR_METAL_MATH_MODE}" UOCR_METAL_MATH_MODE_UPPER)
if(UOCR_METAL_MATH_MODE_UPPER STREQUAL "FAST")
    set(UOCR_METAL_MATH_MODE_VALUE 2)
    set(UOCR_METAL_MATH_MODE_FLAG fast)
    set(UOCR_METAL_MATH_FP32_FUNCTIONS_FLAG fast)
    set(UOCR_METAL_MATH_FP32_FAST_VALUE 1)
elseif(UOCR_METAL_MATH_MODE_UPPER STREQUAL "RELAXED")
    set(UOCR_METAL_MATH_MODE_VALUE 1)
    set(UOCR_METAL_MATH_MODE_FLAG relaxed)
    set(UOCR_METAL_MATH_FP32_FUNCTIONS_FLAG fast)
    set(UOCR_METAL_MATH_FP32_FAST_VALUE 1)
elseif(UOCR_METAL_MATH_MODE_UPPER STREQUAL "SAFE")
    set(UOCR_METAL_MATH_MODE_VALUE 0)
    set(UOCR_METAL_MATH_MODE_FLAG safe)
    set(UOCR_METAL_MATH_FP32_FUNCTIONS_FLAG precise)
    set(UOCR_METAL_MATH_FP32_FAST_VALUE 0)
else()
    message(FATAL_ERROR "UOCR_METAL_MATH_MODE must be FAST, RELAXED, or SAFE")
endif()
set(UOCR_METAL_MATH_FLAGS
    "-fmetal-math-mode=${UOCR_METAL_MATH_MODE_FLAG}"
    "-fmetal-math-fp32-functions=${UOCR_METAL_MATH_FP32_FUNCTIONS_FLAG}")

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
    src/runtime/uocr_profile.c
    src/runtime/uocr_request_validation.c
    src/runtime/uocr_sequence.c
    src/runtime/uocr_vision.c)
target_include_directories(uocr_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(uocr_core
    PUBLIC
        UOCR_HAVE_CPU_REF=$<BOOL:${UOCR_ENABLE_CPU_REF}>
        UOCR_HAVE_METAL=$<BOOL:${UOCR_METAL_ENABLED}>
    PRIVATE
        UOCR_ENABLE_PROFILING=$<BOOL:${UOCR_ENABLE_PROFILING}>)
set_target_properties(uocr_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
uocr_apply_common_warnings(uocr_core)
uocr_apply_sanitizers(uocr_core)

if(UOCR_ENABLE_CPU_REF)
    target_sources(uocr_core PRIVATE src/backend/cpu_ref/uocr_cpu_ref.c)
    if(NOT APPLE AND NOT WIN32)
        target_link_libraries(uocr_core PUBLIC m)
    endif()
endif()

if(UOCR_METAL_ENABLED)
    enable_language(OBJC)
    set(UOCR_METAL_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels")
    set(UOCR_METALLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/unlimitedocr.metallib")
    file(GLOB UOCR_METAL_SOURCES "${UOCR_METAL_KERNEL_DIR}/*.metal")
    if(UOCR_METAL_ENABLE_MPP_TENSOROPS)
        set(UOCR_METAL_ENABLE_MPP_TENSOROPS_VALUE 1)
    else()
        set(UOCR_METAL_ENABLE_MPP_TENSOROPS_VALUE 0)
    endif()

    if(UOCR_METAL_PRECOMPILE)
        find_program(UOCR_XCRUN_EXECUTABLE xcrun)
        if(NOT UOCR_XCRUN_EXECUTABLE)
            message(FATAL_ERROR "UOCR_METAL_PRECOMPILE=ON requires xcrun")
        endif()

        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(UOCR_METAL_COMBINED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/uocr_all.metal")
        set(UOCR_METAL_AIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/uocr_all.air")

        add_custom_command(
            OUTPUT "${UOCR_METAL_COMBINED_SOURCE}"
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_metal.py"
                --kernels "${UOCR_METAL_KERNEL_DIR}"
                --output "${UOCR_METAL_COMBINED_SOURCE}"
            DEPENDS ${UOCR_METAL_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_metal.py"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Generating combined Metal source → uocr_all.metal"
            VERBATIM)

        add_custom_command(
            OUTPUT "${UOCR_METAL_AIR_FILE}"
            COMMAND "${UOCR_XCRUN_EXECUTABLE}" -sdk macosx metal
                -I"${UOCR_METAL_KERNEL_DIR}"
                -DUOCR_METAL_ENABLE_MPP_TENSOROPS=${UOCR_METAL_ENABLE_MPP_TENSOROPS_VALUE}
                ${UOCR_METAL_MATH_FLAGS}
                -c "${UOCR_METAL_COMBINED_SOURCE}" -o "${UOCR_METAL_AIR_FILE}"
            DEPENDS "${UOCR_METAL_COMBINED_SOURCE}"
            COMMENT "Compiling combined Metal source → uocr_all.air"
            VERBATIM)

        add_custom_command(
            OUTPUT "${UOCR_METALLIB_OUTPUT}"
            COMMAND "${UOCR_XCRUN_EXECUTABLE}" -sdk macosx metallib "${UOCR_METAL_AIR_FILE}" -o "${UOCR_METALLIB_OUTPUT}"
            DEPENDS "${UOCR_METAL_AIR_FILE}"
            COMMENT "Linking Metal library → unlimitedocr.metallib"
            VERBATIM)
        add_custom_target(unlimitedocr_metallib ALL DEPENDS "${UOCR_METALLIB_OUTPUT}")
    endif()

    add_library(uocr_metal STATIC
        src/backend/metal/uocr_metal.m)
    if(UOCR_METAL_PRECOMPILE)
        add_dependencies(uocr_metal unlimitedocr_metallib)
    endif()
    target_include_directories(uocr_metal
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(uocr_metal
        PRIVATE
            UOCR_ENABLE_PROFILING=$<BOOL:${UOCR_ENABLE_PROFILING}>
            UOCR_METAL_RUNTIME_COMPILE=$<BOOL:${UOCR_METAL_RUNTIME_COMPILE}>
            UOCR_METAL_ENABLE_MPP_TENSOROPS=${UOCR_METAL_ENABLE_MPP_TENSOROPS_VALUE}
            UOCR_METAL_MATH_MODE=${UOCR_METAL_MATH_MODE_VALUE}
            UOCR_METAL_MATH_FP32_FAST=${UOCR_METAL_MATH_FP32_FAST_VALUE}
            UOCR_DEFAULT_RESOURCE_PATH=\"${UOCR_METAL_KERNEL_DIR}\"
            UOCR_DEFAULT_METALLIB_PATH=\"${UOCR_METALLIB_OUTPUT}\")
    target_link_libraries(uocr_metal
        PUBLIC
            "-framework Foundation"
            "-framework Metal"
            "-framework MetalPerformanceShaders")
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
        UOCR_BUILDING_SHARED=1
        UOCR_ENABLE_PROFILING=$<BOOL:${UOCR_ENABLE_PROFILING}>)
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

    add_executable(test_constants tests/test_constants.c)
    target_include_directories(test_constants PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_constants PRIVATE uocr_core)
    uocr_apply_common_warnings(test_constants)
    uocr_apply_sanitizers(test_constants)
    add_test(NAME uocr_constants COMMAND test_constants)

    add_executable(test_uocr_model_file tests/test_uocr_model_file.c)
    target_include_directories(test_uocr_model_file PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_uocr_model_file PRIVATE uocr_core)
    uocr_apply_common_warnings(test_uocr_model_file)
    uocr_apply_sanitizers(test_uocr_model_file)
    add_test(NAME uocr_model_file COMMAND test_uocr_model_file)

    if(UOCR_BUILD_TOOLS)
        add_executable(test_uocr_dump tests/test_uocr_dump.c)
        target_include_directories(test_uocr_dump PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        uocr_apply_common_warnings(test_uocr_dump)
        uocr_apply_sanitizers(test_uocr_dump)
        add_test(NAME uocr_dump COMMAND test_uocr_dump $<TARGET_FILE:uocr-dump>)
    endif()

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

    add_executable(test_request_validation tests/test_request_validation.c)
    target_include_directories(test_request_validation PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_request_validation PRIVATE uocr_core)
    uocr_apply_common_warnings(test_request_validation)
    uocr_apply_sanitizers(test_request_validation)
    add_test(NAME uocr_request_validation COMMAND test_request_validation)

    add_executable(test_sequence tests/test_sequence.c)
    target_include_directories(test_sequence PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_sequence PRIVATE uocr_core)
    uocr_apply_common_warnings(test_sequence)
    uocr_apply_sanitizers(test_sequence)
    add_test(NAME uocr_sequence COMMAND test_sequence)

    add_executable(test_vision tests/test_vision.c)
    target_include_directories(test_vision PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_vision PRIVATE uocr_core)
    uocr_apply_common_warnings(test_vision)
    uocr_apply_sanitizers(test_vision)
    add_test(NAME uocr_vision COMMAND test_vision)

    if(UOCR_ENABLE_CPU_REF)
        add_executable(test_cpu_ref tests/test_cpu_ref.c)
        target_include_directories(test_cpu_ref PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(test_cpu_ref PRIVATE uocr_core)
        uocr_apply_common_warnings(test_cpu_ref)
        uocr_apply_sanitizers(test_cpu_ref)
        add_test(NAME uocr_cpu_ref COMMAND test_cpu_ref)
    endif()

endif()

install(TARGETS unlimitedocr
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib)
install(FILES include/unlimitedocr.h DESTINATION include)
install(DIRECTORY src/backend/metal/kernels DESTINATION share/unlimitedocr/metal FILES_MATCHING PATTERN "*.metal")
if(UOCR_METAL_ENABLED AND UOCR_METAL_PRECOMPILE)
    install(FILES "${UOCR_METALLIB_OUTPUT}" DESTINATION share/unlimitedocr/metal)
endif()
