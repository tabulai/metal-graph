# metal-graph — Metal-native graph analytics for Apple Silicon.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)

# pyproject.toml is the single source of truth for the release version.
# Parse only the [project] table and require a numeric CMake-compatible
# version so Python metadata, the extension, and the C ABI cannot drift.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _mg_pyproject_lines)
set(_mg_in_project_table FALSE)
set(_mg_project_versions)
foreach(_line IN LISTS _mg_pyproject_lines)
  if(_line MATCHES "^[ \t]*\\[project\\][ \t]*$")
    set(_mg_in_project_table TRUE)
  elseif(_line MATCHES "^[ \t]*\\[")
    set(_mg_in_project_table FALSE)
  elseif(_mg_in_project_table AND _line MATCHES "^[ \t]*version[ \t]*=")
    if(NOT _line MATCHES
       "^[ \t]*version[ \t]*=[ \t]*\"((0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*))\"[ \t]*$")
      message(FATAL_ERROR
        "[project].version in pyproject.toml must use numeric X.Y.Z form.")
    endif()
    list(APPEND _mg_project_versions "${CMAKE_MATCH_1}")
  endif()
endforeach()
list(LENGTH _mg_project_versions _mg_project_version_count)
if(NOT _mg_project_version_count EQUAL 1)
  message(FATAL_ERROR
    "Expected exactly one [project].version in pyproject.toml; found "
    "${_mg_project_version_count}.")
endif()
list(GET _mg_project_versions 0 MG_PROJECT_VERSION)

# Source builds are supported only from native Apple Silicon toolchains.
# Reject Rosetta/Intel hosts and universal or x86_64 target requests before
# enabling the compilers, then make the arm64 target explicit.
if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  message(FATAL_ERROR "metal-graph targets macOS on Apple Silicon only.")
endif()
set(_mg_host_arch "${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(NOT _mg_host_arch)
  execute_process(
    COMMAND uname -m
    RESULT_VARIABLE _mg_uname_result
    OUTPUT_VARIABLE _mg_host_arch
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _mg_uname_result EQUAL 0)
    message(FATAL_ERROR "Could not determine the source-build host architecture.")
  endif()
endif()
if(NOT _mg_host_arch STREQUAL "arm64")
  message(FATAL_ERROR
    "metal-graph source builds require a native Apple Silicon (arm64) "
    "toolchain; found host architecture '${_mg_host_arch}'. "
    "If this is an Apple Silicon Mac, run CMake from a native arm64 shell.")
endif()
if(CMAKE_OSX_ARCHITECTURES AND
   NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
  message(FATAL_ERROR
    "metal-graph supports only CMAKE_OSX_ARCHITECTURES=arm64; got "
    "'${CMAKE_OSX_ARCHITECTURES}'.")
endif()
if(NOT CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING
      "metal-graph target architecture" FORCE)
endif()

# This must be set before project(): CMake initializes compiler and linker
# deployment flags while enabling languages.
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING
      "macOS floor (Metal 3)" FORCE)
endif()

project(metal_graph VERSION "${MG_PROJECT_VERSION}" LANGUAGES C CXX)

if(NOT APPLE)
  message(FATAL_ERROR "metal-graph targets macOS (Apple Silicon) only.")
endif()

include(CheckCSourceCompiles)
check_c_source_compiles(
  "
  #if !defined(__APPLE__) || !defined(__arm64__)
  #error metal-graph requires an Apple arm64 compiler target
  #endif
  int main(void) { return 0; }
  "
  MG_COMPILER_TARGETS_APPLE_ARM64)
if(NOT MG_COMPILER_TARGETS_APPLE_ARM64)
  message(FATAL_ERROR
    "The configured C compiler does not target Apple arm64. Remove "
    "x86_64/universal compiler flags and configure with "
    "-DCMAKE_OSX_ARCHITECTURES=arm64.")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(MG_BUILD_PYTHON "Build the nanobind Python module" ON)
option(MG_BUILD_TESTS "Build C++ smoke tests" ON)

set(MG_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY "${MG_GENERATED_INCLUDE_DIR}")
configure_file(
  cmake/mg_version.h.in
  "${MG_GENERATED_INCLUDE_DIR}/mg_version.h"
  @ONLY)

# ---------------------------------------------------------------------------
# Metal kernels -> default.metallib -> embedded C array
# ---------------------------------------------------------------------------
find_program(XCRUN xcrun REQUIRED)

set(MG_METAL_SOURCES
  src/kernels/spmv.metal
  src/kernels/reduce.metal
  src/kernels/scan.metal
  src/kernels/frontier.metal
  src/kernels/wcc.metal
  src/kernels/select.metal
)
set(MG_METAL_HEADERS
  src/kernels/mg_params.h
  src/kernels/mg_common_msl.h
)

set(_air_files)
foreach(_src ${MG_METAL_SOURCES})
  get_filename_component(_name ${_src} NAME_WE)
  set(_air ${CMAKE_CURRENT_BINARY_DIR}/${_name}.air)
  add_custom_command(
    OUTPUT ${_air}
    COMMAND ${XCRUN} -sdk macosx metal -std=metal3.1 -O2
            -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/${_src} -o ${_air}
    DEPENDS ${_src} ${MG_METAL_HEADERS}
    COMMENT "Compiling ${_src}"
    VERBATIM)
  list(APPEND _air_files ${_air})
endforeach()

set(MG_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/mg_default.metallib)
add_custom_command(
  OUTPUT ${MG_METALLIB}
  COMMAND ${XCRUN} -sdk macosx metallib ${_air_files} -o ${MG_METALLIB}
  DEPENDS ${_air_files}
  COMMENT "Linking mg_default.metallib"
  VERBATIM)

set(MG_METALLIB_C ${CMAKE_CURRENT_BINARY_DIR}/mg_metallib_embedded.c)
add_custom_command(
  OUTPUT ${MG_METALLIB_C}
  COMMAND ${CMAKE_COMMAND} -DIN=${MG_METALLIB} -DOUT=${MG_METALLIB_C}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetallib.cmake
  DEPENDS ${MG_METALLIB} cmake/EmbedMetallib.cmake
  COMMENT "Embedding metallib"
  VERBATIM)

# ---------------------------------------------------------------------------
# Core static library
# ---------------------------------------------------------------------------
add_library(mgcore STATIC
  src/runtime/metalcpp_impl.cpp
  src/runtime/runtime.cpp
  src/graph/build.cpp
  src/engines/spmv_engine.cpp
  src/engines/reduce_engine.cpp
  src/engines/frontier_engine.cpp
  src/engines/scan_engine.cpp
  src/algos/pagerank.cpp
  src/algos/bfs.cpp
  src/algos/khop.cpp
  src/algos/wcc.cpp
  src/algos/cpu/cpu_pagerank.cpp
  src/algos/cpu/cpu_bfs.cpp
  src/algos/cpu/cpu_khop.cpp
  src/algos/cpu/cpu_wcc.cpp
  ${MG_METALLIB_C}
)
target_include_directories(mgcore PUBLIC
  ${MG_GENERATED_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp
)
target_compile_definitions(mgcore PRIVATE METALCPP_SYMBOL_VISIBILITY_HIDDEN)
target_compile_options(mgcore PRIVATE -Wall -Wextra)
set_target_properties(mgcore PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
target_link_libraries(mgcore PUBLIC
  "-framework Metal" "-framework Foundation")

# ---------------------------------------------------------------------------
# C ABI shared library
# ---------------------------------------------------------------------------
if(NOT SKBUILD)
  add_library(metalgraph SHARED src/c_api/mg_c.cpp)
  target_link_libraries(metalgraph PRIVATE mgcore)
  set_target_properties(metalgraph PROPERTIES
    OUTPUT_NAME "metalgraph"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

  # A normal CMake install provides the stable C ABI library, header, and
  # notices. scikit-build-core wheels intentionally contain only _core.
  install(TARGETS metalgraph
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES
          include/mg.h
          "${MG_GENERATED_INCLUDE_DIR}/mg_version.h"
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES LICENSE THIRD_PARTY_NOTICES.md
          DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/metal-graph)
  install(FILES third_party/metal-cpp/LICENSE.txt
          DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/metal-graph
          RENAME metal-cpp-LICENSE.txt)
  install(FILES third_party/nanobind/LICENSE.txt
          DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/metal-graph
          RENAME nanobind-LICENSE.txt)
  install(FILES third_party/nanobind/robin_map/LICENSE.txt
          DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/metal-graph
          RENAME robin-map-LICENSE.txt)
endif()

# ---------------------------------------------------------------------------
# Python module (nanobind)
# ---------------------------------------------------------------------------
if(MG_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_PREFIX_PATH ${NB_DIR})
  find_package(nanobind CONFIG REQUIRED)

  # Interpreter-specific wheels are intentional for the current release.
  # Fail closed if a future scikit-build configuration starts requesting
  # abi3 without the corresponding release-policy change.
  if(SKBUILD_SABI_COMPONENT)
    message(FATAL_ERROR
      "Stable-ABI wheels are not enabled; build interpreter-specific wheels.")
  endif()
  nanobind_add_module(_core NB_STATIC src/python/bindings.cpp)
  target_link_libraries(_core PRIVATE mgcore)
  # Dev-loop convenience only (PYTHONPATH=python): never write into the
  # source tree during scikit-build-core wheel builds.
  if(NOT SKBUILD)
    add_custom_command(TARGET _core POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_core>
              ${CMAKE_CURRENT_SOURCE_DIR}/python/metal_graph/
      COMMENT "Copying _core into python/metal_graph/")
  endif()
  install(TARGETS _core LIBRARY DESTINATION metal_graph)
endif()

# ---------------------------------------------------------------------------
# C++ smoke tests
# ---------------------------------------------------------------------------
if(MG_BUILD_TESTS AND NOT SKBUILD)
  find_program(STRINGS strings REQUIRED)
  enable_testing()
  add_executable(mg_smoke tests/c/smoke.c)
  set_target_properties(mg_smoke PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON)
  target_include_directories(mg_smoke PRIVATE
    ${MG_GENERATED_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(mg_smoke PRIVATE metalgraph)
  add_test(NAME mg_smoke COMMAND mg_smoke)
  add_test(
    NAME mg_exports
    COMMAND ${CMAKE_COMMAND}
      -DMG_NM=${CMAKE_NM}
      -DMG_LIBRARY=$<TARGET_FILE:metalgraph>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/check_exports.cmake)
  add_test(
    NAME mg_metallib_target
    COMMAND ${CMAKE_COMMAND}
      -DMG_STRINGS=${STRINGS}
      -DMG_METALLIB=${MG_METALLIB}
      -DMG_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/check_metallib_target.cmake)
endif()
