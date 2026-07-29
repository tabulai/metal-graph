# metal-graph — Metal-native graph analytics for Apple Silicon.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)

# This must be set before project(): CMake initializes compiler and linker
# deployment flags while enabling languages.
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING
      "macOS floor (Metal 3)" FORCE)
endif()

project(metal_graph VERSION 0.1.0 LANGUAGES C CXX)

if(NOT APPLE)
  message(FATAL_ERROR "metal-graph targets macOS (Apple Silicon) only.")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(MG_BUILD_PYTHON "Build the nanobind Python module" ON)
option(MG_BUILD_TESTS "Build C++ smoke tests" ON)

# ---------------------------------------------------------------------------
# Metal kernels -> default.metallib -> embedded C array
# ---------------------------------------------------------------------------
find_program(XCRUN xcrun REQUIRED)
find_program(STRINGS strings REQUIRED)

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
add_library(metalgraph SHARED src/c_api/mg_c.cpp)
target_link_libraries(metalgraph PRIVATE mgcore)
set_target_properties(metalgraph PROPERTIES
  OUTPUT_NAME "metalgraph"
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

# scikit-build-core packages only the Python extension below. A normal CMake
# install additionally provides the stable C ABI library, header, and notices.
if(NOT SKBUILD)
  install(TARGETS metalgraph
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES include/mg.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES LICENSE third_party/metal-cpp/LICENSE.txt
                THIRD_PARTY_NOTICES.md
          DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/metal-graph)
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

  nanobind_add_module(_core src/python/bindings.cpp)
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
if(MG_BUILD_TESTS)
  enable_testing()
  add_executable(mg_smoke tests/c/smoke.c)
  set_target_properties(mg_smoke PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON)
  target_include_directories(mg_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
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
