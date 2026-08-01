# Standalone build for the hip_validate harness (PLAN.md §0.3).
#
# Deliberately NOT wired into drjit-core's top-level CMakeLists: this is a
# development tool, not part of the library, and keeping it standalone means it
# contributes nothing to the upstream seam we have to carry across merges (§9).
# It also builds in about a second instead of rebuilding drjit-core.
#
#   cmake -S tools/hip_validate -B build-hip -G Ninja
#   cmake --build build-hip
#
# No CUDA or ROCm packages are needed at build time: libnvrtc and libcuda are
# both resolved with dlopen at runtime, exactly as drjit-core resolves libcuda
# in src/cuda_api.cpp. The tool therefore builds anywhere and degrades to
# whichever arm is actually available on the host.

cmake_minimum_required(VERSION 3.13)
project(hip_validate CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hip_validate hip_validate.cpp)

# The preludes are read at run time, not compiled in, so the tool needs to know
# where they live. Overridable at run time with $HIP_VALIDATE_PRELUDE_DIR.
target_compile_definitions(hip_validate PRIVATE
  DRJIT_HIP_VALIDATE_PRELUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(hip_validate PRIVATE ${CMAKE_DL_LIBS})

# Emits the generated prologue so run_tests.sh can compile it for real gfx90a.
# tests/hip_prologue.cpp checks the shape; this checks it is valid HIP.
add_executable(gen_prologue gen_prologue.cpp)

if (CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  target_compile_options(hip_validate PRIVATE -Wall -Wextra)
endif()
