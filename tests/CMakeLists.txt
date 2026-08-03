enable_testing()

set(TEST_FILES basics.cpp mem.cpp loop.cpp vcall.cpp graphviz.cpp reductions.cpp array.cpp record.cpp)

add_executable(test_half half.cpp)
add_test(NAME test_half COMMAND test_half
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_half PRIVATE drjit-core)

foreach (TEST_FILE ${TEST_FILES})
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  add_executable(test_${TEST_NAME} ${TEST_NAME}.cpp test.h test.cpp ${TEST_NAME}.cpp)
  target_link_libraries(test_${TEST_NAME} PRIVATE drjit-core)
  target_compile_definitions(test_${TEST_NAME} PRIVATE -DTEST_NAME="${TEST_NAME}")
  target_compile_features(test_${TEST_NAME} PRIVATE cxx_std_11)

  if (DRJIT_ENABLE_OPTIX)
    target_compile_definitions(test_${TEST_NAME} PRIVATE -DDRJIT_ENABLE_OPTIX=1)
  endif()
  add_test(
    NAME ${TEST_NAME}
    COMMAND test_${TEST_NAME}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endforeach()

if (NOT TARGET check)
  add_custom_target(check
          ${CMAKE_COMMAND} -E echo CWD=${CMAKE_BINARY_DIR}
          COMMAND ${CMAKE_COMMAND} -E echo CMD=${CMAKE_CTEST_COMMAND} -C $<CONFIG>
          COMMAND ${CMAKE_COMMAND} -E echo ----------------------------------
          COMMAND ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1
              ${CMAKE_CTEST_COMMAND} -C $<CONFIG>
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      USES_TERMINAL
  )
endif()

set_property(TARGET test_basics PROPERTY CXX_STANDARD 17)
#set_property(TARGET test_vcall PROPERTY CXX_STANDARD 17)
set_property(TARGET test_loop PROPERTY CXX_STANDARD 17)

if (DRJIT_ENABLE_OPTIX)
 add_executable(optix_triangle optix_triangle.cpp optix_stubs.h optix_stubs.cpp)
 set_property(TARGET optix_triangle PROPERTY CXX_STANDARD 17)
 target_link_libraries(optix_triangle PRIVATE drjit-core)
endif()

if (DRJIT_ENABLE_METAL)
  # Metal analog of the OptiX `triangle` test. Native Objective-C++ (ARC).
  add_executable(metal_triangle metal_triangle.mm)
  set_property(TARGET metal_triangle PROPERTY CXX_STANDARD 17)
  set_source_files_properties(metal_triangle.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_link_libraries(metal_triangle PRIVATE drjit-core
    "-framework Foundation" "-framework Metal")
  add_test(NAME metal_triangle COMMAND metal_triangle
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if (DRJIT_ENABLE_HIP)
  # Type mapping. Pure lookup, no device needed -- runs anywhere.
  add_executable(hip_types hip_types.cpp)
  set_property(TARGET hip_types PROPERTY CXX_STANDARD 17)
  target_link_libraries(hip_types PRIVATE drjit-core)
  add_test(NAME hip_types COMMAND hip_types
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  # Does src/hip_api.h agree with a real ROCm install? Needs no AMD GPU --
  # libamdhip64.so loads and exports its symbols anywhere.
  add_executable(hip_api_abi hip_api_abi.cpp hip_api_abi_rocm.cpp)
  set_property(TARGET hip_api_abi PROPERTY CXX_STANDARD 17)
  target_link_libraries(hip_api_abi PRIVATE drjit-core ${CMAKE_DL_LIBS})
  # DRJIT_ENABLE_HIP / DRJIT_DYNAMIC_HIP are PRIVATE to the library target, so
  # the test must opt in explicitly or hip_api.h compiles away to nothing and
  # the test silently reports "backend disabled".
  target_compile_definitions(hip_api_abi PRIVATE -DDRJIT_ENABLE_HIP=1)
  if (DRJIT_DYNAMIC_HIP)
    target_compile_definitions(hip_api_abi PRIVATE -DDRJIT_DYNAMIC_HIP=1)
  endif()
  if (DEFINED ENV{ROCM_INCLUDE_PATH} AND
      EXISTS "$ENV{ROCM_INCLUDE_PATH}/hip/hip_runtime_api.h")
    target_include_directories(hip_api_abi PRIVATE "$ENV{ROCM_INCLUDE_PATH}")
    target_compile_definitions(hip_api_abi PRIVATE
      -DDRJIT_HIP_HAVE_HEADERS=1 -D__HIP_PLATFORM_AMD__=1)
    message(STATUS "Dr.Jit-Core: HIP ABI test will cross-check constants "
                   "against $ENV{ROCM_INCLUDE_PATH}")
  endif()
  add_test(NAME hip_api_abi COMMAND hip_api_abi
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  set_tests_properties(hip_api_abi PROPERTIES SKIP_RETURN_CODE 77)

  # Packed backend fields. Pure lookup/arithmetic, no device needed.
  add_executable(hip_packing hip_packing.cpp)
  set_property(TARGET hip_packing PROPERTY CXX_STANDARD 17)
  # src/malloc.h reaches into the internal hash map and hashing types.
  target_include_directories(hip_packing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../ext/robin_map/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ext/lz4)
  target_link_libraries(hip_packing PRIVATE drjit-core)
  add_test(NAME hip_packing COMMAND hip_packing
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  # Kernel prologue. Pure, no device needed.
  add_executable(hip_prologue hip_prologue.cpp)
  set_property(TARGET hip_prologue PROPERTY CXX_STANDARD 17)
  target_link_libraries(hip_prologue PRIVATE drjit-core)
  add_test(NAME hip_prologue COMMAND hip_prologue
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  # Literal materialisation. Pure, no device needed.
  add_executable(hip_literal hip_literal.cpp)
  set_property(TARGET hip_literal PROPERTY CXX_STANDARD 17)
  target_link_libraries(hip_literal PRIVATE drjit-core)
  add_test(NAME hip_literal COMMAND hip_literal
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  # Reindent pass. Pure string transform, no device and no JIT state.
  add_executable(hip_format hip_format.cpp)
  set_property(TARGET hip_format PROPERTY CXX_STANDARD 17)
  target_link_libraries(hip_format PRIVATE drjit-core)
  add_test(NAME hip_format COMMAND hip_format
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  # Phase 0a invariants: the backend is registered but inert, and adding it
  # perturbs nothing else.
  add_executable(hip_wiring hip_wiring.cpp)
  set_property(TARGET hip_wiring PROPERTY CXX_STANDARD 17)
  if (DRJIT_HIP_CUDA_SHIM)
    target_compile_definitions(hip_wiring PRIVATE -DDRJIT_HIP_CUDA_SHIM=1)
  endif()
  target_link_libraries(hip_wiring PRIVATE drjit-core)
  add_test(NAME hip_wiring COMMAND hip_wiring
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  # Phase 2 acceptance gate, written before the implementation.
  #
  # WILL_FAIL was removed when Phase 2 started working -- the tripwire fired as
  # designed. The test now guards the milestone rather than predicting it.
  add_executable(hip_codegen hip_codegen.cpp)
  set_property(TARGET hip_codegen PROPERTY CXX_STANDARD 17)
  target_link_libraries(hip_codegen PRIVATE drjit-core)
  add_test(NAME hip_codegen COMMAND hip_codegen
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  # Skips rather than fails when the backend is inert (a default Phase 0a
  # build), so an untouched configuration does not look broken.
  set_tests_properties(hip_codegen PROPERTIES SKIP_RETURN_CODE 77)

  # Phase 5: what the ray-tracing emitter actually produces. Runs twice --
  # closest-hit and shadow select different HIP-RT traversal objects, and
  # emitting the wrong one is invisible until a render comes back wrong.
  #
  # $DRJIT_HIP_TRACE_OUT makes it write the emitted source out; run_tests.sh
  # sets that and then compiles the result for real gfx90a, which is the only
  # place the HIP-RT link requirement gets checked.
  add_executable(hip_trace hip_trace.cpp)
  set_property(TARGET hip_trace PROPERTY CXX_STANDARD 17)
  target_link_libraries(hip_trace PRIVATE drjit-core)
  add_test(NAME hip_trace COMMAND hip_trace
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME hip_trace_shadow COMMAND hip_trace shadow
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  set_tests_properties(hip_trace hip_trace_shadow
                       PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_executable(microbenchmark microbenchmark.cpp)
set_property(TARGET microbenchmark PROPERTY CXX_STANDARD 17)
target_link_libraries(microbenchmark PRIVATE drjit-core)
