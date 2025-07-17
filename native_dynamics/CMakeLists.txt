cmake_minimum_required(VERSION 3.15)
project(native_dynamics LANGUAGES CXX)

add_library(native_dynamics SHARED dynamics.cpp)
target_include_directories(native_dynamics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(native_dynamics PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(native_dynamics PRIVATE /O2 /utf-8)
  target_compile_definitions(native_dynamics PRIVATE NDYN_EXPORTS)
else()
  target_compile_options(native_dynamics PRIVATE -O3 -fPIC)
  target_compile_definitions(native_dynamics PRIVATE NDYN_EXPORTS)
endif()

set_target_properties(native_dynamics PROPERTIES
  OUTPUT_NAME native_dynamics
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/$<CONFIG>
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/$<CONFIG>
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/$<CONFIG>
)
