cmake_minimum_required(VERSION 3.20)
project(native_overlay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(overlay SHARED overlay.cpp)

# Define export macro for dll
if (MSVC)
  target_compile_definitions(overlay PRIVATE NATIVE_OVERLAY_EXPORTS)
endif()

# Windows SDK libs
find_library(D2D1_LIB d2d1.lib REQUIRED)
find_library(DWRITE_LIB dwrite.lib REQUIRED)
find_library(WIC_LIB windowscodecs.lib REQUIRED)

target_link_libraries(overlay PRIVATE ${D2D1_LIB} ${DWRITE_LIB} ${WIC_LIB})

target_include_directories(overlay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(overlay PROPERTIES
  OUTPUT_NAME "overlay"
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
