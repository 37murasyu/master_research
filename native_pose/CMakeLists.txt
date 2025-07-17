cmake_minimum_required(VERSION 3.20)
project(native_pose LANGUAGES CXX)

add_library(native_pose SHARED pose_landmarker.cpp)

set_target_properties(native_pose PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  OUTPUT_NAME "native_pose"
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if (MSVC)
  target_compile_options(native_pose PRIVATE /utf-8)
endif()

target_compile_definitions(native_pose PRIVATE NATIVE_POSE_EXPORTS)

# Backend selection (OFF by default)
option(NPOSE_WITH_MEDIAPIPE "Use MediaPipe Tasks C++ backend" OFF)
option(NPOSE_WITH_ONNXRUNTIME "Use ONNX Runtime backend" OFF)

if (NPOSE_WITH_MEDIAPIPE)
  target_compile_definitions(native_pose PRIVATE NPOSE_WITH_MEDIAPIPE)
  # TODO: add include/lib paths for MediaPipe when available
  # target_link_libraries(native_pose PRIVATE mediapipe::tasks_vision ...)
endif()

if (NPOSE_WITH_ONNXRUNTIME)
  target_compile_definitions(native_pose PRIVATE NPOSE_WITH_ONNXRUNTIME)
  # TODO: add include/lib paths for ONNX Runtime
  # target_include_directories(native_pose PRIVATE third_party/onnxruntime/include)
  # target_link_libraries(native_pose PRIVATE onnxruntime)
endif()

# 将来: MediaPipe / abseil / protobuf / flatbuffers / TFLite などの依存をここに追加
