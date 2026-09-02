cmake_minimum_required(VERSION 3.14)
project(DepthAnythingEngine CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimization and AVX2 SIMD hardware instructions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -mfma -pthread")

#OpenCV
find_package(OpenCV REQUIRED)

#ONNX Runtime
set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.18.0")
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_DIR}/include
)
link_directories("/home/aritra/Desktop/OptimisingDepthAnythingV2/Depth-Anything-V2/DepthAnythingCPP/openvino_env/lib64/python3.13/site-packages/onnxruntime/capi")

#Executable
add_executable(depth_pipeline
    src/main.cpp
    src/ImageProcessor.cpp
    src/DepthModel.cpp
)

target_link_libraries(depth_pipeline
    ${OpenCV_LIBS}
    onnxruntime
    pthread
)