set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_minimum_required(VERSION 3.24) # support arch native # 3.18
project(microbench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

list(APPEND CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-O3" "-lineinfo" "--ptxas-options=-v")

include_directories("/home/v-feiychen/attention_autotuner/third_party/cutlass/include")

# must delete CMakeCache.txt(命令行指定的参数会缓存, CMAkeLists中的不会;通过 set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA ARCH") 初始化缓存变量)
set(PROJECT_CUDA_ARCH native CACHE STRING "CUDA ARCH")
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES) # 默认定义为52
set(CMAKE_CUDA_ARCHITECTURES ${PROJECT_CUDA_ARCH})

add_executable(1_L2 1_L2.cu)

target_link_libraries(1_L2
    ${CUDA_LIBRARIES}
)
target_compile_options(1_L2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
)

# add_executable(1_L2_vec 1_L2_vec.cu)
# target_link_libraries(1_L2_vec
#     ${CUDA_LIBRARIES}
# )
# target_compile_options(1_L2_vec PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
# )

add_executable(1_L2_loadonly 1_L2_loadonly.cu)
target_link_libraries(1_L2_loadonly
    ${CUDA_LIBRARIES}
)
target_compile_options(1_L2_loadonly PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
)

add_executable(1_L2_cp_async 1_L2_cp_async.cu)
target_link_libraries(1_L2_cp_async
    ${CUDA_LIBRARIES}
)
target_compile_options(1_L2_cp_async PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
)