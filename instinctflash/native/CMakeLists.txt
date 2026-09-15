cmake_minimum_required(VERSION 3.24)
project(instinctflash_sm120 LANGUAGES CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)

add_library(instinctflash_sm120 SHARED wan_residual_sm120.cu)
set_target_properties(instinctflash_sm120 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PREFIX "lib"
  OUTPUT_NAME "instinctflash_sm120"
)
target_compile_options(instinctflash_sm120 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -gencode=arch=compute_120,code=sm_120
  >
)

add_library(instinctflash_sm120_wan_stage2 SHARED wan_stage2_sm120.cu)
set_target_properties(instinctflash_sm120_wan_stage2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PREFIX "lib"
  OUTPUT_NAME "instinctflash_sm120_wan_stage2"
)
target_compile_options(instinctflash_sm120_wan_stage2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -gencode=arch=compute_120,code=sm_120
  >
)

add_library(instinctflash_sm120_wan_stage3 SHARED wan_stage3_sm120.cu)
set_target_properties(instinctflash_sm120_wan_stage3 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PREFIX "lib"
  OUTPUT_NAME "instinctflash_sm120_wan_stage3"
)
target_compile_options(instinctflash_sm120_wan_stage3 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -gencode=arch=compute_120,code=sm_120
  >
)

add_library(instinctflash_sm120_wan_qk_rope SHARED wan_qk_rope_sm120.cu)
set_target_properties(instinctflash_sm120_wan_qk_rope PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PREFIX "lib"
  OUTPUT_NAME "instinctflash_sm120_wan_qk_rope"
)
target_compile_options(instinctflash_sm120_wan_qk_rope PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -gencode=arch=compute_120,code=sm_120
  >
)

add_library(instinctflash_sm120_wan_gemm SHARED wan_gemm_sm120.cu)
set_target_properties(instinctflash_sm120_wan_gemm PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PREFIX "lib"
  OUTPUT_NAME "instinctflash_sm120_wan_gemm"
)
target_compile_options(instinctflash_sm120_wan_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -gencode=arch=compute_120,code=sm_120
  >
)
target_link_libraries(instinctflash_sm120_wan_gemm PRIVATE CUDA::cublasLt)
