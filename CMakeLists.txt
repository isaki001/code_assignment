cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mipmap_code_assignment VERSION 0.0.1 LANGUAGES CXX CUDA)
SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Turn on compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(mipmap main.cu)
set_property(TARGET mipmap PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(mipmap PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(mipmap PRIVATE "-lineinfo")