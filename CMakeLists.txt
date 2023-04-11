cmake_minimum_required(VERSION 3.22)
project(kmeansPPML CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(kmeansPPML main.cu)

set_target_properties(kmeansPPML PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)



