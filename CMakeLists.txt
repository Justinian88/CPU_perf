cmake_minimum_required(VERSION 3.16)

project(cpu_perf_lab1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")

add_executable(${PROJECT_NAME} main.cpp)
