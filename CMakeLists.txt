cmake_minimum_required(VERSION 3.16)
project(tracker)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(tracker src/main.cpp)

target_link_libraries(tracker ${OpenCV_LIBS})