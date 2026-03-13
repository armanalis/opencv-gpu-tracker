cmake_minimum_required(VERSION 3.16)
project(tracker)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# This tells the compiler where to look for .hpp files
include_directories(src)

# Added UavTracker.cpp to the build process
add_executable(tracker src/main.cpp src/UavTracker.cpp)

target_link_libraries(tracker ${OpenCV_LIBS})