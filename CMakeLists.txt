cmake_minimum_required(VERSION 3.16)
project(tracker)

set(CMAKE_CXX_STANDARD 17)

# Find necessary packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED) # Locates the system's threading library

add_executable(tracker src/main.cpp UavTracker.cpp)

# Link both OpenCV and Thread libraries to the executable
target_link_libraries(tracker ${OpenCV_LIBS} Threads::Threads)