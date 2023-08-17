cmake_minimum_required(VERSION 3.25)
project("Feed Neural Network")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include_directories(include)

add_library(neural
    src/edge.cpp
    src/node.cpp
    src/neural_network.cpp
    src/data_set.cpp
)


add_executable(main main.cpp)
target_link_libraries(main neural)