cmake_minimum_required(VERSION 3.25)
project("Neural Network")

set(CMAKE_CXX_STANDARD 17)


add_library(neural
    edge.cpp
    neural_network.cpp
    node.cpp
)


add_executable(main main.cpp)
target_link_libraries(main neural)