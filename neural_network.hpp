#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <array> 
#include "node.hpp"

class NeuralNetwork{

private:
    std::vector<std::vector<Node>> layers;

public:
    NeuralNetwork(int inputLayerSize, std::vector<int> hiddenLayerSizes, int outputLayerSize);

};



#endif