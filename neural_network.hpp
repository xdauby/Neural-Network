#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <array> 
#include "node.hpp"
#include "edge.hpp"


class NeuralNetwork{

private:
    std::vector< std::vector< Node > > layers;

public:
    NeuralNetwork(int inputLayerSize, std::vector<int> hiddenLayerSizes, int outputLayerSize);
    double forwardPropagation(std::vector<double> inputData);
};



#endif