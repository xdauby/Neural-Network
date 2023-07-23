#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <vector> 
#include "node.hpp"
#include "edge.hpp"


class NeuralNetwork{

private:
    std::vector< std::vector< Node > > layers;
    std::string lossFunction;

public:
    NeuralNetwork(int inputLayerSize, std::vector<int> hiddenLayerSizes, int outputLayerSize, std::string lossFunction);
    double forwardPropagation(std::vector<double> inputData, std::vector<double> inputLabels);
    std::vector<double> getOutputValues();
    std::vector<double> getWeights();
};



#endif