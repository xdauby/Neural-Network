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
    unsigned int nWeights;

public:
    NeuralNetwork(int inputLayerSize, std::vector<int> hiddenLayerSizes, int outputLayerSize, std::string lossFunction);
    void setWeights(std::vector<double> values);
    unsigned int getnWeights();
    std::vector<double> getWeights();
    std::vector<double> getOutputValues();
    double forwardPropagation(std::vector<double> inputData, std::vector<double> inputLabels);
    
};



#endif