#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <vector> 
#include "node.hpp"
#include "edge.hpp"
#include "data_set.hpp"

class NeuralNetwork{

private:
    std::vector< std::vector< Node > > layers;
    std::string lossFunction;
    unsigned int nWeights;

public:
    NeuralNetwork(int inputLayerSize, 
                  std::vector<int> hiddenLayerSizes, 
                  int outputLayerSize,
                  std::string hiddenActivationType,
                  std::string outputActivationType, 
                  std::string lossFunction);

    void setWeights(std::vector<double> values);
    void reset();
    unsigned int getnWeights();
    std::vector<double> getWeights();
    std::vector<double> getDeltas();
    std::vector<double> getOutputValues();
    double forwardPropagation(std::vector<double> inputData, std::vector<double> inputLabels);
    void backwardPropagation();
    void train(DataSet dataSet, int epochs, double learningRate);
};



#endif