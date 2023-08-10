#include <iostream>
#include "neural_network.hpp"
#include "edge.hpp"
#include "node.hpp"


int main()
{

    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2};
    int outputLayerSize {1};

    std::string lossFunction = "None";
    std::vector<double> inputData = {1.5, 0.5};
    std::vector<double> inputLabels = {1};

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "leakyrelu", "sigmoid", lossFunction);
    
    nn.forwardPropagation(inputData, inputLabels);


}