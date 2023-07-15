#include <iostream>
#include "neural_network.hpp"
#include "edge.hpp"
#include "node.hpp"


int main()
{

    int inputLayerSize {10};
    std::vector<int> hiddenLayerSizes {2,2};
    int outputLayerSize {2};

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize);

}