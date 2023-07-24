#include <iostream>
#include <vector>
#include <cassert>
#include "neural_network.hpp"
#include "edge.hpp"
#include "node.hpp"

void weightsNumberTest()
{
    int inputLayerSize {4};
    std::vector<int> hiddenLayerSizes {3,2};
    int outputLayerSize {2};
    std::string lossFunction = "None";
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, lossFunction);    
    //expected number of weights is 33 : 4 + 4*3 + 3 + 3*2 + 2 + 2*2 + 2 = 33
    unsigned int expectednWeights = 33;
    unsigned int nWeights = nn.getnWeights();
    
    assert(expectednWeights == nWeights);
}

void getWeightsTest()
{
    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2,2};
    int outputLayerSize {1};
    std::string lossFunction = "None";
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, lossFunction);    

    std::vector<double> expectedWeights = {0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1};
    std::vector<double> weights = nn.getWeights();
    assert(expectedWeights == weights);
}

void setWeightsTest()
{
    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2,2};
    int outputLayerSize {1};
    std::string lossFunction = "None";
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, lossFunction);    

    std::vector<double> expectedWeights = {0.5, 0.25, -0.25, -1, 1, 0.5, 1.5, 1.5, 0, 8, -5, 0, 1, 3, 0, 1, 1};
    nn.setWeights(expectedWeights);
    std::vector<double> weights = nn.getWeights();
    
    assert(expectedWeights == weights);
}


int main()
{
    weightsNumberTest();
    getWeightsTest();
    setWeightsTest();

}
