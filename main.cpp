#include <iostream>
#include "neural_network.hpp"
#include "edge.hpp"
#include "node.hpp"
#include "data_set.hpp"
#include <algorithm>
#include <random>
#include <iomanip>

int main()
{
    DataSet trainingSet("./src/datasets/regression_data.csv","./src/datasets/regression_labels.csv");
    NeuralNetwork nn({3}, {50,50,10,10}, {1}, "leakyrelu", "linear", "L2 norm");
    nn.randomInitialization(0);
    nn.train(trainingSet, 10000, 0.0001);

    std::vector<std::vector<double>> inputData = trainingSet.getInputData();
    std::vector<std::vector<double>> inputLabels =  trainingSet.getInputLabels();
    std::vector<double> predictions;

    for(unsigned int rowNumber = 0; rowNumber < inputData.size(); rowNumber++){
        nn.forwardPropagation(inputData[rowNumber],inputLabels[rowNumber]);
        std::vector<double> predicted = nn.getOutputValues();
        std::cout<<predicted[0]<<std::endl;

    }
    
    
}