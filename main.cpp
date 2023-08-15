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
    DataSet trainingSet("./src/datasets/classification_data.csv","./src/datasets/classification_labels.csv");
    trainingSet.shuffle();
    NeuralNetwork nn({100}, {8,8,8}, {2}, "leakyrelu", "sigmoid", "Softmax");
    nn.randomInitialization(0);
    nn.train(trainingSet, 100, 0.01);

    DataSet trainingSet2("./src/datasets/classification_data.csv","./src/datasets/classification_labels.csv");
    std::vector<std::vector<double>> inputData = trainingSet2.getInputData();
    std::vector<std::vector<double>> inputLabels =  trainingSet2.getInputLabels();
    std::vector<double> predictions;

    for(unsigned int rowNumber = 0; rowNumber < inputData.size(); rowNumber++){
        nn.forwardPropagation(inputData[rowNumber],inputLabels[rowNumber]);
        std::vector<double> predicted = nn.getOutputValues();
        std::cout<< " proba 0 : "<<predicted[0]<< " proba 1 : " << predicted[1] <<std::endl;

    }
    
    
}