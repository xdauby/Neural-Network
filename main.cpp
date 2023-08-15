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
    DataSet classificationTrainingSet("./src/datasets/classification_data.csv","./src/datasets/classification_labels.csv", "classification");
    DataSet regressionTrainingSet("./src/datasets/regression_data.csv","./src/datasets/regression_labels.csv", "regression");
    classificationTrainingSet.shuffle();
    NeuralNetwork nn({3}, {8,8,8}, {2}, "leakyrelu", "linear", "L2 norm");
    nn.randomInitialization(0);
    nn.train(regressionTrainingSet, 10000, 0.00001, 0.95);

    DataSet trainingSet2("./src/datasets/classification_data.csv","./src/datasets/classification_labels.csv", "classification");
    std::vector<std::vector<double>> inputData = trainingSet2.getInputData();
    std::vector<std::vector<double>> inputLabels =  trainingSet2.getInputLabels();
    
    
}