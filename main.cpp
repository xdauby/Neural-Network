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
    NeuralNetwork nn({3}, {4,4,4,4}, {1}, "leakyrelu", "linear", "L2 norm");
    nn.randomInitialization(0);
    nn.train(trainingSet, 1000, 0.001);

}